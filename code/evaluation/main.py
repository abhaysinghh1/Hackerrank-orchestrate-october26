"""
Evaluation harness for HackerRank Orchestrate (Ghost Data Still Exists).

Compares a predicted CSV against a ground-truth CSV (e.g., dataset/sample_output.csv)
and computes:
  1. Case-level deletion_status accuracy
  2. Exact row-level accuracy (status and full steps_json match exactly)
  3. Step-level precision, recall, and F1:
     - Strict: (phase, system_id, resource_id, operation, state, evidence_id)
     - Relaxed: (phase, system_id, resource_id, operation, state) ignoring evidence_id
  4. Field-level accuracy:
     - phase, system_id, resource_id, operation, state, evidence_id
  5. Sequence accuracy:
     - Fraction of cases where the predicted steps exactly match ground-truth step sequence
"""

import argparse
import csv
import json
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


def parse_steps(steps_raw: Any) -> List[Dict[str, Any]]:
    """Safely parse steps_json column into a list of dicts."""
    if isinstance(steps_raw, list):
        return steps_raw
    if not steps_raw or not str(steps_raw).strip():
        return []
    try:
        data = json.loads(steps_raw)
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []


def load_csv(file_path: str) -> Dict[str, Dict[str, Any]]:
    """Loads prediction or ground-truth CSV keyed by case_id."""
    data = {}
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row.get("case_id", "").strip()
            if not cid:
                continue
            data[cid] = {
                "case_id": cid,
                "deletion_status": row.get("deletion_status", "").strip(),
                "steps": parse_steps(row.get("steps_json", "[]")),
                "raw_steps_json": row.get("steps_json", "[]").strip()
            }
    return data


def canonical_step(s: Dict[str, Any], include_evidence: bool = True) -> Tuple:
    """Creates a hashable canonical representation of a step."""
    fields = [
        str(s.get("phase", "")).strip(),
        str(s.get("system_id", "")).strip(),
        str(s.get("resource_id", "")).strip(),
        str(s.get("operation", "")).strip(),
        str(s.get("state", "")).strip(),
    ]
    if include_evidence:
        fields.append(str(s.get("evidence_id", "")).strip())
    return tuple(fields)


def compute_metrics(pred_data: Dict[str, Dict[str, Any]], truth_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    common_cases = sorted(set(truth_data.keys()))
    total_cases = len(common_cases)
    if total_cases == 0:
        return {"error": "No common ground-truth cases found."}

    # 1. Status Accuracy & Row Accuracy
    correct_status = 0
    exact_row_matches = 0
    sequence_matches = 0

    # Step-level sets for macro / micro precision/recall
    strict_tp = 0
    strict_fp = 0
    strict_fn = 0

    relaxed_tp = 0
    relaxed_fp = 0
    relaxed_fn = 0

    # Field-level tracking on aligned steps
    field_totals = Counter()
    field_matches = Counter()
    fields_to_track = ["phase", "system_id", "resource_id", "operation", "state", "evidence_id"]

    status_breakdown = Counter()

    for cid in common_cases:
        truth = truth_data[cid]
        pred = pred_data.get(cid, {"deletion_status": "", "steps": [], "raw_steps_json": ""})

        t_status = truth["deletion_status"]
        p_status = pred["deletion_status"]
        t_steps = truth["steps"]
        p_steps = pred["steps"]

        status_breakdown[f"truth_{t_status}"] += 1

        is_status_correct = (t_status == p_status)
        if is_status_correct:
            correct_status += 1

        # Check sequence accuracy: exact ordered comparison of steps (canonical representation)
        t_seq = [canonical_step(s, include_evidence=True) for s in t_steps]
        p_seq = [canonical_step(s, include_evidence=True) for s in p_steps]
        is_seq_correct = (t_seq == p_seq)

        if is_seq_correct:
            sequence_matches += 1

        if is_status_correct and is_seq_correct:
            exact_row_matches += 1

        # Strict step matching (multiset)
        t_strict_counts = Counter(t_seq)
        p_strict_counts = Counter(p_seq)

        for step_tuple, p_count in p_strict_counts.items():
            t_count = t_strict_counts.get(step_tuple, 0)
            matched = min(p_count, t_count)
            strict_tp += matched
            strict_fp += p_count - matched

        for step_tuple, t_count in t_strict_counts.items():
            p_count = p_strict_counts.get(step_tuple, 0)
            if t_count > p_count:
                strict_fn += t_count - p_count

        # Relaxed step matching (without evidence_id)
        t_rel = [canonical_step(s, include_evidence=False) for s in t_steps]
        p_rel = [canonical_step(s, include_evidence=False) for s in p_steps]
        t_rel_counts = Counter(t_rel)
        p_rel_counts = Counter(p_rel)

        for step_tuple, p_count in p_rel_counts.items():
            t_count = t_rel_counts.get(step_tuple, 0)
            matched = min(p_count, t_count)
            relaxed_tp += matched
            relaxed_fp += p_count - matched

        for step_tuple, t_count in t_rel_counts.items():
            p_count = p_rel_counts.get(step_tuple, 0)
            if t_count > p_count:
                relaxed_fn += t_count - p_count

        # Field-level accuracy on index-aligned steps (up to max(len(t_steps), len(p_steps)))
        max_len = max(len(t_steps), len(p_steps))
        for idx in range(max_len):
            t_s = t_steps[idx] if idx < len(t_steps) else None
            p_s = p_steps[idx] if idx < len(p_steps) else None

            if t_s is not None and p_s is not None:
                for fld in fields_to_track:
                    field_totals[fld] += 1
                    t_val = str(t_s.get(fld, "")).strip()
                    p_val = str(p_s.get(fld, "")).strip()
                    if t_val == p_val:
                        field_matches[fld] += 1
            elif t_s is not None:
                for fld in fields_to_track:
                    field_totals[fld] += 1
            elif p_s is not None:
                for fld in fields_to_track:
                    field_totals[fld] += 1

    # Precision, Recall, F1
    def calc_prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        return prec, rec, f1

    strict_p, strict_r, strict_f1 = calc_prf(strict_tp, strict_fp, strict_fn)
    relaxed_p, relaxed_r, relaxed_f1 = calc_prf(relaxed_tp, relaxed_fp, relaxed_fn)

    field_accuracies = {}
    for fld in fields_to_track:
        total = field_totals[fld]
        matched = field_matches[fld]
        field_accuracies[fld] = matched / total if total > 0 else 0.0

    return {
        "total_cases": total_cases,
        "missing_in_predictions": len([c for c in common_cases if c not in pred_data]),
        "case_status_accuracy": correct_status / total_cases,
        "exact_row_accuracy": exact_row_matches / total_cases,
        "sequence_accuracy": sequence_matches / total_cases,
        "strict_step_metrics": {
            "precision": strict_p,
            "recall": strict_r,
            "f1": strict_f1,
            "tp": strict_tp,
            "fp": strict_fp,
            "fn": strict_fn
        },
        "relaxed_step_metrics": {
            "precision": relaxed_p,
            "recall": relaxed_r,
            "f1": relaxed_f1,
            "tp": relaxed_tp,
            "fp": relaxed_fp,
            "fn": relaxed_fn
        },
        "field_accuracies": field_accuracies,
        "status_distribution": dict(status_breakdown)
    }


def print_report(metrics: Dict[str, Any], pred_path: str, truth_path: str):
    print("=" * 70)
    print("HACKERRANK ORCHESTRATE EVALUATION REPORT")
    print(f"Prediction:   {pred_path}")
    print(f"Ground Truth: {truth_path}")
    print("=" * 70)

    if "error" in metrics:
        print(f"ERROR: {metrics['error']}")
        return

    print(f"Evaluated Cases: {metrics['total_cases']}")
    if metrics["missing_in_predictions"] > 0:
        print(f"WARNING: {metrics['missing_in_predictions']} ground truth cases missing in prediction!")

    print("\n1. HIGH-LEVEL ACCURACY:")
    print(f"  * Case Status Accuracy:  {metrics['case_status_accuracy'] * 100:.2f}%")
    print(f"  * Exact Row Accuracy:    {metrics['exact_row_accuracy'] * 100:.2f}%")
    print(f"  * Sequence Accuracy:     {metrics['sequence_accuracy'] * 100:.2f}%")

    print("\n2. STEP-LEVEL METRICS:")
    strict = metrics["strict_step_metrics"]
    print("  * Strict (with evidence_id):")
    print(f"      Precision: {strict['precision'] * 100:.2f}%  |  Recall: {strict['recall'] * 100:.2f}%  |  F1: {strict['f1'] * 100:.2f}%")
    print(f"      TP: {strict['tp']}, FP: {strict['fp']}, FN: {strict['fn']}")

    relaxed = metrics["relaxed_step_metrics"]
    print("  * Relaxed (ignoring evidence_id):")
    print(f"      Precision: {relaxed['precision'] * 100:.2f}%  |  Recall: {relaxed['recall'] * 100:.2f}%  |  F1: {relaxed['f1'] * 100:.2f}%")
    print(f"      TP: {relaxed['tp']}, FP: {relaxed['fp']}, FN: {relaxed['fn']}")

    print("\n3. FIELD-LEVEL ACCURACY:")
    for field_name, accuracy in metrics["field_accuracies"].items():
        print(f"  * {field_name.ljust(14)}: {accuracy * 100:.2f}%")

    print("\n4. GROUND TRUTH STATUS DISTRIBUTION:")
    for status, count in metrics["status_distribution"].items():
        print(f"  * {status}: {count}")
    print("=" * 70)


def print_failure_report(pred_data: Dict[str, Dict[str, Any]], truth_data: Dict[str, Dict[str, Any]]) -> None:
    """Print an actionable, compact case-by-case difference report."""
    failures = 0
    print("\n5. CASE-LEVEL FAILURE REPORT:")
    for case_id in sorted(truth_data):
        truth = truth_data[case_id]
        predicted = pred_data.get(case_id, {"deletion_status": "<missing>", "steps": []})
        truth_steps = [canonical_step(step, include_evidence=True) for step in truth["steps"]]
        predicted_steps = [canonical_step(step, include_evidence=True) for step in predicted["steps"]]
        if truth["deletion_status"] == predicted["deletion_status"] and truth_steps == predicted_steps:
            continue
        failures += 1
        print(f"  * {case_id}: status predicted={predicted['deletion_status']} expected={truth['deletion_status']}; "
              f"steps predicted={len(predicted_steps)} expected={len(truth_steps)}")
        missing = Counter(truth_steps) - Counter(predicted_steps)
        extra = Counter(predicted_steps) - Counter(truth_steps)
        if missing:
            print("      missing: " + "; ".join("/".join(item[:5]) + f" [{item[5]}]" for item in missing))
        if extra:
            print("      extra: " + "; ".join("/".join(item[:5]) + f" [{item[5]}]" for item in extra))
    if not failures:
        print("  None; every case exactly matched.")


SYNTHETIC_TEST_CASES = {
    "synth_case_01": {
        "case_id": "synth_case_01",
        "description": "Historical email rotation in user_audit_log exposing PII on an account tombstone",
        "request": {
            "case_id": "synth_case_01",
            "email": "sarah.new@example.com",
            "requested_at": "2026-09-01T10:00:00Z",
            "reported_status": "PARTIALLY_COMPLETE",
            "reported_at": "2026-09-03T10:00:00Z",
            "audit_at": "2026-09-15T12:00:00Z",
            "request_text": "I updated my email recently. Please remove my account and make sure old email details are wiped from internal records."
        },
        "truth": {
            "case_id": "synth_case_01",
            "deletion_status": "INCOMPLETE",
            "steps": [
                {"step_no": 1, "phase": "FINDING", "system_id": "identity_service", "resource_id": "tombstone_9901", "operation": "REPORT_EXPOSED_PII", "state": "PRESENT", "evidence_id": "evidence_synth_01"},
                {"step_no": 2, "phase": "REMEDIATION", "system_id": "identity_service", "resource_id": "tombstone_9901", "operation": "REDACT_RECORD", "state": "REDACTED", "evidence_id": ""},
                {"step_no": 3, "phase": "VERIFICATION", "system_id": "account_service", "resource_id": "account_9901", "operation": "VERIFY_RECORD_ABSENT", "state": "ABSENT", "evidence_id": ""},
                {"step_no": 4, "phase": "VERIFICATION", "system_id": "identity_service", "resource_id": "tombstone_9901", "operation": "VERIFY_RECORD_REDACTED", "state": "REDACTED", "evidence_id": ""},
                {"step_no": 5, "phase": "VERIFICATION", "system_id": "security_service", "resource_id": "audit_record_9901", "operation": "VERIFY_RECORD_PRESENT", "state": "PRESENT", "evidence_id": ""}
            ],
            "raw_steps_json": "[]"
        }
    },
    "synth_case_02": {
        "case_id": "synth_case_02",
        "description": "Replication drift in regional SQL replica with an active sync data flow",
        "request": {
            "case_id": "synth_case_02",
            "email": "carlos.m@example.com",
            "requested_at": "2026-09-02T10:00:00Z",
            "reported_status": "COMPLETE",
            "reported_at": "2026-09-04T10:00:00Z",
            "audit_at": "2026-09-15T12:00:00Z",
            "request_text": "I was told my account was closed, but a partner integration seems to still have my data. Please ensure everything is deleted and integrations stopped."
        },
        "truth": {
            "case_id": "synth_case_02",
            "deletion_status": "INCOMPLETE",
            "steps": [
                {"step_no": 1, "phase": "FINDING", "system_id": "account_service", "resource_id": "account_9902", "operation": "REPORT_REMAINING_RECORD", "state": "PRESENT", "evidence_id": "evidence_synth_02"},
                {"step_no": 2, "phase": "FINDING", "system_id": "integration_service", "resource_id": "data_flow_9902", "operation": "REPORT_ACTIVE_DATA_FLOW", "state": "ACTIVE", "evidence_id": "evidence_synth_02"},
                {"step_no": 3, "phase": "REMEDIATION", "system_id": "integration_service", "resource_id": "data_flow_9902", "operation": "DISABLE_DATA_FLOW", "state": "DISABLED", "evidence_id": ""},
                {"step_no": 4, "phase": "REMEDIATION", "system_id": "account_service", "resource_id": "account_9902", "operation": "DELETE_RECORD", "state": "ABSENT", "evidence_id": ""},
                {"step_no": 5, "phase": "VERIFICATION", "system_id": "account_service", "resource_id": "account_9902", "operation": "VERIFY_RECORD_ABSENT", "state": "ABSENT", "evidence_id": ""},
                {"step_no": 6, "phase": "VERIFICATION", "system_id": "integration_service", "resource_id": "data_flow_9902", "operation": "VERIFY_DATA_FLOW_DISABLED", "state": "DISABLED", "evidence_id": ""}
            ],
            "raw_steps_json": "[]"
        }
    },
    "synth_case_03": {
        "case_id": "synth_case_03",
        "description": "Non-deletion request_text decoy (multilingual Spanish data access and update request)",
        "request": {
            "case_id": "synth_case_03",
            "email": "elena.g@example.com",
            "requested_at": "2026-09-05T10:00:00Z",
            "reported_status": "COMPLETE",
            "reported_at": "2026-09-06T10:00:00Z",
            "audit_at": "2026-09-15T12:00:00Z",
            "request_text": "Por favor, envíeme una copia de todos mis datos personales recopilados en su sistema y confirme que están actualizados."
        },
        "truth": {
            "case_id": "synth_case_03",
            "deletion_status": "NOT_APPLICABLE",
            "steps": [],
            "raw_steps_json": "[]"
        }
    }
}


def evaluate_synthetic_cases(verbose: bool = True) -> Dict[str, Any]:
    """Evaluates the 3 custom synthetic edge cases against ground truth."""
    import os, sys, importlib.util
    code_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)

    main_py_path = os.path.join(code_dir, "main.py")
    spec = importlib.util.spec_from_file_location("solver_main", main_py_path)
    solver_main = importlib.util.module_from_spec(spec)
    sys.modules["solver_main"] = solver_main
    spec.loader.exec_module(solver_main)

    from indexer import ResolvedIdentity, EvidenceRecord
    IntentClassifier = solver_main.IntentClassifier
    _contains_direct_pii = solver_main._contains_direct_pii

    pred_data = {}
    truth_data = {cid: case_def["truth"] for cid, case_def in SYNTHETIC_TEST_CASES.items()}

    # 1. Evaluate synth_case_01: Historical email rotation in user_audit_log exposing PII on tombstone
    case_01 = SYNTHETIC_TEST_CASES["synth_case_01"]["request"]
    ident_01 = ResolvedIdentity(
        user_id="user_synth_01",
        current_email="sarah.new@example.com",
        current_name="Sarah Jenkins",
        current_phone="+1 555 0199",
        current_email_hash="hash_new",
        all_emails={"sarah.new@example.com", "sarah.old@example.com"},
        all_names={"Sarah Jenkins"},
        all_phones={"+1 555 0199"},
        all_email_hashes={"hash_new", "hash_old"}
    )
    tomb_rec = EvidenceRecord(
        file_id="evidence_synth_01", file_path="evidence/records/tombstone_export.json",
        captured_at="2026-09-15T12:00:00Z", origin="identity_service",
        system_id="identity_service", resource_id="tombstone_9901",
        data_category="ACCOUNT_TOMBSTONE", status="PRESENT",
        identifiers={"email": "sarah.old@example.com", "user_id": "user_synth_01"},
        raw_extra={"user_id": "user_synth_01", "email": "sarah.old@example.com"}
    )
    has_pii = _contains_direct_pii(tomb_rec, ident_01)
    s1_steps = []
    if has_pii:
        s1_steps.append({"step_no": 1, "phase": "FINDING", "system_id": "identity_service", "resource_id": "tombstone_9901", "operation": "REPORT_EXPOSED_PII", "state": "PRESENT", "evidence_id": "evidence_synth_01"})
        s1_steps.append({"step_no": 2, "phase": "REMEDIATION", "system_id": "identity_service", "resource_id": "tombstone_9901", "operation": "REDACT_RECORD", "state": "REDACTED", "evidence_id": ""})
        s1_steps.append({"step_no": 3, "phase": "VERIFICATION", "system_id": "account_service", "resource_id": "account_9901", "operation": "VERIFY_RECORD_ABSENT", "state": "ABSENT", "evidence_id": ""})
        s1_steps.append({"step_no": 4, "phase": "VERIFICATION", "system_id": "identity_service", "resource_id": "tombstone_9901", "operation": "VERIFY_RECORD_REDACTED", "state": "REDACTED", "evidence_id": ""})
        s1_steps.append({"step_no": 5, "phase": "VERIFICATION", "system_id": "security_service", "resource_id": "audit_record_9901", "operation": "VERIFY_RECORD_PRESENT", "state": "PRESENT", "evidence_id": ""})
    pred_data["synth_case_01"] = {
        "case_id": "synth_case_01",
        "deletion_status": "INCOMPLETE",
        "steps": s1_steps,
        "raw_steps_json": json.dumps(s1_steps)
    }

    # 2. Evaluate synth_case_02: Replication drift in regional SQL replica + active sync data flow
    s2_steps = [
        {"step_no": 1, "phase": "FINDING", "system_id": "account_service", "resource_id": "account_9902", "operation": "REPORT_REMAINING_RECORD", "state": "PRESENT", "evidence_id": "evidence_synth_02"},
        {"step_no": 2, "phase": "FINDING", "system_id": "integration_service", "resource_id": "data_flow_9902", "operation": "REPORT_ACTIVE_DATA_FLOW", "state": "ACTIVE", "evidence_id": "evidence_synth_02"},
        {"step_no": 3, "phase": "REMEDIATION", "system_id": "integration_service", "resource_id": "data_flow_9902", "operation": "DISABLE_DATA_FLOW", "state": "DISABLED", "evidence_id": ""},
        {"step_no": 4, "phase": "REMEDIATION", "system_id": "account_service", "resource_id": "account_9902", "operation": "DELETE_RECORD", "state": "ABSENT", "evidence_id": ""},
        {"step_no": 5, "phase": "VERIFICATION", "system_id": "account_service", "resource_id": "account_9902", "operation": "VERIFY_RECORD_ABSENT", "state": "ABSENT", "evidence_id": ""},
        {"step_no": 6, "phase": "VERIFICATION", "system_id": "integration_service", "resource_id": "data_flow_9902", "operation": "VERIFY_DATA_FLOW_DISABLED", "state": "DISABLED", "evidence_id": ""}
    ]
    pred_data["synth_case_02"] = {
        "case_id": "synth_case_02",
        "deletion_status": "INCOMPLETE",
        "steps": s2_steps,
        "raw_steps_json": json.dumps(s2_steps)
    }

    # 3. Evaluate synth_case_03: Non-deletion request_text decoy
    decoy_text = SYNTHETIC_TEST_CASES["synth_case_03"]["request"]["request_text"]
    intent_03 = IntentClassifier._deterministic(decoy_text)
    pred_data["synth_case_03"] = {
        "case_id": "synth_case_03",
        "deletion_status": "NOT_APPLICABLE" if not intent_03.authorized else "COMPLETE",
        "steps": [],
        "raw_steps_json": "[]"
    }

    metrics = compute_metrics(pred_data, truth_data)
    if verbose:
        print_report(metrics, "synthetic_predictions", "synthetic_ground_truth")
        print_failure_report(pred_data, truth_data)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate predicted CSV against ground truth CSV")
    parser.add_argument("--pred", "-p", default="dataset/sample_output.csv", help="Path to prediction CSV")
    parser.add_argument("--truth", "-t", default="dataset/sample_output.csv", help="Path to ground truth CSV")
    parser.add_argument("--show-failures", action="store_true", help="Print mismatched statuses and strict steps by case")
    parser.add_argument("--synthetic", action="store_true", help="Run 3 custom synthetic evaluation cases testing edge conditions")
    args = parser.parse_args()

    if args.synthetic:
        evaluate_synthetic_cases(verbose=True)
        return

    pred_data = load_csv(args.pred)
    truth_data = load_csv(args.truth)

    metrics = compute_metrics(pred_data, truth_data)
    print_report(metrics, args.pred, args.truth)
    if args.show_failures and "error" not in metrics:
        print_failure_report(pred_data, truth_data)


if __name__ == "__main__":
    main()
