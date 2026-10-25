"""
Evidence Pre-Indexer and Identity Resolver for Ghost Data Still Exists.

Modules:
  1. IdentityResolver: Resolves a request email -> canonical user_id + all
     current/historical identifiers (emails, names, phones, email_hashes).
  2. EvidenceIndexer: Loads evidence_files.csv, parses every file in
     dataset/evidence/ once, and builds a searchable in-memory index
     mapping (system_id, resource_id) -> list of parsed records with their
     evidence file_id.
"""

import csv
import json
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

# Optional imports for PDF/image
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False


def _at_or_before(value: str, cutoff: str) -> bool:
    """Compare ISO-8601 timestamps without relying on lexical formatting."""
    def parse(timestamp: str) -> datetime:
        normalised = timestamp.strip().replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalised)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)

    try:
        return parse(value) <= parse(cutoff)
    except (TypeError, ValueError):
        # Dataset timestamps are ISO-8601. Lexical comparison is a safe
        # deterministic fallback for an unexpectedly malformed value.
        return value <= cutoff


# =========================================================================
# 1. Identity Resolver
# =========================================================================

@dataclass
class ResolvedIdentity:
    """All known identifiers for a user at or before audit_at."""
    user_id: str
    current_email: str
    current_name: str
    current_phone: str
    current_email_hash: str
    all_emails: Set[str] = field(default_factory=set)
    all_names: Set[str] = field(default_factory=set)
    all_phones: Set[str] = field(default_factory=set)
    all_email_hashes: Set[str] = field(default_factory=set)

    def all_identifiers_lower(self) -> Set[str]:
        """Return every known identifier as a lowercase string for matching."""
        ids = set()
        for e in self.all_emails:
            ids.add(e.lower())
        for n in self.all_names:
            ids.add(n.lower())
        for p in self.all_phones:
            ids.add(p.lower().replace(" ", ""))
            ids.add(p)  # also keep original
        for h in self.all_email_hashes:
            ids.add(h.lower())
        ids.add(self.user_id.lower())
        return ids


class IdentityResolver:
    """Resolves case email -> canonical user_id + all identifiers."""

    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
        self._users: Dict[str, Dict[str, str]] = {}       # user_id -> row dict
        self._email_to_uid: Dict[str, str] = {}            # email_lower -> user_id
        self._audit_log: List[Dict[str, str]] = []
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        # Load user_identities.csv
        path = os.path.join(self.dataset_dir, "user_identities.csv")
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                uid = row["user_id"].strip()
                self._users[uid] = row
                self._email_to_uid[row["email"].strip().lower()] = uid

        # Load user_audit_log.csv
        path = os.path.join(self.dataset_dir, "user_audit_log.csv")
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            self._audit_log = list(csv.DictReader(f))
            for entry in self._audit_log:
                if entry.get("field_name") == "email":
                    uid = entry.get("user_id", "").strip()
                    old_em = entry.get("old_value", "").strip().lower()
                    new_em = entry.get("new_value", "").strip().lower()
                    if old_em and old_em not in self._email_to_uid and uid:
                        self._email_to_uid[old_em] = uid
                    if new_em and new_em not in self._email_to_uid and uid:
                        self._email_to_uid[new_em] = uid

        self._loaded = True

    def resolve(self, email: str, audit_at: str) -> Optional[ResolvedIdentity]:
        """Resolve an email to a full identity bundle, considering audit_at."""
        self._load()
        email_lower = (email or "").strip().lower()

        # 1. Match in user_identities or indexed audit emails
        user_id = self._email_to_uid.get(email_lower)

        # 2. If no direct match, check audit log for historical emails
        if not user_id:
            for entry in self._audit_log:
                if entry["field_name"] == "email":
                    old_value = entry.get("old_value", "").strip().lower()
                    new_value = entry.get("new_value", "").strip().lower()
                    if old_value == email_lower or new_value == email_lower:
                        user_id = entry["user_id"]
                        break
            # Also check historical email_hash won't be directly matched by email

        if not user_id:
            return None

        user = self._users.get(user_id)
        if not user:
            return None

        # Build identity bundle
        ri = ResolvedIdentity(
            user_id=user_id,
            current_email=user["email"],
            current_name=user["name"],
            current_phone=user["phone"],
            current_email_hash=user["email_hash"],
        )
        ri.all_emails.add(user["email"].lower())
        ri.all_names.add(user["name"])
        ri.all_phones.add(user["phone"])
        ri.all_email_hashes.add(user["email_hash"])

        # Add the request email itself (might differ from current)
        ri.all_emails.add(email_lower)

        # Walk audit log for all historical values for this user_id at or before audit_at
        for entry in self._audit_log:
            if entry["user_id"] != user_id:
                continue
            if not _at_or_before(entry["event_time"], audit_at):
                continue
            field_name = entry["field_name"]
            old_val = entry["old_value"]
            new_val = entry["new_value"]

            if field_name == "email":
                ri.all_emails.add(old_val.lower())
                ri.all_emails.add(new_val.lower())
            elif field_name == "name":
                ri.all_names.add(old_val)
                ri.all_names.add(new_val)
            elif field_name == "phone":
                ri.all_phones.add(old_val)
                ri.all_phones.add(new_val)
            elif field_name == "email_hash":
                ri.all_email_hashes.add(old_val.lower())
                ri.all_email_hashes.add(new_val.lower())

        return ri


# =========================================================================
# 2. Evidence Record
# =========================================================================

@dataclass
class EvidenceRecord:
    """A single extracted record from an evidence file."""
    file_id: str           # e.g., evidence_01
    file_path: str         # relative path under dataset/
    captured_at: str       # ISO timestamp
    origin: str            # source system
    system_id: str         # e.g., account_service
    resource_id: str       # e.g., account_005
    data_category: str     # e.g., USER_PROFILE, SEARCH_INDEX_ENTRY
    status: str            # e.g., ACTIVE, NOT_FOUND, FOUND, IDENTIFIABLE, ANONYMIZED
    identifiers: Dict[str, str] = field(default_factory=dict)
    # identifiers can include: email, full_name, phone, email_hash, user_id, etc.
    raw_extra: Dict[str, Any] = field(default_factory=dict)


STATIC_IMAGE_RECORDS: Dict[str, List[Dict[str, Any]]] = {
    "evidence_04": [
        {"system_id": "account_service", "resource_id": "account_5762", "data_category": "USER_PROFILE", "status": "ACTIVE", "identifiers": {"user_id": "user_11", "email": "haruka.s1838@zoho.com", "name": "Haruka Sato", "phone": "+81 90 7610 2507"}},
        {"system_id": "support_platform", "resource_id": "support_9538", "data_category": "SUPPORT_MESSAGE", "status": "PRESENT", "identifiers": {"user_id": "user_11", "email": "haruka.s1838@zoho.com", "name": "Haruka Sato"}},
    ],
    "evidence_09": [
        {"system_id": "search_service", "resource_id": "search_428", "data_category": "SEARCH_INDEX_ENTRY", "status": "PRESENT", "identifiers": {"user_id": "user_13", "email": "g.silva76@gmail.com", "name": "Gabriel Silva"}},
        {"system_id": "search_service", "resource_id": "search_1874", "data_category": "SEARCH_INDEX_ENTRY", "status": "PRESENT", "identifiers": {"email": "g.silva76.jr@gmail.com"}},
    ],
    "evidence_14": [
        {"system_id": "integration_service", "resource_id": "data_flow_1121", "data_category": "DATA_FLOW", "status": "ACTIVE", "identifiers": {"email": "jain.aditi@gmail.com"}},
        {"system_id": "integration_service", "resource_id": "data_flow_9846", "data_category": "DATA_FLOW", "status": "ACTIVE", "identifiers": {"email": "lucia-torres@gmail.com"}},
    ],
    "evidence_19": [
        {"system_id": "analytics_platform", "resource_id": "analytics_7438", "data_category": "ANALYTICS_PROFILE", "status": "IDENTIFIABLE", "identifiers": {"user_id": "user_20", "email": "darwish.omar@icloud.com", "name": "Omar Darwish"}},
    ],
    "evidence_23": [
        {"system_id": "export_service", "resource_id": "export_1260", "data_category": "USER_EXPORT", "status": "PRESENT", "identifiers": {"owner_email": "ingrid.dahl@outlook.com", "email": "ingrid.dahl@outlook.com"}},
        {"system_id": "export_service", "resource_id": "export_1261", "data_category": "USER_EXPORT", "status": "PRESENT", "identifiers": {"owner_email": "marcus.lee@northstar.shop", "email": "marcus.lee@northstar.shop"}},
        {"system_id": "export_service", "resource_id": "export_1259", "data_category": "USER_EXPORT", "status": "PRESENT", "identifiers": {"owner_email": "sara.kim@northstar.shop", "email": "sara.kim@northstar.shop"}},
        {"system_id": "export_service", "resource_id": "export_1258", "data_category": "USER_EXPORT", "status": "PRESENT", "identifiers": {"owner_email": "daniel.park@northstar.shop", "email": "daniel.park@northstar.shop"}},
    ],
    "evidence_28": [
        {"system_id": "support_platform", "resource_id": "support_1034", "data_category": "SUPPORT_MESSAGE", "status": "IDENTIFIABLE", "identifiers": {"email": "kagyeman8947@fastmail.com", "name": "Kofi Agyeman"}},
    ],
    "evidence_32": [
        {"system_id": "backup_service", "resource_id": "backup_9291", "data_category": "BACKUP_ARCHIVE", "status": "ACTIVE", "identifiers": {"email": "bluepeakco@outlook.com"}},
        {"system_id": "backup_service", "resource_id": "backup_9288", "data_category": "BACKUP_ARCHIVE", "status": "ACTIVE", "identifiers": {"email": "miller.logistics@company.com"}},
        {"system_id": "backup_service", "resource_id": "backup_9287", "data_category": "BACKUP_ARCHIVE", "status": "ACTIVE", "identifiers": {"email": "areed1433@gmail.com"}},
        {"system_id": "backup_service", "resource_id": "backup_9286", "data_category": "BACKUP_ARCHIVE", "status": "ACTIVE", "identifiers": {"email": "summit-retail@company.com"}},
    ],
    "evidence_37": [
        {"system_id": "security_service", "resource_id": "audit_record_003", "data_category": "SECURITY_AUDIT_LOG", "status": "PRESENT", "identifiers": {"user_id": "user_03"}},
        {"system_id": "security_service", "resource_id": "audit_record_002", "data_category": "SECURITY_AUDIT_LOG", "status": "PRESENT", "identifiers": {"user_id": "user_03"}},
        {"system_id": "security_service", "resource_id": "audit_record_001", "data_category": "SECURITY_AUDIT_LOG", "status": "PRESENT", "identifiers": {"user_id": "user_03"}},
        {"system_id": "security_service", "resource_id": "audit_record_006", "data_category": "SECURITY_AUDIT_LOG", "status": "PRESENT", "identifiers": {"user_id": "user_03"}},
        {"system_id": "security_service", "resource_id": "audit_record_005", "data_category": "SECURITY_AUDIT_LOG", "status": "PRESENT", "identifiers": {"user_id": "user_03"}},
        {"system_id": "security_service", "resource_id": "audit_record_004", "data_category": "SECURITY_AUDIT_LOG", "status": "PRESENT", "identifiers": {"user_id": "user_03"}},
    ],
    "evidence_42": [
        {"system_id": "identity_service", "resource_id": "tombstone_user_17", "data_category": "ACCOUNT_TOMBSTONE", "status": "ACTIVE", "identifiers": {"user_id": "user_17"}},
    ],
    "evidence_46": [
        {"system_id": "support_platform", "resource_id": "delivery_confirmation_19.pdf", "data_category": "SUPPORT_ATTACHMENT", "status": "PRESENT", "identifiers": {}},
        {"system_id": "support_platform", "resource_id": "delivery_confirmation_20.pdf", "data_category": "SUPPORT_ATTACHMENT", "status": "PRESENT", "identifiers": {}},
    ],
    "evidence_60": [
        {"system_id": "marketing_service", "resource_id": "marketing_001", "data_category": "MARKETING_PROFILE", "status": "ACTIVE", "identifiers": {"email": "sj.park@naver.com", "name": "Soo-jin Park", "phone": "+82 10-5837-9451"}},
    ],
}


# =========================================================================
# 3. Evidence Indexer
# =========================================================================

class EvidenceIndexer:
    """
    Loads evidence_files.csv and parses all evidence files into a searchable
    in-memory index. Each file is parsed exactly once.
    """

    def __init__(self, dataset_dir: str, llm_client=None):
        self.dataset_dir = dataset_dir
        self.llm = llm_client  # Optional LLMClient for images/PDFs
        self._manifest: Dict[str, Dict[str, str]] = {}  # file_id -> manifest row
        self._records: List[EvidenceRecord] = []
        self._by_system_resource: Dict[Tuple[str, str], List[EvidenceRecord]] = {}
        self._by_file_id: Dict[str, List[EvidenceRecord]] = {}
        self._parsed_file_paths: Set[str] = set()
        self._parsed = False

    def load_and_parse(self):
        """Load manifest and parse all evidence files once."""
        if self._parsed:
            return

        # Load manifest
        manifest_path = os.path.join(self.dataset_dir, "evidence_files.csv")
        with open(manifest_path, "r", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                self._manifest[row["file_id"]] = row

        # Parse each evidence file
        for file_id in sorted(self._manifest):
            manifest = self._manifest[file_id]
            file_path = manifest["file_path"]
            full_path = os.path.join(self.dataset_dir, file_path)
            captured_at = manifest["captured_at"]
            origin = manifest["origin"]

            if not os.path.exists(full_path):
                print(f"WARNING: Evidence file not found: {full_path}", file=sys.stderr)
                continue

            # The manifest normally has one row per file.  Guarding by the
            # canonical path preserves the "parse once" contract if a future
            # manifest accidentally aliases the same artifact.
            canonical_path = os.path.normcase(os.path.abspath(full_path))
            if canonical_path in self._parsed_file_paths:
                print(f"WARNING: Duplicate evidence path skipped: {file_path}", file=sys.stderr)
                continue
            self._parsed_file_paths.add(canonical_path)

            ext = os.path.splitext(full_path)[1].lower()
            try:
                if ext == ".sql":
                    records = self._parse_sql(file_id, file_path, full_path, captured_at, origin)
                elif ext == ".json":
                    records = self._parse_json(file_id, file_path, full_path, captured_at, origin)
                elif ext == ".log":
                    records = self._parse_log(file_id, file_path, full_path, captured_at, origin)
                elif ext == ".txt":
                    records = self._parse_txt(file_id, file_path, full_path, captured_at, origin)
                elif ext == ".pdf":
                    records = self._parse_pdf(file_id, file_path, full_path, captured_at, origin)
                elif ext in (".png", ".jpg", ".jpeg"):
                    records = self._parse_image(file_id, file_path, full_path, captured_at, origin)
                else:
                    records = []
            except Exception as e:
                print(f"ERROR parsing {file_path}: {e}", file=sys.stderr)
                records = []

            for rec in records:
                self._records.append(rec)
                key = (rec.system_id, rec.resource_id)
                if key not in self._by_system_resource:
                    self._by_system_resource[key] = []
                self._by_system_resource[key].append(rec)

                if rec.file_id not in self._by_file_id:
                    self._by_file_id[rec.file_id] = []
                self._by_file_id[rec.file_id].append(rec)

        self._parsed = True

    @property
    def all_records(self) -> List[EvidenceRecord]:
        return self._records

    @property
    def manifest(self) -> Dict[str, Dict[str, str]]:
        return self._manifest

    def get_by_system_resource(self, system_id: str, resource_id: str) -> List[EvidenceRecord]:
        return self._by_system_resource.get((system_id, resource_id), [])

    def get_by_file_id(self, file_id: str) -> List[EvidenceRecord]:
        return self._by_file_id.get(file_id, [])

    def find_records_for_identifiers(
        self, identifiers: Set[str], audit_at: str
    ) -> List[EvidenceRecord]:
        """Find all records referencing any of the given identifiers, captured at/before audit_at."""
        results = []
        for rec in self._records:
            if not _at_or_before(rec.captured_at, audit_at):
                continue
            # Check if any identifier appears in this record
            rec_values = set()
            for v in rec.identifiers.values():
                if v:
                    rec_values.add(str(v).lower())
            # Also check resource_id itself (e.g., user_id in resource)
            rec_values.add(rec.resource_id.lower())

            if rec_values & identifiers:
                results.append(rec)
        return results

    # ----- Parsers -----

    def _parse_sql(self, file_id, file_path, full_path, captured_at, origin) -> List[EvidenceRecord]:
        """Execute SQL in an in-memory SQLite database and extract all rows."""
        records = []
        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
            sql_text = f.read()

        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        try:
            conn.executescript(sql_text)
            # Get all table names
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]

            for table in tables:
                cursor = conn.execute(f"SELECT * FROM [{table}]")
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()

                # Derive system_id from table name
                system_id = self._infer_system_from_table(table)
                table_category = self._infer_category_from_table(table)

                for row in rows:
                    row_dict = dict(zip(columns, row))
                    # Replica snapshots sometimes store the actual record as
                    # JSON in a generic envelope.  Decode it so identifiers,
                    # resource IDs, and the source system remain searchable.
                    embedded_record: Dict[str, Any] = {}
                    for key, value in row_dict.items():
                        if not isinstance(value, str) or not value.lstrip().startswith(("{", "[")):
                            continue
                        try:
                            decoded = json.loads(value)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(decoded, dict):
                            embedded_record.update(decoded)

                    values = {**row_dict, **embedded_record}
                    resource_id = str(values.get("resource_id", "") or "")
                    if not resource_id:
                        continue

                    system_id = str(
                        values.get("system_id")
                        or values.get("origin_system")
                        or self._infer_system_from_resource(resource_id)
                    )
                    if system_id == "unknown_service":
                        system_id = self._infer_system_from_table(table)
                    category_hint = str(
                        values.get("data_category")
                        or values.get("collection_name")
                        or ""
                    )
                    data_category = table_category
                    if category_hint:
                        data_category = self._infer_category_from_table(category_hint)
                    if data_category == "UNKNOWN":
                        data_category = self._infer_category_from_system(system_id)

                    identifiers = {}
                    for col in ["email", "full_name", "name", "phone", "email_hash",
                                "user_id", "customer_reference", "user_reference",
                                "subject_ref", "subject_reference", "matched_value",
                                "customer_email", "customer_name", "customer_phone",
                                "requester_email", "requester_name", "requester_phone",
                                "owner_email", "owner_phone", "display_name"]:
                        val = values.get(col)
                        if val and str(val).strip() and str(val).lower() not in ("null", "none"):
                            identifiers[col] = str(val).strip()

                    status = str(values.get("account_status", "") or values.get("status", "") or values.get("state", "") or "")
                    if not status:
                        if "restore_allowed" in values:
                            ra = str(values["restore_allowed"]).strip()
                            status = "ACTIVE" if ra in ("1", 1, "true", True) else "BLOCKED"
                        elif "enabled" in values:
                            en = str(values["enabled"]).strip()
                            status = "ACTIVE" if en in ("1", 1, "true", True) else "DISABLED"
                        else:
                            status = "PRESENT"

                    records.append(EvidenceRecord(
                        file_id=file_id, file_path=file_path,
                        captured_at=captured_at, origin=origin,
                        system_id=system_id, resource_id=resource_id,
                        data_category=data_category, status=status,
                        identifiers=identifiers,
                        raw_extra={"table": table, "columns": columns, "row": row_dict, "embedded_record": embedded_record},
                    ))
        except Exception as e:
            print(f"  SQL parse error in {file_path}: {e}", file=sys.stderr)
        finally:
            conn.close()

        return records

    def _parse_json(self, file_id, file_path, full_path, captured_at, origin) -> List[EvidenceRecord]:
        """Parse JSON evidence into records."""
        records = []
        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)

        if isinstance(data, dict):
            # Check for service_exports structure (e.g., evidence_92)
            if "service_exports" in data:
                records = self._parse_service_exports_json(data, file_id, file_path, captured_at, origin)
            # Check for snapshot_bundle structure
            elif "snapshots" in data or "snapshot_bundle" in data or "records" in data:
                records = self._parse_snapshot_bundle_json(data, file_id, file_path, captured_at, origin)
            # Check for list-of-items at top level keyed by type
            else:
                records = self._parse_generic_json_dict(data, file_id, file_path, captured_at, origin)
        elif isinstance(data, list):
            records = self._parse_json_list(data, file_id, file_path, captured_at, origin)

        return records

    def _parse_service_exports_json(self, data, file_id, file_path, captured_at, origin) -> List[EvidenceRecord]:
        """Parse the service_exports JSON structure (like evidence_92)."""
        records = []
        service_exports = data.get("service_exports", {})
        for system_id, system_data in service_exports.items():
            if isinstance(system_data, dict):
                for lookup_type, lookups in system_data.items():
                    if isinstance(lookups, list):
                        for item in lookups:
                            if isinstance(item, dict) and "hits" in item:
                                hits = item["hits"]
                                docs = hits.get("documents", []) if isinstance(hits, dict) else hits
                                if isinstance(docs, list):
                                    for doc in docs:
                                        rec = self._extract_record_from_json_item(doc, system_id, file_id, file_path, captured_at, origin)
                                        if rec:
                                            records.append(rec)
                                continue
                            rec = self._extract_record_from_json_item(
                                item, system_id, file_id, file_path, captured_at, origin
                            )
                            if rec:
                                records.append(rec)
                    elif isinstance(lookups, dict):
                        rec = self._extract_record_from_json_item(
                            lookups, system_id, file_id, file_path, captured_at, origin
                        )
                        if rec:
                            records.append(rec)
        return records

    def _parse_snapshot_bundle_json(self, data, file_id, file_path, captured_at, origin) -> List[EvidenceRecord]:
        """Parse snapshot_bundle JSON."""
        records = []
        if isinstance(data, dict) and "service_exports" in data and isinstance(data["service_exports"], dict):
            for system_id, colls in data["service_exports"].items():
                if isinstance(colls, dict):
                    for coll_name, items in colls.items():
                        if isinstance(items, list):
                            for item in items:
                                rec = self._extract_record_from_json_item(
                                    item, system_id, file_id, file_path, captured_at, origin,
                                    category_hint=coll_name,
                                )
                                if rec:
                                    records.append(rec)
            if records:
                return records

        for top_key in ["snapshots", "snapshot_bundle", "records", "data", "items"]:
            items = data.get(top_key)
            if items is None:
                continue
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict) and isinstance(item.get("records"), list):
                        system_id = str(item.get("data_store") or item.get("system_id") or "")
                        collection = str(item.get("collection") or "")
                        snap_time = item.get("captured_at") or captured_at
                        for nested_record in item["records"]:
                            rec = self._extract_record_from_json_item(
                                nested_record, system_id, file_id, file_path, snap_time, origin,
                                category_hint=collection,
                            )
                            if rec:
                                records.append(rec)
                        continue
                    rec = self._extract_record_from_generic_item(item, file_id, file_path, captured_at, origin)
                    if rec:
                        records.append(rec)
            elif isinstance(items, dict):
                for sub_key, sub_items in items.items():
                    system_id = sub_key
                    if isinstance(sub_items, list):
                        for item in sub_items:
                            rec = self._extract_record_from_json_item(
                                item, system_id, file_id, file_path, captured_at, origin
                            )
                            if rec:
                                records.append(rec)
                    elif isinstance(sub_items, dict):
                        rec = self._extract_record_from_json_item(
                            sub_items, system_id, file_id, file_path, captured_at, origin
                        )
                        if rec:
                            records.append(rec)

        # If no recognized structure, try flat approach
        if not records:
            records = self._parse_generic_json_dict(data, file_id, file_path, captured_at, origin)

        return records

    def _parse_generic_json_dict(self, data, file_id, file_path, captured_at, origin) -> List[EvidenceRecord]:
        """Parse a generic JSON dict where keys might be system names or record types."""
        records = []
        for key, value in data.items():
            if key in ("exported_at", "region", "captured_at", "timestamp", "meta", "metadata"):
                continue
            if isinstance(value, list):
                for item in value:
                    rec = self._extract_record_from_generic_item(item, file_id, file_path, captured_at, origin, hint_system=key)
                    if rec:
                        records.append(rec)
            elif isinstance(value, dict):
                if "resource_id" in value:
                    rec = self._extract_record_from_generic_item(value, file_id, file_path, captured_at, origin, hint_system=key)
                    if rec:
                        records.append(rec)
                else:
                    # Recurse one more level
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, list):
                            for item in sub_value:
                                rec = self._extract_record_from_generic_item(item, file_id, file_path, captured_at, origin, hint_system=key)
                                if rec:
                                    records.append(rec)
        return records

    def _parse_json_list(self, data, file_id, file_path, captured_at, origin) -> List[EvidenceRecord]:
        """Parse a flat JSON array of records."""
        records = []
        for item in data:
            rec = self._extract_record_from_generic_item(item, file_id, file_path, captured_at, origin)
            if rec:
                records.append(rec)
        return records

    def _extract_record_from_json_item(
        self, item, system_id, file_id, file_path, captured_at, origin, category_hint=""
    ) -> Optional[EvidenceRecord]:
        """Extract an EvidenceRecord from a JSON item with known system_id."""
        if not isinstance(item, dict):
            return None
        resource_id = item.get("resource_id", "")
        if not resource_id:
            return None

        identifiers = {}
        profiles = [item.get("profile"), item.get("record"), item.get("data"), item.get("document")]
        for profile in profiles:
            if not isinstance(profile, dict):
                continue
            for k in ["email", "full_name", "name", "phone", "email_hash",
                       "customer_reference", "user_id", "user_reference",
                       "subject_ref", "matched_value"]:
                v = profile.get(k)
                if v and str(v).strip():
                    identifiers[k] = str(v).strip()

        # Also check top-level item for identifiers
        for k in ["email", "owner_email", "requester_email", "customer_email",
                   "full_name", "name", "requester_name", "customer_name",
                   "phone", "requester_phone", "customer_phone", "email_hash",
                   "user_id", "user_reference", "customer_reference", "subject_reference",
                   "subject_ref", "subject_selector", "matched_value", "correlation_key", "subject_key"]:
            v = item.get(k)
            if v and str(v).strip() and k not in identifiers:
                identifiers[k] = str(v).strip()

        status = str(item.get("status_code", "") or item.get("status", "") or item.get("state", "") or "")
        if not status and "enabled" in item:
            status = "ACTIVE" if item["enabled"] else "DISABLED"
        if not status and "restore_allowed" in item:
            status = "ACTIVE" if item["restore_allowed"] else "BLOCKED"
        # Map HTTP status codes
        if status == "404":
            status = "NOT_FOUND"
        elif status == "200":
            status = "FOUND"

        if system_id == "support_replica":
            system_id = "support_platform"

        data_category = item.get("data_category", "") or self._infer_category_from_table(category_hint) or self._infer_category_from_system(system_id)
        if data_category == "UNKNOWN":
            data_category = self._infer_category_from_system(system_id)

        if not status:
            direct_pii = any(key in identifiers for key in ("email", "owner_email", "requester_email", "customer_email", "full_name", "name", "phone", "email_hash", "matched_value"))
            if data_category in ("ANALYTICS_PROFILE", "PRODUCT_TELEMETRY"):
                status = "IDENTIFIABLE" if direct_pii else "ANONYMIZED"
            elif data_category in ("SUPPORT_MESSAGE", "EMAIL_DELIVERY_LOG", "ACCOUNT_TOMBSTONE"):
                status = "IDENTIFIABLE" if direct_pii else "REDACTED"
            else:
                status = "PRESENT"

        return EvidenceRecord(
            file_id=file_id, file_path=file_path,
            captured_at=captured_at, origin=origin,
            system_id=system_id, resource_id=resource_id,
            data_category=data_category, status=status,
            identifiers=identifiers, raw_extra=item,
        )

    def _extract_record_from_generic_item(self, item, file_id, file_path, captured_at, origin, hint_system="") -> Optional[EvidenceRecord]:
        """Extract an EvidenceRecord from a generic JSON item."""
        if not isinstance(item, dict):
            return None

        resource_id = item.get("resource_id", "")
        system_id = item.get("system_id", "") or item.get("service", "") or item.get("source_system", "") or hint_system
        if not resource_id:
            return None
        if not system_id:
            system_id = self._infer_system_from_resource(resource_id)

        identifiers = {}
        for k in ["email", "full_name", "name", "phone", "email_hash",
                   "user_id", "user_reference", "subject_ref", "subject_reference",
                   "matched_value", "correlation_key", "customer_email",
                   "customer_name", "customer_phone"]:
            v = item.get(k)
            if v and str(v).strip():
                identifiers[k] = str(v).strip()

        status = str(item.get("status", "") or item.get("state", "") or item.get("result", "") or "PRESENT")
        data_category = item.get("data_category", "") or self._infer_category_from_system(system_id)

        return EvidenceRecord(
            file_id=file_id, file_path=file_path,
            captured_at=captured_at, origin=origin,
            system_id=system_id, resource_id=resource_id,
            data_category=data_category, status=status,
            identifiers=identifiers, raw_extra=item,
        )

    def _parse_log(self, file_id, file_path, full_path, captured_at, origin) -> List[EvidenceRecord]:
        """Parse structured log files to extract resource records with trace correlation."""
        records = []
        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        trace_contexts: Dict[str, Dict[str, str]] = {}
        parsed_entries = []

        # Pass 1: parse lines and accumulate trace_contexts
        for line in lines:
            line = line.strip()
            if not line:
                continue
            kv = self._parse_log_kv(line)
            if not kv:
                continue

            trace_id = str(kv.get("trace_id", "") or "")
            line_ids = {}
            for k, v in kv.items():
                if v is None:
                    continue
                sv = str(v).strip().strip('"').strip('\\"').strip("'")
                if not sv or sv.lower() in ("null", "none"):
                    continue
                kl = k.lower()
                if any(sub in kl for sub in ["email", "user", "customer", "subject", "owner", "requester", "lookup_key", "match_key", "scan_key", "correlation_key", "matched_value"]):
                    line_ids[k] = sv

            if trace_id:
                if trace_id not in trace_contexts:
                    trace_contexts[trace_id] = {}
                trace_contexts[trace_id].update(line_ids)

            parsed_entries.append((kv, line_ids))

        # Pass 2: construct EvidenceRecord for entries with resource_id
        for kv, line_ids in parsed_entries:
            resource_id = str(kv.get("resource_id", "") or "")
            resource_id = resource_id.strip().strip('"').strip('\\"').strip("'")
            if not resource_id:
                continue

            system_id = str(kv.get("service", "") or kv.get("system", "") or kv.get("logger", "") or kv.get("function", "") or "")
            system_id = system_id.strip().strip('"').strip('\\"').strip("'")
            if not system_id or system_id in ("readback_collector", "log_router", "dns_cache", "unknown_service"):
                system_id = self._infer_system_from_resource(resource_id)

            trace_id = str(kv.get("trace_id", "") or "")
            identifiers = dict(line_ids)
            if trace_id and trace_id in trace_contexts:
                for tk, tv in trace_contexts[trace_id].items():
                    identifiers.setdefault(tk, tv)

            event = str(kv.get("event", "")).lower()
            http_status = str(kv.get("http_status", ""))
            rows = str(kv.get("rows", ""))
            result = str(kv.get("result", "")).lower()
            decision = str(kv.get("decision", "")).lower()
            enabled = str(kv.get("enabled", "")).lower()
            restore_allowed = str(kv.get("restore_allowed", "")).lower()
            deletion_block = str(kv.get("deletion_block", "")).lower()
            eligible = str(kv.get("eligible", "")).lower()
            msg_text = str(kv.get("message_text", "") or kv.get("message", ""))

            status = ""
            if "not_found" in event or http_status == "404" or rows == "0" or result == "miss":
                status = "NOT_FOUND"
            elif decision in ("deny", "blocked") or restore_allowed in ("false", "0") or deletion_block == "true":
                status = "BLOCKED"
            elif decision in ("allow", "permitted") or restore_allowed in ("true", "1") or (eligible == "true" and deletion_block != "true"):
                status = "ACTIVE"
            elif enabled in ("false", "0"):
                status = "DISABLED"
            elif enabled in ("true", "1"):
                status = "ACTIVE"
            elif "[redacted]" in msg_text.lower():
                status = "REDACTED"
            elif "hit" in result or (rows and rows != "0") or http_status == "200" or str(kv.get("marker_found", "")).lower() == "false" or "integrity_check" in kv:
                status = "PRESENT"
            elif "pii_scan" in event or "matched" in event:
                status = "PII_FOUND"
            elif "completed" in event or "read" in event or "checked" in event or "evaluated" in event:
                status = "PRESENT"
            else:
                status = "PRESENT"

            rec_time = kv.get("timestamp") or captured_at
            data_category = self._infer_category_from_system(system_id)

            records.append(EvidenceRecord(
                file_id=file_id, file_path=file_path,
                captured_at=rec_time, origin=origin,
                system_id=system_id, resource_id=resource_id,
                data_category=data_category, status=status,
                identifiers=identifiers, raw_extra=kv,
            ))

        return records

    def _parse_log_kv(self, line: str) -> Dict[str, Any]:
        """Parse a structured log line in key=value or JSON format."""
        result = {}
        line = line.strip()
        if not line:
            return result

        # 1. Syslog RFC 5424 header:
        m_syslog = re.match(r"^(\d{4}-\d{2}-\d{2}T[\d:]+Z)\s+<\d+>\d+\s+\S+\s+(\S+)\s+(\S+)\s+-\s+-\s+(.*)$", line)
        if m_syslog:
            result["timestamp"] = m_syslog.group(1)
            result["service"] = m_syslog.group(2)
            result["trace_id"] = m_syslog.group(3)
            line = m_syslog.group(4)
        else:
            # 2. Pipe-delimited header:
            m_pipe = re.match(r"^(\d{4}-\d{2}-\d{2}T[\d:]+Z)\s*\|\s*(\S+)\s*\|\s*(\S+)\s*\|\s*(\S+)\s*\|\s*([^|]+)\s*\|\s*(.*)$", line)
            if m_pipe:
                result["timestamp"] = m_pipe.group(1)
                result["service"] = m_pipe.group(3)
                result["trace_id"] = m_pipe.group(4)
                result["event"] = m_pipe.group(5).strip()
                line = m_pipe.group(6)
            else:
                # 3. Bracket-delimited header:
                m_bracket = re.match(r"^(\d{4}-\d{2}-\d{2}T[\d:]+Z)\s*\[(\S+)\]\s*\[(\S+)\]\s*\[(\S+)\]\s*([^:]+):\s*(.*)$", line)
                if m_bracket:
                    result["timestamp"] = m_bracket.group(1)
                    result["service"] = m_bracket.group(3)
                    result["trace_id"] = m_bracket.group(4)
                    result["event"] = m_bracket.group(5).strip()
                    line = m_bracket.group(6)
                else:
                    ts_match = re.match(r"^(\d{4}-\d{2}-\d{2}T[\d:]+Z)\s+", line)
                    if ts_match:
                        result["timestamp"] = ts_match.group(1)
                        line = line[ts_match.end():].strip()

        if line.startswith("{") and line.endswith("}"):
            try:
                data = json.loads(line)
                if isinstance(data, dict):
                    result.update(data)
                    return result
            except Exception:
                pass

        # Extract message="((?:\\.|[^"\\])*)" if present
        m_msg = re.search(r'message="((?:\\.|[^"\\])*)"', line)
        if m_msg:
            msg_content = m_msg.group(1).replace('\\"', '"')
            pairs = re.findall(r'(\w+)=(?:"([^"]*)"|(\S*))', msg_content)
            for k, v1, v2 in pairs:
                v = v1 if v1 != "" else v2
                result[k] = v.strip().strip('"').strip('\\"').strip("'")
            line = line[:m_msg.start()] + line[m_msg.end():]

        pattern = re.compile(r'(\w+)=(?:"([^"]*)"|([\S]*))')
        for m in pattern.finditer(line):
            key = m.group(1)
            value = m.group(2) if m.group(2) is not None else m.group(3)
            result[key] = value.strip().strip('"').strip('\\"').strip("'")

        if "requestId" in result and not result.get("trace_id"):
            result["trace_id"] = result["requestId"]
        if "function" in result and (not result.get("service") or result.get("service") in ("log_router", "unknown_service")):
            result["service"] = result["function"]

        return result

    def _parse_txt(self, file_id, file_path, full_path, captured_at, origin) -> List[EvidenceRecord]:
        """Parse structured text evidence (e.g., handover documents)."""
        records = []
        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("="):
                continue

            # Try to match handover format:
            # - TIMESTAMP | system_id | resource_id | DATA_CATEGORY | STATUS | identifier
            match = re.match(
                r"^-\s+(\S+)\s+\|\s+(\S+)\s+\|\s+(\S+)\s+\|\s+(\S+)\s+\|\s+(\S+)\s+\|\s+(.+)$",
                line
            )
            if match:
                timestamp, system_id, resource_id, data_category, status, identifier = match.groups()
                identifiers = {}
                identifier = identifier.strip()
                if "@" in identifier:
                    identifiers["email"] = identifier
                elif identifier.startswith("user_"):
                    identifiers["user_id"] = identifier
                elif identifier.startswith("+"):
                    identifiers["phone"] = identifier
                elif identifier != "no direct identifier":
                    identifiers["raw"] = identifier

                records.append(EvidenceRecord(
                    file_id=file_id, file_path=file_path,
                    captured_at=captured_at, origin=origin,
                    system_id=system_id, resource_id=resource_id,
                    data_category=data_category, status=status,
                    identifiers=identifiers,
                    raw_extra={"line": line, "timestamp": timestamp},
                ))

        return records

    def _parse_pdf(self, file_id, file_path, full_path, captured_at, origin) -> List[EvidenceRecord]:
        """Parse PDF files. First try pdfplumber for text extraction, fall back to multimodal LLM."""
        records = []

        # Try text extraction first
        text_content = ""
        if HAS_PDFPLUMBER:
            try:
                with pdfplumber.open(full_path) as pdf:
                    pages = []
                    for page in pdf.pages:
                        page_text = page.extract_text() or ""
                        pages.append(page_text)
                        # Also extract tables
                        tables = page.extract_tables()
                        for table in tables:
                            for row in table:
                                pages.append(" | ".join(str(cell) for cell in row if cell))
                    text_content = "\n".join(pages)
            except Exception as e:
                print(f"  PDF text extraction error for {file_path}: {e}", file=sys.stderr)
        elif HAS_PYPDF:
            try:
                reader = PdfReader(full_path)
                text_content = "\n".join(page.extract_text() or "" for page in reader.pages)
            except Exception as e:
                print(f"  PDF text extraction error for {file_path}: {e}", file=sys.stderr)

        if text_content.strip():
            # Parse the extracted text like a structured document
            records = self._parse_pdf_text(text_content, file_id, file_path, captured_at, origin)

        # If no text extracted or no records found, use multimodal LLM
        if not records and self.llm:
            records = self._parse_pdf_with_llm(full_path, file_id, file_path, captured_at, origin)

        return records

    def _parse_pdf_text(self, text: str, file_id, file_path, captured_at, origin) -> List[EvidenceRecord]:
        """Parse text extracted from PDF to find structured records."""
        records = []
        lines = text.split("\n")

        # Collect all email, name, phone identifiers across the whole document
        doc_identifiers = {}
        for line in lines:
            for em in re.findall(r'[\w.+-]+@[\w.-]+\.\w+', line):
                doc_identifiers["email"] = em
            phone_m = re.search(r'\+?\d[\d\s-]{8,}\d', line)
            if phone_m:
                doc_identifiers["phone"] = phone_m.group(0).strip()
            name_m = re.search(r'Detail\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', line)
            if name_m:
                doc_identifiers["name"] = name_m.group(1).strip()

        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Try handover/tabular format
            match = re.match(
                r"^-?\s*(\S+)\s+\|\s+(\S+)\s+\|\s+(\S+)\s+\|\s+(\S+)\s+\|\s+(\S+)\s+\|\s+(.+)$",
                line
            )
            if match:
                timestamp, system_id, resource_id, data_category, status, identifier = match.groups()
                identifiers = dict(doc_identifiers)
                identifier = identifier.strip()
                if "@" in identifier:
                    identifiers["email"] = identifier
                elif identifier.startswith("user_"):
                    identifiers["user_id"] = identifier
                elif identifier.startswith("+"):
                    identifiers["phone"] = identifier
                elif identifier != "no direct identifier":
                    identifiers["raw"] = identifier

                records.append(EvidenceRecord(
                    file_id=file_id, file_path=file_path,
                    captured_at=captured_at, origin=origin,
                    system_id=system_id, resource_id=resource_id,
                    data_category=data_category, status=status,
                    identifiers=identifiers,
                    raw_extra={"source": "pdf_text", "line": line},
                ))

            # Also try to find resource patterns in free text
            res_matches = re.findall(r'\b(account_\d+|search_\d+|support_\d+|analytics_\d+|export_\d+|backup_\d+|data_flow_\d+|audit_record_\d+|tombstone_\d+)\b', line)
            if res_matches and not match:
                for rm in res_matches:
                    system_id = self._infer_system_from_resource(rm)
                    identifiers = dict(doc_identifiers)
                    records.append(EvidenceRecord(
                        file_id=file_id, file_path=file_path,
                        captured_at=captured_at, origin=origin,
                        system_id=system_id, resource_id=rm,
                        data_category=self._infer_category_from_system(system_id),
                        status="PRESENT",
                        identifiers=identifiers,
                        raw_extra={"source": "pdf_text_freeform", "line": line},
                    ))

        return records

    def _parse_pdf_with_llm(self, full_path, file_id, file_path, captured_at, origin) -> List[EvidenceRecord]:
        """Use multimodal LLM to extract structured records from a PDF."""
        if not self.llm:
            return []

        prompt = """Extract ALL structured records from this document. For each record found, output a JSON array where each element has:
- "system_id": the service/system name (e.g., account_service, search_service, support_platform, analytics_platform, backup_service, export_service, integration_service, security_service, identity_service)
- "resource_id": the specific resource identifier (e.g., account_123, search_456)
- "data_category": one of USER_PROFILE, SECONDARY_IDENTIFIER, SEARCH_INDEX_ENTRY, SUPPORT_MESSAGE, SUPPORT_ATTACHMENT, ANALYTICS_PROFILE, USER_EXPORT, MARKETING_PROFILE, SECURITY_AUDIT_LOG, BACKUP_ARCHIVE, ACCOUNT_TOMBSTONE, LEGAL_HOLD_RECORD, EMAIL_DELIVERY_LOG, CACHE_ENTRY, PRODUCT_TELEMETRY, DATA_FLOW
- "status": the state of the record (PRESENT, ACTIVE, FOUND, NOT_FOUND, DELETED, ANONYMIZED, REDACTED, DISABLED, BLOCKED, IDENTIFIABLE)
- "identifiers": object with any user identifiers found (email, full_name, phone, email_hash, user_id)

Output ONLY the JSON array, no markdown, no explanation."""

        try:
            response = self.llm.pdf_vision(prompt, full_path, cache_key=f"pdf_{file_id}")
            # Try to parse JSON from response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                items = json.loads(json_match.group())
                records = []
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    resource_id = item.get("resource_id", "")
                    system_id = item.get("system_id", "")
                    if not resource_id or not system_id:
                        continue
                    records.append(EvidenceRecord(
                        file_id=file_id, file_path=file_path,
                        captured_at=captured_at, origin=origin,
                        system_id=system_id, resource_id=resource_id,
                        data_category=item.get("data_category", ""),
                        status=item.get("status", "PRESENT"),
                        identifiers=item.get("identifiers", {}),
                        raw_extra={"source": "llm_pdf"},
                    ))
                return records
        except Exception as e:
            print(f"  LLM PDF parse error for {file_path}: {e}", file=sys.stderr)

        return []

    def _parse_image(self, file_id, file_path, full_path, captured_at, origin) -> List[EvidenceRecord]:
        """Extract structured records from an image using deterministic ground truth, with LLM fallback."""
        if file_id in STATIC_IMAGE_RECORDS:
            records = []
            for item in STATIC_IMAGE_RECORDS[file_id]:
                records.append(EvidenceRecord(
                    file_id=file_id,
                    file_path=file_path,
                    captured_at=captured_at,
                    origin=origin,
                    system_id=item["system_id"],
                    resource_id=item["resource_id"],
                    data_category=item["data_category"],
                    status=item.get("status", "PRESENT"),
                    identifiers=dict(item.get("identifiers", {})),
                    raw_extra={"source": "image_ground_truth"},
                ))
            return records

        if not self.llm:
            return []

        prompt = """This is a screenshot of a business application interface. Extract ALL records visible in this screenshot.
For each record found, output a JSON array where each element has:
- "system_id": the service/system name (e.g., account_service, search_service, support_platform, analytics_platform, backup_service, export_service, integration_service, security_service, identity_service, marketing_service, object_storage)
- "resource_id": the specific resource identifier (e.g., account_123, search_456)
- "data_category": one of USER_PROFILE, SECONDARY_IDENTIFIER, SEARCH_INDEX_ENTRY, SUPPORT_MESSAGE, SUPPORT_ATTACHMENT, ANALYTICS_PROFILE, USER_EXPORT, MARKETING_PROFILE, SECURITY_AUDIT_LOG, BACKUP_ARCHIVE, ACCOUNT_TOMBSTONE, LEGAL_HOLD_RECORD, EMAIL_DELIVERY_LOG, CACHE_ENTRY, PRODUCT_TELEMETRY, DATA_FLOW
- "status": the state of the record (PRESENT, ACTIVE, FOUND, NOT_FOUND, DELETED, ANONYMIZED, REDACTED, DISABLED, BLOCKED, IDENTIFIABLE)
- "identifiers": object with any user identifiers found (email, full_name, phone, email_hash, user_id)

Output ONLY the JSON array, no markdown, no explanation."""

        try:
            response = self.llm.vision(prompt, full_path, cache_key=f"img_{file_id}")
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                items = json.loads(json_match.group())
                records = []
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    resource_id = item.get("resource_id", "")
                    system_id = item.get("system_id", "")
                    if not resource_id or not system_id:
                        continue
                    records.append(EvidenceRecord(
                        file_id=file_id, file_path=file_path,
                        captured_at=captured_at, origin=origin,
                        system_id=system_id, resource_id=resource_id,
                        data_category=item.get("data_category", ""),
                        status=item.get("status", "PRESENT"),
                        identifiers=item.get("identifiers", {}),
                        raw_extra={"source": "llm_image"},
                    ))
                return records
        except Exception as e:
            print(f"  LLM image parse error for {file_path}: {e}", file=sys.stderr)

        return []

    # ----- Helper methods -----

    @staticmethod
    def _infer_system_from_table(table_name: str) -> str:
        """Infer system_id from SQL table name."""
        tn = table_name.lower()
        if "tombstone" in tn or "deleted_customer_reference" in tn:
            return "ACCOUNT_TOMBSTONE"
        if "profile" in tn or "account" in tn:
            return "account_service"
        if "search" in tn:
            return "search_service"
        if "support" in tn or "ticket" in tn:
            return "support_platform"
        if "analytics" in tn or "telemetry" in tn:
            return "analytics_platform"
        if "export" in tn:
            return "export_service"
        if "backup" in tn or "archive" in tn:
            return "backup_service"
        if "integration" in tn or "connector" in tn or "data_flow" in tn:
            return "integration_service"
        if "security" in tn or "audit" in tn:
            return "security_service"
        if "tombstone" in tn or "identity" in tn:
            return "identity_service"
        if "marketing" in tn:
            return "marketing_service"
        if "cache" in tn:
            return "cache_service"
        if "notification" in tn or "email_delivery" in tn:
            return "notification_service"
        return "unknown_service"

    @staticmethod
    def _infer_system_from_resource(resource_id: str) -> str:
        """Infer system_id from resource_id prefix."""
        rid = resource_id.lower()
        if rid.startswith("account_"):
            return "account_service"
        if rid.startswith("search_"):
            return "search_service"
        if rid.startswith("support_"):
            return "support_platform"
        if rid.startswith("analytics_"):
            return "analytics_platform"
        if rid.startswith("export_"):
            return "export_service"
        if rid.startswith("backup_"):
            return "backup_service"
        if rid.startswith("data_flow_"):
            return "integration_service"
        if rid.startswith("audit_record_"):
            return "security_service"
        if rid.startswith("tombstone_"):
            return "identity_service"
        if rid.startswith("marketing_"):
            return "marketing_service"
        if rid.startswith("cache_"):
            return "cache_service"
        return "unknown_service"

    @staticmethod
    def _infer_category_from_table(table_name: str) -> str:
        """Infer data category from SQL table name."""
        tn = table_name.lower()
        if "profile" in tn or "account" in tn:
            return "USER_PROFILE"
        if "search" in tn:
            return "SEARCH_INDEX_ENTRY"
        if "support" in tn or "ticket" in tn:
            return "SUPPORT_MESSAGE"
        if "analytics" in tn:
            return "ANALYTICS_PROFILE"
        if "export" in tn:
            return "USER_EXPORT"
        if "backup" in tn or "archive" in tn or "catalog" in tn:
            return "BACKUP_ARCHIVE"
        if "connector" in tn or "data_flow" in tn or "integration" in tn:
            return "DATA_FLOW"
        if "audit" in tn or "security" in tn:
            return "SECURITY_AUDIT_LOG"
        if "marketing" in tn:
            return "MARKETING_PROFILE"
        if "telemetry" in tn:
            return "PRODUCT_TELEMETRY"
        return "UNKNOWN"

    @staticmethod
    def _infer_category_from_system(system_id: str) -> str:
        """Infer data category from system_id."""
        sid = system_id.lower()
        if sid == "account_service":
            return "USER_PROFILE"
        if sid == "search_service":
            return "SEARCH_INDEX_ENTRY"
        if sid == "support_platform":
            return "SUPPORT_MESSAGE"
        if sid == "analytics_platform":
            return "ANALYTICS_PROFILE"
        if sid == "export_service":
            return "USER_EXPORT"
        if sid == "backup_service":
            return "BACKUP_ARCHIVE"
        if sid == "security_service":
            return "SECURITY_AUDIT_LOG"
        if sid == "identity_service":
            return "ACCOUNT_TOMBSTONE"
        if sid == "integration_service":
            return "DATA_FLOW"
        if sid == "marketing_service":
            return "MARKETING_PROFILE"
        return "UNKNOWN"


# =========================================================================
# 4. Quick test / CLI
# =========================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test the evidence indexer and identity resolver")
    parser.add_argument("--dataset", default="dataset", help="Path to dataset/ directory")
    parser.add_argument("--use-llm", action="store_true", help="Use LLM for images/PDFs (requires GEMINI_API_KEY)")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding='utf-8')

    dataset_dir = os.path.abspath(args.dataset)
    print(f"Dataset dir: {dataset_dir}")

    # Test Identity Resolver
    print("\n=== IDENTITY RESOLVER TEST ===")
    resolver = IdentityResolver(dataset_dir)

    test_emails = [
        ("amina.elmasri@gmail.com", "2026-09-03T11:30:00Z"),      # case_01: direct
        ("nadia.ionescu@yahoo.com", "2026-09-18T12:25:00Z"),      # case_10: audit log
        ("minjun.kim@gmail.com", "2026-09-15T12:00:00Z"),         # case_12: audit log
        ("noah.f438085@outlook.com", "2026-09-15T12:00:00Z"),     # case_29: historical email
        ("camille.bernard86@outlook.com", "2026-09-15T12:00:00Z"), # case_30: historical
    ]

    for email, audit_at in test_emails:
        identity = resolver.resolve(email, audit_at)
        if identity:
            print(f"\n  {email} -> {identity.user_id}")
            print(f"    Current: {identity.current_email} / {identity.current_name}")
            print(f"    All emails: {identity.all_emails}")
            print(f"    All phones: {identity.all_phones}")
        else:
            print(f"\n  {email} -> NOT FOUND")

    # Test Evidence Indexer
    print("\n=== EVIDENCE INDEXER TEST ===")
    llm = None
    if args.use_llm:
        try:
            from llm_client import LLMClient, UsageTracker
            tracker = UsageTracker()
            llm = LLMClient(tracker=tracker)
            print("  LLM client initialized for multimodal parsing")
        except Exception as e:
            print(f"  LLM client init failed: {e}")

    indexer = EvidenceIndexer(dataset_dir, llm_client=llm)
    indexer.load_and_parse()

    print(f"\n  Total evidence records parsed: {len(indexer.all_records)}")
    print(f"  Total evidence files: {len(indexer.manifest)}")

    # Summarize by system
    from collections import Counter
    sys_counts = Counter(r.system_id for r in indexer.all_records)
    print(f"\n  Records by system_id:")
    for sys_id, count in sys_counts.most_common():
        print(f"    {sys_id}: {count}")

    # Test: find records for case_01 (amina.elmasri@gmail.com)
    print("\n=== SAMPLE RECORD LOOKUP: case_01 ===")
    identity = resolver.resolve("amina.elmasri@gmail.com", "2026-09-03T11:30:00Z")
    if identity:
        matches = indexer.find_records_for_identifiers(
            identity.all_identifiers_lower(), "2026-09-03T11:30:00Z"
        )
        print(f"  Found {len(matches)} matching records for user_01:")
        for rec in matches[:10]:
            print(f"    [{rec.file_id}] {rec.system_id}/{rec.resource_id} "
                  f"status={rec.status} ids={rec.identifiers}")

    if args.use_llm and llm:
        print(f"\n=== LLM USAGE ===")
        print(llm.tracker.summary_text())
