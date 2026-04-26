"""Storage helpers for attack result metadata.

This module now defaults to SQLite for local, zero-config metadata storage while
keeping the old MongoDB path available as an explicit opt-in backend.
"""

import glob
import json
import os
import sqlite3
from functools import lru_cache
from typing import Any, Iterable

from omegaconf import OmegaConf

try:
    from pymongo import MongoClient
    from pymongo.synchronous.database import Database as MongoDatabase
except ImportError:  # pragma: no cover - optional at runtime
    MongoClient = None
    MongoDatabase = Any

from .data_analysis import get_nested_value, normalize_value_for_grouping


DEFAULT_SQLITE_PATH = os.path.abspath(
    os.environ.get("ADVERSARIAL_SQLITE_PATH", os.path.join("outputs", "runs.sqlite3"))
)


def get_storage_backend() -> str:
    """Return the configured metadata backend.

    SQLite is the default because it requires no external service. MongoDB
    remains available via ``ADVERSARIAL_DB_BACKEND=mongodb``.
    """
    return os.environ.get("ADVERSARIAL_DB_BACKEND", "sqlite").strip().lower()


def get_sqlite_path() -> str:
    """Return the path of the SQLite metadata database."""
    return os.path.abspath(os.environ.get("ADVERSARIAL_SQLITE_PATH", DEFAULT_SQLITE_PATH))


def _matches_filter(document: dict[str, Any], filter_query: dict[str, Any] | None) -> bool:
    if not filter_query:
        return True
    for key, value in filter_query.items():
        if key not in document or not check_match(document[key], value):
            return False
    return True


class SQLiteRunsCollection:
    """Minimal Mongo-like collection wrapper backed by SQLite."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        parent = os.path.dirname(self.db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    config_json TEXT NOT NULL,
                    log_file TEXT NOT NULL,
                    scored_by_json TEXT NOT NULL DEFAULT '[]'
                )
                """
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _row_to_doc(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "_id": row["id"],
            "config": json.loads(row["config_json"]),
            "log_file": row["log_file"],
            "scored_by": json.loads(row["scored_by_json"]),
        }

    @staticmethod
    def _serialize_doc(doc: dict[str, Any]) -> tuple[str, str, str]:
        return (
            json.dumps(doc["config"], sort_keys=True),
            doc["log_file"],
            json.dumps(doc.get("scored_by", []), sort_keys=True),
        )

    def find(
        self,
        filter_query: dict[str, Any] | None = None,
        projection: dict[str, int] | None = None,
    ) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT id, config_json, log_file, scored_by_json FROM runs").fetchall()
        documents = [self._row_to_doc(row) for row in rows]
        matched = [doc for doc in documents if _matches_filter(doc, filter_query)]
        if projection is None:
            return matched

        include_keys = {key for key, include in projection.items() if include}
        projected = []
        for doc in matched:
            projected.append({key: value for key, value in doc.items() if key in include_keys})
        return projected

    def find_one(
        self,
        filter_query: dict[str, Any] | None = None,
        projection: dict[str, int] | None = None,
    ) -> dict[str, Any] | None:
        matches = self.find(filter_query, projection)
        return matches[0] if matches else None

    def replace_one(self, filter_query: dict[str, Any], replacement: dict[str, Any], upsert: bool = False) -> None:
        existing = self.find(filter_query)
        serialized = self._serialize_doc(replacement)
        with self._connect() as conn:
            if existing:
                conn.execute(
                    "UPDATE runs SET config_json = ?, log_file = ?, scored_by_json = ? WHERE id = ?",
                    (*serialized, existing[0]["_id"]),
                )
            elif upsert:
                conn.execute(
                    "INSERT INTO runs (config_json, log_file, scored_by_json) VALUES (?, ?, ?)",
                    serialized,
                )
            conn.commit()

    def delete_one(self, filter_query: dict[str, Any]) -> None:
        existing = self.find(filter_query)
        if not existing:
            return
        with self._connect() as conn:
            conn.execute("DELETE FROM runs WHERE id = ?", (existing[0]["_id"],))
            conn.commit()

    def insert_many(self, documents: list[dict[str, Any]]) -> None:
        serialized = [self._serialize_doc(doc) for doc in documents]
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO runs (config_json, log_file, scored_by_json) VALUES (?, ?, ?)",
                serialized,
            )
            conn.commit()

    def update_many(self, filter_query: dict[str, Any], update_query: dict[str, Any]) -> None:
        add_to_set = update_query.get("$addToSet", {})
        if set(add_to_set) - {"scored_by"}:
            raise NotImplementedError("SQLite backend currently supports $addToSet only for scored_by")

        existing = self.find(filter_query)
        if not existing:
            return

        with self._connect() as conn:
            for doc in existing:
                scored_by = list(doc.get("scored_by", []))
                value = add_to_set.get("scored_by")
                if value is not None and value not in scored_by:
                    scored_by.append(value)
                    conn.execute(
                        "UPDATE runs SET scored_by_json = ? WHERE id = ?",
                        (json.dumps(scored_by, sort_keys=True), doc["_id"]),
                    )
            conn.commit()


class SQLiteDatabase:
    """Container exposing a ``runs`` collection like the MongoDB client."""

    def __init__(self, db_path: str):
        self.runs = SQLiteRunsCollection(db_path)


def get_mongodb_connection() -> SQLiteDatabase | MongoDatabase:
    """Return the configured metadata connection.

    The historic function name is retained for compatibility with the rest of
    the codebase.
    """
    backend = get_storage_backend()
    if backend == "sqlite":
        return SQLiteDatabase(get_sqlite_path())
    if backend != "mongodb":
        raise ValueError(f"Unsupported ADVERSARIAL_DB_BACKEND={backend!r}. Expected 'sqlite' or 'mongodb'.")

    if MongoClient is None:
        raise ImportError("pymongo is required when ADVERSARIAL_DB_BACKEND=mongodb")

    user = os.environ.get("MONGODB_USER")
    password = os.environ.get("MONGODB_PASSWORD")
    host = os.environ.get("MONGODB_HOST")
    mongo_uri = os.environ.get("MONGODB_URI", f"mongodb://{user}:{password}@{host}?authSource={user}")
    client = MongoClient(mongo_uri)
    db_name = os.environ.get("MONGODB_DB", user)
    return client[db_name]


def log_config_to_db(run_config, result, log_file):
    db = get_mongodb_connection()
    collection = db.runs

    idx = run_config.dataset_params.idx
    if idx is None:
        idx = [i for i in range(len(result.runs))]
    elif isinstance(idx, int):
        idx = [idx]

    for i in idx:
        run_config.dataset_params.idx = i
        config_data = {
            "config": OmegaConf.to_container(OmegaConf.structured(run_config), resolve=True),
            "log_file": log_file,
            "scored_by": []
        }
        # If a run with the same config already exists, replace it
        collection.replace_one(
            {"config": config_data["config"]},
            config_data,
            upsert=True
        )


def delete_orphaned_runs(dry_run: bool = False, direction: str = "both"):
    """
    Delete orphaned runs from the database.

    Args:
        dry_run: If True, only print what would be deleted without actually deleting
        direction: "db_only" - remove DB entries for missing files
                  "files_only" - remove files not tracked in DB
                  "both" - do both operations
    """
    db = get_mongodb_connection()

    if direction in ["db_only", "both"]:
        # Original functionality: remove DB entries for missing files
        items = db.runs.find()
        for item in items:
            log_file = item["log_file"]
            if not os.path.exists(log_file):
                print(f"Log file not found: {log_file}, deleting from database")
                if not dry_run:
                    db.runs.delete_one({"_id": item["_id"]})

    if direction in ["files_only", "both"]:
        tracked_files = set()
        items = db.runs.find({}, {"log_file": 1})  # Only fetch log_file field
        for item in items:
            tracked_files.add(item["log_file"])

        all_run_files = set(glob.glob("outputs/**/run.json", recursive=True))
        all_run_files_absolute = set(os.path.abspath(f) for f in all_run_files)

        # Find untracked files
        untracked_files = all_run_files_absolute - tracked_files
        print(f"Found {len(untracked_files)} untracked files")

        for untracked_file in sorted(list(untracked_files)):
            print(f"Untracked run file found: {untracked_file}")
            if not dry_run:
                print(f"Deleting untracked file: {untracked_file}")
                os.remove(untracked_file)


def check_match(doc_fragment, filter_fragment):
    """
    Recursively checks whether ``doc_fragment`` satisfies ``filter_fragment``.

    Supported filter types
    ----------------------
    * **primitive** (str/int/float/bool/None)  - exact equality
    * **iterable**  (list/tuple/set)           - *any* element of the
      iterable must match  ("gcg **or** autodan", …)
      If the document side is itself an iterable, **intersection ≥ 1** counts
      as a hit.
    * **dict**                                 - every key in the filter dict
      must be present in the document and its value must match recursively
      (this is the behaviour you already had).

    Examples
    --------
    filter_by = {
        "attack": ["gcg", "autodan"],           # <- match-any
    }
    """
    # --- 1. dict → recurse over its keys --------------------------------------
    if isinstance(filter_fragment, dict):
        if not isinstance(doc_fragment, dict):
            return False
        for k, v in filter_fragment.items():
            if k not in doc_fragment or not check_match(doc_fragment[k], v):
                return False
        return True  # every key matched

    # --- 2. iterable → "any of these values is fine" --------------------------
    if isinstance(filter_fragment, (list, tuple, set)):
        # doc side is also iterable  →  true if the two are the same
        if isinstance(doc_fragment, (list, tuple, set)):
            return filter_fragment == doc_fragment
            # return any(item in filter_fragment for item in doc_fragment)
        # doc side is a single value  →  true if it is one of the allowed ones
        return doc_fragment in filter_fragment

    # --- 3. primitive equality ------------------------------------------------
    return doc_fragment == filter_fragment


@lru_cache
def get_all_runs() -> list[dict]:
    """
    Retrieves all runs from the database.
    """
    db = get_mongodb_connection()
    collection = db.runs
    return list(collection.find())


def get_filtered_and_grouped_paths(filter_by: dict, group_by: Iterable[str]|None = None, force_reload: bool = True) -> dict[tuple[str], list[str]]:
    """
    Retrieves log paths from MongoDB filtered by criteria and grouped according to group_by.

    Args:
        filter_by (dict): Filtering criteria. Can contain nested dictionaries.
        group_by (list or tuple): List/tuple of keys to group by from the 'config' field.
        force_reload (bool), default True: If True, reload the cached runs. Setting to False is useful for speeding up repeated calls without adding new runs.

    Returns:
        dict: A dictionary where keys are group identifiers (tuples of strings)
              and values are lists of log paths.
    """
    # Connect to MongoDB
    if force_reload:
        get_all_runs.cache_clear()
    all_results = get_all_runs()

    # Filter in Python using the check_match helper for complex nested conditions
    if filter_by:
        filtered_results = [
            doc for doc in all_results
            if check_match(doc['config'], filter_by)
        ]
    else:
        filtered_results = all_results

    # --- Grouping ---
    if not group_by:
        return {("all",): [r["log_file"] for r in filtered_results if "log_file" in r]}

    grouped_results = {}
    for result in filtered_results:
        # Ensure the result has 'config' and 'log_file' before processing
        if "config" not in result or "log_file" not in result:
            continue  # Skip records missing essential fields

        config_data = result["config"]
        log_path = result["log_file"]

        # Create a group key based on the specified group_by fields
        group_key_parts = []
        for key_spec in group_by:
            if isinstance(key_spec, str):
                value = get_nested_value(config_data, [key_spec])
                normalized_value = normalize_value_for_grouping(value)
                group_key_parts.append(f"{key_spec}={normalized_value}")
            elif isinstance(key_spec, (list, tuple)):
                value = get_nested_value(config_data, key_spec)
                normalized_value = normalize_value_for_grouping(value)
                key_name = '.'.join(map(str, key_spec))
                group_key_parts.append(f"{key_name}={normalized_value}")
            else:
                group_key_parts.append(f"invalid_group_spec={key_spec}")

        # Use a tuple of sorted key parts for consistent group keys
        group_key_tuple = tuple(sorted(group_key_parts))

        # Add the log path to the appropriate group
        if group_key_tuple not in grouped_results:
            grouped_results[group_key_tuple] = []
        grouped_results[group_key_tuple].append(log_path)
    return grouped_results
