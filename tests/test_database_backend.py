from adversariallm.io_utils.database import (
    get_all_runs,
    get_filtered_and_grouped_paths,
    get_mongodb_connection,
    get_sqlite_path,
    get_storage_backend,
)


def test_sqlite_is_the_default_backend(tmp_path, monkeypatch):
    sqlite_path = tmp_path / "metadata.sqlite3"
    monkeypatch.delenv("ADVERSARIAL_DB_BACKEND", raising=False)
    monkeypatch.setenv("ADVERSARIAL_SQLITE_PATH", str(sqlite_path))

    get_all_runs.cache_clear()
    db = get_mongodb_connection()

    assert get_storage_backend() == "sqlite"
    assert get_sqlite_path() == str(sqlite_path)
    assert sqlite_path.exists()
    assert hasattr(db, "runs")


def test_sqlite_backend_supports_existing_run_metadata_flow(tmp_path, monkeypatch):
    sqlite_path = tmp_path / "metadata.sqlite3"
    monkeypatch.delenv("ADVERSARIAL_DB_BACKEND", raising=False)
    monkeypatch.setenv("ADVERSARIAL_SQLITE_PATH", str(sqlite_path))

    get_all_runs.cache_clear()
    db = get_mongodb_connection()
    collection = db.runs

    base_doc = {
        "config": {
            "model": "demo-model",
            "attack": "gcg",
            "attack_params": {"generation_config": {"temperature": 0.0}},
        },
        "log_file": str(tmp_path / "outputs" / "2026-04-21" / "12-00-00" / "0" / "run.json"),
        "scored_by": [],
    }

    collection.replace_one({"config": base_doc["config"]}, base_doc, upsert=True)
    collection.replace_one({"config": base_doc["config"]}, base_doc, upsert=True)

    stored = collection.find()
    assert len(stored) == 1
    assert stored[0]["config"]["attack"] == "gcg"

    collection.update_many({"log_file": base_doc["log_file"]}, {"$addToSet": {"scored_by": "strong_reject"}})
    collection.update_many({"log_file": base_doc["log_file"]}, {"$addToSet": {"scored_by": "strong_reject"}})
    updated = collection.find({"log_file": base_doc["log_file"]})
    assert updated[0]["scored_by"] == ["strong_reject"]

    clone = updated[0].copy()
    clone.pop("_id", None)
    clone["log_file"] = str(tmp_path / "outputs" / "2026-04-21" / "12-05-00" / "0" / "run.json")
    collection.insert_many([clone])

    get_all_runs.cache_clear()
    grouped = get_filtered_and_grouped_paths({"attack": "gcg"}, ["model"], force_reload=True)
    assert grouped == {
        ("model=demo-model",): [
            base_doc["log_file"],
            clone["log_file"],
        ]
    }


def test_sqlite_backend_supports_find_one_for_config_lookup(tmp_path, monkeypatch):
    sqlite_path = tmp_path / "metadata.sqlite3"
    monkeypatch.delenv("ADVERSARIAL_DB_BACKEND", raising=False)
    monkeypatch.setenv("ADVERSARIAL_SQLITE_PATH", str(sqlite_path))

    get_all_runs.cache_clear()
    collection = get_mongodb_connection().runs

    doc = {
        "config": {
            "model": "demo-model",
            "dataset": "jbb_behaviors",
            "attack": "inpainting",
            "dataset_params": {"idx": 0},
        },
        "log_file": str(tmp_path / "outputs" / "2026-04-21" / "12-00-00" / "0" / "run.json"),
        "scored_by": [],
    }

    collection.replace_one({"config": doc["config"]}, doc, upsert=True)

    found = collection.find_one({"config": doc["config"]})
    assert found is not None
    assert found["log_file"] == doc["log_file"]

    missing = collection.find_one({"config": {"model": "missing-model"}})
    assert missing is None
