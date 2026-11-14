"""Tiny SQLite key-value cache for searches and fetched pages."""
import json
import sqlite3
import threading


class Cache:
    def __init__(self, path: str):
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS kv (ns TEXT, key TEXT, value TEXT, PRIMARY KEY (ns, key))"
        )

    def get(self, ns: str, key: str):
        with self._lock:
            row = self._db.execute(
                "SELECT value FROM kv WHERE ns=? AND key=?", (ns, key)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def set(self, ns: str, key: str, value) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO kv VALUES (?,?,?)", (ns, key, json.dumps(value))
            )
            self._db.commit()
