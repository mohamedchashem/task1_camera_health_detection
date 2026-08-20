"""SQLite event store for confirmed fault episodes.

Schema contract (approved Phase 2 spec):
- WAL mode, synchronous=NORMAL, schema versioned via ``PRAGMA user_version``;
- idempotent writes via a composite unique key; the key is
  ``camera_id|session_id|fault_type|started_frame|status``, so re-emissions
  after a restart or stream reconnect are no-ops (first write wins);
- parameterized queries only.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from pipeline.decision_engine import ConfirmedFault

_SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS confirmed_faults (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    event_key           TEXT    NOT NULL UNIQUE,
    camera_id           TEXT    NOT NULL,
    session_id          TEXT    NOT NULL,
    fault_type          TEXT    NOT NULL
                            CHECK (fault_type IN
                                ('tampering','low_light','blur','tilt')),
    status              TEXT    NOT NULL CHECK (status IN ('confirmed','cleared')),
    started_frame       INTEGER NOT NULL,
    started_time_s      REAL    NOT NULL,
    ended_frame         INTEGER,
    ended_time_s        REAL,
    peak_confidence     REAL    NOT NULL,
    window_positive_rate REAL   NOT NULL,
    recorded_at         TEXT    NOT NULL
                            DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_confirmed_faults_camera_time
    ON confirmed_faults(camera_id, started_time_s);
"""


def build_event_key(event: ConfirmedFault) -> str:
    """Idempotency key for a ConfirmedFault event."""
    return "|".join(
        (
            event.camera_id,
            event.session_id,
            event.fault_type,
            str(event.started_frame),
            event.status,
        )
    )


class EventStore:
    """Durable, idempotent event store backed by SQLite."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        version = self._conn.execute("PRAGMA user_version").fetchone()[0]
        if version == 0:
            self._conn.executescript(_SCHEMA)
            self._conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
            self._conn.commit()
        elif version != _SCHEMA_VERSION:
            raise RuntimeError(
                f"Event store schema version {version} is not supported "
                f"(expected {_SCHEMA_VERSION})."
            )

    def insert_confirmed_fault(self, event: ConfirmedFault) -> bool:
        """Persist one ConfirmedFault event.

        Idempotent: returns True when a row was inserted, False when the
        event key already exists (first write wins).
        """
        cursor = self._conn.execute(
            """
            INSERT OR IGNORE INTO confirmed_faults (
                event_key, camera_id, session_id, fault_type, status,
                started_frame, started_time_s, ended_frame, ended_time_s,
                peak_confidence, window_positive_rate
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                build_event_key(event),
                event.camera_id,
                event.session_id,
                event.fault_type,
                event.status,
                event.started_frame,
                event.started_time_s,
                event.ended_frame,
                event.ended_time_s,
                event.peak_confidence,
                event.window_positive_rate,
            ),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def query_events(
        self,
        camera_id: str | None = None,
        fault_type: str | None = None,
        since_time_s: float | None = None,
        limit: int | None = None,
    ) -> list[sqlite3.Row]:
        """Read-only filtered query of persisted events."""
        clauses: list[str] = []
        params: list[object] = []
        if camera_id is not None:
            clauses.append("camera_id = ?")
            params.append(camera_id)
        if fault_type is not None:
            clauses.append("fault_type = ?")
            params.append(fault_type)
        if since_time_s is not None:
            clauses.append("started_time_s >= ?")
            params.append(since_time_s)

        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = "SELECT * FROM confirmed_faults" + where + " ORDER BY started_time_s"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return self._conn.execute(sql, params).fetchall()

    def close(self) -> None:
        self._conn.close()
