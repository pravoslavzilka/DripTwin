"""Infusion log. Tiger Data (TimescaleDB) when DATABASE_URL is set, SQLite otherwise.

Relational and time-series data live side by side in one database:
  infusions    one row per order (rate, drop set, fluid, start/end, status)
  calibration  drop volume per drip rate, and the valve angle -> drop rate table
  events       alarms and agent actions
  drops        one row per detected drop                  (hypertable)
  telemetry    about one row per second from the twin     (hypertable)
Continuous aggregates, compression and the agent's read-only queries are in
timeseries.py and only exist on Timescale.

Writes are queued and done by a background thread: a slow or dropped database
connection must never stall the control loop. If Timescale can't be reached at
startup we fall back to SQLite; if it fails mid-run we keep retrying and drop
rows once the queue is full rather than block.

Each order gets its own infusion_id (a new one starts on the twin's
"order_started" event), so one server run can hold several infusions.

    python -m driptwin.storage                 list infusions
    python -m driptwin.storage show <id>       summary + events of one infusion
    python -m driptwin.storage csv <id> [out]  export its telemetry
"""

from __future__ import annotations

import csv
import json
import math
import os
import queue
import sqlite3
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "driptwin_log.db"
VALVE_CAL = ROOT / "valve_cal.json"

# snapshot key -> SQL type. valve_u is the controller output (0 = wide open,
# 1 = shut), valve_pos what the device reports back, clamp_r the model's clamp
# resistance at valve_u, remaining_ml the estimated volume left in the bag.
TELEMETRY_COLS = {
    "rate_gtt_min": "REAL", "target_gtt_min": "REAL", "rate_ml_h": "REAL",
    "valve_u": "REAL", "valve_pos": "REAL", "clamp_r": "REAL",
    "head_cm": "REAL", "head_baseline_cm": "REAL", "infused_ml": "REAL",
    "remaining_ml": "REAL", "drops_counted": "INTEGER", "rate_confident": "BOOLEAN",
    "mode": "TEXT",
}
INFUSION_COLS = ["rate_ml_h", "drop_factor", "vtbi_ml", "bag_volume_ml", "drug"]
# twin event -> infusion status
STATUS = {"complete": "complete", "paused": "paused", "resumed": "running",
          "occlusion": "alarm: occlusion", "empty_bag": "alarm: empty bag",
          "free_flow": "alarm: free flow"}


def _schema(ts: str, js: str) -> list[str]:
    return [
        f"""CREATE TABLE IF NOT EXISTS telemetry (time {ts} NOT NULL, infusion_id TEXT,
            rate_gtt_min REAL, target_gtt_min REAL, valve_u REAL, head_cm REAL,
            infused_ml REAL, mode TEXT)""",
        f"""CREATE TABLE IF NOT EXISTS events (time {ts} NOT NULL, infusion_id TEXT, kind TEXT,
            severity TEXT, message TEXT, data {js})""",
        f"""CREATE TABLE IF NOT EXISTS drops (time {ts} NOT NULL, infusion_id TEXT,
            interval_s REAL, volume_ml REAL, rate_gtt_min REAL, confident BOOLEAN)""",
        f"""CREATE TABLE IF NOT EXISTS infusions (infusion_id TEXT PRIMARY KEY,
            started {ts} NOT NULL, ended {ts}, rate_ml_h REAL, drop_factor INTEGER,
            vtbi_ml REAL, bag_volume_ml REAL, drug TEXT, target_gtt_min REAL, status TEXT)""",
        f"""CREATE TABLE IF NOT EXISTS calibration (measured_at {ts}, source TEXT,
            drop_factor INTEGER, gtt_min REAL, ml_per_drop REAL, valve_angle REAL)""",
        "CREATE INDEX IF NOT EXISTS telemetry_inf ON telemetry (infusion_id, time)",
        "CREATE INDEX IF NOT EXISTS events_inf ON events (infusion_id, time)",
        "CREATE INDEX IF NOT EXISTS drops_inf ON drops (infusion_id, time)",
    ]


def new_infusion_id() -> str:
    return time.strftime("inf-%Y%m%d-%H%M%S")


def calibration_rows() -> list[tuple]:
    """(measured_at, source, drop_factor, gtt_min, ml_per_drop, valve_angle).

    drop_curve: the drop size fitted to our bench measurements, sampled across
    the usable rate range. valve_cal: the measured valve angle -> drop rate table."""
    from .physics import DROP_CURVES
    rows = [(time.time(), "drop_curve_fit", df, g, round(c.volume_ml(g / 60), 5), None)
            for df, c in DROP_CURVES.items() for g in range(10, 201, 10)]
    if VALVE_CAL.exists():
        cal = json.loads(VALVE_CAL.read_text())
        try:
            t = time.mktime(time.strptime(cal.get("measured", ""), "%Y-%m-%d %H:%M"))
        except ValueError:
            t = VALVE_CAL.stat().st_mtime
        rows += [(t, "valve_cal", cal.get("drop_factor"), r["gtt_min"],
                  round(r["ml_h"] / 60 / r["gtt_min"], 5) if r["gtt_min"] else None,
                  r["angle"]) for r in cal["rows"]]
    return rows


class Store:
    def __init__(self, url: str | None = None, path: Path | str = DB_PATH,
                 max_queue: int = 5000, physics=None, calibrate: bool = True):
        self.url = url if url is not None else os.environ.get("DATABASE_URL")
        self.path = Path(path)
        self.physics = physics       # for clamp resistance; None = the generic model
        self.infusion_id = new_infusion_id()
        self.dropped = 0          # rows lost because the queue was full
        self.last_error: str | None = None
        self._last_drop_t: float | None = None
        self.kind = "sqlite"
        self.conn = None
        if self.url:
            try:
                self.conn = self._connect_pg()
                self.kind = "timescale"
            except Exception as e:
                self.last_error = f"{type(e).__name__}: {e}"
                print(f"[storage] Timescale unreachable ({self.last_error}); using SQLite")
        if self.conn is None:
            self.conn = self._connect_sqlite()
        self._q: queue.Queue = queue.Queue(maxsize=max_queue)
        self._lock = threading.Lock()   # guards self.conn for reads from other threads
        self._writer = threading.Thread(target=self._write_loop, name="store-writer",
                                        daemon=True)
        self._writer.start()
        if calibrate:
            self._load_calibration()

    # ---- connections ----
    def _connect_pg(self):
        import psycopg
        from . import timeseries
        conn = psycopg.connect(self.url, autocommit=True, connect_timeout=5)
        for q in _schema("TIMESTAMPTZ", "JSONB"):
            conn.execute(q)
        for col, typ in TELEMETRY_COLS.items():
            conn.execute(f"ALTER TABLE telemetry ADD COLUMN IF NOT EXISTS {col} {typ}")
        timeseries.setup(conn)
        return conn

    def _connect_sqlite(self):
        conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        for q in _schema("REAL", "TEXT"):
            conn.execute(q)
        have = {r[1] for r in conn.execute("PRAGMA table_info(telemetry)")}
        for col, typ in TELEMETRY_COLS.items():
            if col not in have:
                conn.execute(f"ALTER TABLE telemetry ADD COLUMN {col} {typ}")
        return conn

    @property
    def _ph(self) -> str:
        return "%s" if self.kind == "timescale" else "?"

    def _ts(self, wall: float | None):
        if wall is None or self.kind != "timescale":
            return wall
        from datetime import datetime, timezone
        return datetime.fromtimestamp(wall, timezone.utc)

    # ---- write path (called from the control loop and camera thread; never blocks) ----
    def _insert(self, table: str, row: dict) -> None:
        cols = list(row)
        self._put((f"INSERT INTO {table} ({','.join(cols)}) "
                   f"VALUES ({','.join([self._ph] * len(cols))})", list(row.values())))

    def telemetry(self, s: dict) -> None:
        from .physics import clamp_resistance
        row = {c: s.get(c) for c in TELEMETRY_COLS}
        row["valve_pos"] = (s.get("device") or {}).get("u")
        if s.get("valve_u") is not None:
            r = (clamp_resistance(s["valve_u"], self.physics) if self.physics
                 else clamp_resistance(s["valve_u"]))
            row["clamp_r"] = None if math.isinf(r) else round(r, 4)
        self._insert("telemetry", {"time": self._ts(time.time()),
                                   "infusion_id": self.infusion_id, **row})

    def drop(self, s: dict, t: float) -> None:
        """One detected drop. s is the twin snapshot just after it, t the
        detection time in the twin's clock (for the interval)."""
        interval = None if self._last_drop_t is None else t - self._last_drop_t
        self._last_drop_t = t
        if interval is not None and not 0 < interval <= 60:
            interval = None          # first drop after a pause: not a drip interval
        self._insert("drops", {
            "time": self._ts(time.time()), "infusion_id": self.infusion_id,
            "interval_s": None if interval is None else round(interval, 4),
            "volume_ml": s.get("drop_volume_ml"), "rate_gtt_min": s.get("rate_gtt_min"),
            "confident": s.get("rate_confident")})

    def event(self, kind: str, severity: str, message: str, data: dict) -> None:
        now, ph = time.time(), self._ph
        if kind == "order_started":
            self._put((f"UPDATE infusions SET status = 'replaced', ended = {ph} "
                       f"WHERE infusion_id = {ph} AND ended IS NULL",
                       [self._ts(now), self.infusion_id]))
            self.infusion_id = new_infusion_id()
            target = None
            try:
                from .physics import ml_h_to_gtt_min
                target = round(ml_h_to_gtt_min(data["rate_ml_h"], data["drop_factor"]), 2)
            except (KeyError, TypeError):
                pass
            self._insert("infusions", {"infusion_id": self.infusion_id,
                                       "started": self._ts(now),
                                       **{c: data.get(c) for c in INFUSION_COLS},
                                       "target_gtt_min": target, "status": "running"})
        elif kind in STATUS:
            ended = self._ts(now) if kind == "complete" else None
            self._put((f"UPDATE infusions SET status = {ph}, ended = COALESCE({ph}, ended) "
                       f"WHERE infusion_id = {ph}", [STATUS[kind], ended, self.infusion_id]))
        self._insert("events", {"time": self._ts(now), "infusion_id": self.infusion_id,
                                "kind": kind, "severity": severity, "message": message,
                                "data": json.dumps(data, default=str)})

    def _load_calibration(self) -> None:
        try:
            rows = calibration_rows()
        except Exception as e:
            print(f"[storage] calibration not logged ({type(e).__name__}: {e})")
            return
        ph = self._ph
        for src in {r[1] for r in rows}:
            self._put((f"DELETE FROM calibration WHERE source = {ph}", [src]))
        for r in rows:
            self._put((f"INSERT INTO calibration VALUES ({','.join([ph] * 6)})",
                       [self._ts(r[0]), *r[1:]]))

    def _put(self, item) -> None:
        try:
            self._q.put_nowait(item)
        except queue.Full:
            self.dropped += 1

    def flush(self, timeout: float = 5.0) -> bool:
        """Wait until queued rows are written. True if the queue drained in time."""
        end = time.monotonic() + timeout
        while self._q.unfinished_tasks and time.monotonic() < end:
            time.sleep(0.01)
        return not self._q.unfinished_tasks

    def close(self) -> None:
        self.flush()
        with self._lock:
            self.conn.close()

    def _write_loop(self) -> None:
        pending = None
        while True:
            item = pending or self._q.get()
            try:
                with self._lock:
                    self.conn.execute(*item)
                if pending:
                    print("[storage] database back, writing again")
                pending = None
                self._q.task_done()
            except Exception as e:
                if self.kind != "timescale":     # local SQLite: retrying won't help
                    self.last_error = f"{type(e).__name__}: {e}"
                    print(f"[storage] dropped a row ({self.last_error})")
                    pending = None
                    self._q.task_done()
                    continue
                if not pending:
                    self.last_error = f"{type(e).__name__}: {e}"
                    print(f"[storage] write failed ({self.last_error}); retrying")
                pending = item
                time.sleep(2.0)
                self._reconnect()

    def _reconnect(self) -> None:
        if self.kind != "timescale":
            return
        try:
            import psycopg
            with self._lock:
                try:
                    self.conn.close()
                except Exception:
                    pass
                self.conn = psycopg.connect(self.url, autocommit=True, connect_timeout=5)
        except Exception:
            pass

    # ---- read path (dashboard, CLI) ----
    def _query(self, sql: str, args=()) -> list[dict]:
        with self._lock:
            cur = self.conn.execute(sql, args)
            names = [d[0] for d in cur.description]
            return [dict(zip(names, r)) for r in cur.fetchall()]

    def _epoch(self, col: str = "time") -> str:
        return f"EXTRACT(EPOCH FROM {col})::float8" if self.kind == "timescale" else col

    def infusions(self, limit: int = 20) -> list[dict]:
        """Most recent infusions first, with start/end time and volume infused."""
        e = self._epoch
        return self._query(
            f"""SELECT infusion_id, {e('MIN(time)')} AS start, {e('MAX(time)')} AS "end",
                       COUNT(*) AS samples, MAX(infused_ml) AS infused_ml,
                       MAX(target_gtt_min) AS target_gtt_min
                FROM telemetry GROUP BY infusion_id ORDER BY start DESC LIMIT {int(limit)}""")

    def history(self, infusion_id: str | None = None, since_s: float | None = None,
                bucket_s: int | None = None) -> list[dict]:
        """Telemetry for one infusion (default: current), oldest first.

        since_s keeps only the last N seconds; bucket_s averages into buckets
        (Timescale's time_bucket on Tiger Data, integer division on SQLite)."""
        inf = infusion_id or self.infusion_id
        ph, args = self._ph, [inf]
        where = f"infusion_id = {ph}"
        if since_s is not None:
            where += f" AND time >= {ph}"
            args.append(self._ts(time.time() - since_s))
        if not bucket_s:
            cols = ", ".join(TELEMETRY_COLS)
            return self._query(f"SELECT {self._epoch()} AS time, {cols} FROM telemetry "
                               f"WHERE {where} ORDER BY time", args)
        if self.kind == "timescale":
            b = f"time_bucket(make_interval(secs => {int(bucket_s)}), time)"
            bt = f"EXTRACT(EPOCH FROM {b})::float8"
        else:
            b = bt = f"CAST(time / {int(bucket_s)} AS INTEGER) * {int(bucket_s)}"
        return self._query(
            f"""SELECT {bt} AS time, AVG(rate_gtt_min) AS rate_gtt_min,
                       MAX(target_gtt_min) AS target_gtt_min, AVG(valve_u) AS valve_u,
                       AVG(head_cm) AS head_cm, MAX(infused_ml) AS infused_ml
                FROM telemetry WHERE {where} GROUP BY {b} ORDER BY 1""", args)

    def events(self, infusion_id: str | None = None, limit: int = 100,
               severity: str | None = None) -> list[dict]:
        """Events for one infusion (default: current), newest first."""
        ph, args = self._ph, [infusion_id or self.infusion_id]
        where = f"infusion_id = {ph}"
        if severity:
            where += f" AND severity = {ph}"
            args.append(severity)
        rows = self._query(f"SELECT {self._epoch()} AS time, kind, severity, message, data "
                           f"FROM events WHERE {where} ORDER BY time DESC LIMIT {int(limit)}",
                           args)
        for r in rows:
            if isinstance(r["data"], str):
                try:
                    r["data"] = json.loads(r["data"])
                except ValueError:
                    pass
        return rows

    def summary(self, infusion_id: str | None = None) -> dict:
        """Plain-number recap of an infusion, for the agent and the hand-off log."""
        inf = infusion_id or self.infusion_id
        ph, e = self._ph, self._epoch
        row = self._query(
            f"""SELECT {e('MIN(time)')} AS start, {e('MAX(time)')} AS "end",
                       MAX(infused_ml) AS infused_ml, AVG(rate_gtt_min) AS mean_rate_gtt_min,
                       MAX(target_gtt_min) AS target_gtt_min,
                       AVG(ABS(rate_gtt_min - target_gtt_min)) AS mean_abs_error_gtt_min
                FROM telemetry WHERE infusion_id = {ph}""", [inf])[0]
        counts = self._query(f"SELECT severity, COUNT(*) AS n FROM events "
                             f"WHERE infusion_id = {ph} GROUP BY severity", [inf])
        row["infusion_id"] = inf
        row["events"] = {c["severity"]: c["n"] for c in counts}
        row["duration_min"] = (round((row["end"] - row["start"]) / 60, 1)
                               if row["start"] is not None else None)
        return row

    def status(self) -> dict:
        return {"kind": self.kind, "infusion_id": self.infusion_id,
                "queued": self._q.qsize(), "dropped": self.dropped,
                "last_error": self.last_error}

    def export_csv(self, out, infusion_id: str | None = None) -> int:
        rows = self.history(infusion_id)
        w = csv.DictWriter(out, fieldnames=["time", *TELEMETRY_COLS])
        w.writeheader()
        w.writerows(rows)
        return len(rows)


def _fmt_t(t) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t)) if t else "-"


def main(argv: list[str]) -> None:
    from .timeseries import load_env
    load_env()
    store = Store(calibrate=False)
    print(f"[{store.kind}]")
    match argv:
        case [] | ["list"]:
            for r in store.infusions():
                print(f"{r['infusion_id']}  {_fmt_t(r['start'])}  "
                      f"{(r['end'] - r['start']) / 60:5.1f} min  "
                      f"{r['infused_ml'] or 0:6.1f} mL  {r['samples']} samples")
        case ["show", inf]:
            print(json.dumps(store.summary(inf), indent=2, default=str))
            for e in reversed(store.events(inf, limit=50)):
                print(f"{_fmt_t(e['time'])}  {e['severity']:7} {e['kind']:16} {e['message']}")
        case ["csv", inf, *rest]:
            if rest:
                with open(rest[0], "w", newline="") as f:
                    n = store.export_csv(f, inf)
                print(f"wrote {n} rows to {rest[0]}")
            else:
                store.export_csv(sys.stdout, inf)
        case _:
            print(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
