"""Tiger Data (TimescaleDB) side of the infusion log: aggregates, compression,
SQL detection and the agent's read-only access. storage.py writes the raw rows.

    drops, telemetry          hypertables, 1-hour chunks, columnstore after 1 h,
                              raw rows dropped after RETENTION (aggregates kept)
    drops_10s, drops_1m       continuous aggregates: drops/min, mL/h (sum of each
                              drop's calibrated volume), drop-interval spread
    telemetry_10s/_1m         continuous aggregates: valve, bag volume, clamp R
    flow_10s, flow_1m         views joining the two; what the dashboard and agent read

All four aggregates use real-time aggregation, so the newest seconds are included.

The agent connects as a separate read-only role (AGENT_DATABASE_URL) that can
only SELECT the flow views, events and infusions, and it only gets the fixed
parameterised queries in FlowQueries, never free-form SQL.

    python -m driptwin.timeseries setup        create everything (also done on server start)
    python -m driptwin.timeseries agent-role   create the read-only role, add AGENT_DATABASE_URL to .env
    python -m driptwin.timeseries seed [hours] write a synthetic infusion (ids start with seed-)
    python -m driptwin.timeseries bench        compress old chunks, time raw vs aggregate, print stats
    python -m driptwin.timeseries check        run the agent's queries on the live infusion
"""

from __future__ import annotations

import json
import os
import random
import secrets
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"
RETENTION = "7 days"
AGENT_ROLE = "driptwin_agent"
AGENT_TABLES = ["flow_10s", "flow_1m", "events", "infusions"]
IRREGULAR_CV = 0.25     # drop-interval SD / mean above this = irregular dripping
NEAR_EMPTY_ML = 30.0


def load_env(path: Path = ENV_FILE) -> None:
    """Same rules as server.load_env, without importing the web server."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.removeprefix("export ").split("=", 1)
        key, val = key.strip(), val.strip().strip("'\"")
        if val and not os.environ.get(key):
            os.environ[key] = val


# ---------------------------------------------------------------- setup ----

def _try(conn, sql: str, quiet: tuple[str, ...] = ("already",)) -> bool:
    try:
        conn.execute(sql)
        return True
    except Exception as e:
        msg = str(e).splitlines()[0]
        if not any(q in msg for q in quiet):
            print(f"[timeseries] skipped: {msg}  ({sql.split()[0]} {sql.split()[1]} ...)")
        return False


def _exists(conn, name: str) -> bool:
    return conn.execute("SELECT to_regclass(%s) IS NOT NULL", [name]).fetchone()[0]


def setup(conn) -> None:
    """Idempotent: safe to run on every server start. Needs the base tables
    from storage.py. Each step is independent, so one unsupported feature
    doesn't take the rest down."""
    for t in ("drops", "telemetry"):
        conn.execute(f"SELECT create_hypertable('{t}', by_range('time', INTERVAL '1 hour'), "
                     f"if_not_exists => TRUE, migrate_data => TRUE)")
        _try(conn, f"""ALTER TABLE {t} SET (timescaledb.enable_columnstore = true,
                       timescaledb.segmentby = 'infusion_id', timescaledb.orderby = 'time DESC')""",
             quiet=("already", "compressed"))
        _try(conn, f"CALL add_columnstore_policy('{t}', after => INTERVAL '1 hour', "
                   f"if_not_exists => true)")
        _try(conn, f"SELECT add_retention_policy('{t}', drop_after => INTERVAL '{RETENTION}', "
                   f"if_not_exists => true)")
    toolkit = _try(conn, "CREATE EXTENSION IF NOT EXISTS timescaledb_toolkit")
    for name, width, start in (("10s", "10 seconds", "1 hour"), ("1m", "1 minute", "3 hours")):
        per_min = {"10s": 6, "1m": 1}[name]
        if not _exists(conn, f"drops_{name}"):
            conn.execute(f"""
                CREATE MATERIALIZED VIEW drops_{name} WITH (timescaledb.continuous) AS
                SELECT time_bucket(INTERVAL '{width}', time) AS bucket, infusion_id,
                       count(*) AS drops,
                       count(*)::float8 * {per_min} AS gtt_min,
                       sum(volume_ml)::float8 * {per_min * 60} AS ml_h,
                       avg(interval_s)::float8 AS mean_interval_s,
                       stddev_samp(interval_s)::float8 AS sd_interval_s
                FROM drops GROUP BY bucket, infusion_id WITH NO DATA""")
        if not _exists(conn, f"telemetry_{name}"):
            tw = ", time_weight('LOCF', time, valve_u) AS valve_tw" if toolkit else ""
            conn.execute(f"""
                CREATE MATERIALIZED VIEW telemetry_{name} WITH (timescaledb.continuous) AS
                SELECT time_bucket(INTERVAL '{width}', time) AS bucket, infusion_id,
                       avg(valve_u)::float8 AS valve_u, avg(valve_pos)::float8 AS valve_pos,
                       max(valve_u)::float8 AS valve_u_max,
                       min(remaining_ml)::float8 AS remaining_ml,
                       avg(head_cm)::float8 AS head_cm, avg(clamp_r)::float8 AS clamp_r,
                       max(target_gtt_min)::float8 AS target_gtt_min{tw}
                FROM telemetry GROUP BY bucket, infusion_id WITH NO DATA""")
        for src in ("drops", "telemetry"):
            conn.execute(f"ALTER MATERIALIZED VIEW {src}_{name} "
                         f"SET (timescaledb.materialized_only = false)")
            _try(conn, f"""SELECT add_continuous_aggregate_policy('{src}_{name}',
                           start_offset => INTERVAL '{start}', end_offset => INTERVAL '{width}',
                           schedule_interval => INTERVAL '1 minute', if_not_exists => true)""")
        has_tw = conn.execute(
            "SELECT count(*) FROM information_schema.columns "
            "WHERE table_name = %s AND column_name = 'valve_tw'", [f"telemetry_{name}"]
        ).fetchone()[0]
        tw_col = "average(t.valve_tw)" if has_tw else "t.valve_u"
        _try(conn, f"DROP VIEW IF EXISTS flow_{name}")
        conn.execute(f"""
            CREATE VIEW flow_{name} AS
            SELECT coalesce(d.bucket, t.bucket) AS bucket,
                   coalesce(d.infusion_id, t.infusion_id) AS infusion_id,
                   d.drops, d.gtt_min, d.ml_h, d.mean_interval_s, d.sd_interval_s,
                   d.sd_interval_s / nullif(d.mean_interval_s, 0) AS interval_cv,
                   t.valve_u, {tw_col} AS valve_u_time_weighted, t.valve_u_max, t.valve_pos,
                   t.remaining_ml, t.head_cm, t.clamp_r, t.target_gtt_min
            FROM drops_{name} d FULL JOIN telemetry_{name} t
              ON d.bucket = t.bucket AND d.infusion_id = t.infusion_id""")
    _grant_agent(conn)


def _grant_agent(conn) -> None:
    if conn.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", [AGENT_ROLE]).fetchone():
        conn.execute(f"GRANT USAGE ON SCHEMA public TO {AGENT_ROLE}")
        conn.execute(f"GRANT SELECT ON {', '.join(AGENT_TABLES)} TO {AGENT_ROLE}")


def create_agent_role(conn, url: str) -> str:
    """Create (or reset the password of) the read-only agent role; returns its URL."""
    from urllib.parse import urlsplit, urlunsplit, quote
    pw = secrets.token_urlsafe(24)
    if conn.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", [AGENT_ROLE]).fetchone():
        conn.execute(f"ALTER ROLE {AGENT_ROLE} WITH LOGIN PASSWORD '{pw}'")
    else:
        conn.execute(f"CREATE ROLE {AGENT_ROLE} WITH LOGIN PASSWORD '{pw}'")
    db = conn.execute("SELECT current_database()").fetchone()[0]
    conn.execute(f"GRANT CONNECT ON DATABASE {db} TO {AGENT_ROLE}")
    conn.execute(f"ALTER ROLE {AGENT_ROLE} SET default_transaction_read_only = on")
    conn.execute(f"ALTER ROLE {AGENT_ROLE} SET statement_timeout = '5s'")
    _grant_agent(conn)
    p = urlsplit(url)
    host = p.hostname + (f":{p.port}" if p.port else "")
    return urlunsplit((p.scheme, f"{AGENT_ROLE}:{quote(pw)}@{host}", p.path, p.query, ""))


# -------------------------------------------------------- agent queries ----

class FlowQueries:
    """The only database access the agent and dashboard get: fixed, parameterised
    queries over the aggregates, through the read-only role."""

    def __init__(self, url: str | None = None):
        self.url = url or os.environ.get("AGENT_DATABASE_URL")
        self._conn = None
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return bool(self.url)

    def _q(self, sql: str, args=()) -> list[dict]:
        import psycopg
        with self._lock:
            for attempt in (0, 1):
                try:
                    if self._conn is None or self._conn.closed:
                        self._conn = psycopg.connect(self.url, autocommit=True,
                                                     connect_timeout=5)
                    cur = self._conn.execute(sql, args)
                    names = [d[0] for d in cur.description]
                    return [dict(zip(names, r)) for r in cur.fetchall()]
                except psycopg.OperationalError:
                    self._conn = None
                    if attempt:
                        raise

    def current_infusion(self) -> str | None:
        rows = self._q("""SELECT infusion_id FROM flow_10s
                          WHERE bucket > now() - INTERVAL '1 hour'
                            AND infusion_id NOT LIKE 'seed-%%'
                          ORDER BY bucket DESC LIMIT 1""")
        return rows[0]["infusion_id"] if rows else None

    def _inf(self, infusion_id: str | None) -> str:
        inf = infusion_id or self.current_infusion()
        if not inf:
            raise LookupError("no drop data logged in the last hour")
        return inf

    def flow_history(self, minutes: int = 10, infusion_id: str | None = None) -> dict:
        """Per-minute flow from flow_1m (10 s buckets for 5 minutes or less)."""
        minutes = max(1, min(int(minutes), 180))
        inf = self._inf(infusion_id)
        view = "flow_10s" if minutes <= 5 else "flow_1m"
        rows = self._q(f"""
            SELECT round(extract(epoch FROM now() - bucket)::numeric / 60, 1)::float8 AS minutes_ago,
                   coalesce(drops, 0) AS drops, round(ml_h::numeric, 1)::float8 AS ml_h,
                   round(gtt_min::numeric, 1)::float8 AS drops_per_min,
                   round(interval_cv::numeric, 3)::float8 AS interval_cv,
                   round(valve_u_time_weighted::numeric, 3)::float8 AS valve_closure,
                   round(remaining_ml::numeric)::float8 AS bag_ml_left,
                   round(target_gtt_min::numeric, 1)::float8 AS target_drops_per_min
            FROM {view} WHERE infusion_id = %s AND bucket > now() - make_interval(mins => %s)
            ORDER BY bucket""", [inf, minutes])
        # The first bucket after a stop holds one long interval: that's the gap
        # (find_flow_gaps), not irregular dripping.
        irregular = [r["minutes_ago"] for prev, r in zip([None] + rows, rows)
                     if r["interval_cv"] is not None and r["interval_cv"] > IRREGULAR_CV
                     and prev is not None and prev["drops"]]
        return {"infusion_id": inf, "bucket": view[5:], "rows": rows,
                "irregular_dripping_minutes_ago": irregular,
                "note": "valve_closure 0 = wide open, 1 = shut; interval_cv = SD/mean of "
                        f"drop intervals, above {IRREGULAR_CV} is irregular (partial kink?)"}

    def flow_gaps(self, minutes: int = 10, infusion_id: str | None = None) -> dict:
        """10-second windows with no drops, made visible with time_bucket_gapfill
        (a plain GROUP BY just skips them)."""
        minutes = max(1, min(int(minutes), 180))
        inf = self._inf(infusion_id)
        now = datetime.now(timezone.utc)
        end = now.replace(microsecond=0) - timedelta(seconds=now.second % 10)  # last full bucket
        start = end - timedelta(minutes=minutes)
        first = self._q("SELECT min(bucket) AS b FROM flow_10s WHERE infusion_id = %s "
                        "AND bucket >= %s", [inf, start])[0]["b"]
        start = max(start, first or end)
        rows = self._q("""
            SELECT time_bucket_gapfill(INTERVAL '10 seconds', bucket,
                                       start => %s, finish => %s) AS b,
                   coalesce(sum(drops), 0) AS drops,
                   locf(min(remaining_ml)) AS bag_ml_left, locf(max(valve_u_max)) AS valve_closure
            FROM flow_10s WHERE infusion_id = %s AND bucket >= %s AND bucket < %s
            GROUP BY b ORDER BY b""", [start, end, inf, start, end])
        gaps, run = [], None
        for r in rows + [{"b": end, "drops": 1}]:
            if r["drops"] == 0:
                run = run or r
                last = r
            elif run:
                left, shut = last["bag_ml_left"], (last["valve_closure"] or 0) >= 0.99
                gaps.append({
                    "from_s_ago": round((now - run["b"]).total_seconds()),
                    "to_s_ago": round((now - last["b"]).total_seconds()) - 10,
                    "seconds": round((last["b"] - run["b"]).total_seconds()) + 10,
                    "bag_ml_left": None if left is None else round(left),
                    "likely": ("valve shut (paused or alarm)" if shut else
                               "empty bag" if left is not None and left < NEAR_EMPTY_ML else
                               "occlusion, kink or camera lost the drops")})
                run = None
        return {"infusion_id": inf, "window_min": minutes, "gaps": gaps,
                "dripping_now": not gaps or gaps[-1]["to_s_ago"] >= 10}

    def time_to_empty(self, infusion_id: str | None = None) -> dict:
        """Remaining volume / current flow, plus a regr_slope line through the
        last 10 complete minutes of mL/h, extrapolated to zero flow."""
        inf = self._inf(infusion_id)
        # Minutes more than 25% off the median (a kink, a bag bump) are left out of
        # the fit: the trend we want is the slow decline as the bag drains.
        fit = self._q("""
            WITH m AS (SELECT ml_h, extract(epoch FROM bucket - now()) / 60 AS x FROM flow_1m
                       WHERE infusion_id = %s AND bucket >= now() - INTERVAL '11 minutes'
                         AND bucket < time_bucket(INTERVAL '1 minute', now())),
                 med AS (SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY ml_h) AS q FROM m)
            SELECT regr_slope(ml_h, x) AS slope, regr_intercept(ml_h, x) AS ml_h_now,
                   regr_r2(ml_h, x) AS r2, regr_count(ml_h, x) AS n
            FROM m, med WHERE abs(ml_h - med.q) <= 0.25 * med.q""", [inf])[0]
        left = self._q("""SELECT remaining_ml FROM flow_10s WHERE infusion_id = %s
                          AND remaining_ml IS NOT NULL ORDER BY bucket DESC LIMIT 1""", [inf])
        # Flow now: the last two minutes of buckets that had drops (a stop is
        # find_flow_gaps' job; the fit would be dragged down by it).
        now_q = self._q("""SELECT avg(ml_h) AS q FROM flow_10s WHERE infusion_id = %s
                           AND drops > 0 AND bucket >= now() - INTERVAL '2 minutes'
                           AND bucket < time_bucket(INTERVAL '10 seconds', now())""", [inf])[0]["q"]
        out = {"infusion_id": inf, "minutes_fitted": fit["n"],
               "bag_ml_left": round(left[0]["remaining_ml"]) if left else None,
               "flow_ml_h_now": None if now_q is None else round(now_q, 1),
               "flow_trend_ml_h_per_min": None if fit["slope"] is None else round(fit["slope"], 2),
               "fit_r2": None if fit["r2"] is None else round(fit["r2"], 2)}
        q, s = now_q, fit["slope"]
        if q and q > 0 and out["bag_ml_left"] is not None:
            out["minutes_to_empty_at_current_flow"] = round(out["bag_ml_left"] / q * 60)
        if (fit["ml_h_now"] and s is not None and s < 0 and (fit["r2"] or 0) > 0.5
                and fit["n"] >= 6):
            out["minutes_until_trend_reaches_zero_flow"] = round(-fit["ml_h_now"] / s)
        if (fit["n"] or 0) < 6:
            out["note"] = "under 6 steady minutes logged: no trend yet, use the current-flow estimate"
        return out


TOOLS = [
    {"name": "get_flow_history",
     "description": "Per-minute history of the current infusion from the Tiger Data log: "
                    "drops/min, mL/h from the measured drop size, valve closure, bag volume "
                    "left, and drop-interval variation (interval_cv; irregular dripping hints "
                    "at a partial kink). Use for trends and 'what happened in the last N "
                    "minutes'. Read-only.",
     "input_schema": {"type": "object", "properties": {
         "minutes": {"type": "integer", "description": "1 to 180"}},
         "required": ["minutes"], "additionalProperties": False},
     "strict": True},
    {"name": "find_flow_gaps",
     "description": "Times in the last N minutes when no drops fell (10-second windows, "
                    "gap-filled), each with the likely cause: valve shut, empty bag, or "
                    "occlusion/kink. Read-only.",
     "input_schema": {"type": "object", "properties": {
         "minutes": {"type": "integer", "description": "1 to 180"}},
         "required": ["minutes"], "additionalProperties": False},
     "strict": True},
    {"name": "get_time_to_empty",
     "description": "When the bag will run out: volume left / current flow, and a straight-"
                    "line fit of flow over the last 10 minutes. Read-only.",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
     "strict": True},
]
TOOL_NAMES = {t["name"] for t in TOOLS}
_queries: FlowQueries | None = None


def queries() -> FlowQueries:
    global _queries
    if _queries is None:
        _queries = FlowQueries()
    return _queries


def run_tool(name: str, args: dict) -> tuple[str, bool]:
    q = queries()
    if not q.available:
        return "History not available: no Tiger Data agent role configured.", True
    try:
        if name == "get_flow_history":
            return json.dumps(q.flow_history(args.get("minutes", 10)), default=str), False
        if name == "find_flow_gaps":
            return json.dumps(q.flow_gaps(args.get("minutes", 10)), default=str), False
        if name == "get_time_to_empty":
            return json.dumps(q.time_to_empty(), default=str), False
    except LookupError as e:
        return f"No data: {e}.", True
    except Exception as e:
        return f"History query failed: {type(e).__name__}.", True
    return f"Unknown tool {name}", True


# ------------------------------------------------------------ seed / bench ----

def seed(conn, hours: float = 6.0, end_ago_h: float = 1.5) -> str:
    """A synthetic infusion in the past (so its chunks can be compressed):
    40 drops/min with 5% jitter, a slow decline as the bag drains, a 3-minute
    occlusion and a stretch of irregular dripping. ids start with 'seed-'."""
    from .physics import drop_volume_ml
    rng = random.Random(1)
    end = datetime.now(timezone.utc) - timedelta(hours=end_ago_h)
    t = start = end - timedelta(hours=hours)
    inf = "seed-" + start.strftime("%Y%m%d-%H%M%S")
    bag = 1000.0
    occl = (start + timedelta(hours=hours * 0.6), start + timedelta(hours=hours * 0.6, minutes=3))
    rough = (start + timedelta(hours=hours * 0.3), start + timedelta(hours=hours * 0.3, minutes=10))
    drops, tel = [], []
    next_tel = start
    while t < end:
        frac = (t - start) / (end - start)
        rate = 40.0 * (1 - 0.15 * frac)
        mean = 60.0 / rate
        cv = 0.4 if rough[0] <= t < rough[1] else 0.05
        dt = max(0.05, rng.gauss(mean, mean * cv))
        if occl[0] <= t < occl[1]:
            dt = (occl[1] - t).total_seconds() + mean
        t += timedelta(seconds=dt)
        vol = drop_volume_ml(rate, 20)
        bag -= vol
        drops.append((t, inf, round(dt, 4), round(vol, 5), round(rate, 2), True))
        while next_tel <= t:
            tel.append((next_tel, inf, round(rate, 2), 40.0, round(rate * vol * 60, 1),
                        0.4, 0.4, 12.5, 80.0 - 20 * frac, 80.0, round(1000 - bag, 1),
                        round(bag, 1), len(drops), True, "running"))
            next_tel += timedelta(seconds=1)
    with conn.cursor() as cur:
        with cur.copy("COPY drops (time, infusion_id, interval_s, volume_ml, rate_gtt_min, "
                      "confident) FROM STDIN") as cp:
            for r in drops:
                cp.write_row(r)
        with cur.copy("COPY telemetry (time, infusion_id, rate_gtt_min, target_gtt_min, "
                      "rate_ml_h, valve_u, valve_pos, clamp_r, head_cm, head_baseline_cm, "
                      "infused_ml, remaining_ml, drops_counted, rate_confident, mode) "
                      "FROM STDIN") as cp:
            for r in tel:
                cp.write_row(r)
    conn.execute("""INSERT INTO infusions (infusion_id, started, ended, rate_ml_h, drop_factor,
                    vtbi_ml, bag_volume_ml, drug, target_gtt_min, status)
                    VALUES (%s, %s, %s, 120, 20, 1000, 1000, 'saline (synthetic)', 40, 'seed')
                    ON CONFLICT (infusion_id) DO NOTHING""", [inf, start, end])
    for v in ("drops_10s", "drops_1m", "telemetry_10s", "telemetry_1m"):
        conn.execute(f"CALL refresh_continuous_aggregate('{v}', %s, %s)",
                     [start - timedelta(minutes=1), end + timedelta(minutes=1)])
    print(f"seeded {inf}: {len(drops)} drops, {len(tel)} telemetry rows")
    return inf


def _exec_ms(conn, sql: str, args=(), runs: int = 5) -> float:
    """Server-side execution time (EXPLAIN ANALYZE), so network latency doesn't count."""
    ts = []
    for _ in range(runs):
        plan = conn.execute("EXPLAIN (ANALYZE, FORMAT JSON) " + sql, args).fetchone()[0]
        ts.append(plan[0]["Execution Time"])
    return sorted(ts)[len(ts) // 2]


def bench(conn) -> dict:
    """Compress chunks older than 1 hour now (the policy would do it on its
    schedule), then compare raw vs aggregate query time and report storage."""
    for t in ("drops", "telemetry"):
        n = conn.execute(f"""SELECT count(compress_chunk(c, if_not_compressed => true))
                             FROM show_chunks('{t}', older_than => INTERVAL '1 hour') c"""
                         ).fetchone()[0]
        print(f"{t}: {n} chunks in the columnstore")
    inf = conn.execute("SELECT infusion_id FROM drops GROUP BY 1 ORDER BY count(*) DESC LIMIT 1"
                       ).fetchone()[0]
    raw = """SELECT time_bucket('1 minute', time) AS b, count(*), sum(volume_ml) * 60
             FROM drops WHERE infusion_id = %s GROUP BY b ORDER BY b"""
    agg = "SELECT bucket, drops, ml_h FROM drops_1m WHERE infusion_id = %s ORDER BY bucket"
    out = {"infusion_id": inf,
           "raw_per_minute_ms": round(_exec_ms(conn, raw, [inf]), 2),
           "aggregate_per_minute_ms": round(_exec_ms(conn, agg, [inf]), 2)}
    out["speedup"] = round(out["raw_per_minute_ms"] / max(out["aggregate_per_minute_ms"], 1e-3), 1)
    for t in ("drops", "telemetry"):
        s = conn.execute(f"""SELECT sum(before_compression_total_bytes),
                                    sum(after_compression_total_bytes)
                             FROM hypertable_columnstore_stats('{t}')""").fetchone()
        rows = conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        if s[0]:
            s = (float(s[0]), float(s[1]))
            out[f"{t}_rows"] = rows
            out[f"{t}_bytes_before"], out[f"{t}_bytes_after"] = int(s[0]), int(s[1])
            out[f"{t}_smaller_pct"] = round(100 * (1 - s[1] / s[0]), 1)
            out[f"{t}_ratio"] = round(s[0] / s[1], 1)
    return out


def main(argv: list[str]) -> None:
    import psycopg
    load_env()
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL is not set (put it in .env)")
    if argv[:1] == ["check"]:
        q = queries()
        if not q.available:
            raise SystemExit("AGENT_DATABASE_URL is not set: run  agent-role  first")
        for name in ("get_flow_history", "find_flow_gaps", "get_time_to_empty"):
            print(name, run_tool(name, {"minutes": 10})[0][:800])
        return
    from .storage import Store
    Store(url=url, calibrate=False).close()       # base tables + setup()
    conn = psycopg.connect(url, autocommit=True)
    match argv:
        case ["setup"]:
            print("ok")
        case ["agent-role"]:
            agent_url = create_agent_role(conn, url)
            lines = [l for l in ENV_FILE.read_text(encoding="utf-8-sig").splitlines()
                     if not l.startswith("AGENT_DATABASE_URL=")] if ENV_FILE.exists() else []
            ENV_FILE.write_text("\n".join(lines + [f"AGENT_DATABASE_URL={agent_url}"]) + "\n",
                                encoding="utf-8")
            print(f"role {AGENT_ROLE} ready; AGENT_DATABASE_URL written to {ENV_FILE.name}")
        case ["seed", *h]:
            seed(conn, float(h[0]) if h else 6.0)
        case ["bench"]:
            print(json.dumps(bench(conn), indent=2))
        case _:
            print(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
