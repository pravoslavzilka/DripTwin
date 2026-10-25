import io
import sqlite3
import time

from driptwin.storage import Store


def snap(rate, target=40.0, infused=0.0):
    return {"rate_gtt_min": rate, "target_gtt_min": target, "rate_ml_h": rate * 3,
            "valve_u": 0.5, "head_cm": 80.0, "infused_ml": infused, "remaining_ml": 500 - infused,
            "drops_counted": 10, "rate_confident": True, "mode": "running", "order": {}}


def test_roundtrip_and_new_infusion_per_order(tmp_path):
    s = Store(url="", path=tmp_path / "log.db")
    first = s.infusion_id
    s.telemetry(snap(39.0, infused=1.0))
    time.sleep(1.1)                              # ids have one-second resolution
    s.event("order_started", "info", "120 mL/h", {"ml_h": 120})
    second = s.infusion_id
    assert second != first
    s.telemetry(snap(41.0, infused=2.0))
    s.event("occlusion", "alarm", "line kinked", {"u": 1.0})
    assert s.flush()

    assert [r["rate_gtt_min"] for r in s.history(first)] == [39.0]
    assert [r["rate_gtt_min"] for r in s.history()] == [41.0]
    ev = s.events()
    assert [e["kind"] for e in ev] == ["occlusion", "order_started"]
    assert ev[0]["data"] == {"u": 1.0}
    assert [e["kind"] for e in s.events(severity="alarm")] == ["occlusion"]

    summ = s.summary()
    assert summ["infused_ml"] == 2.0 and summ["events"] == {"alarm": 1, "info": 1}
    assert {r["infusion_id"] for r in s.infusions()} == {first, second}

    out = io.StringIO()
    assert s.export_csv(out) == 1 and "rate_gtt_min" in out.getvalue().splitlines()[0]
    s.close()


def test_bucketed_history(tmp_path):
    s = Store(url="", path=tmp_path / "log.db")
    for r in (38.0, 42.0):
        s.telemetry(snap(r))
    s.flush()
    rows = s.history(bucket_s=3600)
    assert len(rows) == 1 and rows[0]["rate_gtt_min"] == 40.0
    assert len(s.history(since_s=60)) == 2
    s.close()


def test_migrates_old_schema(tmp_path):
    path = tmp_path / "old.db"
    c = sqlite3.connect(path)
    c.execute("""CREATE TABLE telemetry (time REAL, infusion_id TEXT, rate_gtt_min REAL,
        target_gtt_min REAL, valve_u REAL, head_cm REAL, infused_ml REAL, mode TEXT)""")
    c.execute("INSERT INTO telemetry VALUES (1, 'old', 1, 1, 1, 1, 1, 'idle')")
    c.commit()
    c.close()
    s = Store(url="", path=path)
    s.telemetry(snap(40.0))
    s.flush()
    assert s.history()[0]["remaining_ml"] == 500
    assert s.history("old")[0]["remaining_ml"] is None
    s.close()


def test_unreachable_timescale_falls_back_to_sqlite(tmp_path):
    s = Store(url="postgresql://nobody@127.0.0.1:1/x", path=tmp_path / "log.db")
    assert s.kind == "sqlite" and s.last_error
    s.close()


def test_writes_never_block_when_queue_full(tmp_path):
    s = Store(url="", path=tmp_path / "log.db", max_queue=1)
    with s._lock:                               # stall the writer thread
        t = time.monotonic()
        for _ in range(50):
            s.telemetry(snap(40.0))
        assert time.monotonic() - t < 0.5
    assert s.dropped > 0
    s.close()


def test_drops_infusions_and_calibration(tmp_path):
    s = Store(url="", path=tmp_path / "log.db")
    s.event("order_started", "info", "go", {"rate_ml_h": 120, "drop_factor": 20,
                                           "vtbi_ml": 500, "bag_volume_ml": 1000, "drug": "saline"})
    for t in (10.0, 11.5, 13.0, 200.0):          # last one comes after a long pause
        s.drop({"drop_volume_ml": 0.065, "rate_gtt_min": 40.0, "rate_confident": True}, t)
    s.telemetry({**snap(40.0), "device": {"u": 0.45}})
    s.event("occlusion", "alarm", "kink", {})
    assert s.flush()
    q = lambda sql: s.conn.execute(sql).fetchall()
    assert [r[0] for r in q("SELECT interval_s FROM drops ORDER BY time")] == [None, 1.5, 1.5, None]
    inf = q("SELECT infusion_id, rate_ml_h, drug, target_gtt_min, status FROM infusions")
    assert inf[0][0] == s.infusion_id and inf[0][1:3] == (120, "saline")
    assert 30 < inf[0][3] < 45 and inf[0][4] == "alarm: occlusion"
    tel = q("SELECT valve_pos, clamp_r FROM telemetry")[0]
    assert tel[0] == 0.45 and tel[1] > 0
    assert q("SELECT count(*) FROM calibration WHERE source = 'drop_curve_fit'")[0][0] > 0
    s.close()
