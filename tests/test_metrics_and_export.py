from __future__ import annotations

from starlette.testclient import TestClient

from simpliscribe.main import app
from simpliscribe.metrics import generate_prometheus_metrics, get_metrics_snapshot, record_analysis_event, record_cache_lookup, record_http_request


def test_metrics_collector():
    record_http_request("GET", 200)
    record_http_request("POST", 400)
    record_analysis_event("success")
    record_cache_lookup(True)
    record_cache_lookup(False)

    snapshot = get_metrics_snapshot()
    assert "uptime_seconds" in snapshot
    assert snapshot["http_requests"].get("GET_200", 0) >= 1
    assert snapshot["http_requests"].get("POST_400", 0) >= 1
    assert snapshot["analyses"].get("success", 0) >= 1
    assert snapshot["vector_cache"].get("hit", 0) >= 1
    assert snapshot["vector_cache"].get("miss", 0) >= 1

    prom_text = generate_prometheus_metrics()
    assert "simpliscribe_uptime_seconds" in prom_text
    assert "simpliscribe_http_requests_total" in prom_text
    assert "simpliscribe_analyses_total" in prom_text
    assert "simpliscribe_vector_cache_lookups_total" in prom_text


def test_metrics_api_json_and_prometheus():
    client = TestClient(app)

    # JSON format
    res_json = client.get("/api/metrics")
    assert res_json.status_code == 200
    assert "application/json" in res_json.headers.get("content-type", "")
    data = res_json.json()
    assert "uptime_seconds" in data

    # Prometheus format via query param
    res_prom = client.get("/api/metrics?format=prometheus")
    assert res_prom.status_code == 200
    assert "text/plain" in res_prom.headers.get("content-type", "")
    assert "simpliscribe_uptime_seconds" in res_prom.text

    # Prometheus format via Accept header
    res_accept = client.get("/api/metrics", headers={"Accept": "text/plain; version=0.0.4"})
    assert res_accept.status_code == 200
    assert "text/plain" in res_accept.headers.get("content-type", "")


def test_request_id_middleware():
    client = TestClient(app)

    # Automatically generated request ID
    res1 = client.get("/api/live")
    assert res1.status_code == 200
    assert "X-Request-ID" in res1.headers
    assert len(res1.headers["X-Request-ID"]) > 0

    # Custom request ID preserved
    custom_id = "test-custom-trace-12345"
    res2 = client.get("/api/live", headers={"X-Request-ID": custom_id})
    assert res2.status_code == 200
    assert res2.headers.get("X-Request-ID") == custom_id


def test_audit_export_json_and_csv():
    client = TestClient(app)

    # JSON export
    res_json = client.get("/api/audit/export?format=json")
    assert res_json.status_code == 200
    data = res_json.json()
    assert "events" in data
    assert "owner_id" in data

    # CSV export
    res_csv = client.get("/api/audit/export?format=csv")
    assert res_csv.status_code == 200
    assert "text/csv" in res_csv.headers.get("content-type", "")
    assert "attachment; filename=\"simpliscribe_audit_events.csv\"" in res_csv.headers.get("content-disposition", "")
    assert "id,created_at,event_type,analysis_id,metadata" in res_csv.text
