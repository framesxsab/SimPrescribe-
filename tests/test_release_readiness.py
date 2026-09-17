from fastapi.testclient import TestClient

from simpliscribe.main import app


def test_favicon_redirects_to_existing_static_asset():
    response = TestClient(app).get("/favicon.ico", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/static/favicon.svg"
