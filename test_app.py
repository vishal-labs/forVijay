"""
test_app.py — Tests for the Botnet Detection API.

Uses FastAPI TestClient to exercise all endpoints with sample data
from the original training notebook.
"""

import pytest
from fastapi.testclient import TestClient

from app import app

client = TestClient(app)

# ── Sample payloads (from main.py / vishal_version_botnet.py) ───

NORMAL_SAMPLE = {
    "dur": 5.2,
    "proto": "tcp",
    "service": "http",
    "state": "CON",
    "spkts": 80,
    "dpkts": 90,
    "sbytes": 3200,
    "dbytes": 3600,
    "rate": 25.0,
    "sttl": 64,
    "dttl": 64,
    "sload": 100,
    "dload": 110,
    "ct_dst_src_ltm": 3,
    "ct_state_ttl": 2,
}

BOTNET_SAMPLE = {
    "dur": 0.005,
    "proto": "udp",
    "service": "-",
    "state": "INT",
    "spkts": 2000,
    "dpkts": 2,
    "sbytes": 120000,
    "dbytes": 100,
    "rate": 80000,
    "sttl": 255,
    "dttl": 5,
}

MEDIUM_SAMPLE = {
    "dur": 2,
    "proto": "udp",
    "service": "dns",
    "state": "CON",
    "spkts": 300,
    "dpkts": 50,
    "sbytes": 40000,
    "dbytes": 5000,
    "rate": 1500,
    "sttl": 120,
    "dttl": 40,
    "sload": 20000,
    "dload": 3000,
    "ct_dst_src_ltm": 25,
    "ct_state_ttl": 20,
    "ct_srv_dst": 18,
    "ct_dst_sport_ltm": 15,
    "ct_src_dport_ltm": 12,
    "ct_dst_ltm": 14,
}


# ── General endpoints ──────────────────────────────────────────


class TestGeneral:
    def test_root(self):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "service" in data
        assert data["docs"] == "/docs"

    def test_health(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["models_loaded"] is True


# ── Ensemble endpoint ──────────────────────────────────────────


class TestEnsemble:
    def test_normal_sample(self):
        resp = client.post("/predict/ensemble", json=NORMAL_SAMPLE)
        assert resp.status_code == 200
        data = resp.json()
        assert data["prediction"] in ("BOTNET", "SUSPICIOUS", "NORMAL")
        assert 0.0 <= data["confidence"] <= 1.0

    def test_botnet_sample(self):
        resp = client.post("/predict/ensemble", json=BOTNET_SAMPLE)
        assert resp.status_code == 200
        data = resp.json()
        assert data["prediction"] in ("BOTNET", "SUSPICIOUS", "NORMAL")
        assert 0.0 <= data["confidence"] <= 1.0

    def test_medium_sample(self):
        resp = client.post("/predict/ensemble", json=MEDIUM_SAMPLE)
        assert resp.status_code == 200
        data = resp.json()
        assert data["prediction"] in ("BOTNET", "SUSPICIOUS", "NORMAL")

    def test_missing_required_field(self):
        """Omit 'dur' — should return 422 validation error."""
        bad = {k: v for k, v in NORMAL_SAMPLE.items() if k != "dur"}
        resp = client.post("/predict/ensemble", json=bad)
        assert resp.status_code == 422


# ── Autoencoder endpoint ───────────────────────────────────────


class TestAutoencoder:
    def test_normal_sample(self):
        resp = client.post("/predict/autoencoder", json=NORMAL_SAMPLE)
        assert resp.status_code == 200
        data = resp.json()
        assert data["prediction"] in ("BOTNET", "NORMAL")
        assert isinstance(data["anomaly_score"], float)

    def test_botnet_sample(self):
        resp = client.post("/predict/autoencoder", json=BOTNET_SAMPLE)
        assert resp.status_code == 200
        data = resp.json()
        assert data["prediction"] in ("BOTNET", "NORMAL")

    def test_missing_required_field(self):
        bad = {k: v for k, v in BOTNET_SAMPLE.items() if k != "proto"}
        resp = client.post("/predict/autoencoder", json=bad)
        assert resp.status_code == 422


# ── Combined endpoint ──────────────────────────────────────────


class TestCombined:
    def test_normal_sample(self):
        resp = client.post("/predict/combined", json=NORMAL_SAMPLE)
        assert resp.status_code == 200
        data = resp.json()
        assert "ensemble" in data
        assert "autoencoder" in data
        assert data["ensemble"]["prediction"] in ("BOTNET", "SUSPICIOUS", "NORMAL")
        assert data["autoencoder"]["prediction"] in ("BOTNET", "NORMAL")

    def test_botnet_sample(self):
        resp = client.post("/predict/combined", json=BOTNET_SAMPLE)
        assert resp.status_code == 200
        data = resp.json()
        assert "ensemble" in data and "autoencoder" in data

    def test_minimal_fields_only(self):
        """Send only the 11 required fields — optional fields omitted."""
        minimal = {
            "dur": 1.0,
            "proto": "tcp",
            "service": "-",
            "state": "FIN",
            "spkts": 10,
            "dpkts": 10,
            "sbytes": 500,
            "dbytes": 500,
            "rate": 5.0,
            "sttl": 64,
            "dttl": 64,
        }
        resp = client.post("/predict/combined", json=minimal)
        assert resp.status_code == 200


# ── Edge cases ─────────────────────────────────────────────────


class TestEdgeCases:
    def test_zero_values(self):
        """All numeric fields at zero — should not crash."""
        zero = {
            "dur": 0,
            "proto": "tcp",
            "service": "-",
            "state": "CON",
            "spkts": 0,
            "dpkts": 0,
            "sbytes": 0,
            "dbytes": 0,
            "rate": 0,
            "sttl": 0,
            "dttl": 0,
        }
        resp = client.post("/predict/combined", json=zero)
        assert resp.status_code == 200

    def test_large_values(self):
        """Extreme large values — should not crash."""
        large = {
            "dur": 999999,
            "proto": "udp",
            "service": "http",
            "state": "INT",
            "spkts": 999999,
            "dpkts": 999999,
            "sbytes": 999999999,
            "dbytes": 999999999,
            "rate": 999999,
            "sttl": 255,
            "dttl": 255,
        }
        resp = client.post("/predict/combined", json=large)
        assert resp.status_code == 200

    def test_empty_body(self):
        """Empty JSON body — should return 422."""
        resp = client.post("/predict/ensemble", json={})
        assert resp.status_code == 422
