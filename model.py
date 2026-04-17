"""
model.py — Pydantic v2 JSON contracts for the Botnet Detection API.

Defines request/response schemas for:
  - Network flow input
  - Ensemble prediction output
  - Autoencoder anomaly detection output
  - Combined prediction output
  - Health check response
"""

from pydantic import BaseModel, Field
from typing import Optional


# ── Request Schema ──────────────────────────────────────────────


class NetworkFlowInput(BaseModel):
    """
    Raw network-flow features for botnet detection.

    Required fields come from the UNSW-NB15 dataset core columns.
    Optional fields are additional counters that improve accuracy
    when available but are not strictly required.
    """

    # ── Required fields ──
    dur: float = Field(..., description="Connection duration in seconds")
    proto: str = Field(..., description="Protocol (e.g. 'tcp', 'udp')")
    service: str = Field(..., description="Service type (e.g. 'http', 'dns', '-')")
    state: str = Field(..., description="Connection state (e.g. 'CON', 'INT', 'FIN')")
    spkts: int = Field(..., description="Source-to-destination packet count")
    dpkts: int = Field(..., description="Destination-to-source packet count")
    sbytes: int = Field(..., description="Source-to-destination bytes")
    dbytes: int = Field(..., description="Destination-to-source bytes")
    rate: float = Field(..., description="Connection rate (packets/sec)")
    sttl: int = Field(..., description="Source TTL value")
    dttl: int = Field(..., description="Destination TTL value")

    # ── Optional fields ──
    sload: Optional[float] = Field(None, description="Source bits per second")
    dload: Optional[float] = Field(None, description="Destination bits per second")
    ct_dst_src_ltm: Optional[int] = Field(
        None, description="No. of connections of the same src and dst in 100 records"
    )
    ct_state_ttl: Optional[int] = Field(
        None, description="No. of connections of same state and TTL"
    )
    ct_srv_dst: Optional[int] = Field(
        None, description="No. of connections of same service and dst in 100 records"
    )
    ct_dst_sport_ltm: Optional[int] = Field(
        None, description="No. of connections of same dst and src port in 100 records"
    )
    ct_src_dport_ltm: Optional[int] = Field(
        None, description="No. of connections of same src and dst port in 100 records"
    )
    ct_dst_ltm: Optional[int] = Field(
        None, description="No. of connections of same dst in 100 records"
    )

    model_config = {"json_schema_extra": {
        "examples": [
            {
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
        ]
    }}

    def to_dict(self) -> dict:
        """Convert to plain dict, dropping None values (optional fields not provided)."""
        return {k: v for k, v in self.model_dump().items() if v is not None}


# ── Response Schemas ────────────────────────────────────────────


class EnsemblePrediction(BaseModel):
    """Response from the ensemble (RF + XGBoost + LightGBM) classifier."""

    prediction: str = Field(
        ..., description="Classification: 'BOTNET', 'SUSPICIOUS', or 'NORMAL'"
    )
    confidence: float = Field(
        ..., description="Probability of being botnet traffic (0.0–1.0)"
    )


class AutoencoderPrediction(BaseModel):
    """Response from the autoencoder anomaly detector."""

    prediction: str = Field(
        ..., description="Classification: 'BOTNET' or 'NORMAL'"
    )
    anomaly_score: float = Field(
        ..., description="Reconstruction error (higher → more anomalous)"
    )


class CombinedPrediction(BaseModel):
    """Combined response from both models."""

    ensemble: EnsemblePrediction
    autoencoder: AutoencoderPrediction


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    models_loaded: bool = Field(..., description="Whether all models are loaded")
