ensemble_model = joblib.load("drive/MyDrive/botnet_model/botnet_ensemble.pkl")
meta           = joblib.load("drive/MyDrive/botnet_model/botnet_metadata.pkl")
autoencoder    = load_model("drive/MyDrive/botnet_model/botnet_autoencoder.keras")

FEATURE_COLUMNS     = meta["feature_columns"]
AE_FEATURE_COLUMNS  = meta["ae_feature_columns"]
ENGINEERED_FEATURES = meta["engineered_features"]
scaler_ae           = meta["scaler_ae"]
pca_ae              = meta["pca_ae"]
threshold           = meta["ae_threshold"]

# ── Re-define helpers (copy from training script) ──
def safe_log(x):
    return np.log1p(np.clip(x, 0, 1e6))

def add_features(df):
    df = df.copy()
    df["packet_ratio"]      = safe_log(df["spkts"] / (df["dpkts"] + 1))
    df["byte_ratio"]        = safe_log(df["sbytes"] / (df["dbytes"] + 1))
    df["traffic_intensity"] = safe_log(df["rate"] * df["dur"])
    df["avg_packet_size"]   = safe_log(
        (df["sbytes"] + df["dbytes"]) / (df["spkts"] + df["dpkts"] + 1))
    df["ttl_difference"]    = safe_log(abs(df["sttl"] - df["dttl"]))
    df["packet_rate"]       = safe_log(
        (df["spkts"] + df["dpkts"]) / (df["dur"] + 0.001))
    return df

my_sample = {
    "dur":     5.2,
    "proto":   "tcp",       # raw string — OHE is handled internally
    "service": "http",
    "state":   "CON",
    "spkts":   80,
    "dpkts":   90,
    "sbytes":  3200,
    "dbytes":  3600,
    "rate":    25.0,
    "sttl":    64,
    "dttl":    64,
    # optional extra fields if available:
    "sload":   100, "dload": 110,
    "ct_dst_src_ltm": 3, "ct_state_ttl": 2,
}
# ── Now predict normally ──
print(predict_botnet(my_sample))
print(predict_autoencoder(my_sample))