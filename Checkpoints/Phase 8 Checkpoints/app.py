
import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from typing import Optional

import torch
import torch.nn as nn
import lightgbm as lgb
import shap
from xgboost import XGBClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.preprocessing import RobustScaler

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import io

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────
DRIVE = "/content/drive/MyDrive/Colab Notebooks"
P3    = f"{DRIVE}/phase3_checkpoints"
P4    = f"{DRIVE}/phase4_checkpoints"
P5    = f"{DRIVE}/phase5_checkpoints"
P7    = f"{DRIVE}/phase7_checkpoints"

# ── Autoencoder definition (must match Phase 5) ──────────────
class FraudAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        def enc_block(i, o):
            return nn.Sequential(nn.Linear(i,o), nn.BatchNorm1d(o), nn.ReLU(), nn.Dropout(0.2))
        def dec_block(i, o):
            return nn.Sequential(nn.Linear(i,o), nn.BatchNorm1d(o), nn.ReLU(), nn.Dropout(0.2))
        self.encoder = nn.Sequential(enc_block(input_dim,64), enc_block(64,32), enc_block(32,16))
        self.decoder = nn.Sequential(dec_block(16,32), dec_block(32,64), nn.Linear(64,input_dim))
    def forward(self, x):
        return self.decoder(self.encoder(x))

# ── Global artifact store ────────────────────────────────────
artifacts = {}

# ── Lifespan: load all artifacts once at startup ─────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Startup] Loading artifacts...")

    # Column lists
    with open(f"{P3}/phase3_top150_cols.json") as f:
        artifacts["top150_cols"] = json.load(f)
    with open(f"{P3}/phase3_eng_params.json") as f:
        artifacts["eng_params"] = json.load(f)

    # Preprocessing params
    with open(f"{P3}/phase3_top150_cols.json") as f:
        artifacts["top150_cols"] = json.load(f)

    import json as _json
    with open(f"{DRIVE}/phase2_checkpoints/phase2_medians.json") as f:
        artifacts["medians"] = _json.load(f)
    with open(f"{DRIVE}/phase2_checkpoints/phase2_flag_cols.json") as f:
        artifacts["flag_cols"] = _json.load(f)
    with open(f"{DRIVE}/phase2_checkpoints/phase2_feature_cols.json") as f:
        artifacts["feature_cols"] = _json.load(f)

    # Layer 1 models
    xgb = XGBClassifier()
    xgb.load_model(f"{P4}/phase4_xgb_model.json")
    artifacts["xgb"] = xgb

    artifacts["lgb"] = lgb.Booster(model_file=f"{P4}/phase4_lgb_model.txt")

    with open(f"{P4}/phase4_brf_model.pkl", "rb") as f:
        artifacts["brf"] = pickle.load(f)

    with open(f"{P4}/phase4_threshold.json") as f:
        artifacts["l1_threshold"] = json.load(f)["l1_threshold"]

    # Layer 2 model + scaler
    with open(f"{P5}/phase5_scaler.pkl", "rb") as f:
        artifacts["scaler"] = pickle.load(f)

    with open(f"{P5}/phase5_p99_normalization.json") as f:
        artifacts["p99"] = json.load(f)["p99_train"]

    ae = FraudAutoencoder(input_dim=150)
    ae.load_state_dict(torch.load(f"{P5}/phase5_autoencoder.pt",
                                   map_location="cpu"))
    ae.eval()
    artifacts["autoencoder"] = ae

    # Fusion params
    with open(f"{P7}/phase7_fusion_weights.json") as f:
        w = json.load(f)
        artifacts["w1"] = w["w1"]
        artifacts["w2"] = w["w2"]
        artifacts["w3"] = w["w3"]

    with open(f"{P7}/phase7_score_norm_params.json") as f:
        n = json.load(f)
        artifacts["l2_lo"]     = n["l2_lo"]
        artifacts["l2_hi"]     = n["l2_hi"]
        artifacts["fused_lo"]  = n["fused_lo"]
        artifacts["fused_hi"]  = n["fused_hi"]

    with open(f"{P7}/phase7_fused_threshold.json") as f:
        artifacts["fused_threshold"] = json.load(f)["fused_threshold"]

    # SHAP explainer (LightGBM)
    artifacts["explainer"] = shap.TreeExplainer(artifacts["lgb"])

    print("[Startup] All artifacts loaded.")
    yield
    print("[Shutdown] API stopped.")

app = FastAPI(title="Mule Account Detection API", lifespan=lifespan)


# ── Preprocessing pipeline ───────────────────────────────────
def preprocess(df: pd.DataFrame) -> np.ndarray:
    """
    Apply Phase 2-3 preprocessing to a raw feature DataFrame.
    Mirrors the exact pipeline used during training.
    Returns float32 numpy array of shape (n, 150).
    """
    df = df.copy()

    # Drop leakage column if present
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)

    # Standardize NA markers
    df.replace(["NA","na","NaN","nan",""], np.nan, inplace=True)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Binary missingness flags (training flag cols only)
    flag_col_map = {f: f.replace("_missing","") for f in artifacts["flag_cols"]}
    for flag_name, src_col in flag_col_map.items():
        if src_col in df.columns:
            df[flag_name] = df[src_col].isna().astype(np.float32)
        else:
            df[flag_name] = 0.0

    # Median imputation (training medians)
    for col, median_val in artifacts["medians"].items():
        if col in df.columns:
            df[col] = df[col].fillna(median_val)
        else:
            df[col] = median_val   # column absent: fill with training median

    # Feature engineering
    eng = artifacts["eng_params"]
    clip_bounds = eng.get("clip_bounds", {})

    def safe_ratio(a, b):
        return df[a] / (df[b] + 1e-9) if a in df.columns and b in df.columns else None

    r1 = safe_ratio("F115",  "F531")
    r2 = safe_ratio("F670",  "F1692")
    if r1 is not None: df["ratio_F115_F531"]  = r1
    if r2 is not None: df["ratio_F670_F1692"] = r2
    if r1 is not None and r2 is not None:
        df["ratio_interaction"] = r1 * r2

    zscore_cols = []
    for col_name, params in eng.items():
        if col_name == "clip_bounds": continue
        src = params.get("source")
        if src and src in df.columns:
            df[col_name] = (df[src] - params["mean"]) / (params["std"] + 1e-9)
            zscore_cols.append(col_name)

    if zscore_cols:
        df["composite_risk_score"] = df[zscore_cols].mean(axis=1)

    # Clip engineered features
    for col, bounds in clip_bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=bounds["p1"], upper=bounds["p99"])

    # Select top 150 columns (fill missing with 0)
    top150 = artifacts["top150_cols"]
    for col in top150:
        if col not in df.columns:
            df[col] = 0.0

    return df[top150].values.astype(np.float32)


# ── Scoring pipeline ─────────────────────────────────────────
def score_accounts(X_raw: np.ndarray) -> dict:
    """
    Run all three layers + fusion on preprocessed float32 array.
    Returns dict with per-account scores and metadata.
    """
    n = len(X_raw)

    # Layer 1
    p_xgb = artifacts["xgb"].predict_proba(X_raw)[:, 1]
    p_lgb = artifacts["lgb"].predict(X_raw)
    p_brf = artifacts["brf"].predict_proba(X_raw)[:, 1]
    l1    = np.clip((p_xgb + p_lgb + p_brf) / 3.0, 0.0, 1.0)

    # Layer 2
    tensor  = torch.tensor(artifacts["scaler"].transform(X_raw),
                           dtype=torch.float32)
    with torch.no_grad():
        recon = artifacts["autoencoder"](tensor)
    mse    = ((tensor - recon) ** 2).mean(dim=1).numpy()
    l2_raw = np.clip(mse / artifacts["p99"], 0, None)
    l2     = np.clip((l2_raw - artifacts["l2_lo"]) /
                     (artifacts["l2_hi"] - artifacts["l2_lo"] + 1e-9),
                     0.0, 1.0)

    # Layer 3 — no graph available at inference time for new accounts
    # Default to 0.0; graph-based score only meaningful in batch
    # re-scoring against known network (handled in /batch)
    l3 = np.zeros(n, dtype=np.float32)

    # Fusion
    w1, w2, w3 = artifacts["w1"], artifacts["w2"], artifacts["w3"]
    fused_raw  = w1*l1 + w2*l2 + w3*l3
    lo, hi     = artifacts["fused_lo"], artifacts["fused_hi"]
    fused_100  = np.clip((fused_raw - lo) / (hi - lo + 1e-9) * 100, 0, 100)

    return {
        "l1": l1, "l2": l2, "l3": l3,
        "fused_100": fused_100
    }


def get_shap_top5(X_raw_row: np.ndarray) -> list:
    top150 = artifacts["top150_cols"]
    sv     = artifacts["explainer"].shap_values(X_raw_row)
    if isinstance(sv, list):
        sv = sv[1]
    sv = sv[0]
    top_idx = np.argsort(np.abs(sv))[::-1][:5]
    return [
        {
            "feature"   : top150[i],
            "shap_value": float(sv[i]),
            "feat_value": float(X_raw_row[0, i]),
            "direction" : "↑ risk" if sv[i] > 0 else "↓ risk"
        }
        for i in top_idx
    ]


def build_alert(token: str, score: float, tier: str,
                l1: float, l2: float, l3: float,
                shap_feats: list) -> str:
    lines = [
        "=" * 55,
        f"  RISK ALERT — Account {token[:8]}...",
        "=" * 55,
        f"  Risk Score : {score:.1f} / 100   |   Tier: {tier}",
        "",
        "  Layer Breakdown:",
        f"    ML Ensemble Score : {l1*100:.1f}",
        f"    Anomaly Score     : {l2*100:.1f}",
        f"    Network Score     : {l3*100:.1f}",
        "",
        "  Key Risk Drivers:",
    ]
    for rank, f in enumerate(shap_feats, 1):
        word = "elevated" if f["shap_value"] > 0 else "suppressed"
        lines.append(
            f"    {rank}. {f['feature']} = {f['feat_value']:.4f}"
            f"  → risk {word} (SHAP {f['shap_value']:+.4f})"
        )
    action = (
        "Escalate for manual review."   if tier == "High"   else
        "Flag for secondary screening." if tier == "Medium" else
        "Monitor — low priority."
    )
    lines += ["", f"  Recommended Action: {action}", "=" * 55]
    return "\n".join(lines)


def assign_tier(s):
    if s >= 70: return "High"
    if s >= 40: return "Medium"
    return "Low"


# ── Request / Response schemas ────────────────────────────────
class AccountFeatures(BaseModel):
    token: str
    features: dict   # {F1: value, F2: value, ...}


class ScoreResponse(BaseModel):
    token: str
    risk_score: float
    tier: str
    l1_score: float
    l2_score: float
    l3_score: float
    flagged: bool
    top_shap_features: list
    alert: Optional[str] = None


# ── Endpoints ────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "artifacts_loaded": len(artifacts)}


@app.post("/score", response_model=ScoreResponse)
def score_single(account: AccountFeatures):
    """
    Score a single account.
    Input : JSON with token string and features dict.
    Output: risk score, tier, layer breakdown, SHAP, alert.
    """
    try:
        df     = pd.DataFrame([account.features])
        X_raw  = preprocess(df)
        scores = score_accounts(X_raw)

        fused  = float(scores["fused_100"][0])
        l1     = float(scores["l1"][0])
        l2     = float(scores["l2"][0])
        l3     = float(scores["l3"][0])
        tier   = assign_tier(fused)
        flagged = fused >= artifacts["fused_threshold"]

        shap_feats = get_shap_top5(X_raw) if flagged else []
        alert      = (build_alert(account.token, fused, tier,
                                  l1, l2, l3, shap_feats)
                      if flagged else None)

        return ScoreResponse(
            token              = account.token,
            risk_score         = round(fused, 2),
            tier               = tier,
            l1_score           = round(l1, 4),
            l2_score           = round(l2, 4),
            l3_score           = round(l3, 4),
            flagged            = flagged,
            top_shap_features  = shap_feats,
            alert              = alert
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch")
async def score_batch(file: UploadFile = File(...)):
    """
    Score a batch of accounts from a CSV file.
    CSV must contain feature columns F1–F3923 and optionally
    a token column. Unnamed:0 dropped automatically.
    Returns CSV with appended risk_score, tier, flagged columns.
    """
    try:
        contents = await file.read()
        df_input = pd.read_csv(io.BytesIO(contents), na_values=["NA"])

        # Extract or generate tokens
        if "token" in df_input.columns:
            tokens = df_input["token"].astype(str).tolist()
            df_feat = df_input.drop(columns=["token"])
        else:
            tokens = [f"account_{i}" for i in range(len(df_input))]
            df_feat = df_input.copy()

        X_raw  = preprocess(df_feat)
        scores = score_accounts(X_raw)

        fused_scores = scores["fused_100"]
        tiers        = [assign_tier(s) for s in fused_scores]
        flagged      = fused_scores >= artifacts["fused_threshold"]

        df_output = df_input.copy()
        df_output["token"]      = tokens
        df_output["risk_score"] = np.round(fused_scores, 2)
        df_output["l1_score"]   = np.round(scores["l1"], 4)
        df_output["l2_score"]   = np.round(scores["l2"], 4)
        df_output["tier"]       = tiers
        df_output["flagged"]    = flagged.astype(bool)

        # Drop Unnamed:0 from output
        if "Unnamed: 0" in df_output.columns:
            df_output.drop(columns=["Unnamed: 0"], inplace=True)

        output_buffer = io.StringIO()
        df_output.to_csv(output_buffer, index=False)
        output_buffer.seek(0)

        return StreamingResponse(
            io.BytesIO(output_buffer.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition":
                     "attachment; filename=scored_accounts.csv"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
