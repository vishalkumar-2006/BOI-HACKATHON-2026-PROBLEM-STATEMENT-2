
import os, json, pickle, time
import numpy as np
import pandas as pd
import networkx as nx
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import requests as req
from pyvis.network import Network
import tempfile, io

st.set_page_config(
    page_title="Mule Account Detection",
    page_icon="🔍",
    layout="wide"
)

DRIVE    = "/content/drive/MyDrive/Colab Notebooks"
P2       = f"{DRIVE}/phase2_checkpoints"
P3       = f"{DRIVE}/phase3_checkpoints"
P4       = f"{DRIVE}/phase4_checkpoints"
P5       = f"{DRIVE}/phase5_checkpoints"
P6       = f"{DRIVE}/phase6_checkpoints"
P7       = f"{DRIVE}/phase7_checkpoints"
API_URL  = "http://localhost:8000"   # Phase 8 API (same Colab session)

# ── Load artifacts (cached) ──────────────────────────────────
@st.cache_resource
def load_artifacts():
    arts = {}
    with open(f"{P2}/phase2_test_tokens.json") as f:
        arts["test_tokens"] = json.load(f)
    with open(f"{P3}/phase3_top150_cols.json") as f:
        arts["top150_cols"] = json.load(f)
    arts["X_test"]  = pd.read_parquet(f"{P3}/phase3_X_test.parquet")
    arts["y_test"]  = pd.read_parquet(f"{P3}/phase3_y_test.parquet")["label"]
    arts["test_fused_scores"] = np.load(f"{P7}/phase7_test_fused_scores.npy")
    arts["test_tiers"]        = np.load(f"{P7}/phase7_test_tiers.npy")
    arts["shap_values"]       = np.load(f"{P7}/phase7_shap_values.npy")
    with open(f"{P7}/phase7_alerts.json") as f:
        arts["alerts"] = json.load(f)
    with open(f"{P7}/phase7_fusion_weights.json") as f:
        arts["weights"] = json.load(f)
    with open(f"{P7}/phase7_fused_threshold.json") as f:
        arts["threshold"] = json.load(f)["fused_threshold"]
    with open(f"{P7}/phase7_test_metrics_fused.json") as f:
        arts["metrics_fused"] = json.load(f)
    arts["test_l1"]        = np.load(f"{P4}/phase4_test_probs_l1.npy")
    arts["test_l2"]        = np.load(f"{P5}/phase5_test_anomaly_scores.npy")
    arts["test_l3"]        = np.load(f"{P6}/phase6_test_l3_scores.npy")
    arts["layer_comparison"] = pd.read_csv(f"{P7}/phase7_layer_comparison.csv")
    arts["graph"]          = nx.read_graphml(f"{P6}/phase6_ego_subgraph.graphml")
    with open(f"{P6}/phase6_token_to_idx.json") as f:
        arts["token_to_idx"] = json.load(f)
    return arts

arts = load_artifacts()

tokens     = arts["test_tokens"]
X_test     = arts["X_test"]
y_test     = arts["y_test"]
top150     = arts["top150_cols"]
scores     = arts["test_fused_scores"]
tiers      = arts["test_tiers"]
shap_vals  = arts["shap_values"]
alerts     = arts["alerts"]
threshold  = arts["threshold"]
weights    = arts["weights"]
l1_scores  = arts["test_l1"]
l2_scores  = arts["test_l2"]
l3_scores  = arts["test_l3"]
metrics    = arts["metrics_fused"]
layer_comp = arts["layer_comparison"]
G          = arts["graph"]
tok_to_idx = arts["token_to_idx"]

HINT_FEATURES = [
    "F115","F321","F527","F531","F670","F1692","F2082","F2122",
    "F2582","F2678","F2737","F2956","F3043","F3836","F3887",
    "F3889","F3891","F3894"
]

scored_df = pd.DataFrame({
    "token"      : tokens,
    "risk_score" : np.round(scores, 1),
    "tier"       : tiers,
    "l1_score"   : np.round(l1_scores * 100, 1),
    "l2_score"   : np.round(l2_scores * 100, 1),
    "l3_score"   : np.round(l3_scores * 100, 1),
    "flagged"    : scores >= threshold,
    "true_label" : y_test.values
})
flagged_df = scored_df[scored_df["flagged"]].sort_values(
    "risk_score", ascending=False
).reset_index(drop=True)


# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════
st.sidebar.title("🔍 Mule Detection")
st.sidebar.markdown("---")

tier_filter  = st.sidebar.multiselect(
    "Filter by Risk Tier",
    options=["High","Medium","Low"],
    default=["High","Medium"]
)
score_range  = st.sidebar.slider(
    "Risk Score Range", 0, 100,
    (int(threshold), 100)
)
token_search = st.sidebar.text_input("Search Token (partial match)")

st.sidebar.markdown("---")
st.sidebar.markdown("**Fusion Weights**")
st.sidebar.markdown(f"- ML Ensemble : `{weights['w1']:.3f}`")
st.sidebar.markdown(f"- Autoencoder : `{weights['w2']:.3f}`")
st.sidebar.markdown(f"- Sim Network : `{weights['w3']:.3f}`")
st.sidebar.markdown(f"\n**Flag Threshold** : `{threshold:.1f} / 100`")

# API status indicator in sidebar
st.sidebar.markdown("---")
try:
    health = req.get(f"{API_URL}/health", timeout=3)
    if health.status_code == 200:
        st.sidebar.success("🟢 Scoring API: Online")
    else:
        st.sidebar.warning("🟡 Scoring API: Degraded")
except Exception:
    st.sidebar.error("🔴 Scoring API: Offline")

view_df = scored_df[
    (scored_df["tier"].isin(tier_filter)) &
    (scored_df["risk_score"] >= score_range[0]) &
    (scored_df["risk_score"] <= score_range[1])
]
if token_search:
    view_df = view_df[view_df["token"].str.contains(token_search)]


# ════════════════════════════════════════════════════════════
# SECTION 1 — KPI CARDS
# ════════════════════════════════════════════════════════════
st.title("Mule Account Detection — Investigator Dashboard")
st.markdown("---")

c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric("Total Accounts",  len(scored_df))
c2.metric("Flagged",         int(scored_df["flagged"].sum()))
c3.metric("🔴 High Risk",    int((tiers=="High").sum()))
c4.metric("🟡 Medium Risk",  int((tiers=="Medium").sum()))
c5.metric("🟢 Low Risk",     int((tiers=="Low").sum()))
c6.metric("System F2",       f"{metrics['f2']:.3f}")
st.markdown("---")


# ════════════════════════════════════════════════════════════
# SECTION 2 — SCORE DISTRIBUTION
# ════════════════════════════════════════════════════════════
st.subheader("Risk Score Distribution")
mule_sc  = scores[y_test.values==1]
legit_sc = scores[y_test.values==0]

fig_dist = go.Figure()
fig_dist.add_trace(go.Histogram(x=legit_sc, name="Legitimate",
    nbinsx=40, opacity=0.6, marker_color="#2196F3"))
fig_dist.add_trace(go.Histogram(x=mule_sc, name="Mule",
    nbinsx=40, opacity=0.85, marker_color="#F44336"))
fig_dist.add_vline(x=threshold, line_dash="dash", line_color="#FF9800",
    annotation_text=f"Threshold={threshold:.1f}",
    annotation_position="top right")
fig_dist.update_layout(barmode="overlay", height=300,
    xaxis_title="Risk Score (0–100)", yaxis_title="Count",
    template="plotly_white", margin=dict(t=20,b=40))
st.plotly_chart(fig_dist, use_container_width=True)
st.markdown("---")


# ════════════════════════════════════════════════════════════
# SECTION 3 — FLAGGED ACCOUNTS TABLE
# ════════════════════════════════════════════════════════════
st.subheader(f"Flagged Accounts ({len(view_df)} matching filters)")

def tier_color(t):
    return {"High":"🔴","Medium":"🟡","Low":"🟢"}.get(t,"")

display_df = view_df.copy()
display_df["tier"]        = display_df["tier"].apply(lambda t: f"{tier_color(t)} {t}")
display_df["token_short"] = display_df["token"].str[:16] + "..."

st.dataframe(
    display_df[["token_short","risk_score","tier",
                "l1_score","l2_score","l3_score","true_label"]].rename(columns={
        "token_short":"Token","risk_score":"Risk Score","tier":"Tier",
        "l1_score":"ML Score","l2_score":"Anomaly Score",
        "l3_score":"Network Score","true_label":"True Label"
    }),
    use_container_width=True, height=300
)
st.markdown("---")


# ════════════════════════════════════════════════════════════
# SECTION 4 — ACCOUNT DRILLDOWN
# ════════════════════════════════════════════════════════════
st.subheader("Account Drilldown")
drilldown_tokens = flagged_df["token"].tolist()

if not drilldown_tokens:
    st.info("No flagged accounts match current filters.")
else:
    selected_token = st.selectbox(
        "Select account token",
        options=drilldown_tokens,
        format_func=lambda t: t[:24]+"..."
    )
    sel_row = scored_df[scored_df["token"]==selected_token].iloc[0]
    sel_idx = tokens.index(selected_token)

    col_left, col_right = st.columns([1,2])

    with col_left:
        tier_col = {"High":"#F44336","Medium":"#FF9800","Low":"#4CAF50"}
        color    = tier_col.get(sel_row["tier"],"#9E9E9E")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(sel_row["risk_score"]),
            title={"text": f"Risk Score — {sel_row['tier']} Risk"},
            gauge={
                "axis":{"range":[0,100]},
                "bar":{"color":color},
                "steps":[
                    {"range":[0,40],"color":"#E8F5E9"},
                    {"range":[40,70],"color":"#FFF8E1"},
                    {"range":[70,100],"color":"#FFEBEE"},
                ],
                "threshold":{"line":{"color":"#FF9800","width":3},
                             "thickness":0.75,"value":threshold}
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(t=40,b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        fig_bar = go.Figure(go.Bar(
            x=["ML Ensemble","Anomaly","Network"],
            y=[float(sel_row["l1_score"]),
               float(sel_row["l2_score"]),
               float(sel_row["l3_score"])],
            marker_color=["#2196F3","#9C27B0","#FF9800"],
            text=[f"{v:.1f}" for v in [sel_row["l1_score"],
                                        sel_row["l2_score"],
                                        sel_row["l3_score"]]],
            textposition="outside"
        ))
        fig_bar.update_layout(title="Layer Score Breakdown (0–100)",
            yaxis=dict(range=[0,110]), height=280,
            template="plotly_white", margin=dict(t=40,b=20))
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_right:
        st.markdown("**Top SHAP Risk Drivers**")
        sv      = shap_vals[sel_idx]
        top_idx = np.argsort(np.abs(sv))[::-1][:10]
        shap_df = pd.DataFrame([{
            "Feature"    : top150[i],
            "SHAP Value" : round(float(sv[i]),4),
            "Direction"  : "↑ Risk" if sv[i]>0 else "↓ Risk",
            "Feat Value" : round(float(X_test.iloc[sel_idx,i]),4)
        } for i in top_idx])

        fig_shap = go.Figure(go.Bar(
            x=shap_df["SHAP Value"], y=shap_df["Feature"],
            orientation="h",
            marker_color=["#F44336" if v>0 else "#2196F3"
                          for v in shap_df["SHAP Value"]]
        ))
        fig_shap.update_layout(
            title="SHAP Values (red=increases risk, blue=decreases)",
            height=320, template="plotly_white",
            yaxis=dict(autorange="reversed"),
            margin=dict(t=40,b=20)
        )
        st.plotly_chart(fig_shap, use_container_width=True)
        st.dataframe(shap_df, use_container_width=True, height=200)

    if selected_token in alerts:
        with st.expander("📋 Full Alert Text", expanded=False):
            st.code(alerts[selected_token], language=None)

    st.markdown("**Similarity Network Neighborhood**")
    node_id = tok_to_idx.get(selected_token)
    if node_id is not None and str(node_id) in G.nodes():
        node_id_str = str(node_id)
        neighbors   = list(G.neighbors(node_id_str))
        sub_nodes   = [node_id_str] + neighbors[:30]
        H           = G.subgraph(sub_nodes)

        net = Network(height="400px", width="100%",
                      bgcolor="#1a1a2e", font_color="white")
        net.barnes_hut()

        for n in H.nodes():
            nd         = H.nodes[n]
            is_center  = (n == node_id_str)
            is_mule    = nd.get("is_mule_seed",0)==1
            prop_score = float(nd.get("prop_score",0))
            color = ("#FF9800" if is_center else
                     "#F44336" if is_mule   else
                     "#F44336" if prop_score>=0.5 else
                     "#FF9800" if prop_score>=0.25 else "#4CAF50")
            size  = 25 if is_center else (18 if is_mule else 10)
            label = nd.get("token",n)[:8]+"..."
            net.add_node(n, label=label, color=color,
                         size=size, title=f"Score: {prop_score:.2f}")

        for u,v,d in H.edges(data=True):
            net.add_edge(u,v,value=float(d.get("weight",0.92)),
                         color="#555577")

        with tempfile.NamedTemporaryFile(
            suffix=".html", delete=False, mode="w"
        ) as tmp:
            net.save_graph(tmp.name)
            html_content = open(tmp.name).read()

        st.components.v1.html(html_content, height=420, scrolling=False)
        st.caption("🔴 Confirmed mule  🟠 Selected/high propagation  "
                   "🟡 Hop-2 propagation  🟢 No propagation")
    else:
        st.info("Account not found in similarity network "
                "(no edges above threshold 0.92).")

st.markdown("---")


# ════════════════════════════════════════════════════════════
# SECTION 5 — SYSTEM PERFORMANCE
# ════════════════════════════════════════════════════════════
st.subheader("System Performance")
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("**Layer Comparison**")
    perf_df = layer_comp.copy()
    for col in ["auroc","prauc","f2"]:
        perf_df[col] = perf_df[col].round(3)
    st.dataframe(
        perf_df[["system","auroc","prauc","f2","tp","fp","fn"]].rename(columns={
            "system":"System","auroc":"AUROC","prauc":"PR-AUC",
            "f2":"F2","tp":"TP","fp":"FP","fn":"FN"
        }),
        use_container_width=True, hide_index=True
    )

with col_b:
    st.markdown("**Fused System Metrics**")
    m = metrics
    fig_met = go.Figure(go.Bar(
        x=["AUROC","PR-AUC","F2 Score"],
        y=[m["auroc"],m["prauc"],m["f2"]],
        marker_color=["#2196F3","#9C27B0","#4CAF50"],
        text=[f"{v:.3f}" for v in [m["auroc"],m["prauc"],m["f2"]]],
        textposition="outside"
    ))
    fig_met.update_layout(yaxis=dict(range=[0,1.1]), height=300,
        template="plotly_white", margin=dict(t=20,b=20))
    st.plotly_chart(fig_met, use_container_width=True)

st.markdown("---")
st.caption(
    f"Test set: {len(scored_df)} accounts | "
    f"{int(y_test.sum())} confirmed mules | "
    f"Primary metric: F2 Score | "
    f"TP={m['tp']} FP={m['fp']} FN={m['fn']}"
)
st.markdown("---")


# ════════════════════════════════════════════════════════════
# SECTION 6 — LIVE SCORING
# ════════════════════════════════════════════════════════════
st.subheader("🧪 Live Account Scoring")
st.markdown(
    "Score new accounts in real time using the Phase 8 inference API. "
    "Upload a CSV for batch scoring or fill the form for a single account."
)

# ── API health gate ───────────────────────────────────────────
try:
    api_ok = req.get(f"{API_URL}/health", timeout=3).status_code == 200
except Exception:
    api_ok = False

if not api_ok:
    st.error(
        "⚠️ Scoring API is offline. "
        "Run Cell 75 / 75c in the Colab notebook to start it, then refresh."
    )
else:
    live_tab1, live_tab2 = st.tabs(["📁 CSV Batch Upload", "✏️ Manual Single Account"])

    # ── TAB 1: CSV BATCH UPLOAD ───────────────────────────────
    with live_tab1:
        st.markdown(
            "Upload a CSV file with feature columns. "
            "Optionally include a **token** column for account identification. "
            "`Unnamed: 0` is dropped automatically."
        )

        uploaded_file = st.file_uploader(
            "Choose CSV file", type=["csv"], key="batch_upload"
        )

        if uploaded_file is not None:
            try:
                preview_df = pd.read_csv(uploaded_file, nrows=5)
                st.markdown(f"**Preview** ({uploaded_file.name}) — first 5 rows:")
                st.dataframe(preview_df, use_container_width=True)

                uploaded_file.seek(0)
                col_run, col_dl = st.columns([1,3])

                with col_run:
                    run_batch = st.button("▶ Score All Accounts",
                                          key="run_batch",
                                          type="primary")

                if run_batch:
                    with st.spinner("Sending to API..."):
                        uploaded_file.seek(0)
                        response = req.post(
                            f"{API_URL}/batch",
                            files={"file": (
                                uploaded_file.name,
                                uploaded_file.read(),
                                "text/csv"
                            )},
                            timeout=120
                        )

                    if response.status_code == 200:
                        result_df = pd.read_csv(
                            io.StringIO(response.content.decode())
                        )

                        # ── Summary metrics ───────────────────
                        n_total   = len(result_df)
                        n_flagged = int(result_df["flagged"].sum())
                        n_high    = int((result_df["tier"]=="High").sum())
                        n_med     = int((result_df["tier"]=="Medium").sum())
                        n_low     = int((result_df["tier"]=="Low").sum())

                        st.success(f"✓ Scored {n_total} accounts successfully.")

                        m1,m2,m3,m4,m5 = st.columns(5)
                        m1.metric("Total",   n_total)
                        m2.metric("Flagged", n_flagged)
                        m3.metric("🔴 High",   n_high)
                        m4.metric("🟡 Medium", n_med)
                        m5.metric("🟢 Low",    n_low)

                        # ── Score distribution ────────────────
                        fig_live = go.Figure(go.Histogram(
                            x=result_df["risk_score"],
                            nbinsx=30,
                            marker_color="#2196F3",
                            opacity=0.8,
                            name="Risk Scores"
                        ))
                        fig_live.add_vline(
                            x=threshold,
                            line_dash="dash",
                            line_color="#F44336",
                            annotation_text=f"Threshold={threshold:.1f}"
                        )
                        fig_live.update_layout(
                            title="Uploaded Batch — Risk Score Distribution",
                            xaxis_title="Risk Score (0–100)",
                            yaxis_title="Count",
                            height=280,
                            template="plotly_white",
                            margin=dict(t=40,b=30)
                        )
                        st.plotly_chart(fig_live, use_container_width=True)

                        # ── Results table ─────────────────────
                        st.markdown("**Scored Results**")
                        display_cols = ["token","risk_score","tier",
                                        "l1_score","l2_score","flagged"]
                        available    = [c for c in display_cols
                                        if c in result_df.columns]
                        st.dataframe(
                            result_df[available].sort_values(
                                "risk_score", ascending=False
                            ),
                            use_container_width=True,
                            height=350
                        )

                        # ── Download button ───────────────────
                        csv_out = result_df.to_csv(index=False).encode()
                        st.download_button(
                            label="⬇ Download Scored CSV",
                            data=csv_out,
                            file_name="scored_results.csv",
                            mime="text/csv"
                        )

                        # ── Flagged account drilldown ─────────
                        flagged_live = result_df[
                            result_df["flagged"]==True
                        ].sort_values("risk_score", ascending=False)

                        if len(flagged_live) > 0:
                            st.markdown("**Flagged Account Details**")
                            sel_live = st.selectbox(
                                "Select flagged account",
                                options=flagged_live["token"].tolist(),
                                format_func=lambda t: str(t)[:24]+"...",
                                key="live_drilldown"
                            )
                            row = flagged_live[
                                flagged_live["token"]==sel_live
                            ].iloc[0]

                            tier_col_map = {
                                "High":"#F44336",
                                "Medium":"#FF9800",
                                "Low":"#4CAF50"
                            }
                            dc1, dc2 = st.columns(2)

                            with dc1:
                                fig_g2 = go.Figure(go.Indicator(
                                    mode="gauge+number",
                                    value=float(row["risk_score"]),
                                    title={"text": f"{row['tier']} Risk"},
                                    gauge={
                                        "axis":{"range":[0,100]},
                                        "bar":{"color": tier_col_map.get(
                                            row["tier"],"#9E9E9E")},
                                        "steps":[
                                            {"range":[0,40],
                                             "color":"#E8F5E9"},
                                            {"range":[40,70],
                                             "color":"#FFF8E1"},
                                            {"range":[70,100],
                                             "color":"#FFEBEE"},
                                        ],
                                        "threshold":{
                                            "line":{"color":"#FF9800",
                                                    "width":3},
                                            "thickness":0.75,
                                            "value":threshold
                                        }
                                    }
                                ))
                                fig_g2.update_layout(
                                    height=260,
                                    margin=dict(t=40,b=10)
                                )
                                st.plotly_chart(fig_g2,
                                                use_container_width=True)

                            with dc2:
                                l1v = float(row.get("l1_score",0))
                                l2v = float(row.get("l2_score",0))
                                fig_b2 = go.Figure(go.Bar(
                                    x=["ML Ensemble","Anomaly","Network"],
                                    y=[l1v, l2v, 0],
                                    marker_color=["#2196F3","#9C27B0",
                                                  "#FF9800"],
                                    text=[f"{v:.1f}"
                                          for v in [l1v, l2v, 0]],
                                    textposition="outside"
                                ))
                                fig_b2.update_layout(
                                    title="Layer Breakdown",
                                    yaxis=dict(range=[0,110]),
                                    height=260,
                                    template="plotly_white",
                                    margin=dict(t=40,b=10)
                                )
                                st.plotly_chart(fig_b2,
                                                use_container_width=True)

                    else:
                        st.error(
                            f"API error {response.status_code}: "
                            f"{response.text[:300]}"
                        )

            except Exception as e:
                st.error(f"Error processing file: {e}")

    # ── TAB 2: MANUAL SINGLE ACCOUNT ─────────────────────────
    with live_tab2:
        st.markdown(
            "Enter values for the 18 domain hint features. "
            "All other features will be imputed using training-set medians. "
            "Leave a field blank to treat it as missing."
        )

        with open(
            f"{DRIVE}/phase3_checkpoints/phase3_eng_params.json"
        ) as f:
            eng_params = json.load(f)

        with open(
            f"{DRIVE}/phase2_checkpoints/phase2_medians.json"
        ) as f:
            medians_dict = json.load(f)

        # Display hint features in 3 columns of 6
        manual_values = {}
        cols_form = st.columns(3)
        for i, feat in enumerate(HINT_FEATURES):
            default_val = medians_dict.get(feat, 0.0)
            val_str = cols_form[i % 3].text_input(
                label=f"{feat}",
                value="",
                placeholder=f"median={default_val:.4f}",
                key=f"manual_{feat}"
            )
            if val_str.strip() == "":
                manual_values[feat] = default_val
            else:
                try:
                    manual_values[feat] = float(val_str)
                except ValueError:
                    manual_values[feat] = default_val
                    cols_form[i % 3].warning(f"{feat}: invalid — using median")

        token_input = st.text_input(
            "Account Token (optional identifier)",
            value="manual_account_001",
            key="manual_token"
        )

        run_manual = st.button("▶ Score This Account",
                               key="run_manual",
                               type="primary")

        if run_manual:
            # Build full feature dict: hints + all top150 filled with medians
            with open(
                f"{DRIVE}/phase3_checkpoints/phase3_top150_cols.json"
            ) as f:
                top150_live = json.load(f)

            feature_dict = {}
            for col in top150_live:
                if col in manual_values:
                    feature_dict[col] = manual_values[col]
                else:
                    feature_dict[col] = float(medians_dict.get(col, 0.0))

            payload = {
                "token"   : token_input,
                "features": feature_dict
            }

            with st.spinner("Scoring..."):
                try:
                    t0       = time.perf_counter()
                    response = req.post(
                        f"{API_URL}/score",
                        json=payload,
                        timeout=30
                    )
                    latency  = (time.perf_counter() - t0) * 1000

                    if response.status_code == 200:
                        result = response.json()

                        st.success(
                            f"✓ Scored in {latency:.0f}ms"
                        )

                        # ── Result display ────────────────────
                        r1, r2, r3, r4 = st.columns(4)
                        r1.metric("Risk Score",
                                  f"{result['risk_score']:.1f} / 100")
                        r2.metric("Risk Tier",  result["tier"])
                        r3.metric("Flagged",
                                  "YES 🚨" if result["flagged"] else "NO ✓")
                        r4.metric("Latency",    f"{latency:.0f} ms")

                        # ── Gauge ─────────────────────────────
                        tc = {"High":"#F44336",
                              "Medium":"#FF9800",
                              "Low":"#4CAF50"}
                        fig_m = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=result["risk_score"],
                            title={"text": f"{result['tier']} Risk"},
                            gauge={
                                "axis":{"range":[0,100]},
                                "bar":{"color":tc.get(
                                    result["tier"],"#9E9E9E")},
                                "steps":[
                                    {"range":[0,40],
                                     "color":"#E8F5E9"},
                                    {"range":[40,70],
                                     "color":"#FFF8E1"},
                                    {"range":[70,100],
                                     "color":"#FFEBEE"},
                                ],
                                "threshold":{
                                    "line":{"color":"#FF9800",
                                            "width":3},
                                    "thickness":0.75,
                                    "value":threshold
                                }
                            }
                        ))
                        fig_m.update_layout(
                            height=280,
                            margin=dict(t=40,b=20)
                        )
                        st.plotly_chart(fig_m, use_container_width=True)

                        # ── Layer breakdown ───────────────────
                        fig_lb = go.Figure(go.Bar(
                            x=["ML Ensemble","Anomaly","Network"],
                            y=[result["l1_score"]*100,
                               result["l2_score"]*100,
                               result["l3_score"]*100],
                            marker_color=["#2196F3","#9C27B0","#FF9800"],
                            text=[f"{v*100:.1f}" for v in [
                                result["l1_score"],
                                result["l2_score"],
                                result["l3_score"]
                            ]],
                            textposition="outside"
                        ))
                        fig_lb.update_layout(
                            title="Layer Score Breakdown (0–100)",
                            yaxis=dict(range=[0,110]),
                            height=260,
                            template="plotly_white",
                            margin=dict(t=40,b=10)
                        )
                        st.plotly_chart(fig_lb, use_container_width=True)

                        # ── SHAP features ─────────────────────
                        if result["top_shap_features"]:
                            st.markdown("**Top Risk Drivers (SHAP)**")
                            shap_live = pd.DataFrame(
                                result["top_shap_features"]
                            )
                            fig_sl = go.Figure(go.Bar(
                                x=shap_live["shap_value"],
                                y=shap_live["feature"],
                                orientation="h",
                                marker_color=[
                                    "#F44336" if v>0 else "#2196F3"
                                    for v in shap_live["shap_value"]
                                ]
                            ))
                            fig_sl.update_layout(
                                height=280,
                                template="plotly_white",
                                yaxis=dict(autorange="reversed"),
                                margin=dict(t=20,b=20)
                            )
                            st.plotly_chart(fig_sl,
                                            use_container_width=True)
                            st.dataframe(shap_live,
                                         use_container_width=True)

                        # ── Alert text ────────────────────────
                        if result["alert"]:
                            with st.expander(
                                "📋 Full Alert Text", expanded=True
                            ):
                                st.code(result["alert"], language=None)

                    else:
                        st.error(
                            f"API error {response.status_code}: "
                            f"{response.text[:300]}"
                        )

                except Exception as e:
                    st.error(f"Scoring failed: {e}")
