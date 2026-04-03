# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from PIL import Image
import os
import cv2
import json
import plotly.graph_objects as go
from ultralytics import YOLO

st.set_page_config(
    page_title="AeroScan — Bird vs Drone AI",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Colours ────────────────────────────────────────────────────────────────────
TEAL   = "#4f98a3"
TEAL2  = "#7ec8d3"
RED    = "#f85149"
GREEN  = "#3fb950"
ORANGE = "#d29922"
BG     = "#0d1117"
CARD   = "#161b22"
BORDER = "#21262d"
BORDER2= "#30363d"
TEXT   = "#e6edf3"
MUTED  = "#8b949e"

# ── CSS — same pattern as PhonePe: style sidebar bg only, never touch toggle ──
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    background: {BG};
    color: {TEXT};
  }}

  #MainMenu {{ visibility: hidden; }}
  footer {{ visibility: hidden; }}
  .block-container {{ padding: 1.5rem 2rem; max-width: 1400px; }}

  /* Sidebar — background + border only, nothing else */
  section[data-testid="stSidebar"] {{
    background: {BG} !important;
    border-right: 1px solid {BORDER} !important;
  }}
  section[data-testid="stSidebar"] label {{ color: #c9d1d9 !important; }}

  /* KPI cards */
  .kpi {{
    background: linear-gradient(135deg, {CARD}, #1c2333);
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 1.2rem 1rem;
    text-align: center;
    margin: 4px;
  }}
  .kpi h3 {{ font-size: 0.72rem; color: {MUTED}; margin: 0;
    text-transform: uppercase; letter-spacing: 0.08em; }}
  .kpi h1 {{ font-size: 1.7rem; font-weight: 700; color: {TEAL}; margin: 6px 0 0; }}
  .kpi p  {{ font-size: 0.72rem; color: {GREEN}; margin: 4px 0 0; }}

  /* Hero */
  .hero {{
    background: linear-gradient(135deg, {BG} 0%, {CARD} 50%, {BG} 100%);
    border: 1px solid {BORDER};
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
  }}
  .hero::before {{
    content: '';
    position: absolute; top: -80px; right: -80px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(79,152,163,0.15) 0%, transparent 70%);
    border-radius: 50%;
  }}
  .hero-badge {{
    display: inline-block;
    background: rgba(79,152,163,0.12);
    border: 1px solid rgba(79,152,163,0.3);
    color: {TEAL}; font-size: 0.7rem; font-weight: 600;
    padding: 0.2rem 0.6rem; border-radius: 20px;
    letter-spacing: 0.08em; text-transform: uppercase;
    margin-bottom: 0.8rem;
  }}
  .hero-title {{
    font-size: 2rem; font-weight: 700;
    background: linear-gradient(90deg, {TEAL}, {TEAL2});
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0; line-height: 1.2;
  }}
  .hero-sub {{ color: {MUTED}; font-size: 0.95rem; margin-top: 0.4rem; }}

  /* Panel */
  .panel {{
    background: {CARD}; border: 1px solid {BORDER};
    border-radius: 12px; padding: 1.5rem;
  }}

  /* Result cards */
  .result-bird {{
    background: rgba(79,152,163,0.08); border: 1px solid rgba(79,152,163,0.35);
    border-radius: 12px; padding: 1.25rem 1.5rem; margin: 0.5rem 0;
  }}
  .result-drone {{
    background: rgba(248,81,73,0.08); border: 1px solid rgba(248,81,73,0.35);
    border-radius: 12px; padding: 1.25rem 1.5rem; margin: 0.5rem 0;
  }}

  /* Section heading */
  .sec-head {{
    color: {TEXT}; font-size: 1rem; font-weight: 600;
    border-bottom: 1px solid {BORDER};
    padding-bottom: 0.6rem; margin-bottom: 1rem;
  }}

  /* Alerts */
  .alert-danger {{
    background: rgba(248,81,73,0.1); border: 1px solid rgba(248,81,73,0.3);
    border-radius: 8px; padding: 0.75rem 1rem; color: {RED}; font-size: 0.85rem;
  }}
  .alert-info {{
    background: rgba(79,152,163,0.1); border: 1px solid rgba(79,152,163,0.3);
    border-radius: 8px; padding: 0.75rem 1rem; color: {TEAL}; font-size: 0.85rem;
  }}

  /* Buttons */
  .stButton > button {{
    background: linear-gradient(135deg, {TEAL}, #3a7a85) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important; width: 100% !important;
  }}
  .stButton > button:hover {{ opacity: 0.85 !important; }}

  [data-testid="stFileUploaderDropzone"] {{
    background: {CARD} !important; border: 2px dashed {BORDER2} !important;
    border-radius: 12px !important;
  }}
  [data-testid="stFileUploaderDropzone"]:hover {{ border-color: {TEAL} !important; }}

  .stSelectbox > div > div {{
    background: #21262d !important; border-color: {BORDER2} !important; color: {TEXT} !important;
  }}
  .stSpinner > div {{ border-top-color: {TEAL} !important; }}
  div[data-testid="stImage"] img {{ border-radius: 10px; }}
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
def kpi(col, label, value, delta=None):
    d = f"<p>{delta}</p>" if delta else ""
    col.markdown(f'<div class="kpi"><h3>{label}</h3><h1>{value}</h1>{d}</div>',
                 unsafe_allow_html=True)


def theme(fig, h=320):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=CARD,
        font=dict(family="Inter", color=MUTED, size=12), height=h,
        margin=dict(l=30, r=30, t=40, b=30),
        xaxis=dict(gridcolor=BORDER, linecolor=BORDER2),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER2),
    )
    return fig


# ── Model loaders ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_classification_models():
    models = {}
    if os.path.exists("models/transfer_model.h5"):
        models["EfficientNetB0 (99.07%)"] = ("efficientnet", load_model("models/transfer_model.h5"))
    if os.path.exists("models/custom_cnn.h5"):
        models["Custom CNN (86.05%)"] = ("cnn", load_model("models/custom_cnn.h5"))
    return models


@st.cache_resource
def load_yolo_model():
    for p in ["runs/detect/runs/detect/bird_drone/weights/best.pt",
              "runs/detect/bird_drone/weights/best.pt"]:
        if os.path.exists(p):
            return YOLO(p)
    return None


def preprocess_image(image, model_type="cnn"):
    arr = img_to_array(image.resize((224, 224)))
    arr = efficientnet_preprocess(arr) if model_type == "efficientnet" else arr / 255.0
    return np.expand_dims(arr, axis=0)


def predict_class(model, arr):
    conf = float(model.predict(arr, verbose=0)[0][0])
    return ("Drone", conf) if conf > 0.5 else ("Bird", 1 - conf)


def load_metrics():
    cnn_acc, tr_acc = 0.8605, 0.9907
    try:
        if os.path.exists("logs/cnn_metrics.json"):
            cnn_acc = json.load(open("logs/cnn_metrics.json")).get("test_accuracy", cnn_acc)
    except Exception:
        pass
    try:
        if os.path.exists("logs/transfer_metrics.json"):
            tr_acc = json.load(open("logs/transfer_metrics.json")).get("test_accuracy", tr_acc)
    except Exception:
        pass
    return cnn_acc, tr_acc


cnn_acc, tr_acc = load_metrics()


# ── Charts ─────────────────────────────────────────────────────────────────────
def gauge_chart(value, label):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value * 100,
        number=dict(suffix="%", font=dict(size=28, color=TEXT, family="Inter")),
        title=dict(text=label, font=dict(size=13, color=MUTED, family="Inter")),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor=BORDER2, tickfont=dict(color=MUTED, size=10)),
            bar=dict(color=TEAL, thickness=0.6), bgcolor=BORDER, borderwidth=0,
            steps=[dict(range=[0,50], color="rgba(248,81,73,0.12)"),
                   dict(range=[50,75], color="rgba(210,153,34,0.12)"),
                   dict(range=[75,100], color="rgba(63,185,80,0.12)")],
            threshold=dict(line=dict(color=TEAL2, width=2), thickness=0.7, value=value*100),
        )
    ))
    fig.update_layout(paper_bgcolor=CARD, plot_bgcolor=CARD,
                      font=dict(family="Inter", color=MUTED, size=12),
                      height=200, margin=dict(l=20, r=20, t=30, b=10))
    return fig


def model_comparison_chart():
    fig = go.Figure()
    for name, val, color in [("Custom CNN", cnn_acc*100, ORANGE),
                              ("EfficientNetB0", tr_acc*100, TEAL)]:
        fig.add_trace(go.Bar(x=[name], y=[val], name=name,
                             marker=dict(color=color, opacity=0.85),
                             text=[f"{val:.2f}%"], textposition="outside",
                             textfont=dict(color=TEXT, size=13), width=0.4))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=CARD,
                      font=dict(family="Inter", color=MUTED, size=12),
                      title=dict(text="Model Accuracy Comparison", font=dict(size=14, color=TEXT)),
                      height=300, showlegend=False, bargap=0.4,
                      xaxis=dict(gridcolor=BORDER, linecolor=BORDER2),
                      yaxis=dict(range=[0,115], ticksuffix="%", gridcolor=BORDER, linecolor=BORDER2),
                      margin=dict(l=30, r=30, t=40, b=30))
    return fig


def class_dist_chart():
    fig = go.Figure(go.Pie(
        labels=["Bird","Drone"], values=[1661,1658], hole=0.55,
        marker=dict(colors=[TEAL,ORANGE], line=dict(color=BG, width=3)),
        textinfo="label+percent", textfont=dict(color=TEXT, size=12),
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=CARD,
                      font=dict(family="Inter", color=MUTED, size=12),
                      title=dict(text="Dataset Distribution", font=dict(size=14, color=TEXT)),
                      height=280, showlegend=False,
                      margin=dict(l=30, r=30, t=40, b=30),
                      annotations=[dict(text="3,319<br>images", x=0.5, y=0.5, showarrow=False,
                                        font=dict(color=TEXT, size=14, family="Inter"))])
    return fig


def yolo_chart():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1,2,3], y=[0.163,0.226,0.221], name="mAP50",
                             mode="lines+markers", line=dict(color=TEAL, width=2.5),
                             marker=dict(size=7, color=TEAL)))
    fig.add_trace(go.Scatter(x=[1,2,3], y=[0.1337,0.1305,0.1274], name="box_loss/10",
                             mode="lines+markers", line=dict(color=ORANGE, width=2, dash="dot"),
                             marker=dict(size=6, color=ORANGE)))
    fig.add_trace(go.Scatter(x=[1,2,3], y=[0.3146,0.2108,0.1447], name="cls_loss/10",
                             mode="lines+markers", line=dict(color=RED, width=2, dash="dot"),
                             marker=dict(size=6, color=RED)))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=CARD,
                      font=dict(family="Inter", color=MUTED, size=12),
                      title=dict(text="YOLOv8 Training Curves", font=dict(size=14, color=TEXT)),
                      height=280, margin=dict(l=30, r=30, t=40, b=30),
                      legend=dict(font=dict(color="#c9d1d9", size=11), bgcolor="rgba(0,0,0,0)",
                                  orientation="h", yanchor="bottom", y=1.02),
                      xaxis=dict(title="Epoch", gridcolor=BORDER, linecolor=BORDER2),
                      yaxis=dict(title="Value", gridcolor=BORDER, linecolor=BORDER2))
    return fig


def radar_chart():
    cats = ["Precision<br>Bird","Recall<br>Bird","F1<br>Bird",
            "Precision<br>Drone","Recall<br>Drone","F1<br>Drone"]
    fig = go.Figure()
    for name, vals, color, fill in [
        ("Custom CNN",     [0.93,0.82,0.87,0.80,0.91,0.85,0.93], ORANGE, "rgba(210,153,34,0.15)"),
        ("EfficientNetB0", [0.98,0.99,0.98,0.99,0.97,0.98,0.98], TEAL,   "rgba(79,152,163,0.15)"),
    ]:
        fig.add_trace(go.Scatterpolar(r=vals, theta=cats+[cats[0]], fill="toself",
                                      name=name, line=dict(color=color, width=2),
                                      fillcolor=fill))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=CARD,
        font=dict(family="Inter", color=MUTED, size=12),
        polar=dict(bgcolor=CARD,
                   radialaxis=dict(visible=True, range=[0,1], gridcolor=BORDER,
                                   tickfont=dict(color=MUTED, size=9), linecolor=BORDER2),
                   angularaxis=dict(gridcolor=BORDER, linecolor=BORDER2,
                                    tickfont=dict(color="#c9d1d9", size=10))),
        legend=dict(font=dict(color="#c9d1d9", size=11), bgcolor="rgba(0,0,0,0)"),
        title=dict(text="Classification Metrics Radar", font=dict(size=14, color=TEXT)),
        height=320, margin=dict(l=30, r=30, t=40, b=30),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — plain with st.sidebar, no CSS hacks on toggle (same as PhonePe)
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center;padding:16px 0 20px'>
      <div style='font-size:2.5rem;margin-bottom:6px'>🛸</div>
      <div style='font-size:1.2rem;font-weight:700;color:{TEAL}'>AeroScan</div>
      <div style='font-size:0.75rem;color:{MUTED}'>Aerial Object Intelligence</div>
      <hr style='border-color:{BORDER};margin:12px 0'>
    </div>
    """, unsafe_allow_html=True)

    task = st.radio("Select Mode",
                    ["Classification", "Object Detection (YOLOv8)"],
                    label_visibility="collapsed")

    models = load_classification_models()
    model_choice = None
    if task == "Classification" and models:
        st.markdown(f"<div style='color:{MUTED};font-size:0.7rem;text-transform:uppercase;"
                    f"letter-spacing:0.1em;font-weight:600;margin:1rem 0 0.4rem'>Model</div>",
                    unsafe_allow_html=True)
        model_choice = st.selectbox("Select Model", list(models.keys()),
                                    label_visibility="collapsed")

    st.markdown(f"<hr style='border-color:{BORDER}'>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='color:{MUTED};font-size:0.78rem;line-height:1.6'>
      Real-time bird vs drone classification.<br><br>
      <b style='color:#c9d1d9'>Dataset:</b> 3,319 aerial images<br>
      <b style='color:#c9d1d9'>Classes:</b> Bird · Drone<br>
      <b style='color:#c9d1d9'>Best Model:</b> EfficientNetB0<br>
      <b style='color:#c9d1d9'>Accuracy:</b> 99.07%
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div class="hero-badge">AI-Powered Detection</div>
  <h1 class="hero-title">Aerial Object Classification & Detection</h1>
  <p class="hero-sub">Deep learning pipeline for real-time Bird vs Drone identification —
  Custom CNN · EfficientNetB0 · YOLOv8</p>
</div>
""", unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
kpi(k1, "Best Accuracy",  f"{tr_acc:.2%}",  "EfficientNetB0")
kpi(k2, "CNN Accuracy",   f"{cnn_acc:.2%}", "Custom CNN")
kpi(k3, "Dataset Size",   "3,319",          "Aerial images")
kpi(k4, "YOLOv8 mAP50",  "22.5%",          "CPU baseline · 3 epochs")

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE ENGINE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-head">🔍 Inference Engine</div>', unsafe_allow_html=True)

# Add CSS to style st.container as a panel card
st.markdown(f"""
<style>
  [data-testid="stVerticalBlockBorderWrapper"] {{
    background: {CARD} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 12px !important;
    padding: 0.5rem !important;
  }}
</style>
""", unsafe_allow_html=True)

col_up, col_res = st.columns(2, gap="large")

with col_up:
    with st.container(border=True):
        st.markdown(f'<p style="color:{MUTED};font-size:0.75rem;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:0.5rem">📁 Upload Image</p>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload aerial image", type=["jpg","jpeg","png"],
                                         label_visibility="collapsed")
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, use_container_width=True)
            w, h = image.size
            st.caption(f"{uploaded_file.name} · {w}×{h}px")
        else:
            st.markdown(f"""
            <div style='text-align:center;padding:2rem 1rem;color:{MUTED}'>
              <div style='font-size:3rem;margin-bottom:0.5rem'>📸</div>
              <p style='font-size:0.9rem;font-weight:500;color:#c9d1d9;margin:0'>Drop an aerial image here</p>
              <p style='font-size:0.75rem;margin-top:0.3rem'>JPG · JPEG · PNG · up to 200MB</p>
            </div>""", unsafe_allow_html=True)

with col_res:
    with st.container(border=True):
        st.markdown(f'<p style="color:{MUTED};font-size:0.75rem;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:0.5rem">🎯 Prediction Results</p>', unsafe_allow_html=True)

        if not uploaded_file:
            # Show model quick-reference and how-it-works guide when idle
            st.markdown(f"""
            <div style='margin-bottom:1rem'>
              <p style='color:{TEXT};font-size:0.85rem;font-weight:600;margin-bottom:0.6rem'>⚡ How It Works</p>
              <div style='display:flex;flex-direction:column;gap:0.4rem'>
                <div style='display:flex;align-items:center;gap:0.75rem;padding:0.5rem 0.75rem;
                             background:rgba(79,152,163,0.06);border-radius:8px;border-left:3px solid {TEAL}'>
                  <span style='font-size:1.1rem'>1️⃣</span>
                  <span style='color:#c9d1d9;font-size:0.8rem'>Upload an aerial image (bird or drone)</span>
                </div>
                <div style='display:flex;align-items:center;gap:0.75rem;padding:0.5rem 0.75rem;
                             background:rgba(79,152,163,0.06);border-radius:8px;border-left:3px solid {TEAL}'>
                  <span style='font-size:1.1rem'>2️⃣</span>
                  <span style='color:#c9d1d9;font-size:0.8rem'>Select model in sidebar (EfficientNetB0 recommended)</span>
                </div>
                <div style='display:flex;align-items:center;gap:0.75rem;padding:0.5rem 0.75rem;
                             background:rgba(79,152,163,0.06);border-radius:8px;border-left:3px solid {TEAL}'>
                  <span style='font-size:1.1rem'>3️⃣</span>
                  <span style='color:#c9d1d9;font-size:0.8rem'>Click Classify Image — get instant prediction</span>
                </div>
                <div style='display:flex;align-items:center;gap:0.75rem;padding:0.5rem 0.75rem;
                             background:rgba(79,152,163,0.06);border-radius:8px;border-left:3px solid {TEAL}'>
                  <span style='font-size:1.1rem'>4️⃣</span>
                  <span style='color:#c9d1d9;font-size:0.8rem'>For bounding boxes, switch to YOLOv8 mode</span>
                </div>
              </div>
            </div>
            <div>
              <p style='color:{TEXT};font-size:0.85rem;font-weight:600;margin-bottom:0.6rem'>🤖 Model Quick Reference</p>
              <table style='width:100%;border-collapse:collapse;font-size:0.78rem'>
                <tr style='border-bottom:1px solid {BORDER}'>
                  <th style='color:{MUTED};font-weight:500;text-align:left;padding:0.35rem 0'>Model</th>
                  <th style='color:{MUTED};font-weight:500;text-align:center;padding:0.35rem 0'>Accuracy</th>
                  <th style='color:{MUTED};font-weight:500;text-align:center;padding:0.35rem 0'>Speed</th>
                  <th style='color:{MUTED};font-weight:500;text-align:center;padding:0.35rem 0'>Task</th>
                </tr>
                <tr style='border-bottom:1px solid {BORDER}'>
                  <td style='color:{TEXT};padding:0.4rem 0;font-weight:500'>EfficientNetB0</td>
                  <td style='color:{GREEN};text-align:center;font-weight:600'>99.07%</td>
                  <td style='color:{TEAL};text-align:center'>Fast</td>
                  <td style='color:{MUTED};text-align:center'>Classification</td>
                </tr>
                <tr style='border-bottom:1px solid {BORDER}'>
                  <td style='color:{TEXT};padding:0.4rem 0;font-weight:500'>Custom CNN</td>
                  <td style='color:{ORANGE};text-align:center;font-weight:600'>86.05%</td>
                  <td style='color:{TEAL};text-align:center'>Fast</td>
                  <td style='color:{MUTED};text-align:center'>Classification</td>
                </tr>
                <tr>
                  <td style='color:{TEXT};padding:0.4rem 0;font-weight:500'>YOLOv8n</td>
                  <td style='color:{RED};text-align:center;font-weight:600'>mAP 22%</td>
                  <td style='color:{ORANGE};text-align:center'>Medium</td>
                  <td style='color:{MUTED};text-align:center'>Detection</td>
                </tr>
              </table>
            </div>
            """, unsafe_allow_html=True)
        else:
            if task == "Classification":
                if not models:
                    st.markdown('<div class="alert-danger">⚠️ No trained models found. '
                                'Run train_cnn.py and train_transfer.py first.</div>',
                                unsafe_allow_html=True)
                else:
                    if st.button("🚀 Classify Image"):
                        with st.spinner("Running inference..."):
                            mtype, mobj = models[model_choice]
                            arr = preprocess_image(image, mtype)
                            label, confidence = predict_class(mobj, arr)
                            if label == "Bird":
                                st.markdown(f"""
                                <div class="result-bird">
                                  <div style='font-size:1.4rem;font-weight:700;color:{TEAL}'>🦅 BIRD</div>
                                  <div style='font-size:0.8rem;color:{MUTED}'>Wildlife — No security threat</div>
                                </div>""", unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class="result-drone">
                                  <div style='font-size:1.4rem;font-weight:700;color:{RED}'>🛸 DRONE</div>
                                  <div style='font-size:0.8rem;color:{MUTED}'>⚠️ Unmanned aerial vehicle detected</div>
                                </div>""", unsafe_allow_html=True)
                            st.markdown("<br>", unsafe_allow_html=True)
                            st.plotly_chart(gauge_chart(confidence, f"Confidence — {model_choice}"),
                                            use_container_width=True, config={"displayModeBar": False})
                            conf_pct = confidence * 100
                            bar_color = TEAL if label == "Bird" else RED
                            st.markdown(f"""
                            <div style='margin-top:0.5rem'>
                              <div style='display:flex;justify-content:space-between;margin-bottom:0.3rem'>
                                <span style='color:{MUTED};font-size:0.75rem'>Confidence Score</span>
                                <span style='color:{TEXT};font-size:0.75rem;font-weight:600'>{conf_pct:.2f}%</span>
                              </div>
                              <div style='height:6px;background:{BORDER};border-radius:4px;overflow:hidden'>
                                <div style='width:{conf_pct}%;height:100%;background:{bar_color};border-radius:4px'></div>
                              </div>
                            </div>""", unsafe_allow_html=True)
            else:
                yolo_model = load_yolo_model()
                if yolo_model is None:
                    st.markdown('<div class="alert-danger">⚠️ YOLOv8 model not found. '
                                'Run python yolo/train_yolo.py first.</div>', unsafe_allow_html=True)
                else:
                    if st.button("🎯 Detect Objects"):
                        with st.spinner("Running YOLOv8..."):
                            results = yolo_model(np.array(image))
                            annotated = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
                            st.image(annotated, use_container_width=True)
                            boxes = results[0].boxes

                            if boxes and len(boxes):
                                st.markdown(f'<div class="alert-info">✅ {len(boxes)} object(s) detected</div>',
                                            unsafe_allow_html=True)
                                for box in boxes:
                                    name = "Bird 🦅" if int(box.cls[0]) == 0 else "Drone 🛸"
                                    conf = float(box.conf[0])
                                    st.markdown(f"""
                                    <div style='display:flex;justify-content:space-between;
                                      padding:0.4rem 0.6rem;background:{BORDER};border-radius:6px;margin-top:0.4rem'>
                                      <span style='color:{TEXT};font-size:0.85rem'>{name}</span>
                                      <span style='color:{TEAL};font-size:0.85rem;font-weight:600'>{conf:.2%}</span>
                                    </div>""", unsafe_allow_html=True)
                            else:
                                # YOLOv8 missed it — only 3 epochs / mAP 22% — auto-fallback to EfficientNetB0
                                st.markdown(f"""
                                <div style='background:rgba(210,153,34,0.1);border:1px solid rgba(210,153,34,0.35);
                                  border-radius:8px;padding:0.6rem 0.9rem;margin-bottom:0.8rem;
                                  font-size:0.8rem;color:{ORANGE}'>
                                  ⚠️ YOLOv8 found no boxes (mAP 22% — model needs more training).
                                  Auto-running EfficientNetB0 classification instead…
                                </div>""", unsafe_allow_html=True)

                                cls_models = load_classification_models()
                                if cls_models:
                                    best_key = "EfficientNetB0 (99.07%)" if "EfficientNetB0 (99.07%)" in cls_models \
                                               else list(cls_models.keys())[0]
                                    mtype, mobj = cls_models[best_key]
                                    arr = preprocess_image(image, mtype)
                                    label, confidence = predict_class(mobj, arr)

                                    if label == "Bird":
                                        st.markdown(f"""
                                        <div class="result-bird">
                                          <div style='font-size:1.4rem;font-weight:700;color:{TEAL}'>🦅 BIRD</div>
                                          <div style='font-size:0.8rem;color:{MUTED}'>Wildlife — No security threat</div>
                                        </div>""", unsafe_allow_html=True)
                                    else:
                                        st.markdown(f"""
                                        <div class="result-drone">
                                          <div style='font-size:1.4rem;font-weight:700;color:{RED}'>🛸 DRONE</div>
                                          <div style='font-size:0.8rem;color:{MUTED}'>⚠️ Unmanned aerial vehicle detected</div>
                                        </div>""", unsafe_allow_html=True)

                                    st.markdown("<br>", unsafe_allow_html=True)
                                    st.plotly_chart(
                                        gauge_chart(confidence, f"Confidence — {best_key}"),
                                        use_container_width=True, config={"displayModeBar": False})
                                    conf_pct = confidence * 100
                                    bar_color = TEAL if label == "Bird" else RED
                                    st.markdown(f"""
                                    <div style='margin-top:0.5rem'>
                                      <div style='display:flex;justify-content:space-between;margin-bottom:0.3rem'>
                                        <span style='color:{MUTED};font-size:0.75rem'>Confidence (EfficientNetB0 fallback)</span>
                                        <span style='color:{TEXT};font-size:0.75rem;font-weight:600'>{conf_pct:.2f}%</span>
                                      </div>
                                      <div style='height:6px;background:{BORDER};border-radius:4px;overflow:hidden'>
                                        <div style='width:{conf_pct}%;height:100%;background:{bar_color};border-radius:4px'></div>
                                      </div>
                                    </div>""", unsafe_allow_html=True)
                                else:
                                    st.markdown('<div class="alert-danger">⚠️ No classification models found either.</div>',
                                                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ANALYTICS DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="sec-head">📊 Model Analytics Dashboard</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns([2, 1.5, 1.5], gap="medium")
with c1: st.plotly_chart(model_comparison_chart(), use_container_width=True, config={"displayModeBar": False})
with c2: st.plotly_chart(class_dist_chart(),       use_container_width=True, config={"displayModeBar": False})
with c3: st.plotly_chart(yolo_chart(),             use_container_width=True, config={"displayModeBar": False})

r1, r2 = st.columns([1.5, 1], gap="medium")
with r1:
    st.plotly_chart(radar_chart(), use_container_width=True, config={"displayModeBar": False})
with r2:
    st.markdown(f"""
    <div class="panel" style='height:320px'>
      <p style='color:{MUTED};font-size:0.8rem;font-weight:500;margin-bottom:1rem'>MODEL SUMMARY</p>
      <table style='width:100%;border-collapse:collapse;font-size:0.8rem'>
        <tr style='border-bottom:1px solid {BORDER}'>
          <th style='color:#6e7681;font-weight:500;text-align:left;padding:0.4rem 0'>Model</th>
          <th style='color:#6e7681;font-weight:500;text-align:right;padding:0.4rem 0'>Acc</th>
          <th style='color:#6e7681;font-weight:500;text-align:right;padding:0.4rem 0'>F1</th>
        </tr>
        <tr style='border-bottom:1px solid {BORDER}'>
          <td style='color:{TEXT};padding:0.5rem 0'>EfficientNetB0</td>
          <td style='color:{TEAL};font-weight:600;text-align:right'>99.07%</td>
          <td style='color:{TEAL};font-weight:600;text-align:right'>0.98</td>
        </tr>
        <tr style='border-bottom:1px solid {BORDER}'>
          <td style='color:{TEXT};padding:0.5rem 0'>Custom CNN</td>
          <td style='color:{ORANGE};font-weight:600;text-align:right'>86.05%</td>
          <td style='color:{ORANGE};font-weight:600;text-align:right'>0.86</td>
        </tr>
        <tr>
          <td style='color:{TEXT};padding:0.5rem 0'>YOLOv8n</td>
          <td style='color:{RED};font-weight:600;text-align:right'>mAP 22%</td>
          <td style='color:{MUTED};font-weight:600;text-align:right'>CPU</td>
        </tr>
      </table>
      <div style='margin-top:1.2rem;padding:0.75rem;background:rgba(79,152,163,0.08);
                  border:1px solid rgba(79,152,163,0.2);border-radius:8px'>
        <p style='color:{TEAL};font-size:0.75rem;font-weight:600;margin-bottom:0.3rem'>🏆 Best Architecture</p>
        <p style='color:{MUTED};font-size:0.73rem;line-height:1.5'>EfficientNetB0 with transfer learning —
        99.07% test accuracy on 215 test images. Fine-tuned on aerial imagery.</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='margin-top:3rem;padding-top:1.5rem;border-top:1px solid {BORDER};
  display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.5rem'>
  <div style='color:#6e7681;font-size:0.75rem'>
    🛸 <b style='color:{TEAL}'>AeroScan</b> — Aerial Object Detection System
  </div>
  <div style='color:#6e7681;font-size:0.75rem;display:flex;gap:1rem'>
    <span>TensorFlow 2.x</span><span>·</span>
    <span>Ultralytics YOLOv8</span><span>·</span>
    <span>Plotly</span><span>·</span><span>Streamlit</span>
  </div>
</div>
""", unsafe_allow_html=True)