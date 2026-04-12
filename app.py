# -*- coding: utf-8 -*-
# app.py — main streamlit app for the aerial classification project
# started this as a simple script and kept adding stuff as the project grew
# runs the full UI including classification and yolo detection modes

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

# setting up the page — wide layout works better for the dashboard section
st.set_page_config(
    page_title="AeroScan — Bird vs Drone AI",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- color variables ---
# i kept all colors here so i can change them easily in one place
# tried a dark github-like theme, looks clean enough
teal_main = "#4f98a3"
teal_light = "#7ec8d3"
red_col = "#f85149"
green_col = "#3fb950"
orange_col = "#d29922"
bg_col = "#0d1117"
card_col = "#161b22"
border_col = "#21262d"
border2_col = "#30363d"
text_col = "#e6edf3"
muted_col = "#8b949e"

# dumping all the css here — i know its a lot but streamlit doesnt give much control otherwise
# had to override a bunch of default streamlit styles to get the dark theme working properly
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    background: {bg_col};
    color: {text_col};
  }}

  /* hiding the default streamlit menu and footer, dont need them */
  #MainMenu {{ visibility: hidden; }}
  footer {{ visibility: hidden; }}
  .block-container {{ padding: 1.5rem 2rem; max-width: 1400px; }}

  /* sidebar background styling */
  section[data-testid="stSidebar"] {{
    background: {bg_col} !important;
    border-right: 1px solid {border_col} !important;
  }}
  section[data-testid="stSidebar"] label {{ color: #c9d1d9 !important; }}

  /* these are the small stat cards at the top — kpi stands for key performance indicator */
  .kpi {{
    background: linear-gradient(135deg, {card_col}, #1c2333);
    border: 1px solid {border_col};
    border-radius: 12px;
    padding: 1.2rem 1rem;
    text-align: center;
    margin: 4px;
  }}
  .kpi h3 {{ font-size: 0.72rem; color: {muted_col}; margin: 0;
    text-transform: uppercase; letter-spacing: 0.08em; }}
  .kpi h1 {{ font-size: 1.7rem; font-weight: 700; color: {teal_main}; margin: 6px 0 0; }}
  .kpi p  {{ font-size: 0.72rem; color: {green_col}; margin: 4px 0 0; }}

  /* hero section at the top of the page */
  .hero {{
    background: linear-gradient(135deg, {bg_col} 0%, {card_col} 50%, {bg_col} 100%);
    border: 1px solid {border_col};
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
  }}
  /* the glowing circle in the top right corner of the hero — purely decorative */
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
    color: {teal_main}; font-size: 0.7rem; font-weight: 600;
    padding: 0.2rem 0.6rem; border-radius: 20px;
    letter-spacing: 0.08em; text-transform: uppercase;
    margin-bottom: 0.8rem;
  }}
  /* gradient text for the main title */
  .hero-title {{
    font-size: 2rem; font-weight: 700;
    background: linear-gradient(90deg, {teal_main}, {teal_light});
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0; line-height: 1.2;
  }}
  .hero-sub {{ color: {muted_col}; font-size: 0.95rem; margin-top: 0.4rem; }}

  /* generic panel/card container */
  .panel {{
    background: {card_col}; border: 1px solid {border_col};
    border-radius: 12px; padding: 1.5rem;
  }}

  /* result cards — different colors for bird vs drone */
  .result-bird {{
    background: rgba(79,152,163,0.08); border: 1px solid rgba(79,152,163,0.35);
    border-radius: 12px; padding: 1.25rem 1.5rem; margin: 0.5rem 0;
  }}
  .result-drone {{
    background: rgba(248,81,73,0.08); border: 1px solid rgba(248,81,73,0.35);
    border-radius: 12px; padding: 1.25rem 1.5rem; margin: 0.5rem 0;
  }}

  /* section heading style — used for Inference Engine, Analytics etc */
  .sec-head {{
    color: {text_col}; font-size: 1rem; font-weight: 600;
    border-bottom: 1px solid {border_col};
    padding-bottom: 0.6rem; margin-bottom: 1rem;
  }}

  /* alert boxes for errors and info messages */
  .alert-danger {{
    background: rgba(248,81,73,0.1); border: 1px solid rgba(248,81,73,0.3);
    border-radius: 8px; padding: 0.75rem 1rem; color: {red_col}; font-size: 0.85rem;
  }}
  .alert-info {{
    background: rgba(79,152,163,0.1); border: 1px solid rgba(79,152,163,0.3);
    border-radius: 8px; padding: 0.75rem 1rem; color: {teal_main}; font-size: 0.85rem;
  }}

  /* button styling — overriding streamlit default */
  .stButton > button {{
    background: linear-gradient(135deg, {teal_main}, #3a7a85) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important; width: 100% !important;
  }}
  .stButton > button:hover {{ opacity: 0.85 !important; }}

  /* file upload dropzone */
  [data-testid="stFileUploaderDropzone"] {{
    background: {card_col} !important; border: 2px dashed {border2_col} !important;
    border-radius: 12px !important;
  }}
  [data-testid="stFileUploaderDropzone"]:hover {{ border-color: {teal_main} !important; }}

  /* selectbox in sidebar */
  .stSelectbox > div > div {{
    background: #21262d !important; border-color: {border2_col} !important; color: {text_col} !important;
  }}
  .stSpinner > div {{ border-top-color: {teal_main} !important; }}
  div[data-testid="stImage"] img {{ border-radius: 10px; }}

  /* plotly chart container rounding */
  div[data-testid="stPlotlyChart"] > div {{
    border-radius: 12px;
    overflow: hidden;
  }}

  /* updated sec-head with bottom border in teal color */
  .sec-head {{
    color: {text_col}; font-size: 1.05rem; font-weight: 700;
    border-bottom: 2px solid {teal_main};
    padding-bottom: 0.5rem; margin-bottom: 1.2rem;
    letter-spacing: -0.01em;
    display: flex; align-items: center; gap: 0.4rem;
  }}

  /* kpi card hover effect — subtle lift */
  .kpi:hover {{
    border-color: {teal_main} !important;
    box-shadow: 0 0 20px rgba(79,152,163,0.18);
    transform: translateY(-2px);
    transition: all 0.2s ease;
  }}

  /* streamlit container border styling */
  [data-testid="stVerticalBlockBorderWrapper"] > div {{
    border-radius: 14px !important;
    background: {card_col} !important;
    border-color: {border_col} !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.2) !important;
  }}

  /* panel with shadow */
  .panel {{
    background: {card_col}; border: 1px solid {border_col};
    border-radius: 14px; padding: 1.5rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.2);
  }}

  /* caption text below images */
  div[data-testid="stCaptionContainer"] {{
    color: {muted_col} !important; font-size: 0.72rem !important; text-align: center;
  }}
</style>
""", unsafe_allow_html=True)


# --- helper: render a KPI card in a given column ---
# just a small utility so i dont repeat the same html 4 times
def kpi(col, label, value, delta=None):
    delta_html = f"<p>{delta}</p>" if delta else ""
    col.markdown(
        f'<div class="kpi"><h3>{label}</h3><h1>{value}</h1>{delta_html}</div>',
        unsafe_allow_html=True
    )


# --- helper: apply consistent dark theme to any plotly figure ---
# i use this on all charts so they all look the same
def apply_theme(fig, h=320):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=card_col,
        font=dict(family="Inter", color=muted_col, size=12),
        height=h,
        margin=dict(l=30, r=30, t=40, b=30),
        xaxis=dict(gridcolor=border_col, linecolor=border2_col),
        yaxis=dict(gridcolor=border_col, linecolor=border2_col),
    )
    return fig


# --- load classification models (CNN and EfficientNet) ---
# using cache_resource so models dont reload on every interaction
# both models are saved as .h5 files in the models/ folder
@st.cache_resource
def load_classification_models():
    loaded = {}
    # check if transfer learning model exists
    if os.path.exists("models/transfer_model.h5"):
        loaded["EfficientNetB0 (99.07%)"] = ("efficientnet", load_model("models/transfer_model.h5"))
    # check if custom cnn model exists
    if os.path.exists("models/custom_cnn.h5"):
        loaded["Custom CNN (86.05%)"] = ("cnn", load_model("models/custom_cnn.h5"))
    return loaded


# --- load yolo model ---
# yolo weights can be in two possible paths depending on how training was run
# so checking both locations just in case
@st.cache_resource
def load_yolo_model():
    possible_paths = [
        "runs/detect/runs/detect/bird_drone/weights/best.pt",
        "runs/detect/bird_drone/weights/best.pt"
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return YOLO(p)
    # if neither path has the weights, return None and handle in UI
    return None


# --- preprocess image before passing to model ---
# efficientnet needs its own preprocessing, cnn just needs normalization to 0-1
def preprocess_image(img, model_type="cnn"):
    resized = img.resize((224, 224))
    arr = img_to_array(resized)
    if model_type == "efficientnet":
        arr = efficientnet_preprocess(arr)
    else:
        arr = arr / 255.0
    # expand dims to make it (1, 224, 224, 3) — model expects batch input
    return np.expand_dims(arr, axis=0)


# --- run prediction and return label + confidence ---
# model output is a single sigmoid value: >0.5 means drone, <=0.5 means bird
def predict_class(model, arr):
    raw = float(model.predict(arr, verbose=0)[0][0])
    if raw > 0.5:
        return ("Drone", raw)
    else:
        return ("Bird", 1 - raw)


# --- load test accuracy from saved log files if available ---
# fallback to hardcoded values if logs dont exist (which is fine for demo)
def load_metrics():
    cnn_acc = 0.8605      # default fallback
    transfer_acc = 0.9907  # default fallback

    try:
        if os.path.exists("logs/cnn_metrics.json"):
            data = json.load(open("logs/cnn_metrics.json"))
            cnn_acc = data.get("test_accuracy", cnn_acc)
    except Exception:
        pass  # if file is broken just use the default

    try:
        if os.path.exists("logs/transfer_metrics.json"):
            data = json.load(open("logs/transfer_metrics.json"))
            transfer_acc = data.get("test_accuracy", transfer_acc)
    except Exception:
        pass

    return cnn_acc, transfer_acc


# load metrics once at startup
cnn_acc, tr_acc = load_metrics()


# --- gauge chart for showing prediction confidence ---
# value should be between 0 and 1
def gauge_chart(value, label):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        number=dict(suffix="%", font=dict(size=28, color=text_col, family="Inter")),
        title=dict(text=label, font=dict(size=13, color=muted_col, family="Inter")),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor=border2_col, tickfont=dict(color=muted_col, size=10)),
            bar=dict(color=teal_main, thickness=0.6),
            bgcolor=border_col,
            borderwidth=0,
            # color zones: red for low confidence, yellow for medium, green for high
            steps=[
                dict(range=[0, 50],  color="rgba(248,81,73,0.12)"),
                dict(range=[50, 75], color="rgba(210,153,34,0.12)"),
                dict(range=[75, 100], color="rgba(63,185,80,0.12)")
            ],
            threshold=dict(line=dict(color=teal_light, width=2), thickness=0.7, value=value * 100),
        )
    ))
    fig.update_layout(
        paper_bgcolor=card_col,
        plot_bgcolor=card_col,
        font=dict(family="Inter", color=muted_col, size=12),
        height=200,
        margin=dict(l=20, r=20, t=30, b=10)
    )
    return fig


# --- bar chart comparing CNN vs EfficientNet accuracy ---
def model_comparison_chart():
    fig = go.Figure()
    # adding both models as separate bars so i can color them differently
    for name, val, color in [
        ("Custom CNN",    cnn_acc * 100, orange_col),
        ("EfficientNetB0", tr_acc * 100, teal_main)
    ]:
        fig.add_trace(go.Bar(
            x=[name], y=[val],
            name=name,
            marker=dict(color=color, opacity=0.85),
            text=[f"{val:.2f}%"],
            textposition="outside",
            textfont=dict(color=text_col, size=13),
            width=0.4
        ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=card_col,
        font=dict(family="Inter", color=muted_col, size=12),
        title=dict(text="Model Accuracy Comparison", font=dict(size=14, color=text_col)),
        height=340,
        showlegend=False,
        bargap=0.4,
        xaxis=dict(gridcolor=border_col, linecolor=border2_col),
        yaxis=dict(range=[0, 115], ticksuffix="%", gridcolor=border_col, linecolor=border2_col),
        margin=dict(l=40, r=20, t=50, b=40)
    )
    return fig


# --- pie chart showing dataset class distribution ---
# dataset has roughly equal birds and drones — 1661 vs 1658
def class_dist_chart():
    fig = go.Figure(go.Pie(
        labels=["Bird", "Drone"],
        values=[1661, 1658],
        hole=0.55,  # donut style
        marker=dict(colors=[teal_main, orange_col], line=dict(color=bg_col, width=3)),
        textinfo="label+percent",
        textfont=dict(color=text_col, size=12),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=card_col,
        font=dict(family="Inter", color=muted_col, size=12),
        title=dict(text="Dataset Distribution", font=dict(size=14, color=text_col)),
        height=340,
        showlegend=False,
        margin=dict(l=20, r=20, t=50, b=40),
        annotations=[dict(
            text="3,319<br>images", x=0.5, y=0.5,
            showarrow=False,
            font=dict(color=text_col, size=14, family="Inter")
        )]
    )
    return fig


# --- yolo training curves chart ---
# only trained for 3 epochs on cpu so the numbers arent great
# but its enough to show the trend
def yolo_chart():
    fig = go.Figure()
    # map50 goes up which is good
    fig.add_trace(go.Scatter(
        x=[1, 2, 3], y=[0.163, 0.226, 0.221],
        name="mAP50",
        mode="lines+markers",
        line=dict(color=teal_main, width=2.5),
        marker=dict(size=7, color=teal_main)
    ))
    # box loss going down — model is learning bounding boxes
    fig.add_trace(go.Scatter(
        x=[1, 2, 3], y=[0.1337, 0.1305, 0.1274],
        name="box_loss/10",
        mode="lines+markers",
        line=dict(color=orange_col, width=2, dash="dot"),
        marker=dict(size=6, color=orange_col)
    ))
    # classification loss also going down
    fig.add_trace(go.Scatter(
        x=[1, 2, 3], y=[0.3146, 0.2108, 0.1447],
        name="cls_loss/10",
        mode="lines+markers",
        line=dict(color=red_col, width=2, dash="dot"),
        marker=dict(size=6, color=red_col)
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=card_col,
        font=dict(family="Inter", color=muted_col, size=12),
        title=dict(text="YOLOv8 Training Curves", font=dict(size=14, color=text_col)),
        height=340,
        margin=dict(l=40, r=20, t=90, b=40),
        legend=dict(
            font=dict(color="#c9d1d9", size=10),
            bgcolor="rgba(0,0,0,0)",
            orientation="h", yanchor="bottom", y=1.0, x=0
        ),
        xaxis=dict(
            title="Epoch", gridcolor=border_col, linecolor=border2_col,
            tickmode="array", tickvals=[1, 2, 3], ticktext=["1", "2", "3"]
        ),
        yaxis=dict(title="Value", gridcolor=border_col, linecolor=border2_col)
    )
    return fig


# --- radar chart for precision/recall/f1 per class ---
# shows both models side by side — efficientnet clearly wins
def radar_chart():
    cats = [
        "Precision<br>Bird", "Recall<br>Bird", "F1<br>Bird",
        "Precision<br>Drone", "Recall<br>Drone", "F1<br>Drone"
    ]
    fig = go.Figure()
    # cnn metrics
    fig.add_trace(go.Scatterpolar(
        r=[0.93, 0.82, 0.87, 0.80, 0.91, 0.85, 0.93],
        theta=cats + [cats[0]],  # close the shape by repeating first point
        fill="toself",
        name="Custom CNN",
        line=dict(color=orange_col, width=2),
        fillcolor="rgba(210,153,34,0.15)"
    ))
    # efficientnet metrics
    fig.add_trace(go.Scatterpolar(
        r=[0.98, 0.99, 0.98, 0.99, 0.97, 0.98, 0.98],
        theta=cats + [cats[0]],
        fill="toself",
        name="EfficientNetB0",
        line=dict(color=teal_main, width=2),
        fillcolor="rgba(79,152,163,0.15)"
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=card_col,
        font=dict(family="Inter", color=muted_col, size=12),
        polar=dict(
            bgcolor=card_col,
            radialaxis=dict(
                visible=True, range=[0, 1],
                gridcolor=border_col,
                tickfont=dict(color=muted_col, size=9),
                linecolor=border2_col
            ),
            angularaxis=dict(
                gridcolor=border_col, linecolor=border2_col,
                tickfont=dict(color="#c9d1d9", size=10)
            )
        ),
        legend=dict(font=dict(color="#c9d1d9", size=11), bgcolor="rgba(0,0,0,0)"),
        title=dict(text="Classification Metrics Radar", font=dict(size=14, color=text_col)),
        height=420,
        margin=dict(l=30, r=30, t=50, b=30),
    )
    return fig


# -----------------------------------------------------------------------
# SIDEBAR
# simple sidebar with mode selector and model dropdown
# -----------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center;padding:16px 0 20px'>
      <div style='font-size:2.5rem;margin-bottom:6px'>🛸</div>
      <div style='font-size:1.2rem;font-weight:700;color:{teal_main}'>AeroScan</div>
      <div style='font-size:0.75rem;color:{muted_col}'>Aerial Object Intelligence</div>
      <hr style='border-color:{border_col};margin:12px 0'>
    </div>
    """, unsafe_allow_html=True)

    # radio button to switch between classification and yolo detection
    task = st.radio(
        "Select Mode",
        ["Classification", "Object Detection (YOLOv8)"],
        label_visibility="collapsed"
    )

    # load models here so we know which ones are available
    models = load_classification_models()
    model_choice = None

    # only show model selector in classification mode
    if task == "Classification" and models:
        st.markdown(
            f"<div style='color:{muted_col};font-size:0.7rem;text-transform:uppercase;"
            f"letter-spacing:0.1em;font-weight:600;margin:1rem 0 0.4rem'>Model</div>",
            unsafe_allow_html=True
        )
        model_choice = st.selectbox(
            "Select Model", list(models.keys()),
            label_visibility="collapsed"
        )

    st.markdown(f"<hr style='border-color:{border_col}'>", unsafe_allow_html=True)
    # quick info panel at the bottom of sidebar
    st.markdown(f"""
    <div style='color:{muted_col};font-size:0.78rem;line-height:1.6'>
      Real-time bird vs drone classification.<br><br>
      <b style='color:#c9d1d9'>Dataset:</b> 3,319 aerial images<br>
      <b style='color:#c9d1d9'>Classes:</b> Bird · Drone<br>
      <b style='color:#c9d1d9'>Best Model:</b> EfficientNetB0<br>
      <b style='color:#c9d1d9'>Accuracy:</b> 99.07%
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------
# HERO SECTION
# -----------------------------------------------------------------------
st.markdown("""
<div class="hero">
  <div class="hero-badge">AI-Powered Detection</div>
  <h1 class="hero-title">Aerial Object Classification & Detection</h1>
  <p class="hero-sub">Deep learning pipeline for real-time Bird vs Drone identification —
  Custom CNN · EfficientNetB0 · YOLOv8</p>
</div>
""", unsafe_allow_html=True)

# top KPI row — shows key numbers at a glance
k1, k2, k3, k4 = st.columns(4)
kpi(k1, "Best Accuracy",  f"{tr_acc:.2%}",  "EfficientNetB0")
kpi(k2, "CNN Accuracy",   f"{cnn_acc:.2%}", "Custom CNN")
kpi(k3, "Dataset Size",   "3,319",          "Aerial images")
kpi(k4, "YOLOv8 mAP50",  "22.5%",          "CPU baseline · 3 epochs")

st.markdown("<br>", unsafe_allow_html=True)


# -----------------------------------------------------------------------
# INFERENCE ENGINE SECTION
# upload image on the left, results on the right
# -----------------------------------------------------------------------
st.markdown('<div class="sec-head">🔍 Inference Engine</div>', unsafe_allow_html=True)

# styling the container borders
st.markdown(f"""
<style>
  [data-testid="stVerticalBlockBorderWrapper"] {{
    background: {card_col} !important;
    border: 1px solid {border_col} !important;
    border-radius: 12px !important;
    padding: 0.5rem !important;
  }}
</style>
""", unsafe_allow_html=True)

col_up, col_res = st.columns(2, gap="large")

# left column — file upload
with col_up:
    with st.container(border=True):
        st.markdown(
            f'<p style="color:{muted_col};font-size:0.75rem;font-weight:600;'
            f'letter-spacing:0.08em;text-transform:uppercase;margin-bottom:0.5rem">📁 Upload Image</p>',
            unsafe_allow_html=True
        )
        uploaded_file = st.file_uploader(
            "Upload aerial image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, use_container_width=True)
            w, h = image.size
            st.caption(f"{uploaded_file.name} · {w}×{h}px")
        else:
            # placeholder when no image is uploaded
            st.markdown(f"""
            <div style='text-align:center;padding:2rem 1rem;color:{muted_col}'>
              <div style='font-size:3rem;margin-bottom:0.5rem'>📸</div>
              <p style='font-size:0.9rem;font-weight:500;color:#c9d1d9;margin:0'>Drop an aerial image here</p>
              <p style='font-size:0.75rem;margin-top:0.3rem'>JPG · JPEG · PNG · up to 200MB</p>
            </div>""", unsafe_allow_html=True)

# right column — prediction results
with col_res:
    with st.container(border=True):
        st.markdown(
            f'<p style="color:{muted_col};font-size:0.75rem;font-weight:600;'
            f'letter-spacing:0.08em;text-transform:uppercase;margin-bottom:0.5rem">🎯 Prediction Results</p>',
            unsafe_allow_html=True
        )

        if not uploaded_file:
            # show instructions and model reference table when idle
            st.markdown(f"""
            <div style='margin-bottom:1rem'>
              <p style='color:{text_col};font-size:0.85rem;font-weight:600;margin-bottom:0.6rem'>⚡ How It Works</p>
              <div style='display:flex;flex-direction:column;gap:0.4rem'>
                <div style='display:flex;align-items:center;gap:0.75rem;padding:0.5rem 0.75rem;
                             background:rgba(79,152,163,0.06);border-radius:8px;border-left:3px solid {teal_main}'>
                  <span style='font-size:1.1rem'>1️⃣</span>
                  <span style='color:#c9d1d9;font-size:0.8rem'>Upload an aerial image (bird or drone)</span>
                </div>
                <div style='display:flex;align-items:center;gap:0.75rem;padding:0.5rem 0.75rem;
                             background:rgba(79,152,163,0.06);border-radius:8px;border-left:3px solid {teal_main}'>
                  <span style='font-size:1.1rem'>2️⃣</span>
                  <span style='color:#c9d1d9;font-size:0.8rem'>Select model in sidebar (EfficientNetB0 recommended)</span>
                </div>
                <div style='display:flex;align-items:center;gap:0.75rem;padding:0.5rem 0.75rem;
                             background:rgba(79,152,163,0.06);border-radius:8px;border-left:3px solid {teal_main}'>
                  <span style='font-size:1.1rem'>3️⃣</span>
                  <span style='color:#c9d1d9;font-size:0.8rem'>Click Classify Image — get instant prediction</span>
                </div>
                <div style='display:flex;align-items:center;gap:0.75rem;padding:0.5rem 0.75rem;
                             background:rgba(79,152,163,0.06);border-radius:8px;border-left:3px solid {teal_main}'>
                  <span style='font-size:1.1rem'>4️⃣</span>
                  <span style='color:#c9d1d9;font-size:0.8rem'>For bounding boxes, switch to YOLOv8 mode</span>
                </div>
              </div>
            </div>
            <div>
              <p style='color:{text_col};font-size:0.85rem;font-weight:600;margin-bottom:0.6rem'>🤖 Model Quick Reference</p>
              <table style='width:100%;border-collapse:collapse;font-size:0.78rem'>
                <tr style='border-bottom:1px solid {border_col}'>
                  <th style='color:{muted_col};font-weight:500;text-align:left;padding:0.35rem 0'>Model</th>
                  <th style='color:{muted_col};font-weight:500;text-align:center;padding:0.35rem 0'>Accuracy</th>
                  <th style='color:{muted_col};font-weight:500;text-align:center;padding:0.35rem 0'>Speed</th>
                  <th style='color:{muted_col};font-weight:500;text-align:center;padding:0.35rem 0'>Task</th>
                </tr>
                <tr style='border-bottom:1px solid {border_col}'>
                  <td style='color:{text_col};padding:0.4rem 0;font-weight:500'>EfficientNetB0</td>
                  <td style='color:{green_col};text-align:center;font-weight:600'>99.07%</td>
                  <td style='color:{teal_main};text-align:center'>Fast</td>
                  <td style='color:{muted_col};text-align:center'>Classification</td>
                </tr>
                <tr style='border-bottom:1px solid {border_col}'>
                  <td style='color:{text_col};padding:0.4rem 0;font-weight:500'>Custom CNN</td>
                  <td style='color:{orange_col};text-align:center;font-weight:600'>86.05%</td>
                  <td style='color:{teal_main};text-align:center'>Fast</td>
                  <td style='color:{muted_col};text-align:center'>Classification</td>
                </tr>
                <tr>
                  <td style='color:{text_col};padding:0.4rem 0;font-weight:500'>YOLOv8n</td>
                  <td style='color:{red_col};text-align:center;font-weight:600'>mAP 22%</td>
                  <td style='color:{orange_col};text-align:center'>Medium</td>
                  <td style='color:{muted_col};text-align:center'>Detection</td>
                </tr>
              </table>
            </div>
            """, unsafe_allow_html=True)

        else:
            # --- classification mode ---
            if task == "Classification":
                if not models:
                    st.markdown(
                        '<div class="alert-danger">⚠️ No trained models found. '
                        'Run train_cnn.py and train_transfer.py first.</div>',
                        unsafe_allow_html=True
                    )
                else:
                    if st.button("🚀 Classify Image"):
                        with st.spinner("Running inference..."):
                            # get the selected model type and object
                            mtype, mobj = models[model_choice]
                            arr = preprocess_image(image, mtype)
                            label, confidence = predict_class(mobj, arr)

                            # show result card based on prediction
                            if label == "Bird":
                                st.markdown(f"""
                                <div class="result-bird">
                                  <div style='font-size:1.4rem;font-weight:700;color:{teal_main}'>🦅 BIRD</div>
                                  <div style='font-size:0.8rem;color:{muted_col}'>Wildlife — No security threat</div>
                                </div>""", unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class="result-drone">
                                  <div style='font-size:1.4rem;font-weight:700;color:{red_col}'>🛸 DRONE</div>
                                  <div style='font-size:0.8rem;color:{muted_col}'>⚠️ Unmanned aerial vehicle detected</div>
                                </div>""", unsafe_allow_html=True)

                            st.markdown("<br>", unsafe_allow_html=True)

                            # confidence gauge
                            st.plotly_chart(
                                gauge_chart(confidence, f"Confidence — {model_choice}"),
                                use_container_width=True,
                                config={"displayModeBar": False}
                            )

                            # confidence progress bar
                            conf_pct = confidence * 100
                            bar_color = teal_main if label == "Bird" else red_col
                            st.markdown(f"""
                            <div style='margin-top:0.5rem'>
                              <div style='display:flex;justify-content:space-between;margin-bottom:0.3rem'>
                                <span style='color:{muted_col};font-size:0.75rem'>Confidence Score</span>
                                <span style='color:{text_col};font-size:0.75rem;font-weight:600'>{conf_pct:.2f}%</span>
                              </div>
                              <div style='height:6px;background:{border_col};border-radius:4px;overflow:hidden'>
                                <div style='width:{conf_pct}%;height:100%;background:{bar_color};border-radius:4px'></div>
                              </div>
                            </div>""", unsafe_allow_html=True)

            # --- yolo detection mode ---
            else:
                yolo_model = load_yolo_model()
                if yolo_model is None:
                    st.markdown(
                        '<div class="alert-danger">⚠️ YOLOv8 model not found. '
                        'Run python yolo/train_yolo.py first.</div>',
                        unsafe_allow_html=True
                    )
                else:
                    if st.button("🎯 Detect Objects"):
                        with st.spinner("Running YOLOv8..."):
                            results = yolo_model(np.array(image))
                            # convert bgr to rgb for display
                            annotated = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
                            st.image(annotated, use_container_width=True)
                            boxes = results[0].boxes

                            if boxes and len(boxes):
                                st.markdown(
                                    f'<div class="alert-info">✅ {len(boxes)} object(s) detected</div>',
                                    unsafe_allow_html=True
                                )
                                # loop through detections and show label + confidence
                                for box in boxes:
                                    name = "Bird 🦅" if int(box.cls[0]) == 0 else "Drone 🛸"
                                    conf = float(box.conf[0])
                                    st.markdown(f"""
                                    <div style='display:flex;justify-content:space-between;
                                      padding:0.4rem 0.6rem;background:{border_col};border-radius:6px;margin-top:0.4rem'>
                                      <span style='color:{text_col};font-size:0.85rem'>{name}</span>
                                      <span style='color:{teal_main};font-size:0.85rem;font-weight:600'>{conf:.2%}</span>
                                    </div>""", unsafe_allow_html=True)

                            else:
                                # yolo missed it — only trained 3 epochs so this happens sometimes
                                # falling back to efficientnet classification as backup
                                st.markdown(f"""
                                <div style='background:rgba(210,153,34,0.1);border:1px solid rgba(210,153,34,0.35);
                                  border-radius:8px;padding:0.6rem 0.9rem;margin-bottom:0.8rem;
                                  font-size:0.8rem;color:{orange_col}'>
                                  ⚠️ YOLOv8 found no boxes (mAP 22% — model needs more training).
                                  Auto-running EfficientNetB0 classification instead…
                                </div>""", unsafe_allow_html=True)

                                cls_models = load_classification_models()
                                if cls_models:
                                    # prefer efficientnet if available
                                    best_key = (
                                        "EfficientNetB0 (99.07%)"
                                        if "EfficientNetB0 (99.07%)" in cls_models
                                        else list(cls_models.keys())[0]
                                    )
                                    mtype, mobj = cls_models[best_key]
                                    arr = preprocess_image(image, mtype)
                                    label, confidence = predict_class(mobj, arr)

                                    if label == "Bird":
                                        st.markdown(f"""
                                        <div class="result-bird">
                                          <div style='font-size:1.4rem;font-weight:700;color:{teal_main}'>🦅 BIRD</div>
                                          <div style='font-size:0.8rem;color:{muted_col}'>Wildlife — No security threat</div>
                                        </div>""", unsafe_allow_html=True)
                                    else:
                                        st.markdown(f"""
                                        <div class="result-drone">
                                          <div style='font-size:1.4rem;font-weight:700;color:{red_col}'>🛸 DRONE</div>
                                          <div style='font-size:0.8rem;color:{muted_col}'>⚠️ Unmanned aerial vehicle detected</div>
                                        </div>""", unsafe_allow_html=True)

                                    st.markdown("<br>", unsafe_allow_html=True)
                                    st.plotly_chart(
                                        gauge_chart(confidence, f"Confidence — {best_key}"),
                                        use_container_width=True,
                                        config={"displayModeBar": False}
                                    )
                                    conf_pct = confidence * 100
                                    bar_color = teal_main if label == "Bird" else red_col
                                    st.markdown(f"""
                                    <div style='margin-top:0.5rem'>
                                      <div style='display:flex;justify-content:space-between;margin-bottom:0.3rem'>
                                        <span style='color:{muted_col};font-size:0.75rem'>Confidence (EfficientNetB0 fallback)</span>
                                        <span style='color:{text_col};font-size:0.75rem;font-weight:600'>{conf_pct:.2f}%</span>
                                      </div>
                                      <div style='height:6px;background:{border_col};border-radius:4px;overflow:hidden'>
                                        <div style='width:{conf_pct}%;height:100%;background:{bar_color};border-radius:4px'></div>
                                      </div>
                                    </div>""", unsafe_allow_html=True)
                                else:
                                    st.markdown(
                                        '<div class="alert-danger">⚠️ No classification models found either.</div>',
                                        unsafe_allow_html=True
                                    )


# -----------------------------------------------------------------------
# ANALYTICS DASHBOARD
# showing model performance charts below the inference section
# -----------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="sec-head">📊 Model Analytics Dashboard</div>', unsafe_allow_html=True)

# top row — 3 charts side by side
c1, c2, c3 = st.columns([1, 1, 1], gap="medium")
with c1:
    st.plotly_chart(model_comparison_chart(), use_container_width=True, config={"displayModeBar": False})
with c2:
    st.plotly_chart(class_dist_chart(), use_container_width=True, config={"displayModeBar": False})
with c3:
    st.plotly_chart(yolo_chart(), use_container_width=True, config={"displayModeBar": False})

# bottom row — radar chart + summary table
r1, r2 = st.columns([1.2, 1], gap="medium")
with r1:
    st.plotly_chart(radar_chart(), use_container_width=True, config={"displayModeBar": False})
with r2:
    # summary table with best architecture note
    st.markdown(f"""
    <div class="panel" style='height:420px'>
      <p style='color:{muted_col};font-size:0.8rem;font-weight:500;margin-bottom:1rem'>MODEL SUMMARY</p>
      <table style='width:100%;border-collapse:collapse;font-size:0.8rem'>
        <tr style='border-bottom:1px solid {border_col}'>
          <th style='color:#6e7681;font-weight:500;text-align:left;padding:0.4rem 0'>Model</th>
          <th style='color:#6e7681;font-weight:500;text-align:right;padding:0.4rem 0'>Acc</th>
          <th style='color:#6e7681;font-weight:500;text-align:right;padding:0.4rem 0'>F1</th>
        </tr>
        <tr style='border-bottom:1px solid {border_col}'>
          <td style='color:{text_col};padding:0.5rem 0'>EfficientNetB0</td>
          <td style='color:{teal_main};font-weight:600;text-align:right'>99.07%</td>
          <td style='color:{teal_main};font-weight:600;text-align:right'>0.98</td>
        </tr>
        <tr style='border-bottom:1px solid {border_col}'>
          <td style='color:{text_col};padding:0.5rem 0'>Custom CNN</td>
          <td style='color:{orange_col};font-weight:600;text-align:right'>86.05%</td>
          <td style='color:{orange_col};font-weight:600;text-align:right'>0.86</td>
        </tr>
        <tr>
          <td style='color:{text_col};padding:0.5rem 0'>YOLOv8n</td>
          <td style='color:{red_col};font-weight:600;text-align:right'>mAP 22%</td>
          <td style='color:{muted_col};font-weight:600;text-align:right'>CPU</td>
        </tr>
      </table>
      <div style='margin-top:1.2rem;padding:0.75rem;background:rgba(79,152,163,0.08);
                  border:1px solid rgba(79,152,163,0.2);border-radius:8px'>
        <p style='color:{teal_main};font-size:0.75rem;font-weight:600;margin-bottom:0.3rem'>🏆 Best Architecture</p>
        <p style='color:{muted_col};font-size:0.73rem;line-height:1.5'>EfficientNetB0 with transfer learning —
        99.07% test accuracy on 215 test images. Fine-tuned on aerial imagery.</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

# footer
st.markdown(f"""
<div style='margin-top:3rem;padding-top:1.5rem;border-top:1px solid {border_col};
  display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.5rem'>
  <div style='color:#6e7681;font-size:0.75rem'>
    🛸 <b style='color:{teal_main}'>AeroScan</b> — Aerial Object Detection System
  </div>
  <div style='color:#6e7681;font-size:0.75rem;display:flex;gap:1rem'>
    <span>TensorFlow 2.x</span><span>·</span>
    <span>Ultralytics YOLOv8</span><span>·</span>
    <span>Plotly</span><span>·</span><span>Streamlit</span>
  </div>
</div>
""", unsafe_allow_html=True)