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
import plotly.express as px
from ultralytics import YOLO

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AeroScan — Bird vs Drone AI",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }

  /* Hide Streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1400px; }

  /* Dark background */
  .stApp { background-color: #0d1117; }

  /* ── Fix sidebar collapse/expand arrow visibility ── */
  [data-testid="collapsedControl"] {
    color: #4f98a3 !important;
    background-color: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 50% !important;
    visibility: visible !important;
    opacity: 1 !important;
  }
  [data-testid="collapsedControl"] svg {
    fill: #4f98a3 !important;
    stroke: #4f98a3 !important;
  }
  button[kind="header"] {
    color: #4f98a3 !important;
    background-color: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 50% !important;
    visibility: visible !important;
    opacity: 1 !important;
  }
  button[kind="header"] svg {
    fill: #4f98a3 !important;
    stroke: #4f98a3 !important;
  }
  /* Sidebar toggle arrow (various Streamlit versions) */
  .st-emotion-cache-1cypcdb,
  .st-emotion-cache-6qob1r,
  [data-testid="stSidebarCollapseButton"],
  [data-testid="stSidebarCollapsedControl"] {
    visibility: visible !important;
    opacity: 1 !important;
    color: #4f98a3 !important;
  }
  [data-testid="stSidebarCollapseButton"] svg,
  [data-testid="stSidebarCollapsedControl"] svg {
    fill: #4f98a3 !important;
  }

  /* ── Hero header ── */
  .hero-header {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
    border: 1px solid #21262d;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
  }
  .hero-header::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(79,152,163,0.15) 0%, transparent 70%);
    border-radius: 50%;
  }
  .hero-title {
    font-size: 2rem; font-weight: 700;
    background: linear-gradient(90deg, #4f98a3, #7ec8d3);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0; line-height: 1.2;
  }
  .hero-subtitle {
    color: #8b949e; font-size: 0.95rem; margin-top: 0.4rem; font-weight: 400;
  }
  .hero-badge {
    display: inline-block;
    background: rgba(79,152,163,0.12);
    border: 1px solid rgba(79,152,163,0.3);
    color: #4f98a3; font-size: 0.7rem; font-weight: 600;
    padding: 0.2rem 0.6rem; border-radius: 20px;
    letter-spacing: 0.08em; text-transform: uppercase;
    margin-bottom: 0.8rem;
  }

  /* ── Metric cards ── */
  .metric-card {
    background: #161b22; border: 1px solid #21262d;
    border-radius: 12px; padding: 1.25rem 1.5rem;
    transition: border-color 0.2s, transform 0.2s;
  }
  .metric-card:hover { border-color: #4f98a3; transform: translateY(-2px); }
  .metric-label { color: #8b949e; font-size: 0.75rem; font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.4rem; }
  .metric-value { color: #e6edf3; font-size: 1.8rem; font-weight: 700; line-height: 1; }
  .metric-sub { color: #4f98a3; font-size: 0.75rem; margin-top: 0.3rem; }

  /* ── Upload zone ── */
  .upload-zone {
    background: #161b22; border: 2px dashed #30363d;
    border-radius: 12px; padding: 1rem;
    transition: border-color 0.2s;
  }

  /* ── Result card ── */
  .result-card {
    border-radius: 12px; padding: 1.25rem 1.5rem; margin: 0.5rem 0;
    border: 1px solid;
  }
  .result-bird {
    background: rgba(79,152,163,0.08);
    border-color: rgba(79,152,163,0.35);
  }
  .result-drone {
    background: rgba(248,81,73,0.08);
    border-color: rgba(248,81,73,0.35);
  }
  .result-title { font-size: 1.4rem; font-weight: 700; margin: 0; }
  .result-bird .result-title { color: #4f98a3; }
  .result-drone .result-title { color: #f85149; }
  .result-subtitle { font-size: 0.8rem; color: #8b949e; margin-top: 0.2rem; }

  /* ── Section heading ── */
  .section-heading {
    color: #e6edf3; font-size: 1rem; font-weight: 600;
    border-bottom: 1px solid #21262d;
    padding-bottom: 0.6rem; margin-bottom: 1rem;
    display: flex; align-items: center; gap: 0.5rem;
  }
  .section-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #4f98a3; display: inline-block;
  }

  /* ── Panel ── */
  .panel {
    background: #161b22; border: 1px solid #21262d;
    border-radius: 12px; padding: 1.5rem;
    height: 100%;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #21262d !important;
  }
  [data-testid="stSidebar"] .stRadio label { color: #c9d1d9 !important; }
  [data-testid="stSidebar"] .stSelectbox label { color: #c9d1d9 !important; }
  .sidebar-logo {
    font-size: 1.2rem; font-weight: 700; color: #4f98a3;
    padding: 1rem 0 0.5rem; letter-spacing: -0.02em;
  }
  .sidebar-tagline { color: #8b949e; font-size: 0.75rem; margin-bottom: 1.5rem; }
  .sidebar-section {
    color: #6e7681; font-size: 0.7rem; text-transform: uppercase;
    letter-spacing: 0.1em; font-weight: 600; margin: 1rem 0 0.5rem;
  }

  /* ── Alert boxes ── */
  .alert-success {
    background: rgba(63,185,80,0.1); border: 1px solid rgba(63,185,80,0.3);
    border-radius: 8px; padding: 0.75rem 1rem; color: #3fb950; font-size: 0.85rem;
  }
  .alert-danger {
    background: rgba(248,81,73,0.1); border: 1px solid rgba(248,81,73,0.3);
    border-radius: 8px; padding: 0.75rem 1rem; color: #f85149; font-size: 0.85rem;
  }
  .alert-info {
    background: rgba(79,152,163,0.1); border: 1px solid rgba(79,152,163,0.3);
    border-radius: 8px; padding: 0.75rem 1rem; color: #4f98a3; font-size: 0.85rem;
  }

  /* Streamlit overrides */
  .stButton>button {
    background: linear-gradient(135deg, #4f98a3, #3a7a85) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important; width: 100% !important;
    transition: opacity 0.2s !important;
  }
  .stButton>button:hover { opacity: 0.85 !important; }
  div[data-testid="stMetricValue"] { color: #e6edf3 !important; font-family: 'Inter' !important; }
  div[data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 0.75rem !important; }
  .stSpinner > div { border-top-color: #4f98a3 !important; }
  .stProgress > div > div { background: linear-gradient(90deg, #4f98a3, #7ec8d3) !important; border-radius: 4px !important; }
  [data-testid="stFileUploaderDropzone"] {
    background: #161b22 !important; border: 2px dashed #30363d !important;
    border-radius: 12px !important;
  }
  [data-testid="stFileUploaderDropzone"]:hover { border-color: #4f98a3 !important; }
  .stSelectbox > div > div {
    background: #21262d !important; border-color: #30363d !important;
    color: #e6edf3 !important;
  }
  .stRadio > div { gap: 0.5rem; }
  .stRadio > div > label { color: #c9d1d9 !important; }
  div[data-testid="stImage"] img { border-radius: 10px; }
  .stMarkdown h3 { color: #e6edf3 !important; }

  /* Plotly chart background */
  .js-plotly-plot .plotly { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


# ── Plotly theme ───────────────────────────────────────────────────────────────
# NOTE: PLOTLY_LAYOUT must NOT include 'margin' — it is passed separately per chart
PLOTLY_LAYOUT = dict(
    paper_bgcolor='#161b22',
    plot_bgcolor='#161b22',
    font=dict(family='Inter', color='#c9d1d9', size=12),
    xaxis=dict(gridcolor='#21262d', linecolor='#30363d', zerolinecolor='#21262d'),
    yaxis=dict(gridcolor='#21262d', linecolor='#30363d', zerolinecolor='#21262d'),
)
TEAL    = '#4f98a3'
TEAL2   = '#7ec8d3'
RED     = '#f85149'
GREEN   = '#3fb950'
ORANGE  = '#d29922'
PURPLE  = '#bc8cff'


# ── Cached model loaders ───────────────────────────────────────────────────────
@st.cache_resource
def load_classification_models():
    models = {}
    if os.path.exists('models/transfer_model.h5'):
        models['EfficientNetB0 (99.07%)'] = ('efficientnet', load_model('models/transfer_model.h5'))
    if os.path.exists('models/custom_cnn.h5'):
        models['Custom CNN (86.05%)'] = ('cnn', load_model('models/custom_cnn.h5'))
    return models


@st.cache_resource
def load_yolo_model():
    paths = [
        'runs/detect/runs/detect/bird_drone/weights/best.pt',
        'runs/detect/bird_drone/weights/best.pt',
    ]
    for p in paths:
        if os.path.exists(p):
            return YOLO(p)
    return None


# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess_image(image, model_type='cnn'):
    img = image.resize((224, 224))
    img_array = img_to_array(img)
    if model_type == 'efficientnet':
        img_array = efficientnet_preprocess(img_array)
    else:
        img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)


def predict_class(model, img_array):
    prediction = model.predict(img_array, verbose=0)
    confidence = float(prediction[0][0])
    if confidence > 0.5:
        return 'Drone', confidence
    else:
        return 'Bird', 1 - confidence


# ── Load metrics ───────────────────────────────────────────────────────────────
def load_metrics():
    cnn_acc, tr_acc = 0.8605, 0.9907
    try:
        if os.path.exists('logs/cnn_metrics.json'):
            with open('logs/cnn_metrics.json') as f:
                cnn_acc = json.load(f).get('test_accuracy', cnn_acc)
    except Exception:
        pass
    try:
        if os.path.exists('logs/transfer_metrics.json'):
            with open('logs/transfer_metrics.json') as f:
                tr_acc = json.load(f).get('test_accuracy', tr_acc)
    except Exception:
        pass
    return cnn_acc, tr_acc


cnn_acc, tr_acc = load_metrics()


# ── Plotly charts ──────────────────────────────────────────────────────────────
def gauge_chart(value, label):
    """
    FIX: Do NOT spread **PLOTLY_LAYOUT here — it contains xaxis/yaxis keys that
    conflict with Indicator figures, and passing margin both inside PLOTLY_LAYOUT
    (if it were there) and explicitly would raise TypeError.
    Instead, set all layout properties individually.
    """
    fig = go.Figure(go.Indicator(
        mode='gauge+number',
        value=value * 100,
        number=dict(suffix='%', font=dict(size=28, color='#e6edf3', family='Inter')),
        title=dict(text=label, font=dict(size=13, color='#8b949e', family='Inter')),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor='#30363d',
                      tickfont=dict(color='#8b949e', size=10)),
            bar=dict(color=TEAL, thickness=0.6),
            bgcolor='#21262d',
            borderwidth=0,
            steps=[
                dict(range=[0, 50],  color='rgba(248,81,73,0.12)'),
                dict(range=[50, 75], color='rgba(210,153,34,0.12)'),
                dict(range=[75, 100], color='rgba(63,185,80,0.12)'),
            ],
            threshold=dict(line=dict(color=TEAL2, width=2), thickness=0.7, value=value * 100),
        )
    ))
    # Set layout properties individually — no **PLOTLY_LAYOUT spread to avoid
    # duplicate-keyword errors with 'margin' or any future key collision.
    fig.update_layout(
        paper_bgcolor='#161b22',
        plot_bgcolor='#161b22',
        font=dict(family='Inter', color='#c9d1d9', size=12),
        height=200,
        margin=dict(l=20, r=20, t=30, b=10),
    )
    return fig


def model_comparison_chart(cnn, tr):
    fig = go.Figure()
    models_names = ['Custom CNN', 'EfficientNetB0']
    values = [cnn * 100, tr * 100]
    colors = [ORANGE, TEAL]
    for i, (name, val, color) in enumerate(zip(models_names, values, colors)):
        fig.add_trace(go.Bar(
            x=[name], y=[val], name=name,
            marker=dict(color=color, opacity=0.85, line=dict(color=color, width=1)),
            text=[f'{val:.2f}%'], textposition='outside',
            textfont=dict(color='#e6edf3', size=13, family='Inter'),
            width=0.4,
        ))
    fig.update_layout(
        paper_bgcolor='#161b22',
        plot_bgcolor='#161b22',
        font=dict(family='Inter', color='#c9d1d9', size=12),
        xaxis=dict(gridcolor='#21262d', linecolor='#30363d', zerolinecolor='#21262d'),
        yaxis=dict(range=[0, 115], ticksuffix='%', gridcolor='#21262d',
                   linecolor='#30363d', zerolinecolor='#21262d'),
        title=dict(text='Model Accuracy Comparison', font=dict(size=14, color='#e6edf3')),
        height=300,
        showlegend=False,
        bargap=0.4,
    )
    return fig


def class_distribution_chart():
    fig = go.Figure(go.Pie(
        labels=['Bird', 'Drone'],
        values=[1661, 1658],
        hole=0.55,
        marker=dict(colors=[TEAL, ORANGE],
                    line=dict(color='#161b22', width=3)),
        textinfo='label+percent',
        textfont=dict(color='#e6edf3', size=12),
    ))
    fig.update_layout(
        paper_bgcolor='#161b22',
        plot_bgcolor='#161b22',
        font=dict(family='Inter', color='#c9d1d9', size=12),
        title=dict(text='Dataset Distribution', font=dict(size=14, color='#e6edf3')),
        height=280,
        showlegend=False,
        annotations=[dict(text='3,319<br><span style="font-size:10px">images</span>',
                          x=0.5, y=0.5, showarrow=False,
                          font=dict(color='#e6edf3', size=14, family='Inter'))],
    )
    return fig


def metrics_radar_chart():
    categories = ['Precision<br>Bird', 'Recall<br>Bird', 'F1<br>Bird',
                  'Precision<br>Drone', 'Recall<br>Drone', 'F1<br>Drone']
    cnn_vals  = [0.93, 0.82, 0.87, 0.80, 0.91, 0.85, 0.93]
    eff_vals  = [0.98, 0.99, 0.98, 0.99, 0.97, 0.98, 0.98]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=cnn_vals, theta=categories + [categories[0]],
        fill='toself', name='Custom CNN',
        line=dict(color=ORANGE, width=2),
        fillcolor='rgba(210,153,34,0.15)',
    ))
    fig.add_trace(go.Scatterpolar(
        r=eff_vals, theta=categories + [categories[0]],
        fill='toself', name='EfficientNetB0',
        line=dict(color=TEAL, width=2),
        fillcolor='rgba(79,152,163,0.15)',
    ))
    fig.update_layout(
        paper_bgcolor='#161b22',
        plot_bgcolor='#161b22',
        font=dict(family='Inter', color='#c9d1d9', size=12),
        polar=dict(
            bgcolor='#161b22',
            radialaxis=dict(visible=True, range=[0, 1], gridcolor='#21262d',
                            tickfont=dict(color='#8b949e', size=9), linecolor='#30363d'),
            angularaxis=dict(gridcolor='#21262d', linecolor='#30363d',
                             tickfont=dict(color='#c9d1d9', size=10)),
        ),
        legend=dict(font=dict(color='#c9d1d9', size=11), bgcolor='rgba(0,0,0,0)'),
        title=dict(text='Classification Metrics Radar', font=dict(size=14, color='#e6edf3')),
        height=320,
    )
    return fig


def yolo_metrics_chart():
    fig = go.Figure()
    epochs = [1, 2, 3]
    map50  = [0.163, 0.226, 0.221]
    box    = [1.337, 1.305, 1.274]
    cls    = [3.146, 2.108, 1.447]
    fig.add_trace(go.Scatter(
        x=epochs, y=map50, name='mAP50', mode='lines+markers',
        line=dict(color=TEAL, width=2.5), marker=dict(size=7, color=TEAL),
    ))
    fig.add_trace(go.Scatter(
        x=epochs, y=[v / 10 for v in box], name='box_loss /10', mode='lines+markers',
        line=dict(color=ORANGE, width=2, dash='dot'), marker=dict(size=6, color=ORANGE),
    ))
    fig.add_trace(go.Scatter(
        x=epochs, y=[v / 10 for v in cls], name='cls_loss /10', mode='lines+markers',
        line=dict(color=RED, width=2, dash='dot'), marker=dict(size=6, color=RED),
    ))
    fig.update_layout(
        paper_bgcolor='#161b22',
        plot_bgcolor='#161b22',
        font=dict(family='Inter', color='#c9d1d9', size=12),
        title=dict(text='YOLOv8 Training Curves', font=dict(size=14, color='#e6edf3')),
        height=280,
        legend=dict(font=dict(color='#c9d1d9', size=11), bgcolor='rgba(0,0,0,0)',
                    orientation='h', yanchor='bottom', y=1.02),
        xaxis=dict(title='Epoch', gridcolor='#21262d', linecolor='#30363d', zerolinecolor='#21262d'),
        yaxis=dict(title='Value', gridcolor='#21262d', linecolor='#30363d', zerolinecolor='#21262d'),
    )
    return fig


# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">🛸 AeroScan</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tagline">Aerial Object Intelligence</div>', unsafe_allow_html=True)
    st.divider()

    st.markdown('<div class="sidebar-section">Mode</div>', unsafe_allow_html=True)
    task = st.radio(
        label="Select Mode",
        options=["Classification", "Object Detection (YOLOv8)"],
        label_visibility="collapsed"
    )

    if task == "Classification":
        models = load_classification_models()
        if models:
            st.markdown('<div class="sidebar-section">Model</div>', unsafe_allow_html=True)
            model_choice = st.selectbox("Select Model", list(models.keys()), label_visibility="collapsed")
        else:
            model_choice = None

    st.divider()
    st.markdown('<div class="sidebar-section">About</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="color:#8b949e; font-size:0.78rem; line-height:1.6;">
    Real-time bird vs drone classification using deep learning.
    <br><br>
    <b style="color:#c9d1d9;">Dataset:</b> 3,319 aerial images<br>
    <b style="color:#c9d1d9;">Classes:</b> Bird, Drone<br>
    <b style="color:#c9d1d9;">Best Model:</b> EfficientNetB0<br>
    <b style="color:#c9d1d9;">Accuracy:</b> 99.07%
    </div>
    """, unsafe_allow_html=True)


# ── HERO ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
  <div class="hero-badge">AI-Powered Detection</div>
  <h1 class="hero-title">Aerial Object Classification & Detection</h1>
  <p class="hero-subtitle">
    Deep learning pipeline for real-time Bird vs Drone identification —
    Custom CNN · EfficientNetB0 · YOLOv8
  </p>
</div>
""", unsafe_allow_html=True)

# ── KPI ROW ────────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Best Accuracy</div>
      <div class="metric-value">{tr_acc:.2%}</div>
      <div class="metric-sub">EfficientNetB0</div>
    </div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">CNN Accuracy</div>
      <div class="metric-value">{cnn_acc:.2%}</div>
      <div class="metric-sub">Custom CNN</div>
    </div>""", unsafe_allow_html=True)
with k3:
    st.markdown("""
    <div class="metric-card">
      <div class="metric-label">Dataset Size</div>
      <div class="metric-value">3,319</div>
      <div class="metric-sub">Aerial images</div>
    </div>""", unsafe_allow_html=True)
with k4:
    st.markdown("""
    <div class="metric-card">
      <div class="metric-label">YOLOv8 mAP50</div>
      <div class="metric-value">22.5%</div>
      <div class="metric-sub">CPU baseline · 3 epochs</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)

# ── INFERENCE PANEL ────────────────────────────────────────────────────────────
st.markdown('<div class="section-heading"><span class="section-dot"></span>Inference Engine</div>',
            unsafe_allow_html=True)

col_upload, col_result = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<p style="color:#8b949e;font-size:0.8rem;margin-bottom:0.5rem;font-weight:500;">UPLOAD IMAGE</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload aerial image", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed"
    )
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_container_width=True)
        w, h = image.size
        st.markdown(f'<p style="color:#8b949e;font-size:0.72rem;text-align:center;margin-top:0.3rem;">{uploaded_file.name} · {w}×{h}px</p>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center;padding:3rem 1rem;color:#8b949e;">
          <div style="font-size:2.5rem;margin-bottom:0.75rem;">📸</div>
          <p style="font-size:0.9rem;font-weight:500;color:#c9d1d9;">Drop an aerial image here</p>
          <p style="font-size:0.78rem;">JPG, JPEG, PNG · up to 200MB</p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_result:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<p style="color:#8b949e;font-size:0.8rem;margin-bottom:0.5rem;font-weight:500;">PREDICTION RESULTS</p>', unsafe_allow_html=True)

    if not uploaded_file:
        st.markdown("""
        <div style="text-align:center;padding:3rem 1rem;color:#8b949e;">
          <div style="font-size:2.5rem;margin-bottom:0.75rem;">🔍</div>
          <p style="font-size:0.9rem;font-weight:500;color:#c9d1d9;">Awaiting image</p>
          <p style="font-size:0.78rem;">Upload an image to start analysis</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        if task == "Classification":
            if not models:
                st.markdown('<div class="alert-danger">⚠️ No trained models found. Run train_cnn.py and train_transfer.py first.</div>', unsafe_allow_html=True)
            else:
                if st.button("🚀 Classify Image"):
                    with st.spinner("Running inference..."):
                        model_type, model_obj = models[model_choice]
                        img_array = preprocess_image(image, model_type)
                        label, confidence = predict_class(model_obj, img_array)

                        if label == 'Bird':
                            st.markdown(f"""
                            <div class="result-card result-bird">
                              <div class="result-title">🦅 BIRD</div>
                              <div class="result-subtitle">Wildlife — No security threat</div>
                            </div>""", unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="result-card result-drone">
                              <div class="result-title">🛸 DRONE</div>
                              <div class="result-subtitle">⚠️ Unmanned aerial vehicle detected</div>
                            </div>""", unsafe_allow_html=True)

                        st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
                        fig_gauge = gauge_chart(confidence, f'Confidence — {model_choice}')
                        st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})

                        conf_pct = confidence * 100
                        bar_color = TEAL if label == 'Bird' else RED
                        st.markdown(f"""
                        <div style="margin-top:0.5rem;">
                          <div style="display:flex;justify-content:space-between;margin-bottom:0.3rem;">
                            <span style="color:#8b949e;font-size:0.75rem;">Confidence Score</span>
                            <span style="color:#e6edf3;font-size:0.75rem;font-weight:600;">{conf_pct:.2f}%</span>
                          </div>
                          <div style="height:6px;background:#21262d;border-radius:4px;overflow:hidden;">
                            <div style="width:{conf_pct}%;height:100%;background:linear-gradient(90deg,{bar_color},{bar_color}cc);border-radius:4px;transition:width 0.5s;"></div>
                          </div>
                        </div>
                        """, unsafe_allow_html=True)

        elif task == "Object Detection (YOLOv8)":
            yolo_model = load_yolo_model()
            if yolo_model is None:
                st.markdown('<div class="alert-danger">⚠️ YOLOv8 model not found. Run python yolo/train_yolo.py first.</div>', unsafe_allow_html=True)
            else:
                if st.button("🎯 Detect Objects"):
                    with st.spinner("Running YOLOv8..."):
                        img_array = np.array(image)
                        results = yolo_model(img_array)
                        annotated = results[0].plot()
                        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        st.image(annotated_rgb, use_container_width=True)

                        boxes = results[0].boxes
                        if boxes and len(boxes):
                            st.markdown(f'<div class="alert-info">✅ {len(boxes)} object(s) detected</div>', unsafe_allow_html=True)
                            for box in boxes:
                                cls  = int(box.cls[0])
                                conf = float(box.conf[0])
                                name = 'Bird 🦅' if cls == 0 else 'Drone 🛸'
                                st.markdown(f"""
                                <div style="display:flex;justify-content:space-between;padding:0.4rem 0.6rem;background:#21262d;border-radius:6px;margin-top:0.4rem;">
                                  <span style="color:#e6edf3;font-size:0.85rem;">{name}</span>
                                  <span style="color:#4f98a3;font-size:0.85rem;font-weight:600;">{conf:.2%}</span>
                                </div>""", unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="alert-info">ℹ️ No objects detected in this image</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ── ANALYTICS DASHBOARD ────────────────────────────────────────────────────────
st.markdown("<div style='margin-top:2rem'></div>", unsafe_allow_html=True)
st.markdown('<div class="section-heading"><span class="section-dot"></span>Model Analytics Dashboard</div>',
            unsafe_allow_html=True)

dash_c1, dash_c2, dash_c3 = st.columns([2, 1.5, 1.5], gap="medium")

with dash_c1:
    st.plotly_chart(model_comparison_chart(cnn_acc, tr_acc),
                    use_container_width=True, config={'displayModeBar': False})

with dash_c2:
    st.plotly_chart(class_distribution_chart(),
                    use_container_width=True, config={'displayModeBar': False})

with dash_c3:
    st.plotly_chart(yolo_metrics_chart(),
                    use_container_width=True, config={'displayModeBar': False})

# ── RADAR ──────────────────────────────────────────────────────────────────────
rad_c1, rad_c2 = st.columns([1.5, 1], gap="medium")

with rad_c1:
    st.plotly_chart(metrics_radar_chart(),
                    use_container_width=True, config={'displayModeBar': False})

with rad_c2:
    st.markdown("""
    <div class="panel" style="height:320px;">
      <p style="color:#8b949e;font-size:0.8rem;font-weight:500;margin-bottom:1rem;">MODEL SUMMARY</p>
      <table style="width:100%;border-collapse:collapse;font-size:0.8rem;">
        <tr style="border-bottom:1px solid #21262d;">
          <th style="color:#6e7681;font-weight:500;text-align:left;padding:0.4rem 0;">Model</th>
          <th style="color:#6e7681;font-weight:500;text-align:right;padding:0.4rem 0;">Acc</th>
          <th style="color:#6e7681;font-weight:500;text-align:right;padding:0.4rem 0;">F1</th>
        </tr>
        <tr style="border-bottom:1px solid #21262d;">
          <td style="color:#e6edf3;padding:0.5rem 0;">EfficientNetB0</td>
          <td style="color:#4f98a3;font-weight:600;text-align:right;">99.07%</td>
          <td style="color:#4f98a3;font-weight:600;text-align:right;">0.98</td>
        </tr>
        <tr style="border-bottom:1px solid #21262d;">
          <td style="color:#e6edf3;padding:0.5rem 0;">Custom CNN</td>
          <td style="color:#d29922;font-weight:600;text-align:right;">86.05%</td>
          <td style="color:#d29922;font-weight:600;text-align:right;">0.86</td>
        </tr>
        <tr>
          <td style="color:#e6edf3;padding:0.5rem 0;">YOLOv8n</td>
          <td style="color:#f85149;font-weight:600;text-align:right;">mAP 22%</td>
          <td style="color:#8b949e;font-weight:600;text-align:right;">CPU</td>
        </tr>
      </table>
      <div style="margin-top:1.2rem;padding:0.75rem;background:rgba(79,152,163,0.08);border:1px solid rgba(79,152,163,0.2);border-radius:8px;">
        <p style="color:#4f98a3;font-size:0.75rem;font-weight:600;margin-bottom:0.3rem;">🏆 Best Architecture</p>
        <p style="color:#8b949e;font-size:0.73rem;line-height:1.5;">EfficientNetB0 with transfer learning — 99.07% test accuracy on 215 test images. Fine-tuned on aerial imagery.</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:3rem;padding-top:1.5rem;border-top:1px solid #21262d;
     display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.5rem;">
  <div style="color:#6e7681;font-size:0.75rem;">
    🛸 <b style="color:#4f98a3;">AeroScan</b> — Aerial Object Detection System
  </div>
  <div style="color:#6e7681;font-size:0.75rem;display:flex;gap:1rem;">
    <span>TensorFlow 2.x</span>
    <span>·</span>
    <span>Ultralytics YOLOv8</span>
    <span>·</span>
    <span>Plotly</span>
    <span>·</span>
    <span>Streamlit</span>
  </div>
</div>
""", unsafe_allow_html=True)