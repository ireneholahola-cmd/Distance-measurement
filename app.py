import streamlit as st
import cv2
import numpy as np
import sys
import os
import time
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))



# ─────────────────────────────────────────────
#  DummyOpt
# ─────────────────────────────────────────────
class DummyOpt:
    def __init__(self):
        self.weights = 'yolov10s.pt'
        self.source = 'lanechange.mp4'
        self.img_size = 640
        self.conf_thres = 0.01
        self.iou_thres = 0.01
        self.device = ''
        self.view_img = False
        self.save_txt = False
        self.save_conf = False
        self.nosave = True
        self.classes = None
        self.agnostic_nms = False
        self.augment = False
        self.update = False
        self.project = 'runs/detect'
        self.name = 'exp'
        self.exist_ok = False
        self.config_deepsort = 'deep_sort/configs/deep_sort.yaml'
        self.show_img = False
        self.save_img = False
        self.road_model_dir = 'code/models'
        self.road_conf_thres = 0.25
        self.max_frames = None
        self.save_jsonl = False
        self.structured_dir = 'structured'

import detect_3d_with_surface as detect_3d
detect_3d.opt = DummyOpt()


# ─────────────────────────────────────────────
#  Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="驭安 DriveSafe",
    layout="wide",
    page_icon="🚗",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  Global CSS — Tesla Dark Tech Theme
# ─────────────────────────────────────────────
GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800;900&family=Rajdhani:wght@300;400;500;600;700&family=Share+Tech+Mono&display=swap');

/* ── Root palette ── */
:root {
    --bg-primary:    #0a0c10;
    --bg-secondary:  #0d1117;
    --bg-card:       #111620;
    --bg-card2:      #141b26;
    --accent-blue:   #00bfff;
    --accent-red:    #e8303a;
    --accent-green:  #00e5a0;
    --accent-amber:  #ffb300;
    --border:        rgba(0,191,255,0.18);
    --border-hot:    rgba(0,191,255,0.55);
    --text-primary:  #e8eaf0;
    --text-muted:    #6b7a99;
    --glow-blue:     0 0 18px rgba(0,191,255,0.45);
    --glow-red:      0 0 18px rgba(232,48,58,0.55);
    --glow-green:    0 0 18px rgba(0,229,160,0.4);
}

/* ── App background ── */
.stApp {
    background: var(--bg-primary) !important;
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(0,191,255,0.08) 0%, transparent 70%),
        linear-gradient(180deg, #0a0c10 0%, #080b0f 100%) !important;
}

/* ── Animated grid overlay ── */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(0,191,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,191,255,0.03) 1px, transparent 1px);
    background-size: 60px 60px;
    pointer-events: none;
    z-index: 0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0c1018 0%, #090d14 100%) !important;
    border-right: 1px solid var(--border) !important;
    box-shadow: 4px 0 30px rgba(0,0,0,0.6) !important;
}
[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}

/* ── All text ── */
html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif !important;
    color: var(--text-primary) !important;
}

/* ── Main headings ── */
h1, h2, h3 {
    font-family: 'Orbitron', monospace !important;
    letter-spacing: 0.06em !important;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
    box-shadow: 0 0 20px rgba(0,191,255,0.06) !important;
    transition: box-shadow .3s, border-color .3s !important;
}
[data-testid="metric-container"]:hover {
    border-color: var(--border-hot) !important;
    box-shadow: var(--glow-blue) !important;
}
[data-testid="metric-container"] label {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.85rem !important;
    color: #00e5ff !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Orbitron', monospace !important;
    font-size: 1.6rem !important;
    font-weight: 800 !important;
    color: var(--text-primary) !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.75rem !important;
}

/* ── Buttons ── */
.stButton > button {
    background: transparent !important;
    color: var(--accent-blue) !important;
    border: 1px solid var(--accent-blue) !important;
    border-radius: 6px !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 10px 22px !important;
    transition: all .3s ease !important;
}
.stButton > button:hover {
    background: var(--accent-blue) !important;
    color: #000 !important;
    box-shadow: var(--glow-blue) !important;
    transform: translateY(-1px) !important;
}

/* ── Primary action button (start) ── */
.btn-start > button {
    background: linear-gradient(135deg, #00bfff22, #00bfff11) !important;
    border: 1px solid var(--accent-blue) !important;
    color: var(--accent-blue) !important;
    box-shadow: inset 0 0 20px rgba(0,191,255,0.05) !important;
}

/* ── Stop button ── */
.btn-stop > button {
    border-color: var(--accent-red) !important;
    color: var(--accent-red) !important;
}
.btn-stop > button:hover {
    background: var(--accent-red) !important;
    color: #fff !important;
    box-shadow: var(--glow-red) !important;
}

/* ── Video container styling ── */
.video-container {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 12px;
    box-shadow: 0 0 20px rgba(0,191,255,0.08);
    height: 100%;
}

.video-header {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.85rem;
    color: #aabbcc;
    letter-spacing: 0.14em;
    margin-bottom: 10px;
    text-align: center;
    text-transform: uppercase;
}

/* ── Dataframe / table ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}
.dataframe {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.82rem !important;
}

/* ── Progress bar ── */
.stProgress > div > div {
    background: linear-gradient(90deg, var(--accent-blue), #0080ff) !important;
    box-shadow: 0 0 8px var(--accent-blue) !important;
}
.stProgress > div {
    background: rgba(0,191,255,0.08) !important;
    border-radius: 4px !important;
}

/* ── Status badge ── */
.status-badge {
    display: inline-block;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    padding: 3px 12px;
    border-radius: 20px;
    border: 1px solid;
}
.badge-online  { color: var(--accent-green); border-color: var(--accent-green); background: rgba(0,229,160,0.08); }
.badge-warning { color: var(--accent-amber); border-color: var(--accent-amber); background: rgba(255,179,0,0.08); }
.badge-danger  { color: var(--accent-red);   border-color: var(--accent-red);   background: rgba(232,48,58,0.08); }

/* ── Risk level bar ── */
.risk-bar-wrap {
    background: rgba(0,191,255,0.06);
    border-radius: 4px;
    height: 8px;
    overflow: hidden;
    margin-top: 4px;
}
.risk-bar-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, var(--accent-green), var(--accent-amber), var(--accent-red));
}

/* ── Section header ── */
.section-header {
    font-family: 'Orbitron', monospace;
    font-size: 0.95rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #00d4ff;
    border-bottom: 1px solid var(--border);
    padding-bottom: 10px;
    margin-bottom: 18px;
}

/* ── Sidebar logo ── */
.sidebar-logo {
    font-family: 'Orbitron', monospace;
    font-size: 1.05rem;
    font-weight: 800;
    color: var(--accent-blue);
    text-align: center;
    letter-spacing: 0.06em;
    padding: 8px 0 16px 0;
    text-shadow: var(--glow-blue);
}
.sidebar-version {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    color: var(--text-muted);
    text-align: center;
    letter-spacing: 0.2em;
    margin-top: -12px;
    margin-bottom: 20px;
}

/* ── Page title area ── */
.page-title {
    font-family: 'Orbitron', monospace;
    font-size: 1.45rem;
    font-weight: 900;
    letter-spacing: 0.06em;
    color: var(--text-primary);
    margin: 0;
}
.page-title span { color: var(--accent-blue); }
.page-subtitle {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.9rem;
    letter-spacing: 0.22em;
    color: #99aabb;
    text-transform: uppercase;
    margin-top: 4px;
}

/* ── Login form ── */
.login-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 48px 56px;
    box-shadow: 0 0 60px rgba(0,191,255,0.08), 0 0 120px rgba(0,0,0,0.6);
    max-width: 460px;
    width: 100%;
    text-align: center;
}
.login-logo {
    font-family: 'Orbitron', monospace;
    font-size: 1.8rem;
    font-weight: 900;
    letter-spacing: 0.08em;
    color: var(--accent-blue);
    text-shadow: var(--glow-blue);
    margin-bottom: 4px;
}
.login-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent-blue), transparent);
    margin: 24px 0;
    opacity: 0.4;
}

/* ── Logout btn small ── */
.logout-btn > button {
    background: transparent !important;
    border: 1px solid var(--accent-red) !important;
    color: var(--accent-red) !important;
    font-size: 0.65rem !important;
    padding: 6px 14px !important;
}
.logout-btn > button:hover {
    background: var(--accent-red) !important;
    color: #fff !important;
}
/* ── FileUploader 全部强制白色 ── */
[data-testid="stFileUploader"] {
    color: #ffffff !important;
}

[data-testid="stFileUploader"] * {
    color: #ffffff !important;
}

/* 文件名 */
[data-testid="stFileUploader"] span {
    color: #ffffff !important;
}

/* 文件大小 */
[data-testid="stFileUploader"] small {
    color: #ffffff !important;
}

/* 防止被 markdown 或 label 覆盖 */
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] div,
[data-testid="stFileUploader"] p {
    color: #ffffff !important;
}
/* ── 文件名改成蓝色（只改这一行） ── */
[data-testid="stFileUploader"] span {
    color: #00e5ff !important;   /* 亮科技蓝 */
    font-weight: 600 !important;
    text-shadow: 0 0 6px rgba(0,191,255,0.6); /* 可选：发光 */
}
</style>
"""

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Session State Init
# ─────────────────────────────────────────────
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'detection_done' not in st.session_state:
    st.session_state.detection_done = False
if 'stop_detection' not in st.session_state:
    st.session_state.stop_detection = False
if 'login_error' not in st.session_state:
    st.session_state.login_error = False
if 'alert_count' not in st.session_state:
    st.session_state.alert_count = 0
if 'frames_processed' not in st.session_state:
    st.session_state.frames_processed = 0


# ─────────────────────────────────────────────
#  LOGIN PAGE
# ─────────────────────────────────────────────
def render_login():
    # Center using columns
    _, center, _ = st.columns([1, 1.4, 1])
    with center:
        st.markdown("""
        <div style="height:60px"></div>
        <div class="login-card">
            <div class="login-logo">&#9651; DriveSafe</div>
            <div style="font-family:'Orbitron',monospace;font-size:0.78rem;
                        letter-spacing:0.18em;color:#6b7a99;margin-bottom:2px;">
                驭安DriveSafe主动安全预警系统
            </div>
            <div class="login-divider"></div>
            <div style="font-family:'Share Tech Mono',monospace;font-size:0.68rem;
                        letter-spacing:0.2em;color:#00bfff;text-align:left;
                        margin-bottom:6px;">用户登录</div>
        </div>
        """, unsafe_allow_html=True)

        username = st.text_input("账号", placeholder="输入账号", key="login_user",
                                 label_visibility="collapsed")

        password = st.text_input("密码", placeholder="输入密码", type="password",
                                 key="login_pass", label_visibility="collapsed")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        if st.button("登录", use_container_width=True):
            if username == "123" and password == "123":
                st.session_state.logged_in = True
                st.session_state.login_error = False
                st.rerun()
            else:
                st.session_state.login_error = True
                st.rerun()

        if st.session_state.login_error:
            st.markdown("""
            <div style="margin-top:12px;font-family:'Share Tech Mono',monospace;
                        font-size:0.75rem;color:#e8303a;letter-spacing:0.1em;
                        text-align:center;">
                访问被拒绝 — 凭证无效
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="margin-top:28px;font-family:'Share Tech Mono',monospace;
                    font-size:0.62rem;color:#3a4560;letter-spacing:0.16em;
                    text-align:center;border-top:1px solid rgba(0,191,255,0.1);
                    padding-top:16px;">
            DRIVESAFE v2.1.0 &nbsp;|&nbsp; YOLOV10 + DEEPSORT &nbsp;|&nbsp; 3D风险引擎
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-logo">&#9651; DRIVESAFE</div>
        <div class="sidebar-version">主动安全预警系统 v2.1</div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # System status
        st.markdown("""
        <div class="section-header">系统状态</div>
        <div style="display:flex;justify-content:space-between;align-items:center;
                    padding:8px 4px;font-family:'Share Tech Mono',monospace;
                    font-size:0.78rem;color:#6b7a99;">
            <span>AI引擎</span>
            <span class="status-badge badge-online">在线</span>
        </div>
        <div style="display:flex;justify-content:space-between;align-items:center;
                    padding:8px 4px;font-family:'Share Tech Mono',monospace;
                    font-size:0.78rem;color:#6b7a99;">
            <span>GPU加速</span>
            <span class="status-badge badge-warning">待命</span>
        </div>
        <div style="display:flex;justify-content:space-between;align-items:center;
                    padding:8px 4px;font-family:'Share Tech Mono',monospace;
                    font-size:0.78rem;color:#6b7a99;">
            <span>深度排序</span>
            <span class="status-badge badge-online">就绪</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="section-header">视频源</div>', unsafe_allow_html=True)
        source_option = st.radio("", ["摄像头", "视频文件"], label_visibility="collapsed")

        source = "lanechange.mp4"
        if source_option == "视频文件":
            video_file = st.file_uploader("上传视频", type=["mp4", "avi", "mov"])
            if video_file:
                with open(video_file.name, "wb") as f:
                    f.write(video_file.getbuffer())
                source = video_file.name
        else:
            cam_id = st.text_input("摄像头编号", value="0")
            source = cam_id

        st.markdown("---")
        st.markdown('<div class="section-header">检测参数</div>', unsafe_allow_html=True)

        with st.expander("高级参数"):
            conf_val = st.slider("置信度阈值", 0.01, 1.0, 0.01, 0.01)
            iou_val = st.slider("IOU 阈值", 0.01, 1.0, 0.01, 0.01)
            detect_3d.opt.conf_thres = conf_val
            detect_3d.opt.iou_thres = iou_val

        st.markdown("<br>", unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<div class="btn-start">', unsafe_allow_html=True)
            start_btn = st.button("开始", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col_b:
            st.markdown('<div class="btn-stop">', unsafe_allow_html=True)
            stop_btn = st.button("停止", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")

        # Logout
        st.markdown('<div class="logout-btn">', unsafe_allow_html=True)
        if st.button("退出登录", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.detection_done = False
            st.session_state.stop_detection = False
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""
        <div style="margin-top:24px;font-family:'Share Tech Mono',monospace;
                    font-size:0.58rem;color:#2a3448;letter-spacing:0.14em;
                    text-align:center;">
            (C) 2025 驭安智能科技
        </div>
        """, unsafe_allow_html=True)

    return source_option, source, start_btn, stop_btn


# ─────────────────────────────────────────────
#  METRIC CARDS ROW
# ─────────────────────────────────────────────
def render_metrics(placeholder=None):
    from data_store import data_store

    stats = data_store.get_stats()

    track_count = str(stats['track_count'])
    alert_count = str(stats['total_alerts'])

    html = """
    <div style="display: flex; gap: 16px;">
        <div style="flex: 1; background: var(--bg-card); border: 1px solid var(--border); 
                    border-radius: 12px; padding: 20px 24px; text-align: center;
                    box-shadow: 0 0 20px rgba(0,191,255,0.06);">
            <div style="font-family: 'Share Tech Mono', monospace; font-size: 0.9rem; 
                        color: #00e5ff; letter-spacing: 0.1em; text-transform: uppercase;
                        margin-bottom: 8px;">目标追踪数</div>
            <div style="font-family: 'Orbitron', monospace; font-size: 2.2rem; 
                        font-weight: 800; color: #e8eaf0;">""" + track_count + """</div>
        </div>
        <div style="flex: 1; background: var(--bg-card); border: 1px solid var(--border); 
                    border-radius: 12px; padding: 20px 24px; text-align: center;
                    box-shadow: 0 0 20px rgba(0,191,255,0.06);">
            <div style="font-family: 'Share Tech Mono', monospace; font-size: 0.9rem; 
                        color: #00e5ff; letter-spacing: 0.1em; text-transform: uppercase;
                        margin-bottom: 8px;">告警次数</div>
            <div style="font-family: 'Orbitron', monospace; font-size: 2.2rem; 
                        font-weight: 800; color: #e8eaf0;">""" + alert_count + """</div>
        </div>
    </div>
    """

    if placeholder is None:
        st.markdown(html, unsafe_allow_html=True)
    else:
        with placeholder:
            st.markdown(html, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# 风险概览组件
# ═══════════════════════════════════════════════════════════════
def render_risk_overview(placeholder=None):
    from data_store import data_store

    stats = data_store.get_stats()
    current_frame = data_store.get_current_frame()

    if current_frame:
        risk_index = current_frame.max_risk_score
        avg_risk = current_frame.avg_risk_score
    else:
        risk_index = 0.0
        avg_risk = 0.0

    risk_progress = min(max(0, (risk_index - 400) / 120 * 100), 100)

    if risk_index >= 510:
        risk_text = "高风险"
        risk_color = "#e8303a"
    elif risk_index >= 500:
        risk_text = "中风险"
        risk_color = "#ffb300"
    else:
        risk_text = "安全"
        risk_color = "#00e5a0"

    risk_html = f"""
    <div style="background: var(--bg-card); border: 1px solid var(--border); 
                border-radius: 12px; padding: 20px; text-align: center; height: 100%;">
        <div style="font-family: 'Orbitron', monospace; font-size: 2.8rem; 
                    font-weight: 900; color: {risk_color};">
            {risk_index:.3f}
        </div>
        <div style="font-family: 'Share Tech Mono', monospace; font-size: 0.95rem; 
                    color: #aaddff; margin-top: 6px;">
            风险指数
        </div>
        <div style="margin-top: 16px; padding: 0 40px;">
            <div class="risk-bar-wrap">
                <div class="risk-bar-fill" style="width: {risk_progress}%;"></div>
            </div>
        </div>
        <div style="margin-top: 12px; font-family: 'Orbitron', monospace; 
                    font-size: 1.2rem; color: {risk_color};">
            {risk_text}
        </div>
        <div style="margin-top: 16px; font-family: 'Share Tech Mono', monospace; 
                    font-size: 0.88rem; color: #aabbcc;">
            跟踪目标数: {stats['track_count']}<br>
            平均风险: {avg_risk:.4f}
        </div>
    </div>
    """

    if placeholder is None:
        st.markdown(risk_html, unsafe_allow_html=True)
    else:
        with placeholder:
            st.markdown(risk_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MAIN DASHBOARD
# ─────────────────────────────────────────────
def render_dashboard():
    source_option, source, start_btn, stop_btn = render_sidebar()

    # ── Page header ──
    h_col1, h_col2 = st.columns([3, 1])
    with h_col1:
        st.markdown("""
        <div class="page-title">&#9651; DRIVE<span>SAFE</span></div>
        <div class="page-subtitle">主动安全预警系统</div>
        """, unsafe_allow_html=True)
    with h_col2:
        now = time.strftime("%Y-%m-%d  %H:%M:%S")
        st.markdown(f"""
        <div style="text-align:right;font-family:'Share Tech Mono',monospace;
                    font-size:0.72rem;color:#6b7a99;letter-spacing:0.1em;
                    padding-top:6px;">
            {now}<br>
            <span style="color:#00e5a0;">&#9679;</span> 系统运行中
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════
    # 1. 顶部信息区：左侧检测统计 + 右侧风险概览
    # ═══════════════════════════════════════════════════════════════
    top_left_col, top_right_col = st.columns([1, 1])

    with top_left_col:
        st.markdown('<div class="section-header">检测数据统计</div>', unsafe_allow_html=True)
        metrics_placeholder = st.empty()
        with metrics_placeholder:
            render_metrics()

    with top_right_col:
        st.markdown('<div class="section-header">风险概览</div>', unsafe_allow_html=True)
        risk_overview_placeholder = st.empty()
        with risk_overview_placeholder:
            render_risk_overview()

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════
    # 2. 中部主显示区：左侧实时画面 + 右侧热力图
    # ═══════════════════════════════════════════════════════════════
    st.markdown('<div class="section-header">实时画面</div>', unsafe_allow_html=True)

    vid_col, risk_col = st.columns([3, 1])

    with vid_col:
        st.markdown("""
        <div class="video-container">
            <div class="video-header">📹 摄像头画面</div>
        </div>
        """, unsafe_allow_html=True)
        video_placeholder = st.empty()
        video_placeholder.markdown(
            '<div style="background:var(--bg-card);border:1px solid var(--border);border-radius:12px;'
            'padding:40px;text-align:center;">'
            '<span style="font-family:Share Tech Mono,monospace;color:#6b7a99;">等待视频源...</span>'
            '</div>',
            unsafe_allow_html=True
        )

    with risk_col:
        st.markdown("""
        <div class="video-container">
            <div class="video-header">🔥 风险热力图</div>
        </div>
        """, unsafe_allow_html=True)
        risk_placeholder = st.empty()
        risk_placeholder.markdown(
            '<div style="background:var(--bg-card);border:1px solid var(--border);border-radius:12px;'
            'padding:40px;text-align:center;">'
            '<span style="font-family:Share Tech Mono,monospace;color:#99aabb;">等待风险数据...</span>'
            '</div>',
            unsafe_allow_html=True
        )

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════
    # 3. 底部详情区：当前检测目标实时数据
    # ═══════════════════════════════════════════════════════════════
    st.markdown('<div class="section-header">当前检测目标</div>', unsafe_allow_html=True)
    realtime_placeholder = st.empty()
    with realtime_placeholder:
        render_realtime_panel()

    # ── Footer ──
    st.markdown("""
    <div style="margin-top:32px;padding-top:16px;border-top:1px solid rgba(0,191,255,0.1);
                font-family:'Share Tech Mono',monospace;font-size:0.7rem;
                color:#667788;letter-spacing:0.16em;text-align:center;">
        DRIVESAFE 主动安全预警系统 &nbsp;|&nbsp; YOLOV10S + DEEPSORT + 3D风险引擎
        &nbsp;|&nbsp; (C) 2026 驭安智能科技有限公司
    </div>
    """, unsafe_allow_html=True)

    # ─── DETECTION LOGIC ───
    if stop_btn:
        st.session_state.stop_detection = True
        st.session_state.detection_done = True
        st.rerun()

    if start_btn and not st.session_state.detection_done:
        st.session_state.detection_done = True
        st.session_state.stop_detection = False
        st.info("[ 系统 ] 检测已启动 — 正在处理帧流...")

        detect_3d.opt.source = source
        detect_3d.opt.view_img = False

        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()

        def callback(im0, risk_img, frame_idx=None, total_frames=None, risk_sources=None):
            import streamlit as st
            from data_store import data_store

            if st.session_state.get('stop_detection', False):
                return False

            if im0 is not None:
                im0_rgb = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
                video_placeholder.image(im0_rgb, channels="RGB", use_container_width=True)
                if frame_idx is not None:
                    st.session_state.frames_processed = frame_idx + 1

            if risk_img is not None:
                risk_rgb = cv2.cvtColor(risk_img, cv2.COLOR_BGR2RGB)
                risk_placeholder.image(risk_rgb, channels="RGB", use_container_width=True)

            if risk_sources is not None:
                alert_triggered = any(src.get('scf', 0) > 510 for src in risk_sources)
                data_store.update_frame(frame_idx if frame_idx is not None else 0, risk_sources, alert_triggered)

                # 更新所有占位符
                with metrics_placeholder:
                    render_metrics()
                with risk_overview_placeholder:
                    render_risk_overview()
                with realtime_placeholder:
                    render_realtime_panel()

            if frame_idx is not None and total_frames is not None:
                progress = (frame_idx + 1) / total_frames
                progress_bar.progress(min(progress, 1.0))
                status_text.markdown(
                    f"<span style='font-family:Share Tech Mono,monospace;"
                    f"font-size:0.72rem;color:#00bfff;'>"
                    f"处理中: {frame_idx + 1} / {total_frames}</span>",
                    unsafe_allow_html=True
                )
            return True

        try:
            if source_option != "摄像头":
                cap = cv2.VideoCapture(source)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
            else:
                total_frames = None

            detect_3d.detect(save_img=False, callback=callback)

            if not st.session_state.get('stop_detection', False):
                st.success("[ 完成 ] 检测完成！")
            else:
                st.warning("[ 停止 ] 检测已停止")

        except Exception as e:
            st.error(f"[ 错误 ] 识别出错: {e}")
        finally:
            st.session_state.detection_done = False
            st.session_state.stop_detection = False
            progress_bar.empty()
            status_text.empty()

    if st.session_state.detection_done and not st.session_state.get('stop_detection', False):
        st.sidebar.markdown("<br>", unsafe_allow_html=True)
        if st.sidebar.button("重新开始"):
            st.session_state.detection_done = False
            st.session_state.stop_detection = False
            st.rerun()
    else:
        if not start_btn:
            st.info("[ 就绪 ] 点击左侧「开始」按钮启动检测")


def render_realtime_panel():
    from data_store import data_store

    st.markdown("---")
    st.markdown('<div class="section-header">当前检测目标</div>', unsafe_allow_html=True)

    current_frame = data_store.get_current_frame()
    current_objects = data_store.get_current_objects()

    if current_objects:
        html = ""
        for obj in current_objects:
            risk_color = "#e8303a" if obj.risk_level == "高风险" else "#ffb300" if obj.risk_level == "中风险" else "#00e5a0"

            html += f"""
            <div style="background: var(--bg-card); border: 1px solid var(--border); 
                        border-radius: 8px; padding: 12px; margin-bottom: 8px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-family: 'Share Tech Mono', monospace; color: #00bfff; font-size: 1.1rem; font-weight: bold;">
                        ID: {obj.track_id} | {obj.class_name}
                    </span>
                    <span style="font-family: 'Orbitron', monospace; font-size: 0.8rem; 
                                color: {risk_color}; border: 1px solid {risk_color}; 
                                padding: 4px 12px; border-radius: 4px;">
                        {obj.risk_level}
                    </span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-top: 12px; 
                            font-family: 'Share Tech Mono', monospace; font-size: 0.9rem; color: #6b7a99;">
                    <span style="flex: 1;">距离: {obj.distance:.2f}m</span>
                    <span style="flex: 1;">速度: {obj.speed:.1f} km/h</span>
                    <span style="flex: 1;">风险类型: {obj.risk_type}</span>
                    <span style="flex: 1;">风险值: {obj.risk_score:.4f}</span>
                </div>
            </div>
            """
        st.markdown(html, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: var(--bg-card); border: 1px solid var(--border); 
                    border-radius: 12px; padding: 40px; text-align: center;">
            <span style="font-family: 'Share Tech Mono', monospace; color: #6b7a99;">
                等待检测数据...
            </span>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  ROUTER
# ─────────────────────────────────────────────
if not st.session_state.logged_in:
    render_login()
else:
    render_dashboard()