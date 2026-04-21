import streamlit as st
import cv2
import numpy as np
import sys
import os
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import detect_3d


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


# 注入 opt
detect_3d.opt = DummyOpt()

# 页面配置
st.set_page_config(page_title="驭安DriveSafe", layout="wide", page_icon="🚗")

# 自定义CSS美化
st.markdown("""
<style>
    /* 背景渐变 */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    /* 主容器背景 */
    .main > div {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 20px;
        margin: 10px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }

    /* 登录框样式 */
    .login-container {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        max-width: 400px;
        margin: 100px auto;
    }

    /* 标题样式 */
    .title-text {
        text-align: center;
        color: #667eea;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 20px;
    }

    /* 功能说明卡片 */
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin: 10px 0;
        transition: transform 0.3s;
    }

    .feature-card:hover {
        transform: translateY(-5px);
    }

    /* 按钮样式 */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s;
    }

    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }

    /* 侧边栏美化 */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        margin: 10px;
        padding: 10px;
    }

    /* 成功/信息提示美化 */
    .stAlert {
        border-radius: 10px;
        border-left: 5px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# 初始化session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'detection_done' not in st.session_state:
    st.session_state.detection_done = False
if 'stop_detection' not in st.session_state:
    st.session_state.stop_detection = False

# 登录页面
if not st.session_state.authenticated:
    # 登录页面布局
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown('<div class="login-container">', unsafe_allow_html=True)

        # 标题
        st.markdown('<div class="title-text">🚗 驭安DriveSafe</div>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; color: #666;">主动安全预警系统</p>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # 登录表单
        with st.form("login_form"):
            username = st.text_input("用户名", placeholder="请输入用户名")
            password = st.text_input("密码", type="password", placeholder="请输入密码")
            submit = st.form_submit_button("登录", use_container_width=True)

            if submit:
                if username == "123" and password == "123":
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("用户名或密码错误")

        # 功能说明（登录页）
        st.markdown("---")
        st.markdown("### ✨ 系统功能")
        st.markdown("""
        - 🚗 **实时目标检测**：精准识别车辆、行人等交通参与者
        - 📊 **3D风险场分析**：可视化展示行车风险等级
        - 🎯 **多目标跟踪**：持续追踪动态目标轨迹
        - ⚡ **毫秒级响应**：实时预警潜在危险
        """)

        st.markdown("</div>", unsafe_allow_html=True)

else:
    # 主页面 - 保持原有设计
    st.title("🚗 驭安DriveSafe 主动安全预警系统")

    # 添加功能说明卡片
    with st.expander("📖 系统功能说明", expanded=False):
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.markdown("""
            <div class="feature-card">
                <h3>🎯 实时检测</h3>
                <p>基于YOLOv10深度学习算法，精准识别车辆、行人、自行车等交通参与者</p>
            </div>
            """, unsafe_allow_html=True)

        with col_b:
            st.markdown("""
            <div class="feature-card">
                <h3>📊 风险场分析</h3>
                <p>创新3D风险场可视化技术，直观展示周边环境风险等级</p>
            </div>
            """, unsafe_allow_html=True)

        with col_c:
            st.markdown("""
            <div class="feature-card">
                <h3>⚡ 主动预警</h3>
                <p>毫秒级响应速度，提前预警潜在碰撞风险，保障行车安全</p>
            </div>
            """, unsafe_allow_html=True)

    # 原有主界面布局
    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("实时视频")
        video_placeholder = st.empty()

    with col2:
        st.subheader("风险场视图")
        risk_placeholder = st.empty()

    # 侧边栏设置
    st.sidebar.title("⚙️ 系统设置")

    # 用户信息
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"👤 当前用户: 管理员")
    if st.sidebar.button("🚪 退出登录", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.detection_done = False
        st.session_state.stop_detection = False
        st.rerun()

    st.sidebar.markdown("---")

    source_option = st.sidebar.radio("📹 视频源", ["摄像头", "视频文件"])

    if source_option == "视频文件":
        video_file = st.sidebar.file_uploader("上传视频", type=["mp4", "avi", "mov"])
        if video_file:
            # 保存上传的文件到临时位置
            with open(video_file.name, "wb") as f:
                f.write(video_file.getbuffer())
            source = video_file.name
        else:
            source = "lanechange.mp4"
    else:
        source = st.sidebar.text_input("摄像头编号", value="0")

    st.sidebar.markdown("---")

    start_btn = st.sidebar.button("▶️ 开始识别", use_container_width=True)
    stop_btn = st.sidebar.button("⏹️ 停止识别", use_container_width=True)

    if stop_btn:
        st.session_state.stop_detection = True
        st.session_state.detection_done = True
        st.rerun()

    if start_btn and not st.session_state.detection_done:
        st.session_state.detection_done = True
        st.session_state.stop_detection = False
        st.info("🔄 正在识别中...")

        detect_3d.opt.source = source
        detect_3d.opt.view_img = False

        # 创建进度条
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()


        def callback(im0, risk_img, frame_idx=None, total_frames=None):
            # 检查是否应该停止
            if st.session_state.get('stop_detection', False):
                return False

            if im0 is not None:
                im0_rgb = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
                video_placeholder.image(im0_rgb, channels="RGB", use_container_width=True)

            if risk_img is not None:
                risk_rgb = cv2.cvtColor(risk_img, cv2.COLOR_BGR2RGB)
                risk_placeholder.image(risk_rgb, channels="RGB", use_container_width=True)

            # 更新进度条
            if frame_idx is not None and total_frames is not None:
                progress = (frame_idx + 1) / total_frames
                progress_bar.progress(min(progress, 1.0))
                status_text.text(f"处理中: {frame_idx + 1}/{total_frames} 帧")

            return True


        try:
            # 获取视频总帧数（用于进度条）
            if source_option != "摄像头":
                cap = cv2.VideoCapture(source)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
            else:
                total_frames = None

            # 修改detect函数调用，传入帧索引
            detect_3d.detect(save_img=False, callback=callback)

            if not st.session_state.get('stop_detection', False):
                st.success("✅ 识别完成！")
            else:
                st.warning("⏸️ 识别已停止")

        except Exception as e:
            st.error(f"❌ 识别出错: {e}")
        finally:
            st.session_state.detection_done = False
            st.session_state.stop_detection = False
            progress_bar.empty()
            status_text.empty()

    if st.session_state.detection_done and not st.session_state.get('stop_detection', False):
        if st.sidebar.button("🔄 重新开始", use_container_width=True):
            st.session_state.detection_done = False
            st.session_state.stop_detection = False
            st.rerun()
    else:
        if not start_btn:
            st.info("💡 提示：点击左侧「开始识别」按钮启动系统")