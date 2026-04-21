import streamlit as st
import cv2
import numpy as np
import sys
import os
import time

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

    /* 登录容器样式 */
    .login-container {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        backdrop-filter: blur(10px);
        max-width: 450px;
        margin: 0 auto;
    }

    /* 主容器样式 */
    .main-container {
        background: rgba(255, 255, 255, 0.92);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
    }

    /* 标题样式 */
    .login-title {
        text-align: center;
        color: #667eea;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }

    .login-subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }

    /* 功能说明卡片 */
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        transition: transform 0.3s;
    }

    .feature-card:hover {
        transform: translateY(-5px);
    }

    /* 侧边栏美化 */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 10px;
        margin: 10px;
    }

    /* 按钮样式 */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        transition: all 0.3s;
    }

    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }

    /* 信息框美化 */
    .stAlert {
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# 初始化登录状态
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# 登录页面
if not st.session_state.logged_in:
    # 创建登录界面容器
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown('<div class="login-container">', unsafe_allow_html=True)

        # 标题
        st.markdown('<div class="login-title">🚗 驭安DriveSafe</div>', unsafe_allow_html=True)
        st.markdown('<div class="login-subtitle">主动安全预警系统</div>', unsafe_allow_html=True)

        # 登录表单
        with st.form("login_form"):
            username = st.text_input("用户名", placeholder="请输入用户名")
            password = st.text_input("密码", type="password", placeholder="请输入密码")
            login_btn = st.form_submit_button("登 录", use_container_width=True)

            if login_btn:
                if username == "123" and password == "123":
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("❌ 用户名或密码错误")

        # 功能说明
        st.markdown("---")
        st.markdown("### 🎯 产品功能")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
            <div class="feature-card">
                🚗 实时车辆检测<br>
                精准识别周围车辆
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="feature-card">
                📊 风险场分析<br>
                智能评估碰撞风险
            </div>
            """, unsafe_allow_html=True)

        with col_b:
            st.markdown("""
            <div class="feature-card">
                🎯 多目标跟踪<br>
                持续追踪动态目标
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="feature-card">
                ⚡ 实时预警<br>
                毫秒级响应速度
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="text-align: center; color: #999; font-size: 0.8rem; margin-top: 1rem;">
            <p>💡 演示账号：123 / 123</p>
            <p>© 2024 驭安DriveSafe 版权所有</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

# 主系统页面（登录后）
else:
    # 主容器
    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    # 顶部退出按钮
    col_top1, col_top2 = st.columns([6, 1])
    with col_top2:
        if st.button("🚪 退出登录"):
            st.session_state.logged_in = False
            st.session_state.detection_done = False
            st.session_state.stop_detection = False
            st.rerun()

    st.title("🚗 驭安DriveSafe 主动安全预警系统")

    # 功能说明栏（紧凑显示）
    with st.expander("📖 系统功能说明", expanded=False):
        col_exp1, col_exp2, col_exp3, col_exp4 = st.columns(4)
        with col_exp1:
            st.info("🎯 **车辆检测**\n精准识别周边车辆")
        with col_exp2:
            st.info("⚠️ **风险预警**\n实时碰撞风险评估")
        with col_exp3:
            st.info("📹 **多目标跟踪**\n持续追踪动态目标")
        with col_exp4:
            st.info("⚡ **毫秒响应**\n极速预警处理")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("实时视频")
        video_placeholder = st.empty()

    with col2:
        st.subheader("风险场视图")
        risk_placeholder = st.empty()

    st.sidebar.title("⚙️ 系统设置")
    source_option = st.sidebar.radio("视频源", ["摄像头", "视频文件"])

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

    if 'detection_done' not in st.session_state:
        st.session_state.detection_done = False
        st.session_state.stop_detection = False

    start_btn = st.sidebar.button("▶️ 开始识别", use_container_width=True)
    stop_btn = st.sidebar.button("⏹️ 停止识别", use_container_width=True)

    if stop_btn:
        st.session_state.stop_detection = True
        st.session_state.detection_done = True
        st.rerun()

    if start_btn and not st.session_state.detection_done:
        st.session_state.detection_done = True
        st.session_state.stop_detection = False
        st.info("🔍 正在识别中...")

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

    st.markdown('</div>', unsafe_allow_html=True)