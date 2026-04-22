import streamlit as st
import cv2
import numpy as np
import sys
import os
import time
import base64

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import detect_3d


class DetectionConfig:
    def __init__(self):
        self.weights = 'yolov10s.pt'
        self.source = 'lanechange.mp4'
        self.img_size = 640
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.device = 'cpu'
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


if 'config' not in st.session_state:
    st.session_state.config = DetectionConfig()

if 'detection_done' not in st.session_state:
    st.session_state.detection_done = False
    st.session_state.stop_detection = False

if 'risk_alerts' not in st.session_state:
    st.session_state.risk_alerts = []

if 'stats' not in st.session_state:
    st.session_state.stats = {
        'total_detections': 0,
        'current_risk': 0.0,
        'high_risk_count': 0,
        'medium_risk_count': 0,
        'low_risk_count': 0
    }

if 'current_risk_img' not in st.session_state:
    st.session_state.current_risk_img = None


def local_css():
    # 读取并编码图片
    def get_base64_image(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    
    global logo_base64
    logo_path = os.path.join("picture", "logo.png")
    bg_path = os.path.join("picture", "background2.jpg")
    
    logo_base64 = get_base64_image(logo_path)
    bg_base64 = get_base64_image(bg_path)
    
    st.markdown(
        '''
        <style>
        body {
            background-image: url('data:image/jpg;base64,%s');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            background-repeat: no-repeat;
        }
        
        .stApp {
            background: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(10px);
        }

        .header-container {
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 2rem;
            position: relative;
        }
        
        .logo {
            width: 80px;
            height: 80px;
            margin-right: 1rem;
            border-radius: 50%%;
            box-shadow: 0 0 20px rgba(0, 153, 204, 0.5);
        }
        
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            color: #0099cc;
            text-align: center;
            padding: 1rem 0;
            text-shadow: 0 0 20px rgba(0, 153, 204, 0.3);
        }

        .sub-header {
            font-size: 1.2rem;
            color: #333333;
            text-align: center;
            margin-bottom: 1.5rem;
        }

        .video-card {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 15px;
            padding: 1rem;
            border: 1px solid rgba(0, 153, 204, 0.2);
            backdrop-filter: blur(5px);
            box-shadow: 0 4px 30px rgba(0, 153, 204, 0.1);
        }

        .risk-card {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 15px;
            padding: 1rem;
            border: 1px solid rgba(0, 153, 204, 0.2);
            backdrop-filter: blur(5px);
            box-shadow: 0 4px 30px rgba(0, 153, 204, 0.1);
        }

        .sidebar .stSidebar {
            background: rgba(245, 247, 250, 0.95);
            border-right: 1px solid rgba(0, 153, 204, 0.2);
        }

        .stButton > button {
            width: 100%%;
            border-radius: 10px;
            height: 3rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }

        .start-btn > button {
            background: linear-gradient(135deg, #0099cc 0%%, #0077aa 100%%);
            color: white;
            border: none;
        }

        .start-btn > button:hover {
            background: linear-gradient(135deg, #00aacc 0%%, #0088bb 100%%);
            box-shadow: 0 0 20px rgba(0, 153, 204, 0.5);
        }

        .stop-btn > button {
            background: linear-gradient(135deg, #ff4757 0%%, #c0392b 100%%);
            color: white;
            border: none;
        }

        .stop-btn > button:hover {
            background: linear-gradient(135deg, #ff6b7a 0%%, #e74c3c 100%%);
            box-shadow: 0 0 20px rgba(255, 71, 87, 0.5);
        }

        .config-section {
            background: rgba(240, 245, 250, 0.8);
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 1rem;
            border-left: 3px solid #0099cc;
        }

        .metric-card {
            background: rgba(240, 245, 250, 0.8);
            border-radius: 10px;
            padding: 1rem;
            text-align: center;
            border: 1px solid rgba(0, 153, 204, 0.2);
            transition: all 0.3s ease;
        }

        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 20px rgba(0, 153, 204, 0.2);
        }

        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #0099cc;
        }

        .metric-label {
            font-size: 0.9rem;
            color: #666666;
            margin-top: 0.5rem;
        }

        .alert-high {
            background: linear-gradient(135deg, rgba(255, 71, 87, 0.1) 0%%, rgba(192, 57, 43, 0.1) 100%%);
            border-left: 4px solid #ff4757;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 0.5rem;
            animation: slideIn 0.3s ease;
        }

        .alert-medium {
            background: linear-gradient(135deg, rgba(255, 193, 7, 0.1) 0%%, rgba(243, 156, 18, 0.1) 100%%);
            border-left: 4px solid #ffc107;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 0.5rem;
            animation: slideIn 0.3s ease;
        }

        .alert-low {
            background: linear-gradient(135deg, rgba(46, 213, 115, 0.1) 0%%, rgba(39, 174, 96, 0.1) 100%%);
            border-left: 4px solid #2ed573;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 0.5rem;
            animation: slideIn 0.3s ease;
        }

        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateX(-20px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }

        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%%;
            margin-right: 8px;
        }

        .status-active {
            background: #2ed573;
            animation: pulse 1.5s infinite;
        }

        .status-inactive {
            background: #636e72;
        }

        @keyframes pulse {
            0%% { box-shadow: 0 0 0 0 rgba(46, 213, 115, 0.7); }
            70%% { box-shadow: 0 0 0 10px rgba(46, 213, 115, 0); }
            100%% { box-shadow: 0 0 0 0 rgba(46, 213, 115, 0); }
        }

        .divider {
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(0, 153, 204, 0.3), transparent);
            margin: 1rem 0;
        }

        .stSelectbox > div > div {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 8px;
            border: 1px solid rgba(0, 153, 204, 0.2);
        }

        .stSlider > div > div > div {
            background: rgba(0, 153, 204, 0.2);
        }

        .image-container {
            display: flex;
            justify-content: center;
            align-items: center;
            background: rgba(240, 245, 250, 0.8);
            border-radius: 10px;
            padding: 0.5rem;
        }

        .image-container img {
            max-height: 300px;
            object-fit: contain;
        }

        .risk-view-container {
            height: 280px;
            display: flex;
            justify-content: center;
            align-items: center;
            background: rgba(240, 245, 250, 0.8);
            border-radius: 10px;
            padding: 0.5rem;
            overflow: hidden;
        }

        .risk-view-container img {
            max-height: 260px;
            width: auto;
            height: auto;
            object-fit: contain;
        }

        section[data-testid="stImage"] {
            max-height: 300px;
            overflow: hidden;
        }

        section[data-testid="stImage"] img {
            max-height: 300px !important;
            width: auto !important;
            height: auto !important;
            object-fit: contain !important;
        }
        
        /* 装饰元素 */
        .decorative-element {
            position: absolute;
            border-radius: 50%%;
            background: radial-gradient(circle, rgba(0, 153, 204, 0.05) 0%%, transparent 70%%);
            z-index: -1;
        }
        
        .decorative-element-1 {
            top: 10%%;
            left: 5%%;
            width: 300px;
            height: 300px;
        }
        
        .decorative-element-2 {
            bottom: 10%%;
            right: 5%%;
            width: 400px;
            height: 400px;
        }
        
        .decorative-element-3 {
            top: 50%%;
            left: 50%%;
            transform: translate(-50%%, -50%%);
            width: 600px;
            height: 600px;
            background: radial-gradient(circle, rgba(0, 153, 204, 0.03) 0%%, transparent 70%%);
        }
        </style>
        ''' % bg_base64, 
        unsafe_allow_html=True
    )

    # 添加装饰元素
    st.markdown(
        """
        <div class="decorative-element decorative-element-1"></div>
        <div class="decorative-element decorative-element-2"></div>
        <div class="decorative-element decorative-element-3"></div>
        """, unsafe_allow_html=True)


st.set_page_config(page_title="驭安DriveSafe", layout="wide", page_icon="🚗")

local_css()

# 显示带logo的标题
st.markdown(f"""
<div class="header-container">
    <img src="data:image/png;base64,{logo_base64}" class="logo">
    <div>
        <h1 class="main-header">驭安DriveSafe 主动安全预警系统</h1>
        <p class="sub-header">基于YOLOv10和深度学习的安全驾驶辅助系统</p>
    </div>
</div>
""", unsafe_allow_html=True)

col_main = st.columns([3, 2])

with col_main[0]:
    # 视频流区域
    st.markdown('<div class="video-card">', unsafe_allow_html=True)
    st.subheader("📹 实时视频流")
    video_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 实时统计区域
    st.markdown('<div class="video-card" style="margin-top: 1rem;">', unsafe_allow_html=True)
    st.subheader("📊 实时统计")
    
    col_stats1, col_stats2, col_stats3 = st.columns(3)
    
    with col_stats1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value" id="total_detections">{st.session_state.stats["total_detections"]}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">检测目标</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_stats2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        risk_level = "高" if st.session_state.stats["current_risk"] > 0.7 else "中" if st.session_state.stats["current_risk"] > 0.4 else "低"
        risk_color = "#ff4757" if risk_level == "高" else "#ffc107" if risk_level == "中" else "#2ed573"
        st.markdown(f'<div class="metric-value" id="risk_level" style="color: {risk_color}">{risk_level}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">风险等级</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_stats3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value" id="high_risk_count">{st.session_state.stats["high_risk_count"]}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">高危目标</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_main[1]:
    # 风险场视图
    st.markdown('<div class="risk-card">', unsafe_allow_html=True)
    st.subheader("⚠️ 风险场视图")
    
    # 创建固定大小的容器
    st.markdown('<div style="width: 100%; height: 200px; display: flex; justify-content: center; align-items: center; background: rgba(0, 0, 0, 0.3); border-radius: 10px; overflow: hidden;">', unsafe_allow_html=True)
    risk_image_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # 实时警报
    st.subheader("🚨 实时警报")
    
    alert_container = st.container()
    
    with alert_container:
        if st.session_state.risk_alerts:
            for alert in st.session_state.risk_alerts[-3:]:
                if alert['level'] == 'high':
                    st.markdown(f"""
                    <div class="alert-high">
                        <strong>🔴 {alert['type']}</strong><br>
                        <span>{alert['message']}</span><br>
                        <small style="color: #a0a0a0;">距离: {alert['distance']:.1f}m | 速度: {alert['speed']:.1f} km/h</small>
                    </div>
                    """, unsafe_allow_html=True)
                elif alert['level'] == 'medium':
                    st.markdown(f"""
                    <div class="alert-medium">
                        <strong>🟡 {alert['type']}</strong><br>
                        <span>{alert['message']}</span><br>
                        <small style="color: #a0a0a0;">距离: {alert['distance']:.1f}m | 速度: {alert['speed']:.1f} km/h</small>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="alert-low">
                        <strong>🟢 {alert['type']}</strong><br>
                        <span>{alert['message']}</span><br>
                        <small style="color: #a0a0a0;">距离: {alert['distance']:.1f}m | 速度: {alert['speed']:.1f} km/h</small>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("暂无警报信息")
    
    st.markdown('</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("---")
    st.title("⚙️ 系统设置")

    st.markdown('<div class="config-section">', unsafe_allow_html=True)
    st.markdown("**📹 视频源选择**")
    source_option = st.radio("输入类型", ["视频文件", "摄像头"], label_visibility="collapsed")

    if source_option == "视频文件":
        video_file = st.file_uploader("上传视频文件", type=["mp4", "avi", "mov"])
        if video_file:
            with open(video_file.name, "wb") as f:
                f.write(video_file.getbuffer())
            st.session_state.config.source = video_file.name
            st.success(f"✅ 已加载: {video_file.name}")
        else:
            st.session_state.config.source = "lanechange.mp4"
    else:
        camera_id = st.text_input("摄像头编号", value="0")
        st.session_state.config.source = camera_id
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="config-section">', unsafe_allow_html=True)
    st.markdown("**🎯 检测参数**")

    conf_thres = st.slider(
        "置信度阈值",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.config.conf_thres,
        step=0.01,
        help="检测目标的最低置信度要求"
    )
    st.session_state.config.conf_thres = conf_thres

    iou_thres = st.slider(
        "IOU阈值",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.config.iou_thres,
        step=0.01,
        help="非极大值抑制的IOU阈值"
    )
    st.session_state.config.iou_thres = iou_thres
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="config-section">', unsafe_allow_html=True)
    st.markdown("**🖥️ 运行设置**")

    device_option = st.selectbox(
        "运行设备",
        options=["cpu", "cuda"],
        index=0 if st.session_state.config.device == "cpu" else 1
    )
    st.session_state.config.device = device_option

    img_size = st.selectbox(
        "图像尺寸",
        options=[320, 416, 640, 1280],
        index=2
    )
    st.session_state.config.img_size = img_size
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="config-section">', unsafe_allow_html=True)
    st.markdown("**🔊 预警设置**")

    enable_sound = st.checkbox("启用声音报警", value=True)
    enable_voice = st.checkbox("启用语音播报", value=False)

    risk_threshold_high = st.slider(
        "高风险阈值",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="超过此阈值视为高风险"
    )

    risk_threshold_medium = st.slider(
        "中风险阈值",
        min_value=0.0,
        max_value=1.0,
        value=0.4,
        step=0.05,
        help="超过此阈值视为中风险"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    col_btn1, col_btn2 = st.columns(2)

    with col_btn1:
        st.markdown('<div class="start-btn">', unsafe_allow_html=True)
        start_btn = st.button("🚀 开始识别", key="start")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_btn2:
        st.markdown('<div class="stop-btn">', unsafe_allow_html=True)
        stop_btn = st.button("⏹️ 停止识别", key="stop")
        st.markdown('</div>', unsafe_allow_html=True)

    progress_bar = st.progress(0)
    status_text = st.empty()

    if stop_btn:
        st.session_state.stop_detection = True
        st.session_state.detection_done = True
        st.rerun()

    if start_btn and not st.session_state.detection_done:
        st.session_state.detection_done = True
        st.session_state.stop_detection = False
        st.session_state.risk_alerts = []
        st.session_state.stats = {
            'total_detections': 0,
            'current_risk': 0.0,
            'high_risk_count': 0,
            'medium_risk_count': 0,
            'low_risk_count': 0
        }

        detect_3d.opt = st.session_state.config

        status_text.info("🚀 正在初始化检测系统...")

        def callback(im0, risk_img, frame_idx=None, total_frames=None, detections=None):
            if st.session_state.get('stop_detection', False):
                return False

            if im0 is not None:
                im0_rgb = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
                video_placeholder.image(im0_rgb, channels="RGB", use_container_width=True)

            if risk_img is not None:
                risk_rgb = cv2.cvtColor(risk_img, cv2.COLOR_BGR2RGB)
                # 调整风险场图像大小，保持纵横比
                risk_image_placeholder.image(risk_rgb, channels="RGB", use_container_width=True, clamp=True)

            if detections is not None:
                st.session_state.stats['total_detections'] = len(detections)

                high_risk = 0
                medium_risk = 0
                low_risk = 0

                for det in detections:
                    if len(det) >= 6:
                        risk_score = det[5] if det[5] > 0 else 0.5
                        distance = det[4] if len(det) > 4 else 10.0
                        speed = det[3] if len(det) > 3 else 0.0

                        obj_type = det[6] if len(det) > 6 else "未知目标"

                        if risk_score > risk_threshold_high:
                            high_risk += 1
                            if len(st.session_state.risk_alerts) == 0 or st.session_state.risk_alerts[-1]['level'] != 'high':
                                st.session_state.risk_alerts.append({
                                    'level': 'high',
                                    'type': obj_type,
                                    'message': '前方碰撞风险高！',
                                    'distance': distance,
                                    'speed': speed
                                })
                        elif risk_score > risk_threshold_medium:
                            medium_risk += 1
                            if len(st.session_state.risk_alerts) == 0 or st.session_state.risk_alerts[-1]['level'] != 'medium':
                                st.session_state.risk_alerts.append({
                                    'level': 'medium',
                                    'type': obj_type,
                                    'message': '注意周围车辆',
                                    'distance': distance,
                                    'speed': speed
                                })
                        else:
                            low_risk += 1

                st.session_state.stats['high_risk_count'] = high_risk
                st.session_state.stats['medium_risk_count'] = medium_risk
                st.session_state.stats['low_risk_count'] = low_risk
                st.session_state.stats['current_risk'] = max(high_risk * 0.8, medium_risk * 0.5)
            else:
                # 重置统计数据
                st.session_state.stats['total_detections'] = 0
                st.session_state.stats['high_risk_count'] = 0
                st.session_state.stats['medium_risk_count'] = 0
                st.session_state.stats['low_risk_count'] = 0
                st.session_state.stats['current_risk'] = 0.0

            if frame_idx is not None and total_frames is not None:
                progress = (frame_idx + 1) / total_frames
                progress_bar.progress(min(progress, 1.0))
                status_text.text(f"处理中: {frame_idx + 1}/{total_frames} 帧")

            return True

        try:
            if source_option != "摄像头":
                cap = cv2.VideoCapture(st.session_state.config.source)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
            else:
                total_frames = None

            status_text.info("🔄 正在加载模型...")

            detect_3d.detect(save_img=False, callback=callback)

            if not st.session_state.get('stop_detection', False):
                status_text.success("✅ 识别完成！")
                st.balloons()
            else:
                status_text.warning("⚠️ 识别已停止")

        except Exception as e:
            status_text.error(f"❌ 识别出错: {e}")
            import traceback
            st.code(traceback.format_exc())
        finally:
            st.session_state.detection_done = False
            st.session_state.stop_detection = False

        st.rerun()

if st.session_state.detection_done and not st.session_state.get('stop_detection', False):
    pass
else:
    if not start_btn:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background: rgba(255, 255, 255, 0.05); border-radius: 15px; margin-top: 2rem;">
            <h3 style="color: #00d4ff;">👈 请在左侧设置参数并点击「开始识别」</h3>
            <p style="color: #a0a0a0;">系统将实时分析视频流并显示检测结果和风险预警</p>
        </div>
        """, unsafe_allow_html=True)