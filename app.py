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

st.set_page_config(page_title="驭安DriveSafe", layout="wide", page_icon="🚗")

st.title("🚗 驭安DriveSafe 主动安全预警系统")

col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("实时视频")
    video_placeholder = st.empty()

with col2:
    st.subheader("风险场视图")
    risk_placeholder = st.empty()

st.sidebar.title("设置")
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

start_btn = st.sidebar.button("开始识别")
stop_btn = st.sidebar.button("停止识别")

if stop_btn:
    st.session_state.stop_detection = True
    st.session_state.detection_done = True
    st.rerun()

if start_btn and not st.session_state.detection_done:
    st.session_state.detection_done = True
    st.session_state.stop_detection = False
    st.info("正在识别中...")

    detect_3d.opt.source = source
    detect_3d.opt.view_img = False

    # 创建进度条
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()


    def callback(im0, risk_img, frame_idx=None, total_frames=None):
        # 检查是否应该停止
        if st.session_state.get('stop_detection', False):
            return False  # 返回False表示停止处理

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

        return True  # 返回True继续处理


    try:
        # 获取视频总帧数（用于进度条）
        if source_option != "摄像头":
            cap = cv2.VideoCapture(source)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        else:
            total_frames = None

        # 修改detect函数调用，传入帧索引
        # 注意：这里需要根据你的detect_3d.py的实际接口调整
        detect_3d.detect(save_img=False, callback=callback)

        if not st.session_state.get('stop_detection', False):
            st.success("识别完成！")
        else:
            st.warning("识别已停止")

    except Exception as e:
        st.error(f"识别出错: {e}")
    finally:
        st.session_state.detection_done = False
        st.session_state.stop_detection = False
        progress_bar.empty()
        status_text.empty()

if st.session_state.detection_done and not st.session_state.get('stop_detection', False):
    if st.sidebar.button("重新开始"):
        st.session_state.detection_done = False
        st.session_state.stop_detection = False
        st.rerun()
else:
    if not start_btn:
        st.info("点击左侧「开始识别」按钮启动")