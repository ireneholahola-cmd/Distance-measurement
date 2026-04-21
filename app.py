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
        source = video_file.name
    else:
        source = "lanechange.mp4"
else:
    source = st.sidebar.text_input("摄像头编号", value="0")

if 'detection_done' not in st.session_state:
    st.session_state.detection_done = False

start_btn = st.sidebar.button("开始识别")

if start_btn and not st.session_state.detection_done:
    st.session_state.detection_done = True
    st.info("正在识别中...")

    detect_3d.opt.source = source
    detect_3d.opt.view_img = False

    frames_displayed = 0
    max_frames = 50


    def callback(im0, risk_img):
        global frames_displayed
        if frames_displayed >= max_frames:
            return

        if im0 is not None:
            im0_rgb = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
            video_placeholder.image(im0_rgb, channels="RGB", use_container_width=True)

        if risk_img is not None:
            risk_rgb = cv2.cvtColor(risk_img, cv2.COLOR_BGR2RGB)
            risk_placeholder.image(risk_rgb, channels="RGB", use_container_width=True)

        frames_displayed += 1


    try:
        detect_3d.detect(save_img=False, callback=callback)
        st.success("识别完成！")
    except Exception as e:
        st.error(f"识别出错: {e}")
        st.session_state.detection_done = False

if st.session_state.detection_done:
    if st.sidebar.button("重新开始"):
        st.session_state.detection_done = False
        st.rerun()
else:
    st.info("点击左侧「开始识别」按钮启动")