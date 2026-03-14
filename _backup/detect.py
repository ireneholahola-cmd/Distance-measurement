import time
import math
import numpy as np
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from yolov10.utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer)
from yolov10.utils.torch_utils import select_device, load_classifier, time_synchronized
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from counter.draw_counter import draw_up_down_counter
import argparse
import platform
import shutil
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import cv2
import os
from PIL import Image
from pylab import *
from matplotlib.pyplot import ginput, ion, ioff
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, './yolov10')

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


# --------------------------
# 风险场计算辅助函数（论文附录A公式依赖）
# --------------------------
def calc_g_v(v):
    """论文2.1节：车速对虚拟质量的影响函数g(v)（v单位：km/h）"""
    return 0.002368 * (v ** 4) - 0.3224 * (v ** 3) + 12.57 * (v ** 2) - 1492


def calc_phi_mu(mu):
    """论文2.2节：路面附着系数影响函数φ(μ)"""
    return -87290 * (mu ** 3) + 151900 * (mu ** 2) - 84630 * mu + 21780


def calc_M_b(vehicle_type, m, v):
    """论文公式(13)：计算风险源（车辆）的虚拟质量M_b"""
    psi_T = 1.000 if vehicle_type == 'car' else 1.443  # 论文表5：汽车=1.000，载货车=1.443
    g_v = calc_g_v(v)
    return psi_T * m * g_v


def calc_R_b(psi_delta, psi_rho_tau, mu):
    """论文公式(14)：计算道路条件影响因子R_b"""
    phi_mu = calc_phi_mu(mu)
    return psi_delta * psi_rho_tau * phi_mu


def SPE_v_bi(x_phi, y_phi, x_obs, y_obs, v_b, phi_v, K, k1, k3, R_b, M_b, R_i, M_i, DR_i):
    """论文附录A公式A10：运动物体安全势能（风险值）计算"""
    r_bi = np.sqrt((x_phi - x_obs) ** 2 + (y_phi - y_obs) ** 2)
    r_bi = np.maximum(r_bi, 1e-6)

    vec_r = np.stack([x_phi - x_obs, y_phi - y_obs], axis=-1)
    vec_v = np.array([np.cos(phi_v), np.sin(phi_v)])
    cos_theta_b = np.dot(vec_r, vec_v) / r_bi
    cos_theta_b = np.clip(cos_theta_b, -1.0, 1.0)

    numerator = K * R_b * M_b * R_i * M_i * (1 + DR_i) * k3
    denominator = (k1 - 1) * (r_bi ** (k1 - 1))
    term1 = np.maximum(k3 - np.abs(v_b) * cos_theta_b, 1e-6)
    term2 = np.maximum(k3 - np.abs(v_b), 1e-6)
    power_term = (term1 ** (1 - k1) / term2) ** (1 / k1)

    return (numerator / denominator) * power_term


def SPE_r_ai(x_phi, y_phi, x_obs, y_obs, K, k1, R_b, M_b, R_i, M_i, DR_i):
    """论文附录A公式A8：静止物体安全势能（风险值）计算"""
    r_bi = np.sqrt((x_phi - x_obs) ** 2 + (y_phi - y_obs) ** 2)
    r_bi = np.maximum(r_bi, 1e-6)
    return (K * R_b * M_b * R_i * M_i * (1 + DR_i)) / ((k1 - 1) * (r_bi ** (k1 - 1)))


# --------------------------
# 风险场热力图生成函数（优化版）
# --------------------------
def generate_risk_map(im0, risk_sources, prev_centers, K=0.1, k1=1.5, k3=160):
    """生成高分层、高对比度风险场热力图，高风险用红色表示"""
    # 1. 论文标定参数初始化
    m = 1500  # 车辆实际质量（kg）
    mu = 0.9  # 干燥路面附着系数
    psi_delta = 1.457  # 能见度>200m
    psi_rho_tau = 1.000  # 直路+平路
    DR_i = 0.5  # 普通驾驶员风险因子
    R_common = calc_R_b(psi_delta, psi_rho_tau, mu)
    M_i = calc_M_b('car', m, v=0)  # 主车虚拟质量（静止）

    # 2. 生成计算网格（300×300平衡细节与速度）
    h, w = im0.shape[:2]
    grid_size = 250
    x_phi = np.linspace(0, w, grid_size)
    y_phi = np.linspace(0, h, grid_size)
    x_phi, y_phi = np.meshgrid(x_phi, y_phi)

    # 3. 计算总风险场（多风险源叠加）
    U_total = np.zeros_like(x_phi)
    for src in risk_sources:
        x_obs, y_obs, v_b, car_type = src['x'], src['y'], src['speed'], src['type']
        M_b = calc_M_b(car_type, m, v_b)

        # 运动物体：计算真实速度方向（基于上一帧位置）
        if v_b > 1e-3:
            # 从prev_centers获取上一帧位置（无则用当前位置）
            prev_x, prev_y = prev_centers.get(src['id'], (x_obs, y_obs))
            delta_x = x_obs - prev_x
            delta_y = y_obs - prev_y
            # 计算方向角（避免除以0）
            phi_v = np.arctan2(delta_y, delta_x) if (delta_x != 0 or delta_y != 0) else 0.0
            single_risk = SPE_v_bi(x_phi, y_phi, x_obs, y_obs, v_b, phi_v, K, k1, k3, R_common, M_b, R_common, M_i,
                                   DR_i)
        # 静止物体：无方向影响
        else:
            single_risk = SPE_r_ai(x_phi, y_phi, x_obs, y_obs, K, k1, R_common, M_b, R_common, M_i, DR_i)

        U_total += single_risk

    # 4. 分段归一化（增强中高风险对比度）
    threshold = np.percentile(U_total, 98)  # 截断2%极端值
    U_total_clipped = np.clip(U_total, 0, threshold)
    # 低风险（0~30%）压缩，中高风险（30%~100%）扩展
    p30 = np.percentile(U_total_clipped, 30)
    U_norm = np.zeros_like(U_total_clipped)
    mask_low = U_total_clipped <= p30
    mask_high = ~mask_low
    U_norm[mask_low] = (U_total_clipped[mask_low] / p30) * 0.3  # 低风险映射到0~0.3
    U_norm[mask_high] = ((U_total_clipped[mask_high] - p30) / (threshold - p30)) * 0.7 + 0.3  # 中高风险映射到0.3~1.0

    # 5. 绘制热力图（200层分层+高对比度配色）
    plt.switch_backend('Agg')  # 非交互式模式
    # 关键修改：使用tight_layout消除边框，确保尺寸准确
    fig, ax = plt.subplots(1, 1, figsize=(w / 100, h / 100), dpi=100, frameon=False)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)  # 翻转y轴对齐OpenCV
    ax.axis('off')
    plt.tight_layout(pad=0)  # 消除边距

    # 200层颜色分层（细腻过渡）
    levels = 400
    # 高对比度配色（深蓝→浅蓝→绿→黄→橙→红）
    colors = [
        (0.1, 0.1, 0.8),  # 深蓝（极低风险）
        (0.3, 0.3, 1.0),  # 浅蓝（低风险）
        (0.2, 0.7, 0.2),  # 绿色（中低风险）
        (0.9, 0.9, 0),  # 黄色（中风险）
        (1.0, 0.5, 0),  # 橙色（中高风险）
        (1.0, 0, 0)  # 红色（高风险）
    ]
    cmap_custom = LinearSegmentedColormap.from_list('high_contrast', colors, N=levels)

    # 绘制热力图（抗锯齿）
    contourf = ax.contourf(
        x_phi, y_phi, U_norm,
        levels=levels,
        cmap=cmap_custom,
        alpha=0.9,
        antialiased=True
    )

    # 叠加白色等高线（强化分层边界）
    contour = ax.contour(
        x_phi, y_phi, U_norm,
        levels=np.linspace(0, 1, 21),  # 20条轮廓线
        colors='white',
        linewidths=0.5,
        alpha=0.6
    )

    # 标记风险源（红色圆点，顶层显示）
    for src in risk_sources:
        marker_size = max(30, min(120, w // 25))
        ax.scatter(
            src['x'], src['y'],
            c='red',
            s=marker_size,
            edgecolors='white',
            linewidths=2,
            zorder=10
        )

    # 6. 转换为OpenCV格式（BGR）- 关键修复
    fig.canvas.draw()
    # 获取实际绘制区域的尺寸
    buf = fig.canvas.tostring_rgb()
    buf_size = len(buf)
    img_size = buf_size // 3
    img_w = int(np.sqrt(img_size * w / h))
    img_h = int(np.sqrt(img_size * h / w))

    # 确保尺寸匹配，不匹配则调整
    if img_w * img_h * 3 != buf_size:
        img_w = w
        img_h = h
        # 直接调整图像尺寸
        risk_rgb = np.frombuffer(buf, dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (3,))
        risk_rgb = cv2.resize(risk_rgb, (w, h), interpolation=cv2.INTER_AREA)
    else:
        risk_rgb = np.frombuffer(buf, dtype=np.uint8).reshape(img_h, img_w, 3)

    risk_bgr = cv2.cvtColor(risk_rgb, cv2.COLOR_RGB2BGR)
    plt.close(fig)  # 释放内存

    return risk_bgr

# --------------------------
# 辅助函数：边界框相对坐标计算
# --------------------------
def bbox_rel(image_width, image_height, *xyxy):
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


# --------------------------
# 速度计算函数
# --------------------------
def Estimated_speed(locations, fps, width):
    present_IDs = []
    prev_IDs = []
    work_IDs = []
    work_IDs_index = []
    work_IDs_prev_index = []
    work_locations = []
    work_prev_locations = []
    speed = []
    for i in range(len(locations[1])):
        present_IDs.append(locations[1][i][2])
    for i in range(len(locations[0])):
        prev_IDs.append(locations[0][i][2])
    for m, n in enumerate(present_IDs):
        if n in prev_IDs:
            work_IDs.append(n)
            work_IDs_index.append(m)
    for x in work_IDs_index:
        work_locations.append(locations[1][x])
    for y, z in enumerate(prev_IDs):
        if z in work_IDs:
            work_IDs_prev_index.append(y)
    for x in work_IDs_prev_index:
        work_prev_locations.append(locations[0][x])
    for i in range(len(work_IDs)):
        speed.append(
            math.sqrt((work_locations[i][0] - work_prev_locations[i][0]) ** 2 +
                      (work_locations[i][1] - work_prev_locations[i][1]) ** 2) *
            width[work_locations[i][3]] / (work_locations[i][4]) * fps / 5 * 3.6 * 2)
    for i in range(len(speed)):
        speed[i] = [round(speed[i], 1), work_locations[i][2]]
    return speed


# --------------------------
# 主检测函数
# --------------------------
def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # 结果保存目录
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # 获取视频帧率
    fps = 30
    if not webcam and source.endswith(('.mp4', '.avi', '.mov')):
        capture = cv2.VideoCapture(source)
        fps = capture.get(cv2.CAP_PROP_FPS)
        capture.release()

    # DeepSort初始化
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # 车型参数
    width = [0, 0.2, 1.85, 0.5, 0, 2.3, 0, 2.5]  # 各类别真实宽度
    time_person = 3  # 刹车反应时间
    locations = []
    speed = []
    current_frame_speeds = {}  # 当前帧车辆速度（ID→speed）
    prev_centers = {}  # 【关键】存储上一帧车辆位置：{ID: (x, y)}

    # 模型初始化
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)
    if half:
        model.half()

        # 二级分类器（默认关闭）
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # 数据加载器
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # 类别名称
    names = model.module.names if hasattr(model, 'module') else model.names

    # 图像显示函数（拼接原画面与风险场）
    def cv_show(p, im0, risk_img=None):
        height, width = im0.shape[:2]
        a = 800 / width  # 缩放比例
        size = (800, int(height * a))
        img_resize = cv2.resize(im0, size, interpolation=cv2.INTER_AREA)

        if risk_img is not None:
            risk_resize = cv2.resize(risk_img, size, interpolation=cv2.INTER_AREA)
            combined_img = np.hstack((img_resize, risk_resize))  # 横向拼接
            cv2.imshow(p, combined_img)
        else:
            cv2.imshow(p, img_resize)
        cv2.waitKey(1)

    # 模型预热
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))
    t0 = time.time()

    # 遍历视频帧
    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        # 图像预处理
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

            # 模型推理
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # 非极大值抑制
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # 二级分类（可选）
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # 处理检测结果
        for i, det in enumerate(pred):
            if webcam:
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')
            s += '%gx%g ' % img.shape[2:]

            current_frame_distances = {}
            risk_sources = []  # 存储当前帧风险源（含ID）

            if det is not None and len(det):
                # 调整检测框到原图尺寸
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # DeepSort输入准备
                bbox_xywh = []
                confs = []
                classes = []
                img_h, img_w, _ = im0.shape
                for *xyxy, conf, cls in det:
                    x_c, y_c, bbox_w, bbox_h = bbox_rel(img_w, img_h, *xyxy)
                    bbox_xywh.append([x_c, y_c, bbox_w, bbox_h])
                    confs.append([conf.item()])
                    classes.append([cls.item()])
                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)
                classes = torch.Tensor(classes)

                # DeepSort跟踪
                outputs = deepsort.update(xywhs, confss, im0, classes)

                # 过滤有效车辆类型（bicycle, car, motorcycle, bus, truck）
                valid_classes = {1, 2, 3, 5, 7}
                valid_outputs = [out for out in outputs if int(out[5]) in valid_classes]

                # 收集风险源信息（含ID）
                # 收集风险源信息（含ID）
                for out in valid_outputs:
                    x1, y1, x2, y2, current_id, cls_id = out
                    # 校验cls_id有效性，跳过无效类别
                    if cls_id < 0 or cls_id >= len(names):
                        print(f"警告：帧{frame_idx}中检测到无效类别ID {cls_id}，已跳过")
                        continue

                    # 类别与车型判断
                    current_class = names[int(cls_id)]
                    car_type = 'truck' if current_class in ['bus', 'truck'] else 'car'

                    # 置信度计算（带匹配校验）
                    class_np = classes.numpy().flatten()
                    match_idx = np.where(class_np == cls_id)[0]
                    if len(match_idx) == 0:
                        conf2 = 0.0
                    else:
                        conf_idx = match_idx[0]
                        conf = confss[conf_idx].item()
                        conf2 = round(conf, 2)

                    # 中心点坐标
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    # 绘制检测框
                    xyxy = [x1, y1, x2, y2]
                    label = f'{current_class} {conf2:.2f}'
                    dis_m = plot_one_box(
                        xyxy, im0, speed, outputs, time_person,
                        label=label, color=[0, 0, 255], line_thickness=3,
                        name=current_class
                    )
                    current_frame_distances[current_id] = round(dis_m, 2)

                    # 【关键】收集风险源（含ID，用于计算方向）
                    src_speed = current_frame_speeds.get(current_id, 0.0)
                    risk_sources.append({
                        'id': current_id,
                        'x': center_x,
                        'y': center_y,
                        'speed': src_speed,
                        'type': car_type
                    })

                # 速度计算（每5帧更新）
                box_centers = []
                for each_box in outputs:
                    if int(each_box[5]) in valid_classes:
                        center_x = (each_box[0] + each_box[2]) / 2
                        center_y = (each_box[1] + each_box[3]) / 2
                        box_centers.append([
                            center_x, center_y, each_box[4], each_box[5], each_box[2] - each_box[0]
                        ])
                location = box_centers
                locations.append(location)

                if len(locations) == 5:
                    if len(locations[0]) and len(locations[-1]) != 0:
                        locations = [locations[0], locations[-1]]
                        speed = Estimated_speed(locations, fps, width)
                        current_frame_speeds.clear()
                        for sp in speed:
                            current_frame_speeds[sp[1]] = sp[0]
                    locations = []

                # 写入car4.txt
                with open('lanechange.txt', 'a', encoding='utf-8') as f:
                    f.write(f"frame:{frame_idx}\n")
                    for current_id in current_frame_distances:
                        center_x = next((bc[0] for bc in box_centers if bc[2] == current_id), 0.0)
                        center_y = next((bc[1] for bc in box_centers if bc[2] == current_id), 0.0)
                        dis_m = current_frame_distances[current_id]
                        sp_kmh = current_frame_speeds.get(current_id, 0.0)
                        f.write(
                            f"id:{current_id},x:{round(center_x, 2)},y:{round(center_y, 2)},distance:{dis_m}m,speed:{sp_kmh}km/h\n")

            # 生成风险场（传入prev_centers）
            risk_img = None
            if len(risk_sources) > 0:
                risk_img = generate_risk_map(im0, risk_sources, prev_centers)

            # 打印耗时
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # 显示结果
            if view_img:
                cv_show(str(p), im0, risk_img)

            # 保存结果
            if save_img:
                if dataset.mode == 'image':
                    if risk_img is not None:
                        combined_img = np.hstack((im0, risk_img))
                        cv2.imwrite(save_path.replace('.jpg', '_with_risk.jpg'), combined_img)
                    else:
                        cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()
                        if vid_cap:
                            fps_vid = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            if risk_img is not None:
                                w *= 2  # 拼接后宽度加倍
                        else:
                            fps_vid, w, h = 30, im0.shape[1] * 2, im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps_vid, (w, h)
                        )
                    if risk_img is not None:
                        combined_img = np.hstack((im0, risk_img))
                        vid_writer.write(combined_img)
                    else:
                        vid_writer.write(im0)

        # 【关键】更新上一帧车辆位置（供下一帧计算方向）
        current_centers = {src['id']: (src['x'], src['y']) for src in risk_sources}
        prev_centers = current_centers

    # 保存信息
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    # 总耗时
    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov10s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='lanechange.mp4', help='source')  # 输入视频路径
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.01, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.01, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results', default=True)
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:
            for opt.weights in ['yolov10s.pt', 'yolov10m.pt', 'yolov10l.pt', 'yolov10x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()