from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml
from deep_sort.deep_sort import DeepSort
from deep_sort.utils.parser import get_config
from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
from utils.general import check_imshow, check_requirements, increment_path, non_max_suppression, scale_coords, set_logging, strip_optimizer
from utils.plots import plot_one_box
from utils.torch_utils import time_synchronized
from yolov10.utils.general import apply_classifier, check_img_size, non_max_suppression, scale_coords, strip_optimizer, xyxy2xywh
from yolov10.utils.torch_utils import load_classifier, select_device, time_synchronized

from bbox3d_utils import BBox3DEstimator, BirdEyeView
from risk_field import RiskFieldEngine
from road_surface_fusion import RoadSurfaceAnalyzer, RoadSurfaceDetector, RoadSurfaceRiskFuser, RoadSurfaceVisualizer, RobustDepthEstimator

import sys

sys.path.insert(0, "./yolov10")


DRAW_WITH_DISTANCE = {"bicycle", "car", "motorcycle", "bus", "truck"}
TRACKED_CLASSES = {0, 1, 2, 3, 5, 7}


def bbox_rel(image_width, image_height, *xyxy):
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = bbox_left + bbox_w / 2
    y_c = bbox_top + bbox_h / 2
    return x_c, y_c, bbox_w, bbox_h


def load_camera_matrix(camera_params_path: str, source: str, webcam: bool) -> np.ndarray:
    cam_params = None
    try:
        with open(camera_params_path, "r", encoding="utf-8") as file_handle:
            cam_params = yaml.safe_load(file_handle)
    except Exception:
        cam_params = None

    if cam_params:
        return np.array(
            [
                [cam_params["camera_matrix"]["fx"], 0, cam_params["camera_matrix"]["cx"]],
                [0, cam_params["camera_matrix"]["fy"], cam_params["camera_matrix"]["cy"]],
                [0, 0, 1],
            ]
        )

    video_width, video_height = 1920, 1080
    if not webcam and source.endswith((".mp4", ".avi", ".mov")):
        capture = cv2.VideoCapture(source)
        if capture.isOpened():
            video_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or video_width
            video_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or video_height
        capture.release()

    est_f = float(video_width)
    est_cx = video_width / 2.0
    est_cy = video_height / 2.0
    return np.array([[est_f, 0, est_cx], [0, est_f, est_cy], [0, 0, 1]])


def draw_actor_box(image: np.ndarray, xyxy, label: str, color) -> None:
    x1, y1, x2, y2 = [int(value) for value in xyxy]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_top = max(y1 - text_h - 6, 0)
    cv2.rectangle(image, (x1, label_top), (x1 + text_w + 6, label_top + text_h + 6), color, -1)
    cv2.putText(
        image,
        label,
        (x1 + 3, label_top + text_h + 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def draw_combined_status_panel(
    image: np.ndarray,
    dynamic_risk: float,
    surface_risk: float,
    combined_risk: float,
    surface_analysis,
) -> None:
    if combined_risk >= 0.8:
        status = "HIGH"
        color = (0, 0, 255)
    elif combined_risk >= 0.55:
        status = "MEDIUM"
        color = (0, 165, 255)
    elif combined_risk >= 0.25:
        status = "LOW"
        color = (0, 215, 255)
    else:
        status = "CLEAR"
        color = (0, 180, 0)

    panel_w = 360
    panel_h = 88
    panel_x = max(image.shape[1] - panel_w - 20, 10)
    panel_y = 20

    overlay = image.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.45, image, 0.55, 0, image)

    lines = [
        f"Decision: {status}",
        f"Dynamic: {dynamic_risk:.2f}  Surface: {surface_risk:.2f}",
        f"P:{surface_analysis.pothole_count}  C:{surface_analysis.crack_count}  Combined:{combined_risk:.2f}",
    ]

    for index, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (panel_x + 10, panel_y + 24 + index * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color if index == 0 else (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def cv_show(window_name: str, im0: np.ndarray, risk_img: Optional[np.ndarray] = None) -> None:
    height, width = im0.shape[:2]
    scale = 800 / width
    size = (800, int(height * scale))
    image_resized = cv2.resize(im0, size, interpolation=cv2.INTER_AREA)

    if risk_img is not None:
        target_h = image_resized.shape[0]
        target_w = 600
        risk_resized = cv2.resize(risk_img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        combined_img = np.hstack((image_resized, risk_resized))
        cv2.imshow(window_name, combined_img)
    else:
        cv2.imshow(window_name, image_resized)
    cv2.waitKey(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default="yolov10s.pt", help="model.pt path(s)")
    parser.add_argument("--source", type=str, default="lanechange.mp4", help="source")
    parser.add_argument("--img-size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument("--conf-thres", type=float, default=0.01, help="object confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.01, help="IOU threshold for NMS")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", dest="view_img", action="store_true", help="display results")
    parser.add_argument("--no-view-img", dest="view_img", action="store_false", help="disable display results")
    parser.set_defaults(view_img=True)
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --class 0, or --class 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default="runs/detect_surface_fusion", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    parser.add_argument("--camera-params", type=str, default="config/camera_params.yaml")
    parser.add_argument("--road-model-dir", type=str, default=str(Path(__file__).resolve().parents[1] / "code" / "models"))
    parser.add_argument("--road-conf-thres", type=float, default=0.25)
    parser.add_argument("--depth-backend", choices=["heuristic", "depth-anything"], default="depth-anything")
    return parser


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith(".txt")
    webcam = source.isnumeric() or source.endswith(".txt") or source.lower().startswith(
        ("rtsp://", "rtmp://", "http://", "https://")
    )

    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    set_logging()
    device = select_device(opt.device)
    use_half = device.type != "cpu"

    camera_matrix = load_camera_matrix(opt.camera_params, source, webcam)
    print(f"Using camera matrix:\n{camera_matrix}")

    print("Initializing depth estimator...")
    depth_estimator = RobustDepthEstimator(
        model_size="small",
        device=device.type if device.type != "cpu" else "cpu",
        backend=opt.depth_backend,
    )
    print(f"Depth backend: {depth_estimator.backend_name}")
    bbox3d_estimator = BBox3DEstimator(camera_matrix=camera_matrix)

    road_detector = RoadSurfaceDetector(model_dir=opt.road_model_dir, preferred_device=device.type)
    road_analyzer = RoadSurfaceAnalyzer()
    road_visualizer = RoadSurfaceVisualizer()
    road_fuser = RoadSurfaceRiskFuser()

    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(
        cfg.DEEPSORT.REID_CKPT,
        max_dist=cfg.DEEPSORT.MAX_DIST,
        min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
        max_age=cfg.DEEPSORT.MAX_AGE,
        n_init=cfg.DEEPSORT.N_INIT,
        nn_budget=cfg.DEEPSORT.NN_BUDGET,
        use_cuda=device.type != "cpu",
    )

    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)
    if use_half:
        model.half()

    classify = False
    if classify:
        modelc = load_classifier(name="resnet101", n=2)
        modelc.load_state_dict(torch.load("weights/resnet101.pt", map_location=device)["model"]).to(device).eval()

    vid_path, vid_writer = None, None
    if webcam:
        if view_img:
            view_img = check_imshow()
        cudnn.benchmark = True
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    names = model.module.names if hasattr(model, "module") else model.names
    if device.type != "cpu":
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    risk_engine = RiskFieldEngine(width_meter=16, depth_meter=25, grid_res=0.1)
    bev_visualizer = BirdEyeView(size=(400, 625), scale=25, camera_height=1.5)
    track_history = defaultdict(list)

    t0 = time.time()
    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if use_half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        pred = non_max_suppression(
            pred,
            opt.conf_thres,
            opt.iou_thres,
            classes=opt.classes,
            agnostic=opt.agnostic_nms,
        )
        t2 = time_synchronized()

        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        for i, det in enumerate(pred):
            if webcam:
                p, s, im0, frame = path[i], f"{i}: ", im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, "", im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)
            save_path = str(save_dir / p.name)
            s += "%gx%g " % img.shape[2:]

            depth_map = depth_estimator.estimate_depth(im0)
            road_main_results, road_aux_results, road_model_label = road_detector.detect(
                im0,
                conf_thres=opt.road_conf_thres,
            )
            surface_analysis = road_analyzer.analyze(
                im0,
                depth_map,
                camera_matrix,
                road_main_results,
                road_aux_results,
                road_model_label,
            )
            im0 = road_visualizer.draw_on_frame(im0, surface_analysis)

            outputs = np.empty((0, 6), dtype=np.int64)
            risk_sources = []
            speed = []
            time_person = 3

            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                bbox_xywh = []
                confs = []
                classes_list = []
                bbox_3d_list = []

                img_h, img_w, _ = im0.shape
                for *xyxy, conf, cls in det:
                    x_c, y_c, bbox_w, bbox_h = bbox_rel(img_w, img_h, *xyxy)
                    bbox_xywh.append([x_c, y_c, bbox_w, bbox_h])
                    confs.append([conf.item()])
                    classes_list.append([cls.item()])

                    xyxy_cpu = [value.item() for value in xyxy]
                    depth_val = depth_estimator.get_depth_in_region(depth_map, xyxy_cpu, method="median")
                    class_name = names[int(cls)]
                    box_3d = bbox3d_estimator.estimate_3d_box(xyxy_cpu, depth_val, class_name)
                    bbox_3d_list.append(
                        (
                            box_3d["location"][0],
                            box_3d["location"][2],
                            box_3d["dimensions"][2],
                            box_3d["dimensions"][1],
                            box_3d["dimensions"][0],
                            box_3d["orientation"],
                        )
                    )

                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)
                classes = torch.Tensor(classes_list)
                outputs = deepsort.update(xywhs, confss, im0, classes, bbox_3d=bbox_3d_list)
                valid_outputs = [item for item in outputs if int(item[5]) in TRACKED_CLASSES]

                for out in valid_outputs:
                    x1, y1, x2, y2, current_id, cls_id = out
                    if cls_id < 0 or cls_id >= len(names):
                        continue

                    current_class = names[int(cls_id)]
                    match_idx = np.where(classes.numpy().flatten() == cls_id)[0]
                    conf = confss[match_idx[0]].item() if len(match_idx) > 0 else 0.0
                    label = f"{current_class} {conf:.2f}"

                    if current_class in DRAW_WITH_DISTANCE:
                        plot_one_box(
                            [x1, y1, x2, y2],
                            im0,
                            speed,
                            outputs,
                            time_person,
                            label=label,
                            color=[0, 0, 255],
                            line_thickness=3,
                            name=current_class,
                        )
                    else:
                        draw_actor_box(im0, [x1, y1, x2, y2], label, (0, 255, 0))

                    track = next((track_item for track_item in deepsort.tracker.tracks if track_item.track_id == current_id), None)
                    if track is None or track.pose_3d is None:
                        continue

                    x_raw, z_raw, length, width, height, yaw_raw = track.pose_3d
                    box_3d_raw = {
                        "location": np.array([x_raw, 0, z_raw]),
                        "dimensions": np.array([height, width, length]),
                        "orientation": yaw_raw,
                        "class_name": current_class,
                    }

                    if hasattr(bbox3d_estimator, "refine_box_3d"):
                        box_3d_refined = bbox3d_estimator.refine_box_3d(box_3d_raw, current_id)
                        x_pos, _, z_pos = box_3d_refined["location"]
                        yaw = box_3d_refined["orientation"]
                    else:
                        x_pos, z_pos, yaw = x_raw, z_raw, yaw_raw

                    object_dims = np.array([height, width, length])
                    box_3d_draw = {
                        "bbox_2d": [x1, y1, x2, y2],
                        "location": np.array([x_pos, 0, z_pos]),
                        "dimensions": object_dims,
                        "orientation": yaw,
                        "class_name": current_class,
                        "object_id": current_id,
                        "depth_value": np.clip((z_pos - 1.0) / 9.0, 0.0, 1.0),
                    }

                    object_type = "truck" if current_class in {"bus", "truck"} else ("person" if current_class == "person" else "car")
                    risk_sources.append(
                        {
                            "id": current_id,
                            "x": x_pos,
                            "z": z_pos,
                            "speed": 0.0,
                            "yaw": yaw,
                            "type": object_type,
                            "dims": object_dims,
                            "box_3d_draw": box_3d_draw,
                            "class_name": current_class,
                        }
                    )

                    track_history[current_id].append((x_pos, z_pos))
                    if len(track_history[current_id]) > 50:
                        track_history[current_id].pop(0)

            total_risk_map = np.zeros((risk_engine.grid_h, risk_engine.grid_w), dtype=np.float32)
            vis_risk_map = np.zeros((risk_engine.grid_h, risk_engine.grid_w), dtype=np.float32)
            dynamic_risk = 0.0
            max_risk_id = None
            ego_v_x, ego_v_z = 0.0, 5.0
            max_dist_m = 10.0

            for src in risk_sources:
                target_x = src["x"]
                target_z = src["z"]
                target_speed = src["speed"] / 3.6
                target_yaw = src["yaw"]

                max_dist_m = max(max_dist_m, target_z)

                target_v_x = target_speed * np.sin(target_yaw)
                target_v_z = target_speed * np.cos(target_yaw)
                scf, ego_field, target_field, overlap = risk_engine.calculate_scf(
                    (0, 0),
                    (target_x, target_z),
                    (ego_v_x, ego_v_z),
                    (target_v_x, target_v_z),
                    weather_factor=1.0,
                )

                vis_target_f = risk_engine.get_visualization_field(
                    target_x,
                    target_z,
                    target_v_x,
                    target_v_z,
                    sigma_x=1.8,
                    sigma_z=5.0,
                    weather_factor=1.0,
                )

                src["scf"] = scf
                total_risk_map += target_field
                vis_risk_map = np.maximum(vis_risk_map, vis_target_f)

                if scf > dynamic_risk:
                    dynamic_risk = scf
                    max_risk_id = src["id"]

            surface_risk_map, surface_vis_map, surface_risk = road_fuser.build_surface_maps(surface_analysis, risk_engine)
            total_risk_map += surface_risk_map
            vis_risk_map = np.maximum(vis_risk_map, surface_vis_map)
            combined_risk = road_fuser.fuse_risk(dynamic_risk, surface_risk)
            if surface_risk >= dynamic_risk and surface_analysis.hazards:
                max_risk_id = "ROAD"

            for src in risk_sources:
                im0 = bbox3d_estimator.draw_box_3d(im0, src["box_3d_draw"], risk_score=src.get("scf", 0.0))

            draw_combined_status_panel(im0, dynamic_risk, surface_risk, combined_risk, surface_analysis)

            for hazard in surface_analysis.hazards:
                max_dist_m = max(max_dist_m, hazard.z_m)

            bev_visualizer.update_scale(max_dist_m)
            bev_visualizer.reset()

            for src in risk_sources:
                history = track_history[src["id"]]
                bev_visualizer.draw_trajectory_fading(history)
                bev_visualizer.draw_future_sector(
                    src["x"],
                    src["z"],
                    src["speed"] / 3.6,
                    src["yaw"],
                    risk_level=src.get("scf", 0.0),
                )

            for src in risk_sources:
                box_3d_bev = {
                    "class_name": src["class_name"],
                    "location": np.array([src["x"], 0, src["z"]]),
                    "dimensions": src["dims"],
                    "orientation": src["yaw"],
                    "object_id": src["id"],
                    "depth_value": np.clip((src["z"] - 1.0) / 9.0, 0.0, 1.0),
                    "bbox_2d": src["box_3d_draw"]["bbox_2d"],
                }
                bev_visualizer.draw_box(box_3d_bev)

            if np.max(vis_risk_map) > 0:
                bev_visualizer.draw_risk_heatmap(cv2.flip(vis_risk_map, 0))

            road_visualizer.draw_on_bev(bev_visualizer, surface_analysis)
            bev_visualizer.draw_hud(combined_risk, max_risk_id, ego_speed=ego_v_z * 3.6)
            risk_img = bev_visualizer.get_image()

            target_h = im0.shape[0]
            target_w = 600
            if risk_img.shape[0] != target_h or risk_img.shape[1] != target_w:
                risk_img = cv2.resize(risk_img, (target_w, target_h))

            print(f"{s}Done. ({t2 - t1:.3f}s)")

            if view_img:
                cv_show(str(p), im0, risk_img)

            if save_img:
                if dataset.mode == "image":
                    combined_img = np.hstack((im0, risk_img))
                    image_output = Path(save_path)
                    output_path = image_output.with_name(f"{image_output.stem}_with_risk{image_output.suffix}")
                    cv2.imwrite(str(output_path), combined_img)
                else:
                    if vid_path != save_path:
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()

                        if vid_cap:
                            fps_vid = vid_cap.get(cv2.CAP_PROP_FPS)
                            width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)) + risk_img.shape[1]
                            height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            fps_vid, width, height = 30, im0.shape[1] + risk_img.shape[1], im0.shape[0]
                            if not save_path.lower().endswith(".mp4"):
                                save_path += ".mp4"

                        vid_writer = cv2.VideoWriter(
                            save_path,
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps_vid,
                            (width, height),
                        )

                    vid_writer.write(np.hstack((im0, risk_img)))

    if save_txt or save_img:
        labels_msg = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        print(f"Results saved to {save_dir}{labels_msg}")

    print(f"Done. ({time.time() - t0:.3f}s)")


if __name__ == "__main__":
    parser = build_parser()
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=("pycocotools", "thop"))

    with torch.no_grad():
        if opt.update:
            for opt.weights in ["yolov10s.pt", "yolov10m.pt", "yolov10l.pt", "yolov10x.pt"]:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
