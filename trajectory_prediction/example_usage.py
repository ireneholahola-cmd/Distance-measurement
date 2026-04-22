#!/usr/bin/env python3
"""
轨迹预测功能示例
"""

import cv2
import numpy as np
from trajectory_predictor import TrajectoryPredictor, RiskFieldIntegrator, TrajectoryVisualizer

# 示例使用轨迹预测器
def example_trajectory_prediction():
    # 假设有一个Track对象列表
    # 这里我们创建一个模拟的Track对象来演示
    class MockTrack:
        def __init__(self, track_id, mean, state=2):  # state=2 表示Confirmed
            self.track_id = track_id
            self.mean = mean  # [x, y, a, h, vx, vy, va, vh]
            self.state = state
        
        def is_confirmed(self):
            return self.state == 2
        
        def predict_future_trajectory(self, steps=10):
            """预测未来轨迹"""
            predictions = []
            current_state = self.mean.copy()
            for _ in range(steps):
                next_x = current_state[0] + current_state[4]
                next_y = current_state[1] + current_state[5]
                predictions.append((next_x, next_y))
                current_state[0] = next_x
                current_state[1] = next_y
            return predictions
    
    # 创建模拟的Track对象
    tracks = [
        MockTrack(1, np.array([100, 200, 1.0, 50, 2, 0, 0, 0])),  # 水平移动
        MockTrack(2, np.array([300, 200, 1.0, 50, -1, 1, 0, 0])),  # 斜向移动
        MockTrack(3, np.array([500, 200, 1.0, 50, 0, -2, 0, 0])),  # 垂直移动
    ]
    
    # 初始化轨迹预测器
    predictor = TrajectoryPredictor()
    
    # 预测轨迹
    trajectories = predictor.predict_trajectories(tracks)
    print("预测轨迹:")
    for track_id, points in trajectories.items():
        print(f"Track {track_id}: {points[:3]}...")  # 只打印前3个点
    
    # 计算风险
    risk_scores = predictor.calculate_risk(trajectories, ego_position=(320, 240))
    print("\n风险评估:")
    for track_id, risk in risk_scores.items():
        print(f"Track {track_id}: 风险值 = {risk:.2f}")
    
    # 可视化轨迹
    # 创建一个空白图像
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 绘制轨迹
    visualizer = TrajectoryVisualizer()
    result = visualizer.draw_trajectories(image, trajectories)
    
    # 显示结果
    cv2.imshow("Trajectory Prediction", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    example_trajectory_prediction()
