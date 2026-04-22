import numpy as np
import cv2

class TrajectoryPredictor:
    """轨迹预测器，用于预测目标的未来运动轨迹"""
    
    def __init__(self, camera_matrix=None):
        """初始化轨迹预测器
        
        Parameters
        ----------
        camera_matrix : ndarray, optional
            相机内参矩阵，用于将2D像素坐标转换为3D物理坐标
        """
        self.camera_matrix = camera_matrix
    
    def predict_trajectories(self, tracks, steps=10):
        """预测多个目标的未来轨迹
        
        Parameters
        ----------
        tracks : list
            Track对象列表
        steps : int
            预测的步数
            
        Returns
        -------
        dict
            以track_id为键，预测轨迹为值的字典
        """
        trajectories = {}
        for track in tracks:
            if track.is_confirmed():
                # 使用Track类的predict_future_trajectory方法预测轨迹
                future_points = track.predict_future_trajectory(steps)
                trajectories[track.track_id] = future_points
        return trajectories
    
    def convert_2d_to_3d(self, points_2d, depth=10):
        """将2D像素坐标转换为3D物理坐标
        
        Parameters
        ----------
        points_2d : list
            2D像素坐标列表 [(x1, y1), (x2, y2), ...]
        depth : float
            假设的深度值（米）
            
        Returns
        -------
        list
            3D物理坐标列表 [(x1, y1, z1), (x2, y2, z2), ...]
        """
        if self.camera_matrix is None:
            # 如果没有相机矩阵，返回带默认深度的坐标
            return [(x, y, depth) for x, y in points_2d]
        
        points_3d = []
        for x, y in points_2d:
            # 使用相机内参将2D坐标转换为3D坐标
            # 这里使用简化的方法，假设深度已知
            x_3d = (x - self.camera_matrix[0, 2]) * depth / self.camera_matrix[0, 0]
            y_3d = (y - self.camera_matrix[1, 2]) * depth / self.camera_matrix[1, 1]
            points_3d.append((x_3d, y_3d, depth))
        return points_3d
    
    def calculate_risk(self, trajectories, ego_position=(0, 0, 0), time_horizon=5):
        """计算轨迹的风险
        
        Parameters
        ----------
        trajectories : dict
            轨迹字典 {track_id: [(x1, y1), ...]}
        ego_position : tuple
            自车位置 (x, y, z)
        time_horizon : int
            时间 horizon（步数）
            
        Returns
        -------
        dict
            以track_id为键，风险值为值的字典
        """
        risk_scores = {}
        for track_id, points in trajectories.items():
            # 计算轨迹与自车的距离
            min_distance = float('inf')
            for i, (x, y) in enumerate(points[:time_horizon]):
                distance = np.sqrt((x - ego_position[0])**2 + (y - ego_position[1])**2)
                if distance < min_distance:
                    min_distance = distance
            
            # 基于距离计算风险分数
            # 距离越近，风险越高
            if min_distance < 5:
                risk_scores[track_id] = 1.0  # 高风险
            elif min_distance < 10:
                risk_scores[track_id] = 0.7  # 中风险
            elif min_distance < 20:
                risk_scores[track_id] = 0.3  # 低风险
            else:
                risk_scores[track_id] = 0.0  # 无风险
        return risk_scores

class RiskFieldIntegrator:
    """风险场集成器，用于将轨迹预测结果集成到风险场中"""
    
    def __init__(self, predictor):
        """初始化风险场集成器
        
        Parameters
        ----------
        predictor : TrajectoryPredictor
            轨迹预测器实例
        """
        self.predictor = predictor
    
    def integrate(self, tracks, risk_field, ego_position=(0, 0, 0)):
        """将轨迹预测集成到风险场中
        
        Parameters
        ----------
        tracks : list
            Track对象列表
        risk_field : dict
            原始风险场
        ego_position : tuple
            自车位置
            
        Returns
        -------
        dict
            集成后的风险场
        """
        # 预测轨迹
        trajectories = self.predictor.predict_trajectories(tracks)
        
        # 计算风险
        risk_scores = self.predictor.calculate_risk(trajectories, ego_position)
        
        # 集成到风险场
        integrated_risk = risk_field.copy()
        for track_id, risk in risk_scores.items():
            if risk > 0:
                # 将轨迹风险添加到风险场
                integrated_risk[f"trajectory_{track_id}"] = {
                    "risk": risk,
                    "track_id": track_id
                }
        
        return integrated_risk

class TrajectoryVisualizer:
    """轨迹可视化器，用于绘制预测轨迹"""
    
    def draw_trajectories(self, image, trajectories, color=(0, 255, 0), thickness=2):
        """在图像上绘制预测轨迹
        
        Parameters
        ----------
        image : ndarray
            输入图像
        trajectories : dict
            轨迹字典 {track_id: [(x1, y1), ...]}
        color : tuple
            轨迹颜色 (B, G, R)
        thickness : int
            轨迹线宽
            
        Returns
        -------
        ndarray
            绘制了轨迹的图像
        """
        result = image.copy()
        for track_id, points in trajectories.items():
            if len(points) > 1:
                # 转换为整数坐标
                points = [(int(x), int(y)) for x, y in points]
                # 绘制轨迹线
                cv2.polylines(result, [np.array(points)], False, color, thickness, cv2.LINE_AA)
                # 绘制轨迹点
                for i, (x, y) in enumerate(points):
                    # 点的大小随预测步数增加而减小
                    radius = max(1, 4 - i // 2)
                    cv2.circle(result, (x, y), radius, color, -1)
        return result
