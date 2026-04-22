import numpy as np
from deep_sort.deep_sort.sort.track import Track

class MotionEngine:
    """运动引擎，负责目标轨迹预测和运动分析"""
    
    def __init__(self, fps=30):
        """初始化运动引擎
        
        Parameters
        ----------
        fps : int
            视频帧率，用于计算时间间隔
        """
        self.fps = fps
        self.dt = 1.0 / fps
    
    def predict_trajectory(self, track, steps=5, method='linear'):
        """预测目标的未来轨迹
        
        Parameters
        ----------
        track : Track
            DeepSORT的Track对象
        steps : int
            预测的步数
        method : str
            预测方法: 'linear' (线性) 或 'polynomial' (多项式)
            
        Returns
        -------
        list
            预测的轨迹点列表 [(x1, y1), (x2, y2), ...]
        """
        if not track.is_confirmed():
            return []
        
        # 从Track对象获取当前状态
        # mean: [x, y, a, h, vx, vy, va, vh]
        current_state = track.mean.copy()
        x, y, vx, vy = current_state[0], current_state[1], current_state[4], current_state[5]
        
        predictions = []
        if method == 'linear':
            # 线性预测 (基于常数速度模型)
            for i in range(steps):
                t = (i + 1) * self.dt
                next_x = x + vx * t
                next_y = y + vy * t
                predictions.append((next_x, next_y))
        elif method == 'polynomial':
            # 多项式预测 (简单二次模型)
            # 假设加速度为常数 (这里简化为0)
            ax, ay = 0, 0
            for i in range(steps):
                t = (i + 1) * self.dt
                next_x = x + vx * t + 0.5 * ax * t**2
                next_y = y + vy * t + 0.5 * ay * t**2
                predictions.append((next_x, next_y))
        
        return predictions
    
    def predict_trajectories(self, tracks, steps=5, method='linear'):
        """预测多个目标的未来轨迹
        
        Parameters
        ----------
        tracks : list
            Track对象列表
        steps : int
            预测的步数
        method : str
            预测方法
            
        Returns
        -------
        dict
            以track_id为键，预测轨迹为值的字典
        """
        trajectories = {}
        for track in tracks:
            if track.is_confirmed():
                trajectory = self.predict_trajectory(track, steps, method)
                if trajectory:
                    trajectories[track.track_id] = trajectory
        return trajectories
    
    def calculate_time_decay(self, steps=5):
        """计算时间衰减因子
        
        Parameters
        ----------
        steps : int
            预测的步数
            
        Returns
        -------
        list
            时间衰减因子列表
        """
        # 指数衰减
        decay = []
        for i in range(steps):
            # 近期预测点权重高，远期预测点权重低
            weight = np.exp(-i * 0.5)  # 0.5是衰减系数
            decay.append(weight)
        # 归一化
        total = sum(decay)
        decay = [w / total for w in decay]
        return decay
    
    def get_motion_info(self, track):
        """获取目标的运动信息
        
        Parameters
        ----------
        track : Track
            DeepSORT的Track对象
            
        Returns
        -------
        dict
            运动信息字典
        """
        if not track.is_confirmed():
            return {}
        
        # 从Track对象获取当前状态
        current_state = track.mean.copy()
        x, y, a, h, vx, vy, va, vh = current_state
        
        # 计算速度大小和方向
        speed = np.sqrt(vx**2 + vy**2)
        direction = np.arctan2(vy, vx) * 180 / np.pi
        
        return {
            'position': (x, y),
            'velocity': (vx, vy),
            'speed': speed,
            'direction': direction,
            'aspect_ratio': a,
            'height': h
        }
