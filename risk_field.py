import numpy as np
import cv2

class RiskFieldEngine:
    def __init__(self, width_meter=16, depth_meter=25, grid_res=0.1):
        """
        Initialize the Risk Field Engine.
        :param width_meter: Width of the field in meters (X-axis)
        :param depth_meter: Depth of the field in meters (Z-axis)
        :param grid_res: Grid resolution in meters
        """
        self.width_meter = width_meter
        self.depth_meter = depth_meter
        self.grid_res = grid_res
        
        # Grid dimensions
        self.grid_w = int(width_meter / grid_res)
        self.grid_h = int(depth_meter / grid_res)
        
        # Create meshgrid for vectorized calculations
        # X: -width/2 to width/2
        # Z: 0 to depth
        # Note: We use meshgrid 'xy' indexing by default in numpy, but for image we want (row, col)
        # Here we align with image coordinates: row -> Z (up/down), col -> X (left/right)
        
        # X axis (columns): -width/2 ... width/2
        x = np.linspace(-width_meter/2, width_meter/2, self.grid_w)
        # Z axis (rows): 0 ... depth (usually 0 is bottom, depth is top in BEV, or vice versa)
        # In our BEV, usually bottom is 0 (near), top is depth (far).
        # We'll map Z=0 to row=height-1, Z=depth to row=0.
        z = np.linspace(0, depth_meter, self.grid_h)
        
        self.X, self.Z = np.meshgrid(x, z)
        
    def get_gaussian_field(self, x_c, z_c, v_x=0, v_z=0, sigma_x=0.8, sigma_z=2.0, v_stretch_factor=0.2, weather_factor=1.0):
        """
        Generate a Gaussian risk field for a vehicle.
        :param x_c, z_c: Center position
        :param v_x, v_z: Velocity vector
        :param sigma_x, sigma_z: Base standard deviations
        :param v_stretch_factor: How much velocity stretches the field
        :param weather_factor: Environmental risk multiplier (e.g., 1.2 for rain)
        :return: Grid of risk values
        """
        # Apply weather factor to base sigma
        eff_sigma_x = sigma_x * weather_factor
        eff_sigma_z_base = sigma_z * weather_factor
        
        # Calculate velocity magnitude
        v_mag = np.sqrt(v_x**2 + v_z**2)
        
        # Stretch sigma_z based on velocity (Driving Risk Field concept)
        eff_sigma_z = eff_sigma_z_base + v_mag * v_stretch_factor
        
        # Rotation angle from Z-axis (which is up in our plot)
        # v_z is forward, v_x is lateral
        angle = np.arctan2(v_x, v_z) 
        
        # Rotate covariance matrix
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        
        # Covariance in local frame (aligned with velocity)
        # We assume X is lateral, Z is longitudinal
        Sigma_local = np.array([[eff_sigma_x**2, 0], [0, eff_sigma_z**2]])
        
        # Sigma_global = R @ Sigma_local @ R.T
        # Note: This standard rotation formula applies if X, Z are standard Cartesian.
        Sigma_global = R @ Sigma_local @ R.T
        
        # Inverse of covariance matrix
        try:
            Sigma_inv = np.linalg.inv(Sigma_global)
        except:
            Sigma_inv = np.eye(2) * (1.0 / (eff_sigma_x * eff_sigma_z))
            
        # Vectorized Mahalanobis distance
        # Stack points: (N, 2)
        # We flatten X and Z to make a list of points
        points = np.stack([self.X.ravel() - x_c, self.Z.ravel() - z_c], axis=1)
        
        # Calculate (x-mu)^T * Sigma^-1 * (x-mu)
        # Term 1: points @ Sigma_inv -> (N, 2)
        term1 = points @ Sigma_inv
        # Term 2: sum(term1 * points, axis=1) -> (N,)
        mahal_sq = np.sum(term1 * points, axis=1)
        
        # Gaussian: exp(-0.5 * mahal_sq)
        grid_vals = np.exp(-0.5 * mahal_sq)
        
        # Reshape back to grid
        return grid_vals.reshape(self.grid_h, self.grid_w)

    def calculate_scf(self, ego_pos, target_pos, v_ego, v_target, weather_factor=1.0):
        """
        Calculate Surrogate Conflict Field (SCF) between ego and target.
        :return: scf_value, ego_field, target_field, overlap_field
        """
        # Ego field (Self)
        ego_field = self.get_gaussian_field(
            ego_pos[0], ego_pos[1], 
            v_ego[0], v_ego[1], 
            sigma_x=1.0, sigma_z=3.0, # Ego field is usually larger
            weather_factor=weather_factor
        )
        
        # Target field
        target_field = self.get_gaussian_field(
            target_pos[0], target_pos[1], 
            v_target[0], v_target[1], 
            sigma_x=0.8, sigma_z=2.0,
            weather_factor=weather_factor
        )
        
        # Conflict overlap
        overlap_field = ego_field * target_field
        
        # Integral (Sum)
        scf_value = np.sum(overlap_field)
        
        return scf_value, ego_field, target_field, overlap_field

    def get_visualization_field(self, x_c, z_c, v_x=0, v_z=0, sigma_x=1.5, sigma_z=4.0, weather_factor=1.0):
        """
        Generate a larger, smoother Gaussian field specifically for visualization (Radar-like feel).
        Parameters are tuned for visual impact (larger sigma).
        """
        return self.get_gaussian_field(
            x_c, z_c, v_x, v_z, 
            sigma_x=sigma_x, sigma_z=sigma_z, 
            v_stretch_factor=0.3, # More stretch for visual effect
            weather_factor=weather_factor
        )
