import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


# 定义公式
def U_obs(x_phi, y_phi, x_obs, y_obs, beta_U, beta_v, v, mu, beta_mu, beta_obs, a, beta_a, phi_1):
    distance_x = (y_phi - x_obs) ** 2 / (1 + beta_v * v * beta_mu * mu * np.abs(np.cos(phi_1))) ** 2
    distance_y = (x_phi - y_obs) ** 2 / (2 * (1 + beta_obs * np.abs(np.sin(phi_1)))) ** 2
    cos_term = beta_a * a * np.cos(np.arctan2(y_phi, x_phi))
    return beta_U * np.exp(-0.5 * (distance_x + distance_y + cos_term))


# 设置参数
beta_U = 0.6
beta_v = 1.0
v = 1.0
mu = 1.0
beta_mu = 1.0
beta_obs = 1.0
a = 1.0
beta_a = 1.0
x_obs_init = 2
y_obs_init = 8
# 生成平面坐标，进一步增加点数以提高分辨率
x_phi = np.linspace(-20, 20, 1000)
y_phi = np.linspace(-15, 30, 1000)
x_phi, y_phi = np.meshgrid(y_phi, x_phi)
# 生成U_obs的值
phi_1_1 = np.pi / 2  # 第一个位置的phi_1的值
U_obs_values_1 = U_obs(x_phi, y_phi, 0, -10, beta_U, beta_v, v, mu, beta_mu, beta_obs, a, beta_a, phi_1_1)

phi_1_2 = np.pi / 4  # 第二个位置的phi_1的值
U_obs_values_2 = U_obs(x_phi, y_phi, x_obs_init, y_obs_init, beta_U, beta_v, v, mu, beta_mu, beta_obs, a,
                       beta_a, phi_1_2)
x3 = -5
y3 = 7
U_obs_values_3 = U_obs(x_phi, y_phi, x3, y3, beta_U, beta_v, v, mu, beta_mu, beta_obs, a, beta_a, phi_1_2)
U_obs_values_4 = U_obs(x_phi, y_phi, -3, 10, beta_U, beta_v, v, mu, beta_mu, beta_obs, a, beta_a, phi_1_2)
U = U_obs_values_1 + U_obs_values_2 + U_obs_values_3 + U_obs_values_4

# 创建 3D 图形对象
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# 绘制三维曲面图，开启抗锯齿并调整着色模式
surf = ax.plot_surface(y_phi, x_phi, U, cmap='hot', antialiased=True, shade=True, rstride=1, cstride=1)

# 添加颜色条
fig.colorbar(surf, shrink=0.5, aspect=5, ticks=[0, 0.37, 0.42, 0.5, 1])

# 设置坐标轴标签和标题
ax.set_xlabel('x_phi')
ax.set_ylabel('y_phi')
ax.set_zlabel('U')
ax.set_title('行车风险场动态仿真三维图')

plt.show()
