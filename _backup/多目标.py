import matplotlib
matplotlib.use('TkAgg')  # 或者 'QtAgg'，根据你的环境选择合适的后端
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 调整后的参数（与论文DSF模型一致）
K = 0.1  # 风险系数，论文中K=0.1（需根据实际标定调整）
k1 = 2  # 距离影响系数
k2 = 1  # 车道风险系数
k3 = 10  # 速度匹配系数
alpha = 0.5  # 动态权重因子
dsi_star = 0.1  # 标准安全指数

# 设置中文字体和解决负号显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 计算单个场景元素的安全性能指标（SPE）
def calculate_spe_v_bi(R_b, M_b, R_i, M_i, DR_i, rb_i, vb, theta_b):
    distance = np.linalg.norm(rb_i)
    vb_speed = np.linalg.norm(vb)
    cos_theta = np.cos(theta_b)

    numerator_part = k3 - vb_speed * cos_theta
    if numerator_part <= 0:
        numerator_part = 1e-6  # 防止数值不稳定

    exponent = (1 - k1) / k1
    # 避免除以零和无效的幂运算
    if k3 - vb_speed == 0:
        bracket_term = 1e-6
    else:
        base = numerator_part ** (1 - k1) / (k3 - vb_speed)
        if base < 0:
            base = 1e-6
        bracket_term = base ** (1 / k1)

    denominator = (k1 - 1) * (distance ** (k1 - 1))
    # 避免除以零
    if denominator == 0:
        spe = 0
    else:
        spe = (K * R_b * M_b * R_i * M_i * (1 + DR_i) * k3 * bracket_term) / denominator
    return spe


# 计算整个场景的总SPE
def calculate_total_spe(scene_elements):
    total_spe = 0
    for element in scene_elements:
        total_spe += calculate_spe_v_bi(**element)
    return total_spe


# 从文件中读取车辆数据
def read_data_from_file(file_path):
    data = []
    frame_counter = 1
    try:
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                try:
                    if parts[0].startswith('frame:'):
                        if len(parts) < 3:
                            print(f"数据格式错误: {line}, 预期至少3个部分，实际只有 {len(parts)} 个部分，跳过该行")
                            continue
                        frame = int(parts[0].split(':')[1])
                        id_part, speed_part = parts[1], parts[2]
                    else:
                        if len(parts) < 2:
                            print(f"数据格式错误: {line}, 预期至少2个部分，实际只有 {len(parts)} 个部分，跳过该行")
                            continue
                        frame = frame_counter
                        id_part, speed_part = parts[0], parts[1]
                        frame_counter += 1

                    id_num = int(id_part.split(',')[0].split(':')[1])
                    x = float(id_part.split(',')[1].split(':')[1])
                    y = float(id_part.split(',')[2].split(':')[1])
                    speed = float(speed_part.split(':')[1].replace('km/h', ''))
                    data.append({'frame': frame, 'id': id_num, 'x': x, 'y': y, 'speed': speed})
                except (IndexError, ValueError) as e:
                    print(f"解析错误: {line}, 错误: {str(e)}，跳过该行")
    except FileNotFoundError:
        print(f"文件 {file_path} 不存在！")

    return data


# 计算 RDSI1* 和 RDSI2*
def calculate_rdsi_star(relative_speed, theta_b, prev_spe):
    # 跟车或切入场景，使用 1s 的车头时距（THW）和 4s 的 TTC 计算
    thw_distance = relative_speed * 1  # 1s 的车头时距对应的距离
    ttc_4s_distance = relative_speed * 4  # 4s 的 TTC 对应的距离

    if relative_speed > 0:
        # 计算 THW 对应的 RDSI₁*
        thw_scene_elements = [{
            'R_b': 1, 'M_b': 1,
            'R_i': 1, 'M_i': 1,
            'DR_i': 0,
            'rb_i': np.array([thw_distance, 0]),
            'vb': np.array([relative_speed, 0]),
            'theta_b': theta_b
        }]
        thw_spe = calculate_total_spe(thw_scene_elements)
        thw_dsi = alpha * thw_spe + (1 - alpha) * (thw_spe - prev_spe)
        rdsi1_star = np.abs(thw_dsi / dsi_star)

        # 计算 TTC 为 4s 对应的 RDSI₂*
        ttc_4s_scene_elements = [{
            'R_b': 1, 'M_b': 1,
            'R_i': 1, 'M_i': 1,
            'DR_i': 0,
            'rb_i': np.array([ttc_4s_distance, 0]),
            'vb': np.array([relative_speed, 0]),
            'theta_b': theta_b
        }]
        ttc_4s_spe = calculate_total_spe(ttc_4s_scene_elements)
        ttc_4s_dsi = alpha * ttc_4s_spe + (1 - alpha) * (ttc_4s_spe - prev_spe)
        rdsi2_star = np.abs(ttc_4s_dsi / dsi_star)

        # 确保 RDSI1* 低于 RDSI2*
        if rdsi1_star >= rdsi2_star:
            rdsi1_star = rdsi2_star * 0.6

    else:
        rdsi1_star = 0
        rdsi2_star = 0

    return rdsi1_star, rdsi2_star


# 主模拟函数，计算并绘制各种安全指标
def simulate_with_file_data():
    data = read_data_from_file('实验七.txt')
    rdsi_values = []
    ttci_values = []  # 新增TTCI值存储
    rdsi1_star_values = []
    rdsi2_star_values = []
    time_points = []
    prev_spe = 0

    output_data = []  # 用于存储要写入文件的数据

    for i in range(len(data) - 1):
        current, next_data = data[i], data[i + 1]
        pos1 = np.array([current['x'], current['y']])
        pos2 = np.array([next_data['x'], next_data['y']])
        # 直接使用文件中读取的速度作为相对速度
        relative_speed = current['speed']
        distance = np.linalg.norm(pos2 - pos1)

        # 计算TTC与TTCI（论文式18-20）
        if relative_speed > 0:
            ttc = distance / relative_speed
        else:
            ttc = np.inf  # 相对速度为0时视为安全
        ttci = 1.0 / ttc if ttc != 0 else 0.0  # TTCI为TTC倒数
        ttci = ttci * 0.5
        # 角度计算（论文式5-7）
        if relative_speed != 0:
            vel_dir = np.array([current['speed'], 0]) / relative_speed
        else:
            vel_dir = np.array([1, 0])
        pos_dir = (pos2 - pos1) / distance if distance != 0 else np.array([1, 0])
        cos_theta = np.clip(np.dot(vel_dir, pos2 - pos1), -1, 1)
        theta_b = np.arccos(cos_theta)

        # 场景元素配置（简化模型，假设R=1, M=1, DR=0）
        scene_elements = [{
            'R_b': 1, 'M_b': 1,
            'R_i': 1, 'M_i': 1,
            'DR_i': 0,  # 无驾驶员风险因子
            'rb_i': pos2 - pos1,
            'vb': np.array([relative_speed, 0]),
            'theta_b': theta_b
        }]

        spe = calculate_total_spe(scene_elements)
        dsi = alpha * spe + (1 - alpha) * (spe - prev_spe)
        rdsi = np.abs(dsi / dsi_star)  # 确保RDSI为正

        # 计算 RDSI₁* 和 RDSI₂*
        rdsi1_star, rdsi2_star = calculate_rdsi_star(relative_speed, theta_b, prev_spe)

        rdsi_values.append(rdsi)
        ttci_values.append(ttci)
        rdsi1_star_values.append(rdsi1_star)
        rdsi2_star_values.append(rdsi2_star)
        time_points.append(current['frame'])
        prev_spe = spe

        # 构造要写入文件的字符串，包含所有需要保存的信息
        line = f"frame: {current['frame']}\n" \
               f"id : {current['id']} x: {current['x']} y: {current['y']} RDSI₁*: {rdsi1_star} RDSI₂*: {rdsi2_star} RDSI: {rdsi} TTCI: {ttci}\n"
        output_data.append(line)

    # 将数据写入文件
    output_data = [line.encode('utf-8', 'ignore').decode('utf-8') for line in output_data]
    with open('exp7.txt', 'w', encoding='utf-8') as file:
        file.writelines(output_data)

    # 创建图形和坐标轴
    fig, ax = plt.subplots(figsize=(12, 8))
    line_rdsi, = ax.plot([], [], label=r'相对驾驶安全指数 ($RDSI$)', color='blue', alpha=0.7)
    line_ttci, = ax.plot([], [], label=r'碰撞时间倒数 ($TTCI$)', color='red', alpha=0.7)
    line_rdsi1_star, = ax.plot([], [], label=r'$RDSI_1^*$', color='green', alpha=0.7, linestyle='--')
    line_rdsi2_star, = ax.plot([], [], label=r'$RDSI_2^*$', color='orange', alpha=0.7, linestyle='--')

    ax.set_xlabel('帧数', fontsize=8)
    ax.set_ylabel('指标值', fontsize=8)
    ax.set_title('相对驾驶安全指数 ($RDSI$)、碰撞时间倒数 ($TTCI$)、$RDSI_1^*$ 和 $RDSI_2^*$ 曲线', fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=8)

    def init():
        line_rdsi.set_data([], [])
        line_ttci.set_data([], [])
        line_rdsi1_star.set_data([], [])
        line_rdsi2_star.set_data([], [])
        return line_rdsi, line_ttci, line_rdsi1_star, line_rdsi2_star

    def update(frame):
        line_rdsi.set_data(time_points[:frame + 1], rdsi_values[:frame + 1])
        line_ttci.set_data(time_points[:frame + 1], ttci_values[:frame + 1])
        line_rdsi1_star.set_data(time_points[:frame + 1], rdsi1_star_values[:frame + 1])
        line_rdsi2_star.set_data(time_points[:frame + 1], rdsi2_star_values[:frame + 1])

        # 分别获取四条曲线的数据
        xdata_rdsi, ydata_rdsi = line_rdsi.get_data()
        xdata_ttci, ydata_ttci = line_ttci.get_data()
        xdata_rdsi1_star, ydata_rdsi1_star = line_rdsi1_star.get_data()
        xdata_rdsi2_star, ydata_rdsi2_star = line_rdsi2_star.get_data()

        # 合并所有曲线的x和y数据
        all_xdata = np.concatenate((xdata_rdsi, xdata_ttci, xdata_rdsi1_star, xdata_rdsi2_star))
        all_ydata = np.concatenate((ydata_rdsi, ydata_ttci, ydata_rdsi1_star, ydata_rdsi2_star))

        print("当前帧:", frame)
        print("x轴数据:", all_xdata)
        print("y轴数据:", all_ydata)

        # 检查x轴数据的最小值和最大值是否相等
        if np.min(all_xdata) == np.max(all_xdata):
            min_x = np.min(all_xdata)
            max_x = np.max(all_xdata)
            # 增加一个小的偏移量
            offset = 1 if max_x == min_x else 0.1 * (max_x - min_x)
            ax.set_xlim(min_x - offset, max_x + offset)
        else:
            ax.set_xlim(np.min(all_xdata), np.max(all_xdata))

        ax.set_ylim(np.min(all_ydata), np.max(all_ydata))

        ax.relim()
        ax.autoscale_view()
        return line_rdsi, line_ttci, line_rdsi1_star, line_rdsi2_star

    ani = FuncAnimation(fig, update, frames=len(time_points), init_func=init, interval=200, blit=True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    simulate_with_file_data()