# utils/analysis.py
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def extract_keypoints_from_result(result, class_names, conf_threshold=0.5):
    """
    从 YOLO 单帧推理结果中提取关键点（检测框中心点坐标）。
    对难以检测的 key_point 使用更低的阈值。
    """
    pts = {}
    if result.boxes is None:
        return pts
    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        
        # 对 key_point（假设其 class_id 为 1）使用更低的阈值
        effective_threshold = conf_threshold
        if cls == 1:  # key_point
            effective_threshold = max(0.05, conf_threshold * 0.2)  # 最低不低于 0.05
        
        if conf < effective_threshold:
            continue
            
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        name = class_names.get(cls, 'unknown')
        pts[name] = (cx, cy)
    return pts

def calc_angular_velocity_from_two_points(p1_prev, p2_prev, p1_curr, p2_curr, fps):
    """
    通过两个关键点构成的向量，计算相邻帧间的角速度。
    返回角速度 (rad/s)
    """
    dt = 1.0 / fps if fps > 0 else 0.033
    v_prev = np.array(p2_prev) - np.array(p1_prev)
    v_curr = np.array(p2_curr) - np.array(p1_curr)
    angle_prev = math.atan2(v_prev[1], v_prev[0])
    angle_curr = math.atan2(v_curr[1], v_curr[0])
    delta = angle_curr - angle_prev
    if delta > math.pi:
        delta -= 2 * math.pi
    elif delta < -math.pi:
        delta += 2 * math.pi
    return delta / dt

def compute_lyapunov_exponent(time_series, dt=1.0, embedding_dim=3, delay=1, return_divergence=False):
    """
    使用 Rosenstein 算法计算最大李雅普诺夫指数。
    
    参数:
        time_series: list or array，角速度时间序列
        dt: 采样间隔（1/fps）
        embedding_dim: 嵌入维度，默认 3
        delay: 延迟步数，默认 1
        return_divergence: 是否返回发散曲线的完整数据（用于绘图）
    
    返回:
        如果 return_divergence=False: float（最大李雅普诺夫指数）
        如果 return_divergence=True: tuple (lyap_exponent, divergence_curve)
            - lyap_exponent: float
            - divergence_curve: dict {'t': [...], 'ln_divergence': [...]}
    """
    import numpy as np
    
    data = np.array(time_series)
    n = len(data)
    
    # 去掉前 20% 的瞬态数据
    data = data[int(0.2 * n):]
    n = len(data)
    
    # 相空间重构
    m = embedding_dim
    tau = delay
    N = n - (m - 1) * tau
    
    if N < 10:
        if return_divergence:
            return 0.0, {'t': [], 'ln_divergence': []}
        return 0.0
    
    # 构造嵌入向量
    vectors = np.zeros((N, m))
    for i in range(N):
        for j in range(m):
            vectors[i, j] = data[i + j * tau]
    
    # Rosenstein 算法
    divergence = []
    max_t = min(50, N // 2)
    
    for t in range(1, max_t):
        dist_sum = 0.0
        count = 0
        
        for i in range(N - t):
            j = i + t
            if j < N:
                dist = np.linalg.norm(vectors[i] - vectors[j])
                if dist > 0:
                    dist_sum += np.log(dist)
                    count += 1
        
        if count > 0:
            divergence.append(dist_sum / count)
    
    if len(divergence) < 2:
        if return_divergence:
            return 0.0, {'t': [], 'ln_divergence': []}
        return 0.0
    
    # 线性拟合
    x = np.arange(len(divergence)) * dt
    y = np.array(divergence)
    
    fit_len = max(5, len(x) // 2)
    x_fit = x[:fit_len]
    y_fit = y[:fit_len]
    
    A = np.vstack([x_fit, np.ones(len(x_fit))]).T
    slope, _ = np.linalg.lstsq(A, y_fit, rcond=None)[0]
    
    if return_divergence:
        return slope, {'t': list(x), 'ln_divergence': list(y)}
    return slope


def compute_full_lyapunov_spectrum(time_series, dt=1.0, embedding_dim=3, delay=1):
    """
    计算李雅普诺夫指数谱的简化版本。
    返回最大指数、判断结论、以及发散曲线数据。
    """
    lyap, div_curve = compute_lyapunov_exponent(
        time_series, dt, embedding_dim, delay, return_divergence=True
    )
    
    if lyap > 0.01:
        conclusion = "系统存在混沌现象（λ > 0）"
    elif lyap < -0.01:
        conclusion = "系统处于周期性运动（λ < 0）"
    else:
        conclusion = "系统处于临界状态（λ ≈ 0）"
    
    return {
        'lyapunov_exponent': round(lyap, 6),
        'conclusion': conclusion,
        'is_chaotic': lyap > 0.01,
        'divergence_curve': div_curve
    }


def plot_lyapunov_divergence(divergence_curve, lyap_value, output_path=None):
    """
    绘制距离对数与时间的关系图（发散曲线拟合图）。
    
    参数:
        divergence_curve: dict {'t': [...], 'ln_divergence': [...]}
        lyap_value: 拟合得到的李雅普诺夫指数
        output_path: 保存路径，为 None 则直接显示
    """
    import numpy as np
    
    t = np.array(divergence_curve['t'])
    ln_div = np.array(divergence_curve['ln_divergence'])
    
    if len(t) == 0:
        return
    
    plt.figure(figsize=(10, 5))
    
    # 原始发散曲线
    plt.plot(t, ln_div, 'b-', linewidth=1.2, alpha=0.7, label='ln(发散距离)')
    
    # 拟合直线
    fit_len = max(5, len(t) // 3)
    t_fit = t[:fit_len]
    fitted_line = lyap_value * t + (ln_div[0] - lyap_value * t[0])
    plt.plot(t, fitted_line, 'r--', linewidth=2, label=f'拟合直线 (λ = {lyap_value:.4f})')
    
    # 标注拟合区域
    plt.axvspan(t[0], t[fit_len - 1], alpha=0.1, color='yellow', label='拟合区域')
    
    plt.xlabel('时间 (s)')
    plt.ylabel('ln(发散距离)')
    plt.title(f'最大李雅普诺夫指数: λ = {lyap_value:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def generate_phase_portrait(angles, angular_velocities, output_path=None, title="相图 (角度-角速度)"):
    """绘制真正的相空间轨迹（角度 vs 角速度）"""
    # 角度解缠绕（unwrap）：消除 0/2π 接缝，使混沌吸引子相图连续、不被切分
    angles = np.unwrap(np.asarray(angles, dtype=float))
    angular_velocities = np.asarray(angular_velocities, dtype=float)
    plt.figure(figsize=(8, 6))
    plt.plot(angles, angular_velocities, linewidth=0.8, color='darkblue', alpha=0.7)
    plt.scatter(angles, angular_velocities, s=5, color='red', alpha=0.3)
    plt.xlabel('角度 (rad)')
    plt.ylabel('角速度 (rad/s)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def generate_time_plot(frame_indices, angular_velocities, output_path=None, title="角速度-时间图"):
    """绘制角速度随时间变化的曲线"""
    plt.figure(figsize=(10, 5))
    plt.plot(frame_indices, angular_velocities, color='blue', linewidth=1.5)
    plt.xlabel('帧索引')
    plt.ylabel('角速度 (rad/s)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()