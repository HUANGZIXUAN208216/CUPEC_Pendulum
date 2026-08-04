# app.py
import os
import glob
import cv2
import math
import base64
import csv
import uuid
import queue
import threading
import time
import numpy as np
from datetime import datetime
from io import BytesIO
from flask import Flask, request, render_template, send_file, jsonify
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
from ultralytics import YOLO

# 必须在使用 pyplot 前设置后端
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.analysis import (
    extract_keypoints_from_result,
    calc_angular_velocity_from_two_points,
    compute_full_lyapunov_spectrum
)

# ---------- 初始化 ----------
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading',
                    ping_timeout=600, ping_interval=30, max_http_buffer_size=1e8)

UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# 自动使用"最新一次训练"生成的 best.pt（按文件夹的修改时间选择）
TRAIN_RESULTS_ROOT = "C:/CUPEC_National_Contest/Model_Training/Results"
def find_latest_best_pt():
    candidates = glob.glob(os.path.join(TRAIN_RESULTS_ROOT, "*", "weights", "best.pt"))
    if candidates:
        return max(candidates, key=os.path.getmtime)
    return "models/best.pt"  # 兜底：没找到训练结果时用本地 models 目录

MODEL_PATH = find_latest_best_pt()
print(f"加载模型: {MODEL_PATH}")

ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

model = YOLO(MODEL_PATH)
class_names = {0: 'central_point', 1: 'key_point'}

client_state_lock = threading.Lock()
client_state = {}
client_threads = {}
client_vel_buffer = {}  # 摄像头实时角速度缓冲区

recent_results = []
recent_results_lock = threading.Lock()

def allowed_video_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS

def safe_emit(event, data, room):
    try:
        socketio.emit(event, data, room=room)
        return True
    except Exception as e:
        print(f"发送失败: {e}")
        return False

# 内存绘图函数：返回 Base64 图片字符串
def plot_divergence_to_base64(divergence_curve, lyap_value):
    t = np.array(divergence_curve['t'])
    ln_div = np.array(divergence_curve['ln_divergence'])
    if len(t) == 0:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, ln_div, 'b-', linewidth=1.2, alpha=0.7, label='ln(Divergence)')
    
    # 拟合区与 compute_lyapunov_exponent 一致：10%~50%
    fit_start = max(1, int(len(t) * 0.10))
    fit_end = max(fit_start + 3, int(len(t) * 0.50))
    fit_end = min(fit_end, len(t))
    x_fit = t[fit_start:fit_end]
    y_fit = ln_div[fit_start:fit_end]
    intercept = np.mean(y_fit) - lyap_value * np.mean(x_fit)
    fitted_line = lyap_value * t + intercept
    ax.plot(t, fitted_line, 'r--', linewidth=2, label=f'Fitting Line (λ = {lyap_value:.4f})')
    ax.axvspan(t[fit_start], t[min(fit_end, len(t)) - 1], alpha=0.1, color='yellow', label='Fitting Region')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('ln(Divergence)')
    ax.set_title(f'Largest Lyapunov Exponent: λ = {lyap_value:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return f'data:image/png;base64,{img_base64}'

# ---------- 路由 ----------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return {'success': False, 'message': '没有文件'}, 400
    file = request.files['video']
    if file.filename == '' or not allowed_video_file(file.filename):
        return {'success': False, 'message': '文件类型不支持'}, 400
    filename = secure_filename(file.filename)
    file.save(os.path.join(UPLOAD_FOLDER, filename))
    return {'success': True, 'filename': filename}

@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(RESULTS_FOLDER, filename)
    if not os.path.exists(file_path):
        return "文件不存在", 404
    return send_file(file_path, as_attachment=True)

@app.route('/api/recent')
def get_recent():
    with recent_results_lock:
        summaries = []
        for r in recent_results:
            summaries.append({
                'id': r['id'],
                'motor_speed': r['motor_speed'],
                'total_frames': r['total_frames'],
                'csv_download': r['csv_download'],
                'lyapunov_exponent': r.get('lyapunov_exponent'),
                'lyapunov_conclusion': r.get('lyapunov_conclusion'),
                'lyapunov_plot': r.get('lyapunov_plot'),
                'timestamp': r['timestamp']
            })
        return jsonify(summaries)

@app.route('/api/velocity_data/<analysis_id>')
def get_velocity_data(analysis_id):
    csv_path = os.path.join(RESULTS_FOLDER, f"{analysis_id}.csv")
    if not os.path.exists(csv_path):
        return jsonify({'error': '数据不存在'}), 404
    velocities = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            velocities.append(float(row['angular_velocity']))
    return jsonify({'velocities': velocities})

# ---------- 视频推理工作线程 ----------
def video_processing_worker(sid, filepath, motor_speed):
    analysis_id = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]
    
    cap = cv2.VideoCapture(filepath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 5

    frame_idx = 0
    csv_rows = []
    prev_pts = {}
    skip = 2

    frame_queue = queue.Queue(maxsize=100)
    socketio.emit('status', {'message': f'开始处理（转速: {motor_speed} rpm）...'}, room=sid)

    def sender_worker():
        while True:
            item = frame_queue.get()
            if item is None:
                break
            data = item['data']
            annotated_frame = item['frame']
            _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 40])
            img_b64 = base64.b64encode(buffer).decode('utf-8')
            data['image'] = img_b64
            if not safe_emit('frame_data', data, sid):
                break
            frame_queue.task_done()

    sender_thread = threading.Thread(target=sender_worker, daemon=True)
    sender_thread.start()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % (skip + 1) != 0:
                frame_idx += 1
                continue

            results = model(frame, verbose=False)[0]
            pts = extract_keypoints_from_result(results, class_names, conf_threshold=0.1)

            angle = 0.0
            angular_velocity = 0.0

            if 'central_point' in pts and 'key_point' in pts:
                cx, cy = pts['central_point']
                kx, ky = pts['key_point']
                angle = math.atan2(ky - cy, kx - cx)
                if angle < 0:
                    angle += 2 * math.pi

                if 'central_point' in prev_pts and 'key_point' in prev_pts:
                    angular_velocity = calc_angular_velocity_from_two_points(
                        prev_pts['central_point'], prev_pts['key_point'],
                        pts['central_point'], pts['key_point'],
                        fps / (skip + 1)   # 修正: 相邻处理帧实隔 (skip+1)/fps 秒
                    )

            annotated_frame = frame.copy()
            for name, (px, py) in pts.items():
                cv2.circle(annotated_frame, (int(px), int(py)), 6, (0, 255, 0), -1)
                cv2.putText(annotated_frame, name, (int(px) + 10, int(py) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f'Frame {frame_idx} | Motor: {motor_speed} rpm | AV: {angular_velocity:.3f} rad/s',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            frame_data = {
                'frame_index': frame_idx,
                'central_point': list(pts.get('central_point', (None, None))),
                'key_point': list(pts.get('key_point', (None, None))),
                'angle': angle,
                'angular_velocity': angular_velocity,
                'motor_speed': motor_speed,
                'analysis_id': analysis_id
            }
            frame_queue.put({'data': frame_data, 'frame': annotated_frame})

            central = pts.get('central_point', (None, None))
            key = pts.get('key_point', (None, None))
            csv_rows.append({
                'frame': frame_idx,
                'central_x': round(central[0], 2) if central[0] else '',
                'central_y': round(central[1], 2) if central[1] else '',
                'key_x': round(key[0], 2) if key[0] else '',
                'key_y': round(key[1], 2) if key[1] else '',
                'angle': round(angle, 4),
                'angular_velocity': round(angular_velocity, 4)
            })

            prev_pts = pts
            frame_idx += 1

        frame_queue.put(None)
        sender_thread.join()
        cap.release()

        # 生成 CSV
        csv_filename = f"{analysis_id}.csv"
        csv_path = os.path.join(RESULTS_FOLDER, csv_filename)
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'frame', 'central_x', 'central_y', 'key_x', 'key_y', 'angle', 'angular_velocity'
            ])
            writer.writeheader()
            writer.writerows(csv_rows)

        # ========== 计算李雅普诺夫指数并生成内存图片 ==========
        # 过滤掉检测失败的帧（角速度与角度同时为 0），避免污染相空间重构
        angular_velocities = [
            row['angular_velocity'] for row in csv_rows
            if not (row['angle'] == 0 and row['angular_velocity'] == 0)
        ]
        dt = (skip + 1) / fps if fps > 0 else 0.033   # 修正: 有效采样间隔 = (skip+1)/fps
        
        lyap_result = compute_full_lyapunov_spectrum(angular_velocities, dt=dt)
        
        # 在内存中生成发散曲线图的 Base64 编码
        lyap_plot_b64 = plot_divergence_to_base64(
            lyap_result['divergence_curve'],
            lyap_result['lyapunov_exponent']
        )
        
        # 发送李雅普诺夫结果（使用 Base64 图片，无文件依赖）
        socketio.emit('lyapunov_result', {
            'lyapunov_exponent': float(lyap_result['lyapunov_exponent']),
            'conclusion': str(lyap_result['conclusion']),
            'is_chaotic': bool(lyap_result['is_chaotic']),
            'plot_base64': lyap_plot_b64
        }, room=sid)

        # 可选：保存 PNG 备份到磁盘
        try:
            lyap_plot_filename = f"lyapunov_{analysis_id}.png"
            lyap_plot_path = os.path.join(RESULTS_FOLDER, lyap_plot_filename)
            if lyap_plot_b64:
                img_data = base64.b64decode(lyap_plot_b64.split(',')[1])
                with open(lyap_plot_path, 'wb') as f:
                    f.write(img_data)
        except:
            pass

        # 存储到最近结果列表
        result_entry = {
            'id': analysis_id,
            'motor_speed': motor_speed,
            'total_frames': frame_idx,
            'csv_download': f'/download/{csv_filename}',
            'lyapunov_exponent': float(lyap_result['lyapunov_exponent']),
            'lyapunov_conclusion': str(lyap_result['conclusion']),
            'lyapunov_plot': lyap_plot_b64,
            'timestamp': datetime.now().isoformat()
        }
        with recent_results_lock:
            recent_results.append(result_entry)
            if len(recent_results) > 2:
                oldest = recent_results.pop(0)
                old_csv = os.path.join(RESULTS_FOLDER, f"{oldest['id']}.csv")
                if os.path.exists(old_csv):
                    os.remove(old_csv)
                old_png = os.path.join(RESULTS_FOLDER, f"lyapunov_{oldest['id']}.png")
                if os.path.exists(old_png):
                    os.remove(old_png)

        socketio.emit('processing_done', {
            'analysis_id': analysis_id,
            'motor_speed': motor_speed,
            'total_frames': frame_idx,
            'csv_download': f'/download/{csv_filename}'
        }, room=sid)

        socketio.emit('recent_updated', {'count': len(recent_results)}, room=sid)

    except Exception as e:
        socketio.emit('error', {'message': f'处理出错: {str(e)}'}, room=sid)
    finally:
        with client_state_lock:
            client_threads.pop(sid, None)

# ---------- WebSocket 事件 ----------
@socketio.on('process_frame')
def handle_realtime_frame(data):
    """处理摄像头实时帧（主线程中轻量执行）"""
    sid = request.sid
    try:
        # 1. Base64 解码图像
        img_data = base64.b64decode(data['image'].split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return

        # 2. 增强对比度：CLAHE 自适应直方图均衡化
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l)
        lab_eq = cv2.merge([l_eq, a, b])
        frame = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

        # 4. YOLO 模型推理
        results = model(frame, verbose=False)[0]
        pts = extract_keypoints_from_result(results, class_names, conf_threshold=0.1)

        # 5. 计算角度和角速度
        angle = 0.0
        angular_velocity = 0.0

        if 'central_point' in pts and 'key_point' in pts:
            cx, cy = pts['central_point']
            kx, ky = pts['key_point']
            angle = math.atan2(ky - cy, kx - cx)
            if angle < 0:
                angle += 2 * math.pi

            with client_state_lock:
                prev_pts = client_state.get(sid, {})
                if 'central_point' in prev_pts and 'key_point' in prev_pts:
                    angular_velocity = calc_angular_velocity_from_two_points(
                        prev_pts['central_point'], prev_pts['key_point'],
                        pts['central_point'], pts['key_point'],
                        fps=5
                    )
                client_state[sid] = pts

                # 记录角速度用于摄像头停止后的李雅普诺夫计算
                if sid not in client_vel_buffer:
                    client_vel_buffer[sid] = []
                client_vel_buffer[sid].append(angular_velocity)

        # 6. 递增帧计数器（每个客户端独立）
        with client_state_lock:
            frame_count = client_state.get(f'{sid}_count', 0)
            client_state[f'{sid}_count'] = frame_count + 1

        # 7. 发送相点数据
        emit('phase_point', {
            'angle': angle,
            'angular_velocity': angular_velocity
        })

        # 8. 绘制标注帧并发送
        annotated_frame = frame.copy()
        for name, (px, py) in pts.items():
            cv2.circle(annotated_frame, (int(px), int(py)), 6, (0, 255, 0), -1)
            cv2.putText(annotated_frame, name, (int(px) + 10, int(py) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f'AV: {angular_velocity:.3f} rad/s',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        img_b64 = base64.b64encode(buffer).decode('utf-8')

        emit('frame_data', {
            'frame_index': frame_count,
            'image': img_b64,
            'angle': angle,
            'angular_velocity': angular_velocity
        })

    except Exception as e:
        print(f"实时帧处理错误: {e}")

@socketio.on('process_video')
def handle_uploaded_video(data):
    sid = request.sid
    filename = data.get('filename')
    motor_speed = data.get('motor_speed', 0)

    if not filename:
        emit('error', {'message': '缺少文件名'})
        return

    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(filepath):
        emit('error', {'message': '文件不存在'})
        return

    with client_state_lock:
        if sid in client_threads and client_threads[sid].is_alive():
            emit('error', {'message': '已有视频正在处理中'})
            return

    thread = threading.Thread(target=video_processing_worker, args=(sid, filepath, motor_speed), daemon=True)
    with client_state_lock:
        client_threads[sid] = thread
    thread.start()

@socketio.on('stop_camera')
def handle_stop_camera():
    """摄像头停止时，计算李雅普诺夫指数并推送结果"""
    sid = request.sid
    with client_state_lock:
        velocities = client_vel_buffer.pop(sid, [])
    
    if len(velocities) < 50:
        emit('error', {'message': '数据点太少，无法计算李雅普诺夫指数（至少需要50帧）'})
        return
    
    dt = 1.0 / 5.0   # 摄像头采样间隔 200ms = 5fps（前端 setInterval(200)）
    lyap_result = compute_full_lyapunov_spectrum(velocities, dt=dt)
    lyap_plot_b64 = plot_divergence_to_base64(
        lyap_result['divergence_curve'],
        lyap_result['lyapunov_exponent']
    )
    
    emit('lyapunov_result', {
        'lyapunov_exponent': float(lyap_result['lyapunov_exponent']),
        'conclusion': str(lyap_result['conclusion']),
        'is_chaotic': bool(lyap_result['is_chaotic']),
        'plot_base64': lyap_plot_b64
    })

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    with client_state_lock:
        client_state.pop(sid, None)
        client_state.pop(f'{sid}_count', None)
        client_vel_buffer.pop(sid, None)

if __name__ == '__main__':
    print(" 混沌摆分析平台：http://127.0.0.1:5000")
    socketio.run(app, debug=False, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)