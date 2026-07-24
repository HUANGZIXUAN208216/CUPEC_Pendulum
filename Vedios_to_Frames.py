import cv2
import os
import numpy as np

def save_frame_robust(frame, img_path):
    """
    强制标准化格式并直接写入二进制文件
    """
    # 1. 强制转换为标准 uint8 格式
    frame = np.clip(frame, 0, 255).astype(np.uint8)
    
    # 2. 将图像编码为 .jpg 格式的内存数据
    ret, img_encoded = cv2.imencode('.jpg', frame)
    
    if ret:
        # 3. 以二进制写入方式直接保存到硬盘
        with open(img_path, 'wb') as f:
            f.write(img_encoded.tobytes())
        return True
    else:
        return False

# 配置参数
video_path = "E:/CUPEC/training_vedios/20260531161256.avi"         # 视频文件路径
output_dir = "E:/CUPEC/outputframes/training_frames_5"             # 输出到 U 盘
frame_interval = 20                     # 抽帧间隔：每10帧保存一张

# --- 初始化 ---
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"❌ 视频打开失败，请检查路径：{video_path}")
    exit()

# 逐帧处理
frame_count = 1286
saved_count = 1286

print("开始处理视频...")
while True:
    ret, frame = cap.read()
    if not ret:
        print(f"视频处理完成！共成功保存 {saved_count} 张图片至：{output_dir}")
        break
    
    if frame_count % frame_interval == 0:
        # 生成文件名（如 frame_0001.jpg）
        img_name = f"frame_{str(saved_count).zfill(4)}.jpg"
        img_path = os.path.join(output_dir, img_name)
        
        # 调用终极保存函数
        success = save_frame_robust(frame, img_path)
        
        if success:
            print(f"已保存：{img_name}")
            saved_count += 1
        else:
            print(f"保存失败：{img_path}（图片编码异常）")
    
    frame_count += 1

# --- 释放资源 ---
cap.release()
cv2.destroyAllWindows()