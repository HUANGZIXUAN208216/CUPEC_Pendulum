from ultralytics import YOLO

# 加载预训练模型
model = YOLO('yolov8n.pt')

# 训练
model.train(
    data="E:/CUPEC/images_and_labels/dataset.yaml",   # 你的数据集配置文件路径
    epochs=200,                                        # 训练轮数
    imgsz=640,                                         # 输入图片尺寸
    batch=4,                                           # 批大小
    device='cpu',                                      # 用CPU训练
    workers=2,                                         # 数据加载子进程数
    cache=False,                                       # 不缓存图片到内存
    plots=True,
    amp=False,
    project='E:/CUPEC/Training_Results',               # 结果保存的根目录
    name='Model_8',                                    # 本次实验名称
    exist_ok=True                                      # 同名文件夹直接覆盖
)
