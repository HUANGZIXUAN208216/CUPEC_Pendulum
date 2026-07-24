from ultralytics import YOLO

# 加载你训练好的模型
model = YOLO('E:/CUPEC/Training_Results/Model_3/weights/best.pt')

# 检测图片（替换成你的测试图路径）
results = model("E:/CUPEC/TEST/frame_0206.jpg")

# 输出每个检测到的目标
for r in results:
    boxes = r.boxes
    if boxes is not None:
        for box in boxes:
            # 坐标信息
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            # 类别、置信度
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls]
            # 中心点
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            # 宽高
            w = x2 - x1
            h = y2 - y1

            print(f"类别: {class_name} | 置信度: {conf:.2%}")
            print(f"  中心点: ({cx:.1f}, {cy:.1f})")
            print(f"  框尺寸: {w:.1f} x {h:.1f}")
            print(f"  边界框: ({x1:.1f}, {y1:.1f}) -> ({x2:.1f}, {y2:.1f})")
            print()

# 保存带标注框的图片
results[0].save('E:/CUPEC/result_annotated_2.jpg')
print("✅ 已保存标注图片到 E:/CUPEC/result_annotated_2.jpg")