import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 设置 matplotlib 参数以支持绘图
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def print_model_structure(model):
    """
    【新增功能】打印模型结构信息
    用途：用于实验报告中的'网络架构分析'部分，展示你对模型深度的了解。
    """
    print("\n" + "="*30)
    print("【模型结构分析】")
    print("="*30)
    # 打印模型详细层级信息
    model.info(detailed=True)
    print(f"模型类型: YOLOv8n")
    print(f"检测类别数: {len(model.names)}")
    print("-" * 30)

def plot_loss_curve(result_dir):
    """
    【新增功能】读取训练日志并绘制 Loss 曲线
    用途：用于实验报告中的'训练过程分析'，证明模型确实在收敛。
    """
    csv_path = os.path.join(result_dir, 'results.csv')
    if not os.path.exists(csv_path):
        print(f"未找到训练日志: {csv_path}")
        return

    print(f"\n正在绘制训练损失曲线，数据来源: {csv_path}")
    
    # 读取 CSV 数据 (去除列名空格)
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    
    plt.figure(figsize=(12, 5))
    
    # 绘制 Box Loss (定位损失) - 衡量框画得准不准
    plt.subplot(1, 2, 1)
    plt.plot(df['epoch'], df['train/box_loss'], label='Train Box Loss', marker='o', color='blue')
    plt.title('Box Loss (Localization)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 绘制 Class Loss (分类损失) - 衡量是什么物体认得准不准
    plt.subplot(1, 2, 2)
    plt.plot(df['epoch'], df['train/cls_loss'], label='Train Cls Loss', marker='o', color='orange')
    plt.title('Cls Loss (Classification)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    save_path = os.path.join(result_dir, 'loss_curve.png')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"损失曲线图已保存至: {save_path}")

def manual_draw_detections(model, image_path, target_cls_id=1):
    """
    【新增功能】手动实现推理结果的解析和绘制
    用途：替代自动保存，展示对 CV 底层绘图和坐标处理的理解。
    """
    print(f"\n正在对 {image_path} 进行手动推理和绘制...")
    
    # 1. 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return

    # 2. 模型推理 (verbose=False 不打印默认日志)
    results = model(image_path, verbose=False)
    
    # 3. 解析结果 Tensor
    result = results[0]
    boxes = result.boxes.cpu().numpy() # 将 Tensor 转为 numpy 数组以便处理
    
    detect_count = 0
    
    for box in boxes:
        # 获取类别 ID
        cls_id = int(box.cls[0])
        
        # 仅处理目标类别 (例如自行车，COCO索引为1)
        if cls_id == target_cls_id:
            detect_count += 1
            
            # 获取坐标 (x1, y1, x2, y2)
            x1, y1, x2, y2 = box.xyxy[0].astype(int)
            
            # 获取置信度
            conf = box.conf[0]
            
            # 获取类别名称
            label_name = result.names[cls_id]
            
            # --- 手动绘制 (OpenCV 核心操作) ---
            # 画矩形框
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 准备标签文本
            label_text = f"{label_name} {conf:.2f}"
            
            # 计算文本大小以便绘制背景底色
            (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            
            # 画文字背景 (实心矩形)
            cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
            
            # 画文字
            cv2.putText(img, label_text, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
            print(f"检测到目标: {label_text} | 坐标: ({x1}, {y1}), ({x2}, {y2})")

    # 4. 保存结果
    output_path = "manual_result.jpg"
    cv2.imwrite(output_path, img)
    print(f"手动绘制的检测结果已保存至: {output_path}")
    print(f"共检测到 {detect_count} 个目标。")

def train_and_detect():
    # 检查 GPU
    if torch.cuda.is_available():
        device = 0
        print(f"已检测到 GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("未检测到 GPU，使用 CPU。")

    print("\n" + "="*30)
    print("开始训练过程 (基于 COCO128 数据集)")
    print("="*30)
    
    # 1. 加载模型
    model = YOLO('yolov8n.pt') 
    
    # 打印模型结构 (新增步骤)
    print_model_structure(model)
    
    # 2. 训练模型
    project_path = 'runs/train'
    exp_name = 'coco_bike_model'
    
    print("正在开始训练...")
    # exist_ok=True 允许覆盖旧的实验文件夹，方便调试
    model.train(data='coco128.yaml', epochs=3, imgsz=640, project=project_path, name=exp_name, device=device, exist_ok=True)
    
    print("\n训练完成！")
    
    # 3. 结果分析 (新增步骤)
    # 绘制 Loss 曲线
    result_dir = os.path.join(project_path, exp_name)
    try:
        plot_loss_curve(result_dir)
    except Exception as e:
        print(f"绘制曲线时出错: {e}")

    # 4. 验证/推理 (修改为手动实现)
    target_img = 'bike.jpg'
    if os.path.exists(target_img):
        manual_draw_detections(model, target_img, target_cls_id=1) # 1 是 bicycle
    else:
        print(f"\n提示: 当前目录下未找到 {target_img}，跳过推理演示。")

if __name__ == '__main__':
    train_and_detect()