import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

#CNN模型定义
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 卷积层 1: 输入 1通道, 输出 32通道, 核大小 3x3
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # 卷积层 2: 输入 32通道, 输出 64通道, 核大小 3x3
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # Dropout 防止过拟合
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        # 全连接层
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

# 2. 模型训练 (Training)
def train_model(save_path='mnist_cnn.pt'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 检查是否已有模型
    if os.path.exists(save_path):
        print(f"Loading existing model from {save_path}")
        model = Net().to(device)
        model.load_state_dict(torch.load(save_path, map_location=device))
        model.eval()
        return model

    print("Training new model on MNIST...")
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 下载 MNIST 数据集
    try:
        dataset1 = datasets.MNIST('./data', train=True, download=True, transform=transform)
    except:
        print("Error downloading MNIST. Please ensure internet connection or manual download.")
        return None

    train_loader = DataLoader(dataset1, batch_size=64, shuffle=True)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)

    epochs = 5 # 增加训练轮数以提高准确率
    model.train()
    
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch+1} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        scheduler.step()

    torch.save(model.state_dict(), save_path)
    print("Model saved.")
    model.eval()
    return model

# 3. 图像预处理与数字分割 (Preprocessing)
def preprocess_image(image_path):
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return []

    img = cv2.imread(image_path)
    # 转换为灰度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. 高斯模糊 (Gaussian Blur) - 关键步骤：去除背景纹理噪点
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2. 二值化 (Adaptive Thresholding)
    # 增大 C 值 (从 2 改为 10)，提高对比度要求，进一步过滤浅色背景纹理
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 45, 10)

    # 3. 形态学操作 (Morphological Operations) - 去除微小噪点
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # 查找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    digit_rects = []
    
    # 筛选轮廓 (根据面积和长宽比筛选出数字)
    img_h, img_w = gray.shape
    min_h = img_h * 0.05  
    max_h = img_h * 0.6   
    
    candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # 几何筛选
        if h > min_h and h < max_h:
            aspect_ratio = w / float(h)
            # 数字通常是瘦长的，或者稍微宽一点
            if 0.2 < aspect_ratio < 1.2: 
                candidates.append((x, y, w, h))

    # 4. 行对齐过滤 (Row Alignment Filtering) - 关键步骤：找到排列成一行的数字
    # 算法思路：找到包含最多矩形的“行”
    if not candidates:
        return []

    # 按 y 坐标排序
    candidates.sort(key=lambda r: r[1])
    
    best_row = []
    current_row = []
    
    # 简单的聚类：如果两个矩形的 y 坐标相差不大，且高度相近，视为同一行
    for i, rect in enumerate(candidates):
        if not current_row:
            current_row.append(rect)
            continue
            
        prev_rect = current_row[-1]
        y_diff = abs(rect[1] - prev_rect[1])
        h_diff = abs(rect[3] - prev_rect[3])
        
        # 允许 y 坐标偏差在 20% 高度以内，高度差异在 30% 以内
        if y_diff < prev_rect[3] * 0.5 and h_diff < prev_rect[3] * 0.5:
            current_row.append(rect)
        else:
            # 结算当前行
            if len(current_row) > len(best_row):
                best_row = current_row
            current_row = [rect]
            
    # 检查最后一行
    if len(current_row) > len(best_row):
        best_row = current_row
        
    digit_rects = best_row

    # 对最终选定的行按 x 坐标排序 (从左到右)
    digit_rects.sort(key=lambda x: x[0])
    
    # 提取 ROI 并预处理成 MNIST 格式 (28x28, 黑底白字)
    processed_digits = []
    debug_img = img.copy()
    
    for (x, y, w, h) in digit_rects:
        # 稍微扩大一点框
        pad = int(h * 0.1)
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = w + 2*pad
        h = h + 2*pad
        
        roi = thresh[y:y+h, x:x+w]
        
        if roi.size == 0: continue

        # 保持纵横比缩放
        h_roi, w_roi = roi.shape
        scale = 20.0 / max(h_roi, w_roi)
        new_w = int(w_roi * scale)
        new_h = int(h_roi * scale)
        resized = cv2.resize(roi, (new_w, new_h))
        
        # 创建 28x28 的黑色背景
        padded = np.zeros((28, 28), dtype=np.uint8)
        
        # 将数字居中
        start_x = (28 - new_w) // 2
        start_y = (28 - new_h) // 2
        padded[start_y:start_y+new_h, start_x:start_x+new_w] = resized
        
        # 归一化
        tensor_img = transforms.ToTensor()(padded)
        tensor_img = transforms.Normalize((0.1307,), (0.3081,))(tensor_img)
        
        processed_digits.append(tensor_img)
        cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # 保存调试图片看看切分效果
    cv2.imwrite('debug_contours.jpg', debug_img)
    
    return processed_digits

# ==========================================
# 5. 主程序 (Main)
# ==========================================
def main():
    # 1. 训练或加载模型
    model = train_model()
    if model is None: return

    # 2. 读取并处理图片
    image_filename = 'id_card.jpg' 
    print(f"Processing {image_filename}...")
    digits_tensors = preprocess_image(image_filename)
    
    if not digits_tensors:
        print("No digits found! Please check 'debug_contours.jpg' and adjust preprocessing parameters.")
        return

    print(f"Found {len(digits_tensors)} digits.")

    # 3. 识别
    result_str = ""
    print("Recognizing...")
    device = next(model.parameters()).device
    with torch.no_grad():
        for dt in digits_tensors:
            dt = dt.unsqueeze(0).to(device) # 增加 batch 维度
            output = model(dt)
            pred = output.argmax(dim=1, keepdim=True)
            result_str += str(pred.item())
            
    print(f"\n{'='*30}")
    print(f"识别结果 (Recognized ID): {result_str}")
    print(f"{'='*30}")
    print("Check 'debug_contours.jpg' to see the detected digit areas.")

if __name__ == '__main__':
    main()
