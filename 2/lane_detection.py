import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def region_of_interest(img, vertices):
    """
    应用图像掩膜。
    只保留由 `vertices` 定义的多边形区域内的图像，其余部分置黑。
    """
    mask = np.zeros_like(img)   
    
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[0, 255, 0], thickness=8):
    """
    改进后的画线函数：
    1. 根据斜率区分左右车道。
    2. 过滤掉斜率过小的噪点（如水平线）。
    3. 对左右车道的点进行线性拟合，画出完整的车道线。
    """
    if lines is None:
        return

    # 存储左右车道的点
    left_x, left_y = [], []
    right_x, right_y = [], []

    # 图像尺寸
    height, width = img.shape[:2]
    # 设定车道线延伸的最高点（对应ROI的顶部比例）
    y_min = int(height * 0.6)
    y_max = height

    for line in lines:
        for x1, y1, x2, y2 in line:
            # 计算斜率
            if x2 == x1:
                continue # 防止除以0
            slope = (y2 - y1) / (x2 - x1)
            
            # 过滤掉斜率绝对值太小的线（近似水平的线通常不是车道线）
            if abs(slope) < 0.4:
                continue
            
            # 根据斜率正负区分左右车道
            # 在图像坐标系中，y向下增大。
            # 左车道：x增大时y减小（向上延伸），斜率 < 0
            # 右车道：x增大时y增大（向下延伸），斜率 > 0
            if slope < 0:
                left_x.extend([x1, x2])
                left_y.extend([y1, y2])
            else:
                right_x.extend([x1, x2])
                right_y.extend([y1, y2])

    # 内部函数：拟合并不画线
    def fit_and_draw(x_points, y_points):
        if len(x_points) >= 2:
            # 1次多项式拟合 (y = mx + b)
            poly = np.polyfit(x_points, y_points, 1)
            m = poly[0]
            b = poly[1]
            
            # 根据直线方程计算端点
            # x = (y - b) / m
            if m != 0:
                x1 = int((y_max - b) / m)
                x2 = int((y_min - b) / m)
                cv2.line(img, (x1, y_max), (x2, y_min), color, thickness)

    fit_and_draw(left_x, left_y)
    fit_and_draw(right_x, right_y)

def lane_detection_pipeline(image_path, output_path):
    # 1. 读取图像
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found.")
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}.")
        return
        
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 2. 灰度化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 3. 高斯模糊
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    
    # 4. Canny 边缘检测
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    
    # 5. 感兴趣区域 (ROI) 选择
    imshape = image.shape
    height = imshape[0]
    width = imshape[1]
    
    # 定义梯形掩膜顶点
    # 注意：如果是校园道路，可能需要根据实际拍摄角度调整这些比例
    vertices = np.array([[(0, height),
                          (width * 0.45, height * 0.6), 
                          (width * 0.55, height * 0.6), 
                          (width, height)]], dtype=np.int32)
    
    masked_edges = region_of_interest(edges, vertices)
    
    # 6. 霍夫变换 (Hough Transform)
    rho = 1             # 距离分辨率 (像素)
    theta = np.pi/180   # 角度分辨率 (弧度)
    threshold = 15      # 最小投票数
    min_line_len = 40   # 最小线段长度
    max_line_gap = 20   # 允许的最大线段间隙
    
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), 
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    
    # 7. 在空白图像上绘制拟合后的车道线
    line_image = np.zeros((height, width, 3), dtype=np.uint8)
    draw_lines(line_image, lines)
    
    # 8. 将车道线叠加回原图
    result = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    
    # 保存结果
    cv2.imwrite(output_path, result)
    print(f"Result saved to {output_path}")
    
    # 可视化展示
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title('Lane Detection Result')
    plt.axis('off')
    
    plt.show()

    return result

# 算法分析 (已更新)
analysis = """
Algorithm Analysis:
1. **Grayscale Conversion**: Converts the color image to grayscale to reduce computational complexity.
2. **Gaussian Blur**: Applies a Gaussian filter to smooth the image and reduce noise.
3. **Canny Edge Detection**: Detects edges in the image by looking for strong gradients.
4. **Region of Interest (ROI)**: A mask is applied to focus only on the road area, eliminating irrelevant edges from the environment.
5. **Hough Transform**: The Probabilistic Hough Transform is used to detect line segments in the edge-detected image.
6. **Line Filtering and Extrapolation**: (Improved Step) The detected line segments are separated into left and right lanes based on their slope. Segments with near-horizontal slopes are filtered out as noise. Linear regression (polyfit) is then used to fit a single solid line for each lane, which is extrapolated to cover the full region of interest, providing a clean visualization of the lane boundaries.
"""

if __name__ == "__main__":
    print(analysis)
    # 请将此处修改为你拍摄的校园道路图片文件名
    input_image = 'campus_road.jpg' 
    output_image = 'campus_road_result.jpg'
    
    # 如果没有图片，代码会提示错误。# filepath: c:\Users\bond\Desktop\cv\ex\2\lane_detection.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def region_of_interest(img, vertices):
    """
    应用图像掩膜。
    只保留由 `vertices` 定义的多边形区域内的图像，其余部分置黑。
    """
    mask = np.zeros_like(img)   
    
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[0, 255, 0], thickness=8):
    """
    改进后的画线函数：
    1. 根据斜率区分左右车道。
    2. 过滤掉斜率过小的噪点（如水平线）。
    3. 对左右车道的点进行线性拟合，画出完整的车道线。
    """
    if lines is None:
        return

    # 存储左右车道的点
    left_x, left_y = [], []
    right_x, right_y = [], []

    # 图像尺寸
    height, width = img.shape[:2]
    # 设定车道线延伸的最高点（对应ROI的顶部比例）
    y_min = int(height * 0.6)
    y_max = height

    for line in lines:
        for x1, y1, x2, y2 in line:
            # 计算斜率
            if x2 == x1:
                continue # 防止除以0
            slope = (y2 - y1) / (x2 - x1)
            
            # 过滤掉斜率绝对值太小的线（近似水平的线通常不是车道线）
            if abs(slope) < 0.4:
                continue
            
            # 根据斜率正负区分左右车道
            # 在图像坐标系中，y向下增大。
            # 左车道：x增大时y减小（向上延伸），斜率 < 0
            # 右车道：x增大时y增大（向下延伸），斜率 > 0
            if slope < 0:
                left_x.extend([x1, x2])
                left_y.extend([y1, y2])
            else:
                right_x.extend([x1, x2])
                right_y.extend([y1, y2])

    # 内部函数：拟合并不画线
    def fit_and_draw(x_points, y_points):
        if len(x_points) >= 2:
            # 1次多项式拟合 (y = mx + b)
            poly = np.polyfit(x_points, y_points, 1)
            m = poly[0]
            b = poly[1]
            
            # 根据直线方程计算端点
            # x = (y - b) / m
            if m != 0:
                x1 = int((y_max - b) / m)
                x2 = int((y_min - b) / m)
                cv2.line(img, (x1, y_max), (x2, y_min), color, thickness)

    fit_and_draw(left_x, left_y)
    fit_and_draw(right_x, right_y)

def lane_detection_pipeline(image_path, output_path):
    # 1. 读取图像
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found.")
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}.")
        return
        
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 2. 灰度化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 3. 高斯模糊
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    
    # 4. Canny 边缘检测
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    
    # 5. 感兴趣区域 (ROI) 选择
    imshape = image.shape
    height = imshape[0]
    width = imshape[1]
    
    # 定义梯形掩膜顶点
    # 注意：如果是校园道路，可能需要根据实际拍摄角度调整这些比例
    vertices = np.array([[(0, height),
                          (width * 0.45, height * 0.6), 
                          (width * 0.55, height * 0.6), 
                          (width, height)]], dtype=np.int32)
    
    masked_edges = region_of_interest(edges, vertices)
    
    # 6. 霍夫变换 (Hough Transform)
    rho = 1             # 距离分辨率 (像素)
    theta = np.pi/180   # 角度分辨率 (弧度)
    threshold = 15      # 最小投票数
    min_line_len = 40   # 最小线段长度
    max_line_gap = 20   # 允许的最大线段间隙
    
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), 
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    
    # 7. 在空白图像上绘制拟合后的车道线
    line_image = np.zeros((height, width, 3), dtype=np.uint8)
    draw_lines(line_image, lines)
    
    # 8. 将车道线叠加回原图
    result = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    
    # 保存结果
    cv2.imwrite(output_path, result)
    print(f"Result saved to {output_path}")
    
    # 可视化展示
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title('Lane Detection Result')
    plt.axis('off')
    
    plt.show()

    return result

# 算法分析 (已更新)
analysis = """
Algorithm Analysis:
1. **Grayscale Conversion**: Converts the color image to grayscale to reduce computational complexity.
2. **Gaussian Blur**: Applies a Gaussian filter to smooth the image and reduce noise.
3. **Canny Edge Detection**: Detects edges in the image by looking for strong gradients.
4. **Region of Interest (ROI)**: A mask is applied to focus only on the road area, eliminating irrelevant edges from the environment.
5. **Hough Transform**: The Probabilistic Hough Transform is used to detect line segments in the edge-detected image.
6. **Line Filtering and Extrapolation**: (Improved Step) The detected line segments are separated into left and right lanes based on their slope. Segments with near-horizontal slopes are filtered out as noise. Linear regression (polyfit) is then used to fit a single solid line for each lane, which is extrapolated to cover the full region of interest, providing a clean visualization of the lane boundaries.
"""

if __name__ == "__main__":
    print(analysis)
    input_image = 'road.png' 
    output_image = 'road_result.png'
    lane_detection_pipeline(input_image, output_image)