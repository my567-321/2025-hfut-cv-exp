import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def manual_convolution(image, kernel):
    """
    Performs 2D convolution manually without using cv2.filter2D or similar.
    Strictly uses loops for calculation to avoid using function packages for the process.
    """
    h, w = image.shape
    k_size = kernel.shape[0]
    pad = k_size // 2
    
    # Create padded image
    padded_image = np.zeros((h + 2 * pad, w + 2 * pad), dtype=np.float32)
    padded_image[pad:pad+h, pad:pad+w] = image
    
    output = np.zeros_like(image, dtype=np.float32)
    
    # Convolution loop - Strictly manual
    for i in range(h):
        for j in range(w):
            # Calculate convolution for this pixel
            val = 0.0
            for ki in range(k_size):
                for kj in range(k_size):
                    # Image pixel value * Kernel value
                    val += padded_image[i + ki, j + kj] * kernel[ki, kj]
            output[i, j] = val
            
    return output

def manual_histogram(image):
    """
    Calculates color histogram manually.
    """
    hist_b = np.zeros(256, dtype=int)
    hist_g = np.zeros(256, dtype=int)
    hist_r = np.zeros(256, dtype=int)
    
    h, w, c = image.shape
    
    for i in range(h):
        for j in range(w):
            b, g, r = image[i, j]
            hist_b[b] += 1
            hist_g[g] += 1
            hist_r[r] += 1
            
    return hist_b, hist_g, hist_r

def manual_glcm_features(image):
    """
    Calculates GLCM and extracts texture features manually.
    Using distance=1, angle=0 (horizontal).
    """
    levels = 256
    glcm = np.zeros((levels, levels), dtype=int)
    h, w = image.shape
    
    # 1. Calculate GLCM (Horizontal: pixel(i,j) and pixel(i, j+1))
    for i in range(h):
        for j in range(w - 1):
            row = image[i, j]
            col = image[i, j+1]
            glcm[row, col] += 1
            
    # 2. Normalize GLCM
    total_pixels = 0
    for i in range(levels):
        for j in range(levels):
            total_pixels += glcm[i, j]
            
    glcm_norm = np.zeros((levels, levels), dtype=float)
    if total_pixels > 0:
        for i in range(levels):
            for j in range(levels):
                glcm_norm[i, j] = glcm[i, j] / total_pixels
        
    # 3. Extract features manually (Loops instead of vectorization)
    contrast = 0.0
    energy = 0.0
    homogeneity = 0.0
    
    # Variables for Correlation
    mean_i = 0.0
    mean_j = 0.0
    
    # First pass for basic features and means
    for i in range(levels):
        for j in range(levels):
            p = glcm_norm[i, j]
            if p == 0: continue
            
            # Contrast: sum P(i,j) * (i-j)^2
            contrast += p * ((i - j) ** 2)
            
            # Energy: sum P(i,j)^2
            energy += p ** 2
            
            # Homogeneity: sum P(i,j) / (1 + |i-j|)
            homogeneity += p / (1 + abs(i - j))
            
            # For Correlation means
            mean_i += i * p
            mean_j += j * p

    # Second pass for standard deviations and correlation
    std_i = 0.0
    std_j = 0.0
    covariance = 0.0
    
    for i in range(levels):
        for j in range(levels):
            p = glcm_norm[i, j]
            if p == 0: continue
            
            std_i += p * ((i - mean_i) ** 2)
            std_j += p * ((j - mean_j) ** 2)
            covariance += p * (i - mean_i) * (j - mean_j)
            
    std_i = std_i ** 0.5
    std_j = std_j ** 0.5
    
    if std_i * std_j == 0:
        correlation = 0
    else:
        correlation = covariance / (std_i * std_j)
        
    features = {
        'contrast': contrast,
        'energy': energy,
        'homogeneity': homogeneity,
        'correlation': correlation
    }
    return features

def main():
    # 1. Load Image
    # 请确保目录下有图片，或者修改此处为你的图片路径
    img_path = 'water_cup.png' 
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found. Please place an image named 'water_cup.png' in the directory or update the code.")
        # Create a dummy image for demonstration if file missing
        print("Creating a dummy image for demonstration...")
        img_color = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.circle(img_color, (50, 50), 30, (255, 255, 255), -1)
    else:
        img_color = cv2.imread(img_path)

    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    
    # 2. Sobel Operator & Given Kernel
    # Given Kernel (Vertical Edge / Horizontal Gradient)
    kernel_given = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ])
    
    print("Applying given convolution kernel...")
    filtered_given = manual_convolution(img_gray, kernel_given)
    
    # Full Sobel Operator
    # Sobel X (Vertical Edge Detection) - Matches the given kernel structure
    sobel_x = kernel_given
    
    # Sobel Y (Horizontal Edge Detection) - Transpose of X
    sobel_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])
    
    print("Applying Sobel operator...")
    grad_x = manual_convolution(img_gray, sobel_x)
    grad_y = manual_convolution(img_gray, sobel_y)
    
    # Gradient magnitude
    sobel_magnitude = np.zeros_like(grad_x)
    h, w = grad_x.shape
    for i in range(h):
        for j in range(w):
            sobel_magnitude[i, j] = (grad_x[i, j]**2 + grad_y[i, j]**2) ** 0.5
    
    # Normalize for display (0-255)
    filtered_given_disp = np.clip(np.abs(filtered_given), 0, 255).astype(np.uint8)
    sobel_disp = np.clip(sobel_magnitude, 0, 255).astype(np.uint8)
    
    # 3. Color Histogram
    print("Calculating color histogram...")
    hist_b, hist_g, hist_r = manual_histogram(img_color)
    
    # 4. Texture Features
    print("Extracting texture features...")
    texture_features = manual_glcm_features(img_gray)
    print("Texture Features:", texture_features)
    
    # Save texture features
    np.save('texture_features.npy', texture_features)
    
    # Visualization
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(filtered_given_disp, cmap='gray')
    plt.title('Filtered by Given Kernel')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(sobel_disp, cmap='gray')
    plt.title('Sobel Operator Result')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.plot(hist_r, color='r', label='Red')
    plt.plot(hist_g, color='g', label='Green')
    plt.plot(hist_b, color='b', label='Blue')
    plt.title('Color Histogram')
    plt.legend()
    plt.xlim([0, 256])
    
    plt.subplot(2, 3, 5)
    # Format the text nicely
    feature_text = "\n".join([f"{k}: {v:.4f}" for k, v in texture_features.items()])
    plt.text(0.1, 0.5, feature_text, fontsize=12)
    plt.title('Texture Features')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('experiment_results.png')
    plt.show()
    
    print("Experiment completed. Results saved to 'experiment_results.png' and 'texture_features.npy'.")

if __name__ == "__main__":
    main()