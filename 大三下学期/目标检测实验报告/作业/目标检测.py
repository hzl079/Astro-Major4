import cv2
import numpy as np
import matplotlib.pyplot as plt

def manual_hog(img):
    # 1. 预处理：归一化+Gamma校正
    img = np.sqrt(img / float(np.max(img))) # Gamma校正
    height, width = img.shape

    # 2. 计算梯度 (使用 Sobel 算子)
    # gx 计算水平梯度，gy 计算垂直梯度
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=1)# cv2.CV_64F 表示输出图像的深度为 64 位浮点数，ksize=1 表示使用 1x1 的 Sobel 核
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=1)

    # 计算幅值 和 方向
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)# cv2.cartToPolar 将 Cartesian 坐标 (gx, gy) 转换为极坐标 (mag, angle)，angleInDegrees=True 表示角度以度为单位
    angle = angle % 180 # 简化为 0-180 度

    # 3. 计算 Cell 直方图 (以 4x4 为一个 Cell)
    cell_size = 4
    bin_size = 9 # 9 个方向，即9维
    angle_unit = 180 / bin_size # 即20度一个 bin
    
    cell_gradient_vector = np.zeros((height // cell_size, width // cell_size, bin_size))

    for i in range(cell_gradient_vector.shape[0]):
        for j in range(cell_gradient_vector.shape[1]):
            # 提取当前 cell 的幅值和角度
            cell_mag = mag[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]# 用切片提出一个矩阵
            cell_angle = angle[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
            
            # 简单的加权投票：根据角度将幅值加进对应的 bin
            for r in range(cell_size):
                for c in range(cell_size):
                    bin_idx = int(cell_angle[r, c] / angle_unit) % bin_size
                    cell_gradient_vector[i, j, bin_idx] += cell_mag[r, c]
    #这里没有做插值
    # 4. Block 归一化 (以 2x2 个 Cell 为一个 Block)
    # 这一步是为了消除局部光照差异
    hog_vector = []
    for i in range(cell_gradient_vector.shape[0] - 1):# -1因为要取 2x2 的 cells，所以最后一个 cell 不能取
        for j in range(cell_gradient_vector.shape[1] - 1):
            # 取 2x2 的 cells
            block_vector = cell_gradient_vector[i:i+2, j:j+2].flatten()
            # L2 归一化
            norm = np.sqrt(np.sum(block_vector**2) + 1e-5) # 加一个小常数防止除零
            hog_vector.extend(block_vector / norm) # extend是将 block_vector 中的元素逐个添加到 hog_vector 中，而不是添加整个 block_vector 作为一个元素

    return np.array(hog_vector), cell_gradient_vector

def draw_hog_descriptors(img_shape, cell_hist_matrix, cell_size=4, bin_size=9, magnification=1):
    height, width = img_shape
    # 创建一个纯黑的空白图像用于绘图
    hog_image = np.zeros((height, width), dtype=np.float32)
    
    # 获取 Cell 矩阵的维度
    n_cells_h, n_cells_w, _ = cell_hist_matrix.shape
    
    # 计算线段的最大长度（ Cell 宽度的一半）
    max_len = cell_size / 2.0
    # 每个 bin 代表的角度角度
    angle_unit = np.pi / bin_size # 弧度
    
    # 对整个图像的 Cell 直方图进行归一化，以便映射亮度 (可选，增强对比度)
    cell_hist_matrix = cell_hist_matrix / (np.max(cell_hist_matrix) + 1e-5)

    for i in range(n_cells_h):
        for j in range(n_cells_w):
            # 获取当前 Cell 中心像素坐标
            cx = (j + 0.5) * cell_size
            cy = (i + 0.5) * cell_size
            
            # 遍历该 Cell 内的 9 个方向 bin
            for bin_idx in range(bin_size):
                # 该方向的幅值强度 (已归一化到 0-1 之间)
                strength = cell_hist_matrix[i, j, bin_idx]
                
                # 如果强度太小，不绘制
                if strength < 0.05: continue 
                
                # 计算该 bin 代表的真实角度中心 (弧度)
                angle = bin_idx * angle_unit + angle_unit / 2.0
                
                # 计算线段的两个端点坐标 (x, y)
                # 方向：角度从 x 轴正方向逆时针计算
                # 这里由于图像坐标系 y 轴向下，所以我们要相应调整
                dx = max_len * np.cos(angle)
                dy = max_len * np.sin(angle)
                
                x1 = int(cx - dx * magnification)
                y1 = int(cy + dy * magnification) # y 轴取反
                x2 = int(cx + dx * magnification)
                y2 = int(cy - dy * magnification) # y 轴取反

                # 绘制线段：亮度对应幅值强度
                # 我们需要在边界检查，防止端点超出图像
                cv2.line(hog_image, (x1, y1), (x2, y2), float(strength), thickness=1)

    return hog_image

# --- hog 特征提取 ---
img = cv2.imread(r'.\jordan.jpeg', 0) # 读取为灰度图
img_resized = cv2.resize(img, (512, 512)) # 按照要求处理（512x512）
features, cell_gradient_vector = manual_hog(img_resized)
print(f"HOG 特征向量维度: {features.shape}")

hog_v_image = draw_hog_descriptors(img_resized.shape, cell_gradient_vector)
fig,(ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(img_resized,cmap='gray')
ax1.set_title('Original Gray Image')
ax2.axis('off')
ax2.imshow(hog_v_image, cmap=plt.cm.gray)
ax2.set_title('HOG Visualization')

plt.show()