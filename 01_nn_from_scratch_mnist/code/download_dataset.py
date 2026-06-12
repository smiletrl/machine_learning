from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt

# 1. 下载并缓存数据集
print("正在从 Hugging Face 下载 MNIST 数据集...")
# 从 Hugging Face 下载数据集。如果是首次下载，会自动缓存在本地的 `~/.cache/huggingface/datasets` 目录下。
ds = load_dataset("ylecun/mnist")

# 2. 保存到本地
print("正在保存数据集到本地 ./data 目录...")
# 将数据集保存到本地磁盘，方便后续直接离线加载
ds.save_to_disk("./data")
print("数据集准备完毕！\n")

# 3. 提取前两张图片和对应的标签
item1 = ds['train'][0]
item2 = ds['train'][1]

img1, label1 = item1['image'], item1['label']
img2, label2 = item2['image'], item2['label']

# 将图片对象转换为 28x28 的二维 NumPy 数组
img_array_2d = np.array(img1)

# 查看转换后的结果
print("--- 第一张图片的数据结构说明 ---")
print(f"真实标签: {label1}")
print(f"二维数组维度: {img_array_2d.shape}")

# 按照 28x28 格式完美打印原始像素值
print("\n--- 28x28 原始像素矩阵实况 (0=纯黑, 255=纯白) ---")
for row in img_array_2d:
    # 为了在控制台对齐，每个数字占用 3 个字符宽度，中间加一个空格
    row_str = " ".join([f"{val:3}" for val in row])
    print(row_str)

# 将二维数组展平为一维数组 (长度为 28 * 28 = 784)
# 这正是我们神经网络输入层所需的 784 维向量 x
pixel_values = img_array_2d.flatten()
print(f"\n展平后的一维数组维度: {pixel_values.shape}")

# 数据归一化 (Normalization)
# 将整数像素值转换为浮点数，并从 [0, 255] 缩放到 [0.0, 1.0] 的区间
# 这一步对于神经网络的稳定训练（防止梯度爆炸/消失）至关重要
x_input = pixel_values.astype('float32') / 255.0

# 按照 28x28 格式完美打印归一化后的像素值
print("\n--- 28x28 归一化后像素矩阵实况 (0.00=纯黑, 1.00=纯白) ---")
x_input_2d = x_input.reshape(28, 28) # 为了打印，将一维数组重新变回 28x28
for row in x_input_2d:
    # 使用 4.2f 格式化，保留两位小数（如 0.00, 1.00），占用 4 个字符宽度对齐
    row_str = " ".join([f"{val:4.2f}" for val in row])
    print(row_str)

# --- 以上简单科普，以第一张图为例说明数据结构 ---
# --- 以下展示数据集中前两张图片 ---

# 4. 可视化展示 (展示第一张和第二张图片)
print("\n正在弹窗展示前两张图片，请查看...")

# 创建一个 1 行 2 列的画布
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

# 在第一个子图上绘制第一张图片
axes[0].imshow(np.array(img1), cmap='gray')  # cmap='gray' 表示使用灰度色彩映射
axes[0].set_title(f"Image 1 - Label: {label1}")
# axes[0].axis('off')  # 隐藏坐标轴

# 在第二个子图上绘制第二张图片
axes[1].imshow(np.array(img2), cmap='gray')
axes[1].set_title(f"Image 2 - Label: {label2}")
# axes[1].axis('off')

# 调整布局并显示
plt.tight_layout()
plt.show()
