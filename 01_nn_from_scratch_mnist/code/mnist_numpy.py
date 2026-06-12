import numpy as np

# ==========================================
# 1. 定义基础数学组件 (激活函数与导数)
# ==========================================

def relu(x):
    """ReLU 激活函数：把负数清零，正数原样输出"""
    return np.maximum(0, x)

def relu_deriv(x):
    """ReLU 的导数：大于0的导数为1，小于等于0的导数为0"""
    return (x > 0).astype(float)

def softmax(x):
    """Softmax 激活函数：将输出转化为概率分布"""
    # 减去最大值是为了防止指数运算溢出 (Numerical Stability)，数学本质不变
    exp_x = np.exp(x - np.max(x)) 
    return exp_x / np.sum(exp_x)

# ==========================================
# 2. 网络初始化 (分配矩阵维度)
# ==========================================
# 这里的维度完全按照手稿里的设计：784 -> 16 -> 15 -> 10

# 权重使用标准正态分布随机初始化并缩小尺度，偏置初始化为0
W0 = np.random.randn(16, 784) * 0.01  
b1 = np.zeros((16, 1))

W1 = np.random.randn(15, 16) * 0.01
b2 = np.zeros((15, 1))

W2 = np.random.randn(10, 15) * 0.01
b3 = np.zeros((10, 1))

# ==========================================
# 3. 核心训练步骤：前向传播 + 反向传播
# ==========================================

def train_step(X, Y_onehot, learning_rate=0.01):
    """
    X: 维度 (784, 1) 的列向量，代表一张图片
    Y_onehot: 维度 (10, 1) 的列向量，真实的 One-Hot 标签
    """
    global W0, b1, W1, b2, W2, b3
    
    # ---------------------------------------
    # 前向传播 (Forward Propagation)
    # ---------------------------------------
    # 隐藏层 1
    h1 = np.dot(W0, X) + b1            # (16, 784) x (784, 1) -> (16, 1)
    a1 = relu(h1)                      # (16, 1)
    
    # 隐藏层 2
    h2 = np.dot(W1, a1) + b2           # (15, 16) x (16, 1) -> (15, 1)
    a2 = relu(h2)                      # (15, 1)
    
    # 输出层
    h3 = np.dot(W2, a2) + b3           # (10, 15) x (15, 1) -> (10, 1)
    p = softmax(h3)                    # (10, 1)
    
    # 计算交叉熵 Loss (仅用于监控，不参与反向传播计算)
    # 加 1e-8 是防止 np.log(0) 报错
    loss = -np.sum(Y_onehot * np.log(p + 1e-8)) 
    
    # ---------------------------------------
    # 反向传播 (Backward Propagation) - 灵魂所在！
    # ---------------------------------------
    
    # 1. 输出层梯度
    # 这里就是你顿悟的极简减法！
    g3 = p - Y_onehot                              # (10, 1)
    dW2 = np.dot(g3, a2.T)                         # (10, 1) x (1, 15) -> (10, 15)
    db3 = g3                                       # (10, 1)
    
    # 2. 隐藏层 2 梯度
    # 注意看 W2.T 转置的完美咬合，以及 relu_deriv 的逐元素相乘 (*)
    g2 = np.dot(W2.T, g3) * relu_deriv(h2)         # (15, 10) x (10, 1) -> (15, 1)
    dW1 = np.dot(g2, a1.T)                         # (15, 1) x (1, 16) -> (15, 16)
    db2 = g2                                       # (15, 1)
    
    # 3. 隐藏层 1 梯度
    g1 = np.dot(W1.T, g2) * relu_deriv(h1)         # (16, 15) x (15, 1) -> (16, 1)
    dW0 = np.dot(g1, X.T)                          # (16, 1) x (1, 784) -> (16, 784)
    db1 = g1                                       # (16, 1)
    
    # ---------------------------------------
    # 参数更新 (Gradient Descent)
    # ---------------------------------------
    W2 = W2 - learning_rate * dW2
    b3 = b3 - learning_rate * db3
    
    W1 = W1 - learning_rate * dW1
    b2 = b2 - learning_rate * db2
    
    W0 = W0 - learning_rate * dW0
    b1 = b1 - learning_rate * db1
    
    return loss, p

# ==========================================
# 4. 来个模拟测试！
# ==========================================
if __name__ == "__main__":
    # 捏造一张假的输入图片 (784维向量) 和一个假标签 (假设正确数字是3)
    dummy_X = np.random.rand(784, 1)
    
    dummy_Y = np.zeros((10, 1))
    dummy_Y[3] = 1.0  # One-Hot 编码：数字3的位置设为1
    
    print("=== 开始训练前的预测概率 ===")
    _, initial_p = train_step(dummy_X, dummy_Y, learning_rate=0.0)
    print(f"预测数字 '3' 的概率: {initial_p[3][0]*100:.4f}%\n")
    
    print("=== 开始狠狠地训练 100 次 ===")
    for epoch in range(100):
        # 让网络反复学习这一张图
        loss, p = train_step(dummy_X, dummy_Y, learning_rate=0.05)
        if epoch % 20 == 0:
            print(f"第 {epoch} 次更新 -> Loss: {loss:.4f}, 数字3的概率: {p[3][0]*100:.2f}%")
            
    print("\n=== 100次训练后的最终预测 ===")
    print(f"预测数字 '3' 的概率飙升至: {p[3][0]*100:.4f}%！")