import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from datasets import load_dataset
import numpy as np
import time

# ==========================================
# 1. 加载并打包数据 (引入测试集)
# ==========================================
print("正在加载数据集...")
ds = load_dataset("ylecun/mnist")

# 【PyTorch 特色 1：张量形状】
# NumPy 版中用了 (784, 60000) 这种 (特征数, 样本数) 的形状。
# PyTorch 默认要求 (样本数, 特征数)，即 (60000, 784)，所以这里不需要 .T 转置了。

# --- 准备训练集 (日常作业，60000张) ---
X_train_np = np.array([np.array(img).flatten() for img in ds['train']['image']])
X_train = torch.tensor(X_train_np, dtype=torch.float32) / 255.0

# 【PyTorch 特色 2：告别 One-Hot】
# PyTorch 算交叉熵时，底层会自动处理类别标签，不需要手动转成 [0,1,0,0...] 的 One-Hot 格式，直接传数字即可！
Y_train = torch.tensor(ds['train']['label'], dtype=torch.long)

# --- 准备测试集 (期末考试，10000张，完全不参与训练) ---
print("正在准备期末考试卷(测试集)...")
X_test_np = np.array([np.array(img).flatten() for img in ds['test']['image']])
X_test = torch.tensor(X_test_np, dtype=torch.float32) / 255.0
Y_test = torch.tensor(ds['test']['label'], dtype=torch.long)

# 【PyTorch 特色 3：DataLoader 数据管家】
# 替代原本手写的 permutation 和 for 循环切片，它会自动帮我们打乱数据并切分成 Batch。
batch_size = 64
train_dataset = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# ==========================================
# 2. 神经网络组件
# ==========================================
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        # PyTorch 把权重初始化（W 和 b）自动藏在了 nn.Linear 里
        # 结构与手写的一致：784 -> 16 -> 15 -> 10
        self.net = nn.Sequential(
            nn.Linear(784, 16),
            nn.ReLU(),
            nn.Linear(16, 15),
            nn.ReLU(),
            nn.Linear(15, 10) 
            # 【注意】PyTorch 的 CrossEntropyLoss 自带 Softmax 效果，
            # 所以这里最后一层不需要单独加 Softmax 层！输出的是纯净的“对数几率 (Logits)”。
        )

    def forward(self, x):
        return self.net(x)

# 实例化网络
model = MNISTNet()


# ==========================================
# 3. 核心算法 (定义损失函数、优化器和学习率策略)
# ==========================================
epochs = 5
learning_rate = 0.5
lr_decay = 0.8

# 相当于之前手动算的交叉熵误差 (内部包含 Softmax)
criterion = nn.CrossEntropyLoss()

# 优化器：负责更新 W 和 b (相当于之前写的 W -= learning_rate * dW)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 学习率调度器：相当于之前写的 learning_rate = learning_rate * lr_decay
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)


# ==========================================
# 4. 训练与测试循环
# ==========================================
def evaluate(X, Y):
    """考试机制：只做题不看答案，不更新权重。测试真正的泛化能力。"""
    model.eval()               # 告诉网络：现在是考试状态（关掉训练特有的机制）
    with torch.no_grad():      # 告诉 PyTorch：不用在后台记账算梯度了，省点内存和算力
        logits = model(X)      # 一次性让全量数据穿过网络
        predictions = torch.argmax(logits, dim=1) # 选出概率最大的数字
        accuracy = (predictions == Y).float().mean() * 100
    return accuracy.item()

print(f"\n=== 开始 PyTorch 硬核实战 (Batch Size: {batch_size}) ===")
for epoch in range(epochs):
    start_time = time.time()
    
    # --- 阶段 1：在训练集上苦练 (做作业) ---
    model.train()              # 告诉网络：现在是做作业状态，准备学习
    for X_batch, Y_batch in train_loader:
        
        # 1. 归零脑回路里的残余记忆 (清空上一轮的梯度)
        optimizer.zero_grad()
        
        # 2. 前向传播：看题、给出预测
        outputs = model(X_batch)
        
        # 3. 计算误差：对比标准答案
        loss = criterion(outputs, Y_batch)
        
        # 4. 反向传播：纠正脑回路 (【核心】PyTorch 自动算 dW 和 db！)
        loss.backward()
        
        # 5. 参数更新：真正把知识记在脑子里
        optimizer.step()
        
    # --- 阶段 2：在测试集上考试 (摸底测验) ---
    train_acc = evaluate(X_train, Y_train) # 看看作业做对多少
    test_acc = evaluate(X_test, Y_test)    # 看看真正考试拿多少分
    
    epoch_time = time.time() - start_time
    print(f"Epoch {epoch+1}/{epochs} | 耗时: {epoch_time:.2f}秒 | 作业准确率(Train): {train_acc:.2f}% | 真实考试成绩(Test): {test_acc:.2f}%")
    
    # 【核心魔法】：跑完一轮，步子缩小到原来的 80%
    scheduler.step()
    