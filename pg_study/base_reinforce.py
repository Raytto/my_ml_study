import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

# 定义策略网络
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)   # 隐藏层
        self.fc2 = nn.Linear(128, output_dim)  # 输出动作概率
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.to(self.device)
        
    def forward(self, x):
        x = x.to(self.device)  # Ensure input tensor is on the correct device
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=-1)
        return x

# 计算折扣奖励
def compute_returns(rewards, gamma):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns

# 设置环境和超参数
env = gym.make('CartPole-v1')
input_dim = env.observation_space.shape[0]  # 输入维度
output_dim = env.action_space.n             # 输出维度
policy_net = PolicyNetwork(input_dim, output_dim)
optimizer = optim.Adam(policy_net.parameters(), lr=0.01)
gamma = 0.99  # 折扣因子

# 开始训练
num_episodes = 1
for episode in range(num_episodes):
    # state = torch.tensor(env.reset()[0], dtype=torch.float32).to(policy_net.device)
    log_probs = []
    rewards = []
    state = env.reset()[0]
    # 采样轨迹
    done = False
    while not done:
        
        # state = torch.tensor(np.array(state), dtype=torch.float32)  # Ensure state tensor is on the correct device
        probs = policy_net(state)
        m = Categorical(probs) # 创建一个类别分布
        action = m.sample()  # 采样动作
        log_probs.append(m.log_prob(action))  # 记录 log(prob)
        # 执行动作
        state, reward, terminated, truncated, info = env.step(action.item())
        state = torch.tensor(state, dtype=torch.float32).to(policy_net.device)
        # print(f"state: {state}")
        done = bool(terminated) or bool(truncated)  # Ensure done is a boolean value
        rewards.append(reward)
        
    
    # 计算累计回报
    returns = compute_returns(rewards, gamma)
    returns = torch.tensor(returns, dtype=torch.float32).to(policy_net.device)  # Ensure returns tensor is on the correct device
    
    # 标准化回报，以减少方差
    returns = (returns - returns.mean()) / (returns.std() + 1e-9)
    
    # 计算损失
    policy_loss = []
    for log_prob, G in zip(log_probs, returns):
        policy_loss.append(-log_prob * G)  # REINFORCE的损失：-log_prob * G
    policy_loss = torch.cat(policy_loss).sum()
    
    # 反向传播和优化
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()
    
    # 输出训练进度
    if episode % 50 == 0:
        print(f"Episode {episode}, Total Reward: {sum(rewards)}")

env.close()
