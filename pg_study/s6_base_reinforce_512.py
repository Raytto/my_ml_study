# %% [markdown]
# ### 1.包导入

# %%
import gym
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

# %% [markdown]
# ### 2.环境尝试

# %%
env = gym.make("CartPole-v1")
env.reset()
env.step(0)

# %% [markdown]
# ### 3.定义策略网络


# %%
# 定义策略网络
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 512)  # 隐藏层
        self.fc3 = nn.Linear(512, output_dim)  # 输出动作概率
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        x = T.tensor(x, dtype=T.float32).to(self.device)
        layer1 = T.relu(self.fc1(x))
        layer2 = T.relu(self.fc2(layer1))
        p = T.softmax(self.fc3(layer2), dim=-1)
        return p


# %% [markdown]
# ### 4.计算折扣奖励函数


# %%
# 计算折扣奖励
def compute_returns(rewards, gamma):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


# %% [markdown]
# ### 5.设置环境和超参数

# %%
# 设置环境
env = gym.make("CartPole-v1")
input_dim = env.observation_space.shape[0]  # 输入维度
output_dim = env.action_space.n  # 输出维度
# 设置超参数
batch_size = 20  # 设置每次更新的轨迹数量
gamma = 0.9  # 奖励折扣因子
avg_episode = 50
save_episode = 100
print_episode = 50
clip_norm = 100
continue_train = True
num_episodes = 5000
learning_rate = 1e-5

# 学习进度保存路径
model_path = "./save/s6_policy_net.pth"
avg_path = "./save/s6_avg_rewards.pth"

# %%
policy_net = PolicyNetwork(input_dim, output_dim)
optimizer = optim.AdamW(policy_net.parameters(), lr=learning_rate)
avg_rewards = []

if continue_train:
    if os.path.exists(model_path) and os.path.exists(avg_path):
        policy_net.load_state_dict(T.load(model_path))
        policy_net.to(policy_net.device)
        with open(avg_path, "rb") as f:
            avg_rewards = pickle.load(f)
len(avg_rewards)

# %% [markdown]
# ### 6.开始训练

# %%
# 开始训练
record_rewards = []
for episode in range(1, num_episodes + 1):
    log_probs = []
    total_returns = []
    for _ in range(batch_size):
        # state = torch.tensor(env.reset()[0], dtype=torch.float32).to(policy_net.device)

        rewards = []
        state = env.reset()[0]
        # 采样轨迹
        done = False
        while not done:

            # state = torch.tensor(np.array(state), dtype=torch.float32)  # Ensure state tensor is on the correct device
            probs = policy_net(state)
            m = Categorical(probs)  # 创建一个类别分布
            action = m.sample()  # 采样动作
            log_probs.append(m.log_prob(action))  # 记录 log(prob)
            # 执行动作
            state, reward, terminated, truncated, info = env.step(action.item())
            state = T.tensor(state, dtype=T.float32).to(policy_net.device)
            # print(f"state: {state}")
            done = bool(terminated) or bool(truncated)  # Ensure done is a boolean value
            rewards.append(reward)

        # 记录本次总得分
        record_rewards.append(sum(rewards))

        # 计算累计回报
        returns = compute_returns(rewards, gamma)
        total_returns += returns
    total_returns = T.tensor(total_returns, dtype=T.float32).to(
        policy_net.device
    )  # Ensure returns tensor is on the correct device

    # 计算损失
    policy_loss = []
    for log_prob, G in zip(log_probs, total_returns):
        policy_loss.append(-log_prob * G)  # REINFORCE的损失：-log_prob * G
    policy_loss = T.stack(policy_loss).sum() / batch_size

    # 反向传播和优化
    optimizer.zero_grad()
    policy_loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=clip_norm)
    optimizer.step()

    # 输出训练进度
    if episode % avg_episode == 0:
        avg_reward = sum(record_rewards[-avg_episode:]) / avg_episode
        avg_rewards.append(avg_reward)
        if episode % print_episode == 0:
            print(
                f"Episode {episode}, Current Reward:{record_rewards[-1]}, Avg Reward: {avg_reward}"
            )
        if episode % save_episode == 0:
            print(f"to save model and avg_rewards {len(avg_rewards)}")
            T.save(policy_net.state_dict(), model_path)
            with open(avg_path, "wb") as f:
                pickle.dump(avg_rewards, f)

# %%
# 绘制得分变化图
plt.plot(range(1, len(avg_rewards) + 1), avg_rewards)
plt.xlabel("Episode")
plt.ylabel("Avg Reward")
plt.title(f"Avg Reward over Episodes x{avg_episode}")
plt.show()

# %%


# %%
