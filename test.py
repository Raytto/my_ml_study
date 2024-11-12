import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# 定义超参数
learning_rate = 1e-3
gamma = 0.99  # 折扣因子
num_episodes = 1000

# 定义 Actor 网络
class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=-1)
        return x

# 定义 Critic 网络
class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        value = self.fc2(x)
        return value

# 初始化环境和网络
env = gym.make("CartPole-v1")
actor = Actor(input_dim=4, hidden_dim=128, output_dim=2)
critic = Critic(input_dim=4, hidden_dim=128)

# 定义优化器
actor_optimizer = optim.Adam(actor.parameters(), lr=learning_rate)
critic_optimizer = optim.Adam(critic.parameters(), lr=learning_rate)

# 训练循环
for episode in range(num_episodes):
    state = env.reset()[0]
    episode_reward = 0

    for t in range(1, 10000):  # 设置最大步长

        # Actor 根据策略选择动作
        probs = actor(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        next_state, reward, done, _ = env.step(action.item())

        # 累计奖励
        episode_reward += reward

        # Critic 估计当前状态值
        value = critic(state)
        next_state_value = critic(torch.FloatTensor(next_state).unsqueeze(0))
        
        # 计算 TD 误差和优势
        td_target = reward + gamma * next_state_value * (1 - int(done))
        td_error = td_target - value

        # Critic 网络更新
        critic_loss = td_error.pow(2)
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # Actor 网络更新
        actor_loss = -dist.log_prob(action) * td_error.detach()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        if done:
            print(f"Episode {episode + 1}: Total Reward: {episode_reward}")
            break

        state = next_state

env.close()
