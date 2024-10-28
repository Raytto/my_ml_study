import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

# 定义策略和价值网络
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # 策略网络
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # 价值网络
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        policy_dist = self.actor(x)
        value = self.critic(x)
        return policy_dist, value

# Hyperparameters
gamma = 0.99         # 折扣因子
clip_eps = 0.2       # PPO clipping 参数
policy_lr = 1e-4     # 策略网络学习率
value_lr = 1e-3      # 值网络学习率
k_epochs = 4         # PPO 迭代次数
batch_size = 64      # 批次大小

# PPO agent
class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.model = ActorCritic(state_dim, action_dim)
        self.policy_optimizer = optim.Adam(self.model.actor.parameters(), lr=policy_lr)
        self.value_optimizer = optim.Adam(self.model.critic.parameters(), lr=value_lr)

    def select_action(self, state):
        state = torch.FloatTensor(np.array(state, dtype=np.float32).flatten()).unsqueeze(0)
        policy_dist, _ = self.model(state)
        action = torch.multinomial(policy_dist, 1).item()
        return action, policy_dist[:, action].item()

    def compute_advantage(self, rewards, values, dones):
        returns = []
        advs = []
        discounted_sum = 0
        for i in reversed(range(len(rewards))):
            if dones[i]:
                discounted_sum = 0
            discounted_sum = rewards[i] + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        returns = torch.FloatTensor(returns)
        values = torch.FloatTensor(np.array(values))
        advs = returns - values
        return advs, returns

    def update(self, memory):
        states = torch.FloatTensor(np.array([m[0] for m in memory], dtype=np.float32))
        actions = torch.LongTensor([m[1] for m in memory])
        old_probs = torch.FloatTensor([m[2] for m in memory])
        rewards = [m[3] for m in memory]
        dones = [m[4] for m in memory]

        _, values = self.model(states)
        advantages, returns = self.compute_advantage(rewards, values.detach().numpy(), dones)

        for _ in range(k_epochs):
            # 计算新的策略分布和价值
            policy_dist, value = self.model(states)
            new_probs = policy_dist.gather(1, actions.unsqueeze(1)).squeeze(1)

            # 计算 ratio 和 clip loss
            ratio = new_probs / old_probs
            clip_adv = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
            policy_loss = -torch.min(ratio * advantages, clip_adv).mean()

            # 价值损失
            value_loss = nn.MSELoss()(value.squeeze(), returns)

            # 更新策略网络
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            # 更新价值网络
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

    def save(self, filepath):
        torch.save(self.model.state_dict(), filepath)

    def load(self, filepath):
        if os.path.exists(filepath):
            self.model.load_state_dict(torch.load(filepath))

# 训练 PPO
env = gym.make("Acrobot-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = PPOAgent(state_dim, action_dim)
num_episodes = 1000
max_timesteps = 500
save_path = os.path.join(os.path.dirname(__file__), "ppo_agent.pth")

# 尝试加载现有模型
agent.load(save_path)

for episode in range(num_episodes):
    state = env.reset()
    memory = []
    total_reward = 0

    for t in range(max_timesteps):
        action, prob = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward

        memory.append((state, action, prob, reward, done))

        state = next_state
        if done:
            break

    # 更新 agent
    agent.update(memory)
    print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    # 保存模型
    agent.save(save_path)

env.close()
