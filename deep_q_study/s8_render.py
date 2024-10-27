import gym
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T
import matplotlib.pyplot as plt
import random
from collections import deque


class DuelingDeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, input_dims):
        super(DuelingDeepQNetwork, self).__init__()

        self.fc1 = nn.Linear(*input_dims, 512)
        self.fc2 = nn.Linear(512, 512)
        # 分别为价值流和优势流创建全连接层
        self.value_stream = nn.Linear(512, 1)
        self.advantage_stream = nn.Linear(512, n_actions)

        self.optimizer = optim.AdamW(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        layer1 = F.leaky_relu(self.fc1(state), negative_slope=0.01)
        layer2 = F.leaky_relu(self.fc2(layer1), negative_slope=0.01)

        value = self.value_stream(layer2)
        advantages = self.advantage_stream(layer2)

        # 合并价值和优势
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))

        return q_values


class Agent:
    def __init__(
        self,
        input_dims,
        n_actions,
        lr,
        gamma=0.99,
        epsilon=1.0,
        ep_dec=1e-5,
        ep_min=0.01,
        mem_size=100000,
        batch_size=64,
    ):
        self.lr = lr
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = ep_dec
        self.eps_min = ep_min
        self.n_action_space = [i for i in range(n_actions)]
        self.batch_size = batch_size
        self.memory = deque(maxlen=mem_size)

        self.Q = DuelingDeepQNetwork(self.lr, self.n_actions, self.input_dims)

    def store_transition(self, state, action, reward, state_, done):
        self.memory.append((state, action, reward, state_, done))

    def sample_memory(self):
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, states_, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(states_),
            np.array(dones),
        )

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            # 确保state是一个批量的二维张量
            state = (
                T.tensor(observation, dtype=T.float32).to(self.Q.device).unsqueeze(0)
            )
            actions = self.Q.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.n_action_space)

        return action

    def decrement_epsilon(self):
        self.epsilon = (
            self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
        )

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        self.Q.optimizer.zero_grad()
        states, actions, rewards, states_, dones = self.sample_memory()

        states = T.tensor(states, dtype=T.float32).to(self.Q.device)
        actions = T.tensor(actions, dtype=T.long).to(self.Q.device)
        rewards = T.tensor(rewards, dtype=T.float32).to(self.Q.device)
        states_ = T.tensor(states_, dtype=T.float32).to(self.Q.device)
        dones = T.tensor(dones, dtype=T.bool).to(self.Q.device)

        q_pred = self.Q.forward(states)[range(self.batch_size), actions]
        q_next = self.Q.forward(states_).max(dim=1)[0]
        q_next[dones] = 0.0

        q_target = rewards + self.gamma * q_next

        loss = self.Q.loss(q_target, q_pred).to(self.Q.device)
        loss.backward()
        self.Q.optimizer.step()
        self.decrement_epsilon()

    def save_model(self, filename):
        T.save(self.Q.state_dict(), filename)

    def load_model(self, filename):
        self.Q.load_state_dict(T.load(filename))
        self.Q.to(self.Q.device)


def plot_learning_curve(x, scores, eps_history, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, eps_history, color="C0")
    ax.set_xlabel("Game", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis="x", colors="C0")
    ax.tick_params(axis="y", colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t - 100) : (t + 1)])

    ax2.scatter(x, running_avg, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel("Score", color="C1")
    ax2.yaxis.set_label_position("right")
    ax2.tick_params(axis="y", colors="C1")

    # plt.savefig(filename)
    return plt


env = gym.make("CartPole-v1")
agent = Agent(
    input_dims=env.observation_space.shape, n_actions=env.action_space.n, lr=0.0001
)
mode = "render_test_model"

if mode == "train_new":
    n_games = 1000
    scores = []
    eps_history = []
    for i in range(n_games):
        score = 0
        done = False
        obs = env.reset()[0]

        while not done:
            action = agent.choose_action(obs)
            obs_, reward, terminated, truncated, info = env.step(action)
            score += reward
            agent.store_transition(obs, action, reward, obs_, terminated or truncated)
            agent.learn()
            obs = obs_
            done = terminated or truncated

        scores.append(score)
        eps_history.append(agent.epsilon)

        if i % 100 == 0:
            avg_score = np.mean(scores[-100:])
            if i % 100 == 0:
                print(
                    f"episode {i} score {score} average score {avg_score} epsilon {agent.epsilon}"
                )

    agent.save_model("cartpole_dueling_deep_q_network.pth")

if mode == "train_continue":
    agent.load_model(
        "D:\ppfiles\myprograms\python_programs\my_ml_study\deep_q_study\cartpole_dueling_deep_q_network.pth"
    )
    n_games = 2000
    scores = []
    eps_history = []
    for i in range(n_games):
        score = 0
        done = False
        obs = env.reset()[0]

        while not done:
            action = agent.choose_action(obs)
            obs_, reward, terminated, truncated, info = env.step(action)
            score += reward
            agent.store_transition(obs, action, reward, obs_, terminated or truncated)
            agent.learn()
            obs = obs_
            done = terminated or truncated

        scores.append(score)
        eps_history.append(agent.epsilon)

        if i % 100 == 0:
            avg_score = np.mean(scores[-100:])
            if i % 100 == 0:
                print(
                    f"episode {i} score {score} average score {avg_score} epsilon {agent.epsilon}"
                )

    agent.save_model(
        "D:\ppfiles\myprograms\python_programs\my_ml_study\deep_q_study\cartpole_dueling_deep_q_network.pth"
    )


if mode == "render_test_model":
    agent.load_model(
        "D:\ppfiles\myprograms\python_programs\my_ml_study\deep_q_study\cartpole_dueling_deep_q_network.pth"
    )

    # 测试加载的模型
    env = gym.make("CartPole-v1", render_mode="human")

    for _ in range(10000):
        done = False
        obs = env.reset()[0]
        while not done:
            action = agent.choose_action(obs)
            obs_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

    print(f"Test score: {score}")
    env.close()
