import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from algorithm.share import *
from algorithm.ppo.continuous.ppo import PPO
from algorithm.DDPG.ddpg import DDPG
from algorithm.SAC.continuous.sac import SACContinuous
from controller.Examples import get_example_by_name, Example
from controller.Env import Env
from controller.Plot import plot


def train_by_ppo(example: Example):
    torch.manual_seed(2024)
    np.random.seed(2024)
    actor_lr = 1e-4
    critic_lr = 5e-3
    num_episodes = 100
    hidden_dim = 128
    gamma = 0.9
    lmbda = 0.9
    epochs = 10
    eps = 0.2
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device('cpu')

    env = Env(example)
    env.reward_gaussian = False
    state_dim = env.n_obs
    action_dim = env.u_dim  # 连续动作空间
    agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)

    return_list = []
    for i_episode in range(num_episodes):
        episode_return = 0
        transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
        state_list = []
        state, info = env.reset(seed=2024)
        done, truncated = False, False
        while not done and not truncated:
            state_list.append(state)
            action = agent.take_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)
            state = next_state
            episode_return += reward
        return_list.append(episode_return)
        agent.update(transition_dict)

        print(f'episode:{i_episode + 1},reward:{episode_return},step:{len(state_list)}')
        if i_episode % 20 == 0:
            state_list = np.array(state_list)
            x = state_list[:, :1]
            y = state_list[:, 1:2]
            plot(env, x, y)

    return return_list


if __name__ == '__main__':
    example = get_example_by_name('Oscillator')
    return_list = train_by_ppo(example)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('PPO')
    plt.show()

    mv_return = moving_average(return_list, 21)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('PPO')
    plt.show()

    # simulation(env_name, agent)
