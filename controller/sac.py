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


def train_by_ddpg(example: Example):
    torch.manual_seed(2024)
    random.seed(2024)
    np.random.seed(2024)

    env = Env(example)
    env.reward_gaussian = False
    state_dim = env.n_obs
    action_dim = env.u_dim
    action_bound = env.u

    actor_lr = 3e-4
    critic_lr = 3e-3
    alpha_lr = 3e-4
    num_episodes = 100
    hidden_dim = 128
    gamma = 0.99
    tau = 0.005  # 软更新参数
    buffer_size = 100000
    minimal_size = 1000
    batch_size = 64

    target_entropy = -action_dim
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device('cpu')

    replay_buffer = ReplayBuffer(buffer_size)
    agent = SACContinuous(state_dim, hidden_dim, action_dim, action_bound, actor_lr, critic_lr, alpha_lr,
                          target_entropy, tau, gamma, device)

    return_list = []
    for i_episode in range(num_episodes):
        episode_return = 0
        state_list = []
        state, info = env.reset(seed=2024)
        done, truncated = False, False
        while not done and not truncated:
            state_list.append(state)
            action = agent.take_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_return += reward
            if replay_buffer.size() > minimal_size:
                b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r,
                                   'dones': b_d}
                agent.update(transition_dict)
        return_list.append(episode_return)

        print(f'episode:{i_episode + 1},reward:{episode_return},step:{len(state_list)}')
        if i_episode % 20 == 0:
            state_list = np.array(state_list)
            x = state_list[:, :1]
            y = state_list[:, 1:2]
            plot(env, x, y)

    return return_list


if __name__ == '__main__':
    example = get_example_by_name('Oscillator')
    return_list = train_by_ddpg(example)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('SAC')
    plt.show()

    mv_return = moving_average(return_list, 21)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('SAC')
    plt.show()

    # simulation(env_name, agent)
