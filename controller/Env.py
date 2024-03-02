import numpy as np


class Zones:  ## 定义一个区域 ：有两种，长方体或者球
    def __init__(self, shape: str, center=None, r=None, low=None, up=None, inner=True):
        self.shape = shape
        self.inner = inner
        if shape == 'ball':
            self.center = np.array(center)
            self.r = r
        elif shape == 'box':
            self.low = np.array(low)
            self.up = np.array(up)
            self.center = (self.low + self.up) / 2  ## 外接球中心
            self.r = sum(((self.up - self.low) / 2) ** 2)  ## 外接球半径平方
        else:
            raise ValueError('没有形状为{}的区域'.format(shape))


class Example:
    def __init__(self, n_obs, u_dim, D_zones, I_zones, f, u, dense, units, name, dt=0.001, G_zones=None,
                 U_zones=None, goal='avoid', max_episode=1000):
        self.n_obs = n_obs  # 变量个数
        self.u_dim = u_dim  # 控制维度
        self.D_zones = D_zones  # 不变式区域
        self.I_zones = I_zones  ## 初始区域
        self.G_zones = G_zones  ##目标区域
        self.U_zones = U_zones  ## 非安全区域
        self.f = f  # 微分方程
        self.u = u  # 输出范围为 [-u,u]
        self.dense = dense  # 网络的层数
        self.units = units  # 每层节点数
        # self.activation = activation  # 激活函数
        # self.k = k  # 提前停止的轨迹条数
        self.name = name  # 标识符
        self.dt = dt  # 步长
        self.goal = goal  # 'avoid','reach','reach-avoid'
        self.max_episode = max_episode


class Env:
    def __init__(self, example: Example):
        self.n_obs = example.n_obs
        self.u_dim = example.u_dim
        self.D_zones = example.D_zones
        self.I_zones = example.I_zones
        self.G_zones = example.G_zones
        self.U_zones = example.U_zones
        self.f = example.f
        # self.path = example.path
        self.u = example.u

        self.dense = example.dense  # 网络的层数
        self.units = example.units  # 每层节点数
        # self.activation = example.activation  # 激活函数
        self.name = example.name
        self.dt = example.dt
        self.goal = example.goal
        self.s = None
        self.beyond_domain = True
        self.episode = 0
        self.max_episode = example.max_episode
        self.reward_gaussian = True

    def reset(self, seed):
        if self.I_zones.shape == 'ball':
            state = np.random.randn(self.n_obs)
            state = state / np.sqrt(sum(state ** 2)) * self.I_zones.r * np.random.random() ** (1 / self.n_obs)
            state = state + self.I_zones.center
        else:
            state = np.random.rand(self.n_obs) - 0.5
            state = state * (self.I_zones.up - self.I_zones.low) + self.I_zones.center
        self.s = state
        self.episode = 0
        return state, 'Reset successfully!'

    def step(self, action):
        self.episode += 1
        ds = np.array([F(self.s, action) for F in self.f])
        self.s = self.s + ds * self.dt
        done = self.check_done()
        truncated = self.check_truncated()
        reward = self.get_reward()

        return self.s, reward, done, truncated, {'episode': self.episode}

    def get_reward(self):
        if self.goal == 'avoid':
            reward = self.avoid_reward()
        elif self.goal == 'reach':
            reward = self.reach_reward()
        else:
            reward = self.avoid_reward() + self.reach_reward()
        return reward

    def avoid_reward(self):
        if self.reward_gaussian:
            if self.U_zones.shape == 'box':
                reward_avoid = -np.exp(
                    -sum((self.s - self.U_zones.center) ** 2 / ((self.U_zones.up - self.U_zones.low) / 2) ** 2))
            else:
                reward_avoid = -np.exp(-sum((self.s - self.U_zones.center) ** 2 / self.U_zones.r ** 2))
        else:
            reward_avoid = np.sqrt(sum((self.s - self.U_zones.center) ** 2)) - self.U_zones.r

        if not self.U_zones.inner:
            reward_avoid = -reward_avoid
        return reward_avoid

    def reach_reward(self):
        if self.reward_gaussian:
            if self.G_zones.shape == 'box':
                reward_reach = np.exp(
                    -sum((self.s - self.G_zones.center) ** 2 / ((self.G_zones.up - self.G_zones.low) / 2) ** 2))
            else:
                reward_reach = np.exp(-sum((self.s - self.G_zones.center) ** 2 / self.G_zones.r ** 2))
        else:
            reward_reach = self.G_zones.r - np.sqrt(sum((self.s - self.G_zones.center) ** 2))

        if not self.G_zones.inner:
            reward_reach = -reward_reach
        return reward_reach

    def check_done(self):
        # 是否进入不安全区域
        done = False
        if self.U_zones.shape == 'box':
            vis = sum([self.U_zones.low[i] <= self.s[i] <= self.U_zones.up[i] for i in range(self.n_obs)]) != self.n_obs
        else:
            vis = sum((self.s - self.U_zones.center) ** 2) >= self.U_zones.r ** 2

        if vis ^ self.U_zones.inner:
            # 进入不安全区域
            done = True

        # 是否超出整个domain
        if self.D_zones.shape == 'box':
            vis = sum([self.D_zones.low[i] <= self.s[i] <= self.D_zones.up[i] for i in range(self.n_obs)]) == self.n_obs
        else:
            vis = sum((self.s - self.D_zones.center) ** 2) <= self.D_zones.r ** 2

        if vis ^ self.U_zones.inner:
            # 超出domain区域,对于超出domain的state的处理
            if self.beyond_domain:
                done = True
            else:
                if self.D_zones.shape == 'box':
                    self.s = np.array(
                        [min(self.D_zones.up[i], max(self.D_zones.low[i], self.s[i])) for i in range(self.n_obs)])
                else:
                    ratio = self.D_zones.r / np.sqrt((self.s - self.D_zones.center) ** 2)
                    self.s = (self.s - self.D_zones.center) * ratio + self.D_zones.center
        return done

    def check_truncated(self):
        # 超过最大回合数
        return self.episode >= self.max_episode
