import numpy as np

from controller.Env import Zones, Example, Env

pi = np.pi

examples = {
    1: Example(n_obs=2,
               u_dim=1,
               D_zones=Zones(shape='box', low=[-pi, -5], up=[pi, 5]),
               I_zones=Zones(shape='ball', center=[0, 0], r=2),
               G_zones=Zones(shape='ball', center=[0, 0], r=1),
               U_zones=Zones(shape='ball', center=[0, 0], r=2.5, inner=False),
               f=[lambda x, u: x[1],
                  lambda x, u: -10 * (0.005621 * x[0] ** 5 - 0.1551 * x[0] ** 3 + 0.9875 * x[0]) - 0.1 * x[1] + u[0]
                  ],
               u=2,
               dense=4,
               units=20,
               dt=0.001,
               max_episode=2000,
               goal='reach',
               name='test'),
    2: Example(
        n_obs=2,
        u_dim=1,
        D_zones=Zones('box', low=[-2, -2], up=[2, 2]),
        I_zones=Zones('box', low=[-0.51, 0.49], up=[-0.49, 0.51]),
        G_zones=Zones('box', low=[-0.05, -0.05], up=[0.05, 0.05]),
        U_zones=Zones('box', low=[-0.4, 0.2], up=[0.1, 0.35]),
        f=[lambda x, u: x[1],
           lambda x, u: (1 - x[0] ** 2) * x[1] - x[0] + u[0]
           ],  # 0.01177-3.01604*x1-19.59416*x2+2.96065*x1^2+27.86854*x1*x2+48.41103*x2^2
        u=3,
        dense=5,
        units=64,
        dt=0.005,
        max_episode=1500,
        goal='reach',
        name='Oscillator')
}


def get_example_by_id(id: int):
    return examples[id]


def get_example_by_name(name: str):
    for ex in examples.values():
        if ex.name == name:
            return ex
    raise ValueError('The example {} was not found.'.format(name))


if __name__ == '__main__':
    example = examples[1]
    env = Env(examples[1])
    env.reward_gaussian = False
    x, y, r = [], [], []
    s, info = env.reset(2024)
    print(s)
    x.append(s[0])
    y.append(s[1])
    done, truncated = False, False
    while not done and not truncated:
        action = np.array([1])
        observation, reward, terminated, truncated, info = env.step(action)
        x.append(observation[0])
        y.append(observation[1])
        r.append(reward)

    from controller.Plot import plot

    plot(env, x, y)
    print(sum(r))
