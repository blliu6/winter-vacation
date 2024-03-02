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
               name='test')
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
