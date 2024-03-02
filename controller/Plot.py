import numpy as np
from matplotlib import pyplot as plt


def draw_zone(zone, label, color):
    if zone.shape == 'box':
        up = zone.up
        low = zone.low
        x1, y1 = up[:2]
        x2, y2 = low[:2]
        plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], label=label, c=color)
    else:
        r = zone.r
        thta = np.linspace(0, 2 * np.pi, 100)
        x = [r * np.cos(v) + zone.center[0] for v in thta]
        y = [r * np.sin(v) + zone.center[1] for v in thta]
        plt.plot(x, y, label=label, c=color)


def plot(env, X, Y):
    draw_zone(env.D_zones, 'domain', 'green')
    draw_zone(env.I_zones, 'initial set', 'b')
    draw_zone(env.G_zones, 'target set', 'y')
    draw_zone(env.U_zones, 'unsafe set', 'r')

    plt.plot(X, Y, linewidth=0.5, c='brown')

    plt.legend()
    plt.axis('equal')
    plt.show()
