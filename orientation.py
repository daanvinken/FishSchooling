""" 
orientation.py 

Plot the distribution of orientation angles of nearest
neighbouring prey fish.
"""

from fishmodel import Environment, Prey, Predator, Food
import matplotlib.pyplot as plt
import numpy as np
from fishmodel import fast_norm
from parameter_fit import fit_params
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from scipy import stats
from progress.bar import Bar

def get_angle(v1, v2):
    cosang = np.dot(v1, v2)
    sinang = fast_norm(np.cross(v1, v2))
    return np.arctan2(sinang, cosang)

def get_closest_angle(prey, neighbors):
    i = get_closest_neighbor(prey, neighbors)

    return get_angle(prey.vel, neighbors[i].vel)

def get_closest_neighbor(prey, neighbors):
    """ Returns distance to nearest neighbor"""
    all_dist = []
    for n in neighbors:
        if n is not prey:
            all_dist.append(fast_norm(prey.pos - n.pos))
    return all_dist.index(np.min(all_dist))

def nn_orientation(env, timesteps=700):
    y = []
    bar = Bar("timesteps", max=timesteps)
    for _ in range(timesteps):
        angles = []
        n = 0
        for prey in env.prey:
            if not prey.active:
                continue

            angles.append(get_closest_angle(prey, env.prey))
            n += 1

        average = np.sum(angles) / n
        y.append(average)
        env.timestep()
        bar.next()
    bar.finish()
    return np.array(y)

if __name__ == "__main__":
    env = Environment(20, 0)
    angles = nn_orientation(env)


    plt.figure(figsize=(8, 7))
    x = np.linspace(-0.7, 1.7, 100)
    plt.plot(x, stats.norm.pdf(x, np.mean(angles), np.std(angles)))
    plt.ylabel('probability density')
    plt.xlabel(r'$\theta_{nn}$')
    print('mean', np.mean(angles), 'std', np.std(angles))

    plt.show()