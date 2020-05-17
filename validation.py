"""
This will produce a plot with the average distance of the nearest neighbour 
of each prey fish. Fit using lognormal law according to

"Experimental evidences of a structural and dynamical transition
in fish school" Becco et al. 2006
"""
from fishmodel import Environment
from fishmodel import Environment, Prey, Predator
from param_sweep import average_nearest_distance_over_time
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import lognorm

if __name__ == "__main__":

    timesteps = 700
    multiple = 1
    sums = np.ones(timesteps - 100)

    for _ in range(multiple):
        env = Environment(20, 1)
        sums += average_nearest_distance_over_time(env, timesteps=timesteps)[100:]
        
    
    interd = sums / multiple
    
    # lognorm fit plot
    s = np.std(interd)
    mu = np.mean(interd)
    dist = lognorm([s], scale=np.exp(mu))
    
    x = np.linspace(0, 10, 100)
    
    plt.figure(figsize=(10, 7))
    # plt.plot(x, lognorm.pdf(x, s, 0, mu))

    # histogram
    # plt.figure(figsize=(10, 7))
    plt.hist(interd, density=True, color='navy')

    plt.show()
    