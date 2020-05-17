"""
2dvisual_scatter.py

Creates a 2D animated scatter plot of the model. 
Will require DIMS in fishmodel.py to be set to 2.

"""

from fishmodel import Environment, Prey, Predator, Food
import matplotlib.pyplot as plt
import numpy as np
from visual_scatter import color_func
from parameter_fit import fit_params
from matplotlib import animation

# 2D test
env = Environment(num_prey=20, num_predator=7)
# env = fit_params(700, 200, 20, 7)

pos = env.get_positions()
fig = plt.figure()
agents = env.prey + env.predators + env.food
scat = plt.scatter(pos[:, 0], pos[:, 1], color=color_func(agents))

plt.xlim(0, env.bounds[0])
plt.ylim(0, env.bounds[1])

def animate(num):
  env.timestep()
  pos = env.get_positions()
  scat.set_offsets(pos)
  
  return scat,

anim = animation.FuncAnimation(fig, animate, frames=1, interval=100)

plt.show()
