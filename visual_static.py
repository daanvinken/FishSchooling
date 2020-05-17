"""
visual_static.py

Creates a static 3D scatter plot of the model.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from fishmodel import Environment, Prey, Predator, Food

def color_func(agents):
  colors = []
  for a in agents:
    if not a.active:
      colors.append('black')
      continue
    if isinstance(a, Prey):
      colors.append("blue")
    elif isinstance(a, Predator):
      colors.append("red")
    elif isinstance(a, Food):
      colors.append("green")
    
    
  return colors

# Static scatter plot of fish locations

exp = Environment(num_prey=20, num_predator=1)

for _ in range(5):
  exp.timestep()

pos = exp.get_positions()
fig = plt.figure(figsize=(10, 7))
ax = Axes3D(fig)
ax.set_xlim(0, exp.bounds[0])
ax.set_ylim(0, exp.bounds[1])
ax.set_zlim(0, exp.bounds[2])

agents = exp.prey + exp.predators + exp.food
ax.scatter(pos[:20,0], pos[:20,1], pos[:20, 2], color=color_func(exp.prey))
ax.scatter(pos[20:21,0], pos[20:21,1], pos[20:21,2], color=color_func(exp.predators))
ax.scatter(pos[21:,0], pos[21:,1], pos[21:,2], color=color_func(exp.food))
ax.legend(['Prey', 'Predator', 'Food'])

plt.show()
