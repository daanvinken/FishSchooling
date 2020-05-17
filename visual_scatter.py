"""
visual_scatter.py

Creates a 3D animated scatter plot of the model.
"""

from fishmodel import Environment, Prey, Predator, Food
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from parameter_fit import fit_params
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

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

if __name__ == '__main__':
  # 3D Animated scatter plot
  exp = Environment(num_prey=20, num_predator=7)
  # exp = fit_params(700, 1000, 20, 7)

  fig = plt.figure(figsize=(11, 9))
  ax = Axes3D(fig)
  ax.set_xlim(0, exp.bounds[0])
  ax.set_ylim(0, exp.bounds[1])
  ax.set_zlim(0, exp.bounds[2])
  x=np.zeros((len(exp.prey + exp.predators + exp.food)))

  agents = exp.prey + exp.predators + exp.food
  scat = ax.scatter(x,x, color=color_func(agents))


  def animate(num):
    exp.timestep()
    pos = exp.get_positions()
    scat._offsets3d = (pos[:,0],pos[:,1], pos[:, 2])
    return scat,

  anim = animation.FuncAnimation(fig, animate, frames=1, 
                                interval=100, blit=False)

  # anim.save('result.gif', dpi=80, writer='imagemagick')
  
  plt.show()