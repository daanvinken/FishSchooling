"""
visual_quiver.py

Creates a static 3D quiver plot of the model.
"""

import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation, rc

from fishmodel import Environment

exp = Environment(num_prey=50, num_predator=0)

fig = plt.figure(figsize=(20,15))
ax = fig.gca(projection='3d')

ax.set_xlim(0, exp.bounds[0])
ax.set_ylim(0, exp.bounds[1])
ax.set_zlim(0, exp.bounds[2])

pos = exp.get_positions()  
vel = exp.get_velocity()
quiv = ax.quiver3D(pos[:,0],pos[:,1], pos[:, 2], 
                   vel[:,0], vel[:,1], vel[:, 2])

def quiver_data_to_segments(X, Y, Z, u, v, w, length=3):
    segments = (X, Y, Z, X+v*length, Y+u*length, Z+w*length)
    segments = np.array(segments).reshape(6,-1)
    return [[[x, y, z], [u, v, w]] for x, y, z, u, v, w in zip(*list(segments))]

def animate(num):
  exp.timestep()
  pos = exp.get_positions()
  vel = exp.get_velocity()
  
  segments = quiver_data_to_segments(pos[:,0],pos[:,1], pos[:, 2], 
                                     vel[:,0], vel[:,1], vel[:, 2])
  quiv.set_segments(segments)

  return quiv,

anim = animation.FuncAnimation(fig, animate, frames=1, interval=120, blit=False)

# anim.save('schooling.mp4', writer='ffmpeg')
plt.show()
