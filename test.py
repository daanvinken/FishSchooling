import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
DIMS = 3
def incone(points):
  colors = []
  for point in points:
    daz = 10
    x = np.array([5,5,5])
    direction = np.array([0, 1, 0])
    cone_dist = np.dot(x - point, direction)

    if cone_dist > daz:
      colors.append('red')
      continue

    r = np.tan(np.pi / 6) * daz
    cone_radius = (cone_dist / daz) * r
    orth_distance = np.linalg.norm((x - point) - cone_dist * direction)

    if orth_distance < cone_radius:
      colors.append('blue')
    else:
      colors.append('red')

  return colors


fig = plt.figure()
ax = Axes3D(fig)
x = np.random.rand(5500)*10
y = np.random.rand(5500)*10
z = np.random.rand(5500)*10


ax.scatter(x, y, z, color=incone(zip(x, y, z)))
plt.show()
