""" Will plot deaths against another property, uses data in data/ folder.
"""

import numpy as np
import matplotlib.pyplot as plt

# load data
scatter = ['num_dead', 'num_food']
datatypes = ['attraction_preference', 'food_preference',
             'anti_pred_preference', 'attack_angle']

plt.rcParams.update({'font.size': 18})


a = np.load(f'data/num_dead.npy')
b = np.load(f'data/anti_pred_preference.npy')

x_dead = a[0]
y_dead = a[1]

x_pref = b[0]
y_pref = b[1]

fig = plt.figure(figsize=(10, 7))

ax1 = fig.subplots()

ax1.set_xlabel('Generations')
ax1.set_ylabel('Deaths', color='firebrick')
ax1.scatter(x_dead, y_dead, color='firebrick')

ax2 = ax1.twinx()
ax2.set_ylabel(r'Anti-predator preference $\Omega^P$', color='blue')
ax2.plot(x_pref, y_pref, color='blue')

fig.tight_layout()

plt.show()