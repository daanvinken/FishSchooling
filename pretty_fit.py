"""
pretty_fit.py

Create a good looking plot of the data in data/ folder.
"""
import numpy as np
import matplotlib.pyplot as plt

# load data
scatter = ['num_dead', 'num_food']
datatypes = ['attraction_preference', 'food_preference',
             'anti_pred_preference', 'attack_angle']

titles = {
    'attraction_preference': r'Attraction preference $\Omega^a$ ',
    'food_preference': r'Food preference $\Omega^f$',
    'anti_pred_preference': r'Anti-predator preference $\Omega^P$',
    'attack_angle': r'Attack angle $\theta^a$'
}

plt.rcParams.update({'font.size': 22})


plt.figure(figsize=(10, 7))

for datatype in datatypes:
    d = np.load(f'data/{datatype}.npy')
    x = d[0]
    y = d[1]
    
    plt.title(titles[datatype])
    plt.plot(x, y)
    plt.xlabel('Generations')
    # plt.show()
    # break

plt.legend(datatypes)
plt.show()