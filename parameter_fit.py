"""
parameter_fit.py

This file will run multiple generations of the fishmodel class
and plot's the evolvable parameters value of each generation.
Plot data is also saved in data/

Running alot of generations might take some time.
"""

import time
from fishmodel import Environment, Prey, Predator
import matplotlib.pyplot as plt
import numpy as np
from progress.bar import Bar

prey_attributes = ['attraction_preference', 'food_preference',
                   'anti_pred_preference']
predator_attributes = ['attack_angle']

def average_params(objects, attributes):
  attrs = dict()
  n = len(objects)
  
  # init to zero
  for attr in attributes:
      attrs[attr] = 0
  
  # count attributes
  for o in objects:
    for attr in attributes:
      attrs[attr] += getattr(o, attr)
  
  if (n > 0):
    for attr in attributes:
      attrs[attr] /= n
  
  return attrs


def fit_params(iterations, generations, num_prey, num_pred):
  start_time = time.time()
  num_dead = []
  num_food = []
  
  # Init empty list for each attribute
  average_prey = dict()
  average_predator = dict()
  for attr in prey_attributes:
      average_prey[attr] = []
  
  for attr in predator_attributes:
      average_predator[attr] = []
  
  bar = Bar('Evolving', max=generations)
  env = Environment(num_prey, num_pred)
  
  for g in range(generations):
    for i in range(iterations):
      env.timestep()
    
    food, dead = env.evolve()
    prey_attrs = average_params(env.prey, prey_attributes)
    predator_attrs = average_params(env.predators, predator_attributes)
    
    # Prey attributes
    for attr in prey_attributes:
      average_prey[attr].append(prey_attrs[attr])
    
    # Predator attributes
    for attr in predator_attributes:
      average_predator[attr].append(predator_attrs[attr])
    
    num_dead.append(dead)
    num_food.append(food)
    
    bar.next()

  bar.finish()
  
  end_time = time.time() - start_time
  
  end_prey = average_params(env.prey, prey_attributes)
  end_predator = average_params(env.predators, predator_attributes)
  
  result = (f"{iterations*generations} timesteps took {end_time:.2f} seconds to "
            f"process or {end_time/60:.2f} minutes\n"
            f"{end_time / (iterations*generations):.4f} seconds per timestep\n"
            f"Average params:\n")
  for attr in prey_attributes:
    result += f"\t {attr}:\t {end_prey[attr]:.4f}\n"
  for attr in predator_attributes:
    result += f"\t {attr}:\t {end_predator[attr]:.4f}\n"

  print(result)
  
  plt.figure(figsize=(20, 12))
  plt.subplot(3, 3, 1)
  plt.scatter(np.arange(generations), num_dead, alpha=0.3)
  np.save('data/num_dead', np.array([np.arange(generations), num_dead]))
  plt.title('Deaths')
  
  plt.subplot(3, 3, 2)
  plt.scatter(np.arange(generations), num_food, alpha=0.3)
  np.save('data/num_food', np.array([np.arange(generations), num_food]))
  plt.title('food consumption')
  
  counter = 3
  for attr in prey_attributes:
    plt.subplot(3, 3, counter)
    plt.plot(np.arange(generations), average_prey[attr])
    np.save('data/' + attr, np.array([np.arange(generations), average_prey[attr]]))
    plt.title(attr)
    counter += 1
  
  for attr in predator_attributes:
    plt.subplot(3, 3, counter)
    plt.plot(np.arange(generations), average_predator[attr])
    np.save('data/' + attr, np.array([np.arange(generations), average_predator[attr]]))
    plt.title(attr)
    counter += 1
  
  plt.savefig('result.png')
  plt.show()
  return env


if __name__ == '__main__':
  fit_params(700, 20, 50, 7)