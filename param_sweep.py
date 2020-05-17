""" 
param_sweep.py

Sensitivity analysis of the model.
Changes one variable of the model and plots the average
nearest neighbour distance of the swarm. Will do this for every parameter.
"""

import time
from fishmodel import Environment, Prey, Predator, fast_norm
import matplotlib.pyplot as plt
import numpy as np
from progress.bar import Bar
from scipy import stats
from IPython.display import HTML
from IPython.display import display
from scipy import stats

def get_closest_dist(prey, neighbors):
    """ Returns distance to nearest neighbor"""
    all_dist = []
    for n in neighbors:
        if n is not prey:
            all_dist.append(fast_norm(prey.pos - n.pos))
    return np.min(all_dist)


def average_nearest_distance_over_time(env, timesteps=500):
    """ Return list of average prey density for t timesteps"""
    y = []
    for _ in range(timesteps):
        dist = []
        n = 0
        for prey in env.prey:
            if not prey.active:
                continue

            dist.append(get_closest_dist(prey, env.prey))
            n += 1

        average = np.sum(dist) / n
        y.append(average)
        env.timestep()
    
    return np.array(y)

def param_sweep(testrange, sweep_param, pred):
  
  bar = Bar(sweep_param, max=len(testrange))
  mean_list = []
  stdev_list = []
  
  counter = 0
  for value in testrange:
    exp = Environment(50, 7)
  
    if pred:
      for pred in exp.predators:
        setattr(pred, sweep_param, value)
    if pred == False:
      for prey in exp.prey:
        setattr(prey, sweep_param, value)
  
    density = average_nearest_distance_over_time(exp, timesteps=700)
    mean_list.append(np.mean(density))
    stdev_list.append(np.std(density))

    bar.next()
    
    counter += 1
      
  return mean_list, stdev_list

 
if __name__ == "__main__":
    params_prey = ['attraction_preference', 'food_preference', 'anti_pred_preference']
    exp = Environment(20,7)
    prey = Prey(exp)
    pred = Predator(exp)

    testranges_prey = []
    for param_name in params_prey:
        param_value = getattr(prey,param_name)
        testrange = np.linspace(param_value*0.1,param_value*10,80)
        testranges_prey.append(testrange)
    
    params_pred = ['attack_angle']

    testranges_pred = []
    for param_name in params_pred:  
        param_value = getattr(pred,param_name)
        testrange = np.linspace(param_value*0.1,param_value*10,80)
        testranges_pred.append(testrange)

    plt.figure(figsize=(15, 12))

    counter = 1
    for i in range(len(params_prey)):
        param = params_prey[i]
        testrange = testranges_prey[i]
    
        
        density_mean, density_stdev = param_sweep(testrange, param,False)
        plt.subplot(5, 2, counter)
        plt.plot(testrange, density_mean)
        plt.title(param + ' mean')

        counter += 1
        plt.subplot(5, 2, counter)
        plt.plot(testrange, density_stdev)
        plt.title(param + ' stdev')

        counter += 1
        
    for i in range(len(params_pred)):
        param = params_pred[i]
        testrange = testranges_pred[i]
    
        
        density_mean, density_stdev = param_sweep(testrange, param,True)
        plt.subplot(5, 2, counter)
        plt.plot(testrange, density_mean)
        plt.title(param + ' mean')

        counter += 1
        plt.subplot(5, 2, counter)
        plt.plot(testrange, density_stdev)
        plt.title(param + ' stdev')

        counter += 1
        
    plt.show()