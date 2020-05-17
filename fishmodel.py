"""
Fishmodel.py

In this file the complete predator, prey schooling behaviour is modelled.
Import the 'Environment' class to instantiate the model.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation, rc
import time
from IPython.display import HTML

prey_attributes = ['attraction_preference', 'food_preference', 'anti_pred_preference']
predator_attributes = ['attack_angle', 'attack_distance']

DIMS = 3

def fast_norm(x):
  # Faster than np.linalg.norm
  return np.sqrt(x.dot(x))

class Agent:
  def __init__(self, environment):
    self.pos = np.zeros(DIMS)
    self.vel = np.zeros(DIMS)
    self.environment = environment
    self.active = True
  
  def update_position(self, neighbors):
    print("No update position defined for this class")
    pass
  
  def __str__(self):
    return f"{self.__class__.__name__}: pos: {self.pos} vel: {self.vel} active:{self.active}"
  __repr__ = __str__


class Prey(Agent):
  def __init__(self, environment):
    super().__init__(environment)
    
    # Model parameters
    self.repulsion = 5
    self.attraction = 5.5
    self.alignment = 0
    self.max_attraction = 6.0
    self.mass = 200
    
    self.fmin = -1200
    self.fmax = 120
    self.vel_max = 5
    self.dt = 0.1
    
    self.food = 0
    self.food_range = 100
    self.f_food = 5*32
    self.predator_range = 37
    self.near_neighbors = []
    
    self.attraction_preference = 2
    self.food_preference = 2
    self.anti_pred_preference = 1
  
  
  def update_position(self):
    """ Update the position of this prey fish"""
    prey_force = np.zeros(DIMS)
    predator_force = np.zeros(DIMS)
    food_force = np.zeros(DIMS)
    environment_force = np.zeros(DIMS)
    
    self.near_neighbors = []
    # Prey Force
    for n in self.environment.prey:

      if n is self or not n.active:
        continue

      distance = fast_norm(self.pos - n.pos)
      if distance < 15:
        self.near_neighbors.append(n)
      
      magnitude = self.attraction_force(n, distance) / self.mass
      direction = (self.pos - n.pos) / distance
      prey_force += -magnitude * direction
       
    # Predator Force
    for predator in self.environment.predators:
      distance = fast_norm(self.pos - predator.pos)
      
      if distance > self.predator_range:
        continue
      
      magnitude = (self.fmin * self.anti_pred_preference) / self.mass
      direction = (self.pos - predator.pos) / distance
      predator_force += -magnitude * direction
      
    # Environment force
    environment_force = self.environment_force()
    
    # Food force
    for food in self.environment.food:
      if not food.active:
        continue
      distance = fast_norm(self.pos - food.pos)
      
      if distance > self.food_range:
        continue
        
      magnitude = (self.f_food * self.food_preference) / self.mass
      direction = (self.pos - food.pos) / distance
      food_force += -magnitude * direction
      
    # Apply forces
    self.vel += (prey_force + predator_force + food_force + environment_force) * self.dt
    
    norm = fast_norm(self.vel)
    if norm > self.vel_max:
      self.vel = (self.vel / norm) * self.vel_max

    self.pos += self.vel
    
    # Check for food collisions
    for food in self.environment.food:
      if not food.active:
        continue
      
      distance = fast_norm(self.pos - food.pos)
      if distance < self.repulsion:
        food.size -= 1
        self.food += 1
        
        if food.size <= 0:
          food.active = False
      
    
  def attraction_force(self, n, distance):
    """
    linear attraction-alignment-repulsion function
    Returns magnitude of force exterted by neighbor
    distance: distance to neighbour
    """    
    if distance < self.repulsion:
      return min(self.fmin, 1 - distance / self.repulsion)
    
    if distance < self.alignment and distance >= self.repulsion:
      return self.alignment
    
    if distance < self.attraction and distance >= self.alignment:
      return (min(self.fmin, self.alignment - distance) / self.repulsion) \
      * self.attraction_preference
    
    return self.fmax * self.attraction_preference


  def environment_force(self):
    """
    Return the force exerted by the boundaries
    """
    a = 1 / (1 + pow(-self.pos, 2))
    b = 1 / (1 + pow(self.environment.bounds - self.pos, 2))
    force = 100 * (a - b)
    return force

    
  
  def __str__(self):
    return "Prey(repul={}, attr={}, align={})"\
            .format(self.repulsion, self.attraction, self.alignment)
  


class Predator(Agent):
  def __init__(self, environment):
    super().__init__(environment)
    
    # Model paramters
    self.mass = 50
    self.attack_distance = 40
    self.target_attraction = 3000
    self.vel_max = 5.5
    self.dt = 0.1
    self.daz = 5
    self.attack_angle = np.pi / 6
    self.target = None
    
    self.emax = 80000
    self.energy = self.emax
    self.e_from_prey = 10
    self.emin = 50
    self.total_eaten = 0
    
  def update_position(self):
    """Update position of the predator fish"""
    # Halt predator if out of energy
    e_vel_factor = self.energy/self.emax
    if self.energy <= self.emin:
      self.vel = np.zeros(DIMS)
      return
    
    target_force = np.zeros(DIMS)
    
    # Assign higher probability to prey with little neighbors
    p = []
    for prey in self.environment.prey:
      if not prey.active:
        p.append(0)
      else:
        fraq = (self.environment.num_prey - len(prey.near_neighbors)) / (self.environment.num_prey - 1)
        p.append(fraq)
     
    if np.sum(p) == 0:
      p = np.ones(self.environment.num_prey) / self.environment.num_prey
    else:
      p = p / np.sum(p)
    
    # First target is random
    if not self.target:
      self.target = np.random.choice(self.environment.prey, p=p)
    
    # Attraction to target
    distance = fast_norm(self.pos - self.target.pos)

    last_pos = np.array(self.pos)
    
    magnitude = self.target_attraction / self.mass
    direction = (self.pos - self.target.pos) / distance
    target_force += -magnitude * direction

    # Apply Forces
    self.vel += target_force * self.dt
    self.vel = np.clip(self.vel, -self.vel_max, self.vel_max)
    self.pos += self.vel
    
    # Change target
    candidates = []
    for prey in self.environment.prey:
      if not prey.active:
        continue
      
      if self.incone(prey, direction):
        candidates.append(prey)

    if len(candidates) > 0:
      self.target = np.random.choice(candidates)
    
    # Check if prey can be eaten
    if distance < self.target.repulsion:
      self.target.active = False
      self.target = None
      self.total_eaten += 1
      self.energy += self.e_from_prey
      
    # Energy loss/input(input from eaten prey)
    work_distance = fast_norm(self.pos - last_pos)
    
    target_force = fast_norm(target_force)
    Energy_for_movement = target_force * work_distance
    self.energy = self.energy - Energy_for_movement
    
    if self.energy >= self.emax:
      self.energy = self.emax
    if self.energy <= self.emin:
      self.energy = 0
  
  def incone(self, prey, direction):
    """ Returns True if prey is witihn predators' attack cone"""
    cone_dist = np.dot(prey.pos - self.target.pos, direction)
    
    if not (0 <= cone_dist and cone_dist <= self.daz):
      return False

    r = np.tan(self.attack_angle) * self.daz
    cone_radius = (cone_dist / self.daz) * r
    orth_distance = fast_norm((prey.pos - self.target.pos) - cone_dist * direction)

    return orth_distance < cone_radius
    

class Food(Agent):
  def __init__(self, environment, size):
    super().__init__(environment)
  
    self.size = size
    
  def update_position(self, _):
    pass

  def __str__(self):
    return '{}: pos: {} size: {}'.format(self.__class__.__name__,
                                         self.pos, self.size)
  __repr__ = __str__



class Environment:
  def __init__(self, num_prey=50, num_predator=1):
    self.prey = []
    self.predators = []
    self.food = []
    self.t = 0
    self.warmup = 30
    
    # Model parameters
    self.bounds = np.array([150, 150, 150])
    if DIMS == 2:
      self.bounds = self.bounds[:2]
    self.num_prey = num_prey
    self.num_predator = num_predator
    self.spawn_distance_pred = 30
    
    self.generate_prey()
    
    # Allow prey fish to form school
    for _ in range(self.warmup):
      self.timestep()
    
    self.generate_predator()
    self.generate_food()

  
  def generate_prey(self):
    for _ in range(self.num_prey):
      agent = Prey(self)
          
      # Position agent uniformly in environment
      agent.pos = np.random.rand(DIMS) * self.bounds
      self.prey.append(agent)
    
  def generate_predator(self):
    for i in range(self.num_predator):
      agent = Predator(self)
      if i != 0:
        agent.active = False
      agent.pos = np.random.rand(DIMS)*self.bounds 
      swarm_pos = self.swarm_position()
      
      while fast_norm(agent.pos-swarm_pos) < self.spawn_distance_pred:
        agent.pos = np.random.rand(DIMS)*self.bounds
      
      self.predators.append(agent)
      
  
  def generate_food(self):
    food_supply = 1024
    total = 0
    left_over = food_supply

    while total < food_supply:
      # Assign food bounds
      size_of_meals = [1, 4, 16, 64, 256]
      pos = self.random_in_bounds()

      # Assign food size        
      choice = np.random.choice(size_of_meals)
      
      if left_over - choice < 0:
        del size_of_meals[-1]
        continue
      else:
        left_over -= choice
        total += choice
      
      food = Food(self, size=choice)
      food.pos = pos
      self.food.append(food)


  def random_in_bounds(self):
    """ Returns position uniformly distrubuted in the bounds, 
    outside of the inner bounds"""
    pos = np.random.uniform(0, self.bounds[0], size=(20, DIMS))
    filtered = []
    for p in pos:
      if (p > self.bounds / 4).all() and (p < self.bounds * 0.75).all():
        continue
      filtered.append(p)
    return filtered[np.random.randint(0, len(filtered))]
  
  def swarm_position(self):
    """ Returns average position of the swarm"""
    avg_positions = []
    for p in self.prey:
      if p:
        avg_positions.append(p.pos)
    return np.mean(avg_positions, axis=0)
  
  def predator_position(self):
    if len(self.predators) == 1:
      return self.predators[0].pos
    return np.zeros(DIMS)
    
  def timestep(self):
    """
    Perform a single timestep of the model
    """
    all = self.prey + self.predators
    for a in all:
      if a.active:
        a.update_position()
    
    if self.t % 70 == 0:
      self.activate_next_predator()

    self.t += 1

  def activate_next_predator(self):
    """ Disable current predator, and activate the next"""
    for i in range(len(self.predators)):
      if self.predators[i].active:
        self.predators[i].active = False
        if len(self.predators) != i + 1:
          self.predators[i + 1].active = True
        break

  def __str__(self):
    return "Number of prey: {}\n{}".format(len(self.prey), self.prey)
    

  def get_positions(self):
    positions = []
    all = self.prey + self.predators + self.food
    for a in all:
      positions.append(a.pos)

    return np.array(positions)
  
  def get_velocity(self):
    vel = []
    all = self.prey + self.predators + self.food
    for a in all:
      vel.append(a.vel / fast_norm(a.vel))
      
    return np.array(vel)
 
  def new_child_values(self, parent, child, attributes):
    """ Give child fish the property values of it's parent, with some 
    mutations """
    noise = 0
    uniform = np.random.uniform
    deviation = 0.05

    for attribute in attributes:
      setattr(child, attribute,  \
      uniform(getattr(parent, attribute) * (1 - deviation), 
              getattr(parent, attribute) * (1 + deviation)) + noise)


  def evolve(self):
    """ Evolve all prey and predators, and reset the environment. """
    p = []
    values = []
    nextgen_prey = []
    total_eaten_food = 0

    nextgen_pred = []
    dead = 0
    
    # Count food consumed, of prey which are not dead
    for prey in self.prey:
      if not prey.active:
        dead += 1
        continue
      total_eaten_food += prey.food
      
    # Calculate probabilities
    if total_eaten_food != 0:
      for prey in self.prey:
        if prey.active:
          p.append(prey.food / total_eaten_food)
          values.append(prey)
    else:
      p = np.ones(self.num_prey) / self.num_prey
      values = self.prey

    # Generate new prey
    for _ in range(self.num_prey):
      parent = np.random.choice(values, p=p)
      new_prey = Prey(environment=self)
      self.new_child_values(parent, new_prey, prey_attributes)

      new_prey.pos = np.random.rand(DIMS) * self.bounds
      nextgen_prey.append(new_prey)  
    
    self.prey = nextgen_prey
    
      
    # Total number of prey eaten
    total_prey_eaten = 0
    for predator in self.predators:
      total_prey_eaten += predator.total_eaten

    p = []
    if total_prey_eaten != 0:
      for predator in self.predators:
        p.append(predator.total_eaten / total_prey_eaten)
    else:
      p = np.ones(self.num_predator) / self.num_predator

    for i in range(self.num_predator):
      parent = np.random.choice(self.predators, p=p)
      new_pred = Predator(environment=self)
      
      if i != 0:
        new_pred.active = False
      
      self.new_child_values(parent, new_pred, predator_attributes)

      swarm_pos = self.swarm_position()
      new_pred.pos = np.random.rand(DIMS)*self.bounds

      while fast_norm(new_pred.pos-swarm_pos) < self.spawn_distance_pred:
        new_pred.pos = np.random.rand(DIMS)*self.bounds
      
      nextgen_pred.append(new_pred)
   
    
    # Warmup prey
    self.food = []
    for _ in range(self.warmup):
      self.timestep()
    
    # Add new predators
    self.predators = nextgen_pred  
    self.generate_food()
    self.t = 0
    
    return total_eaten_food, dead


if __name__ == "__main__":
  print("Running this file alone does not do anything.")