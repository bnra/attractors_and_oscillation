from scipy.stats import uniform, norm
import numpy as np

def draw_uniform(a:float=0., b:float=1., size:int=1):
    scale = b - a
    return uniform.rvs(loc=a, scale=scale, size=size)

def draw_normal(mu:float=0., sigma:float=1., size:int=1):
    return norm.rvs(loc=mu, scale=sigma, size=size)

def draw_uniformely_random_from_values(values:np.ndarray, size:int=1):
    return np.random.choice(values, size=size) 
