import numpy as np
import copy
from collections import defaultdict
import os

def printf(**x): #type printf(a=a) to get "a: 3.14" instead of print("a:", a), slightly faster
    print(str(x)[1:-1].replace("'",""))

def normalize(vector):
    return np.array(vector/(vector[0]**2 + vector[1]**2)**0.5)

def out_of_bound(pos):
    n = 64
    return pos[0] < 0 or pos[0] >= n or pos[1] < 0 or pos[1] >= n         

def magnitude(pos):
    return distance((0,0), pos)

def distance(pos1, pos2):
    return distance_pow_2(pos1, pos2)**0.5

def distance_pow_2(pos1, pos2):
    return (pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2

def sign(a):
    v1 = 1 if (a > 0) else 0
    v2 = 1 if (a < 0) else 0
    return v1 - v2

def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))

def gravity(toward_pos, from_pos):
    toward_pos = np.array(toward_pos)
    from_pos = np.array(from_pos)
    dist = distance(toward_pos, from_pos)
    force_magnitude = 1 / (dist ** 2)
    force_dir = normalize(toward_pos - from_pos)
    return force_dir * force_magnitude

def dot_product(vec1, vec2):
    return vec1[0]*vec2[0] + vec1[1]*vec2[1]

def round_pos(pos):
    return round(pos[0]), round(pos[1])

    #-1..1 => 0..1
def neg1_1_to_0_1(num):
    return (num + 1) * 0.5

def neg1_1_to_0_1_round(num):
    return round(((num + 1) * 0.5), 1)

    #0..1 => -1..1
def pos0_1_to_neg1_1(num):
    return num * 2 - 1

def adjacent(pos):
    x,y = pos
    return [(x-1, y), (x+1, y), (x,y-1), (x,y+1)]

def projection(vec_from, vec_to):
    return vec_to * dot_product(vec_from, vec_to) / dot_product(vec_to, vec_to)

def projection_same_dir(vec_from, vec_to):
    if magnitude(vec_to) == 0:
        return True
    const = dot_product(vec_from, vec_to) / dot_product(vec_to, vec_to)
    return const > 0

def get_parent_folder(path):
    return "\\".join(path.split("\\")[:-1])