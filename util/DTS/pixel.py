import random
import numpy as np

class Pixel: #Gene
    def __init__(self, x, y):
        self._offsetX = 0
        self._offsetY = 0
        self._x = x
        self._y = y
        self._startX = x
        self._startY = y
        self.moveX = 0
        self.moveY = 0
        self.next_moveX = 0
        self.next_moveY = 0
        self.transparency = 0 # How transparent the black pixel is.
        self.stroke = 0
        self.neighboor_indexes = []
        self.neighboor_dist = {}
        self.neighboor_dir = {}
        self.distance_to_nearest = 0
        self.counter_force_last = (0,0)
        self.black_hole_force_last = (0,0)
        self.rubber_force_last = (0,0)
        self.close_black_holes = []
        self.close_black_holes_pos = (0,0)
        self.rubber_counter = 0
        self.hits = 0

    def getX(self):
        return self._x + self.moveX
    
    def getY(self):
        return self._y + self.moveY

    def getPos(self):
        return (self.getX(), self.getY())

    def setPos(self, pos):
        x,y = pos
        self._x = round(x)
        self.moveX = x - round(x)
        self._y = round(y)
        self.moveY = y - round(y)

    def clone(self):
        clone = Pixel(self._x, self._y)
        clone.moveX = self.moveX
        clone.moveY = self.moveY
        clone.transparency = self.transparency
        clone.stroke = self.stroke
        clone.neighboor_indexes = self.neighboor_indexes
        return clone

    def add_neighboor(self, pos, value, index, min_max, index_dict):
        x, y = pos
        if x == self._x and y == self._y:
            return
        if (x,y) in index_dict:
            self.neighboor_indexes.append(index_dict[(x,y)])
            min_max[index] = value

    def connect_neighboors(self, index_dict):
        n = 64
        xMin = max(0, self._x - 1)
        xMax = min(n-1, self._x + 1)    
        yMin = max(0, self._y - 1)
        yMax = min(n-1, self._y + 1)
        min_max = [xMin, xMax, yMin, yMax]
        self.add_neighboor((xMin, self._y), self._x, 0, min_max, index_dict)
        self.add_neighboor((xMax, self._y), self._x, 1, min_max, index_dict)
        self.add_neighboor((self._x, yMin), self._y, 2, min_max, index_dict)
        self.add_neighboor((self._x, yMax), self._y, 3, min_max, index_dict)
        xMin, xMax, yMin, yMax = min_max
        for x in range(xMin, xMax+1):
            for y in range(yMin, yMax+1):
                if x == self._x or y == self._y:
                    continue
                if (x,y) in index_dict:
                    self.neighboor_indexes.append(index_dict[(x,y)])
    def get_neighboors(self, all_pixels): #Can return none after transformation
        return [all_pixels[index] for index in self.neighboor_indexes]

    def out_of_bound(self):
        n = 64
        return self._x < 0 or self._x >= n or self._y < 0 or self._y >= n

    