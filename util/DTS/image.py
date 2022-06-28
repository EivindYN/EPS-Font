from collections import defaultdict
import numpy as np
from util.DTS.pixel import Pixel
import math
import random
from enum import Enum
from util.DTS.util import *
from typing import List
ModifyMode = Enum('ModifyMode', 'APPLY REMOVE DISTANCE COUNT')

class Image(): #Chromosome
    def __init__(self, name, base, goal):
        self.done = False
        self.done_counter = 0
        self.name = name
        self.scaleX = 1.0
        self.scaleY = 1.0
        self.rotation = 0 # In radians
        self.moveX = 0
        self.moveY = 0
        self.stroke = 4
        self.base = base #True = White, False = Black
        self.goal = goal #True = White, False = Black
        self.debug_image = None
        self.old_stroke = 0 #0: skeleton
        self.distance_dict = {}
        self.distance_iterator = None
        self.distance_iterator_half = None

    def make_distance_dict(self, pixels):
        transparency_dict = self.make_stroke(pixels)
        n = 64
        image = np.array([[transparency_dict.get((x,y),self.stroke + 1) for x in range(n)] for y in range(n)])
        return image 

    def make_rubber(self):
        n = 64
        #distance_count = 0
        edge_dict = {}
        #total_dist = 0
        #self.distance_dict = self.make_distance_dict(self.pixels)
        for pixel in self.pixels:
            for neighboor in pixel.get_neighboors(self.pixels):
                lessX = pixel._startX < neighboor._startX
                sameX = pixel._startX == neighboor._startX
                lessY = pixel._startY < neighboor._startY
                sameXlessY = sameX and lessY
                key = (pixel, neighboor) if lessX or sameXlessY else (neighboor, pixel)
                x, y = pixel._startX, pixel._startY
                x2, y2 = neighboor._startX, neighboor._startY
                if key not in edge_dict:
                    #distance_count += 1
                    edge_dict[key] = True
                    #total_dist += distance((x*self.scaleX, y*self.scaleY), (x2*self.scaleX, y2*self.scaleY))
                    me_scaled = np.array((x*self.scaleX, y*self.scaleY))
                    other_scaled = np.array((x2*self.scaleX, y2*self.scaleY))
                    pixel.neighboor_dist[neighboor] = distance(me_scaled, other_scaled)
                    neighboor.neighboor_dist[pixel] = distance(me_scaled, other_scaled)
                    pixel.neighboor_dir[neighboor] = normalize(other_scaled - me_scaled)
                    neighboor.neighboor_dir[pixel] = normalize(me_scaled - other_scaled)


        #self.avg_dist = total_dist/distance_count

    def find_body(self, pos, stroke):
        x_raw, y_raw = pos
        x, y = round(x_raw), round(y_raw)
        n = 64
        stroke_raw = stroke
        stroke = round(stroke_raw) + 1
        counter_hit = 0
        counter_any = 0
        for x_stroke in range(x - stroke, x + stroke + 1):
            for y_stroke in range(y - stroke, y + stroke + 1):
                if x_stroke < 0 or x_stroke >= n or y_stroke < 0 or y_stroke >= n:
                    continue
                x_diff = abs(x_stroke - x_raw)
                y_diff = abs(y_stroke - y_raw)
                #print(x_raw)
                distance = math.sqrt(x_diff**2 + y_diff**2)
                if distance > stroke_raw:
                    continue
                counter_any += 1
                if not self.goal[y_stroke][x_stroke]:
                    counter_hit += 1
        return counter_hit/counter_any

    def make_pixels(self):
        n = 64
        self.pixels : List[Pixel] = []
        index_dict = {}
        index = 0
        for y in range(n):
            for x in range(n):
                if not self.base[y][x]:
                    self.pixels.append(Pixel(x, y))
                    index_dict[(x,y)] = index
                    index += 1
        for pixel in self.pixels:
            pixel.connect_neighboors(index_dict)

    def add_path(self, path_start, pixel, found_pixels):
        if path_start in found_pixels:
            return
        v = [pixel]
        q = [path_start]
        while q:
            current = q.pop()
            v.append(current)
            for neighboor in current.get_neighboors(self.pixels):    
                if neighboor in v:
                    continue
                if len(neighboor.neighboor_indexes) == 2:
                    q.append(neighboor)
                else:
                    v.append(neighboor)
        self.paths.append(v)
        found_pixels += v[:-1]
    
    # find paths in character, t has 4, h has 3, - has 1
    def find_paths(self):
        found_pixels = []
        self.paths = []
        for pixel in self.pixels:
            if len(pixel.neighboor_indexes) != 2:
                for path_start in pixel.get_neighboors(self.pixels):
                    self.add_path(path_start, pixel, found_pixels)
        for pixel in self.pixels:
            if pixel not in found_pixels:
                for path_start in pixel.get_neighboors(self.pixels)[:-1]:
                    self.add_path(path_start, pixel, found_pixels)
        return self.paths

    # find individual pieces in character, e.x. % is 3, " is 2, ' is 1
    def find_endpoints(self):
        found_pixels = []
        self.endpoints = []
        for pixel in self.pixels:
            if pixel in found_pixels:
                continue
            all_neighboors = [pixel]
            self.get_all_neighboors(pixel, all_neighboors)
            endpoint = all_neighboors[0]
            for neighboor in all_neighboors:
                if len(neighboor.neighboor_indexes) == 1:
                    endpoint = neighboor
                    break
            self.endpoints.append(endpoint)
            found_pixels += all_neighboors

    def get_all_neighboors(self, pixel, all_neighboors):
        for neighboor in pixel.get_neighboors(self.pixels):
            if not neighboor in all_neighboors:
                all_neighboors.append(neighboor)
                self.get_all_neighboors(neighboor, all_neighboors)

    def clone(self):
        image = Image(self.goal)
        image.scaleX = self.scaleX
        image.scaleY = self.scaleY
        image.rotation = self.rotation
        image.moveX = self.moveX
        image.moveY = self.moveY
        image.stroke = self.stroke
        image.endpoints = self.endpoints
        image.pixels = self.clone_pixels()
        return image

    def clone_pixels(self):
        pixels = []
        for pixel in self.pixels:
            pixels.append(pixel.clone())
        return pixels

    def make_distance_iterator(self):
        self.distance_iterator = {}
        self.distance_iterator_half = {}
        limit = 33
        for y in range(-limit,limit + 1):
            for x in range(-limit,limit+1):
                distance = math.sqrt(x**2 + y**2)
                if distance > limit:
                    continue
                if math.floor(distance) not in self.distance_iterator:
                    self.distance_iterator[math.floor(distance)] = []
                    self.distance_iterator_half[math.floor(distance)] = []
                self.distance_iterator[math.floor(distance)].append((x,y))
                if x < 0 or (x == 0 and y < 0):
                    self.distance_iterator_half[math.floor(distance)].append((x,y))

    def color_to_skel(self, color_img):
        for pixel in self.pixels:
            color = color_img[pixel._startY][pixel._startX]/255
            offset = self.color_calc(color)
            pixel.moveX = offset[0]
            pixel.moveY = offset[1]


    def setup(self):
        self.make_pixels()
        self.make_distance_iterator()
        #self.find_endpoints()

    def random_section(self, path):
        rnd1 = random.randrange(0,len(path))
        rnd2 = random.randrange(0,len(path))
        return min(rnd1, rnd2), max(rnd1, rnd2)


    def path_from_endpoints(self, endpoints):
        endpoint = random.choice(endpoints)
        path = []
        seen_pixels = [endpoint]
        queue = [endpoint]
        while len(queue) > 0:
            pixel = queue.pop()
            path.append(pixel)
            for neighboor in pixel.get_neighboors(self.pixels):
                if not neighboor in seen_pixels:
                    seen_pixels.append(neighboor)
                    queue.append(neighboor)
            queue.sort(key=lambda x: -distance_pow_2(pixel.getPos(), x.getPos()))
        return path

    def fitness_info(self, image):
        n = 64
        overlap = 0
        miss_goal = 0
        miss_image = 0
        total_goal = 0
        for y in range(n):
            for x in range(n):
                if not self.goal[y][x]:
                    total_goal += 1
                    if not image[y][x]:
                        overlap += 1
                    else:
                        miss_goal += 1
                elif not image[y][x]:
                    miss_image += 1
        return overlap, miss_goal, miss_image#, total_goal

    def fitness_empty_info(self):
        n = 64
        overlap = 0
        miss_goal = 0
        miss_image = 0
        for y in range(n):
            for x in range(n):
                if not self.goal[y][x]:
                    miss_goal += 1
        return overlap, miss_goal, miss_image

    def fitness_empty(self):
        _, miss_goal, miss_image = self.fitness_empty_info()
        return miss_goal * 5 + miss_image

    def make_offset_image(self):
        n = 64
        image = [[np.array([1,1,1]) for x in range(n)] for y in range(n)]
        for pixel in self.pixels:
            x = pixel._startX
            y = pixel._startY
            moveX = pixel.getX() - x
            moveY = pixel.getY() - y
            image[y][x] = self.offset_calc((moveX, moveY))
        return np.array(image)

    # Output color
    def offset_calc(self, move):
        r = neg1_1_to_0_1(self.offset_neg1_1(move[0]))
        g = neg1_1_to_0_1(self.offset_neg1_1(move[1]))
        if r < 0 or r > 1 or g < 0 or g > 1:
            print("Over limit") #Incase running in pool
            raise Exception("Over limit")
        return np.array([r, g, 0])

    # Output offset
    def color_calc(self, color):
        x = self.color_neg1_1(pos0_1_to_neg1_1(color[0]))
        y = self.color_neg1_1(pos0_1_to_neg1_1(color[1]))
        return (x,y)

    def offset_neg1_1(self, move):
        return (move * 4)/255
        #return (sign(move)*(abs(move)**0.5))/8

    def color_neg1_1(self, color): #TODO
        return (color * 255)/4
        #return sign(color) * ((color * 8) ** 2)

    def fitness(self, image):
        _, miss_goal, miss_image = self.fitness_info(image)
        return miss_goal * 5 + miss_image

    def image_and_fitness(self, pixels, debug = False):
        image = self.make_image(pixels, debug)
        error = self.fitness(image)
        error *= 1 + self.rotation/(2*math.pi)
        return image, error

    def calculate_fitness(self, pixels):
        _, error = self.image_and_fitness(pixels)
        return error

    def make_image(self, pixels, debug):
        n = 64
        if self.stroke == 0: #FIX should we only fill gaps when stroke = 0? It could be the optimized way of handling it kinda
            for pixel in self.pixels:
                pixel.stroke = 0
            #image = self.fill_gaps(pixels, False) #FIX
            image1 = self.fill_gaps(pixels, True) #FIX
            image = {}
            for thing in image1:
                x,y = thing
                x1 = [round(x)]
                if round(x * 2) % 2 == 1:
                    x1 = [round(x-0.5), round(x+0.5)]
                y1 = [round(y)]
                if round(y * 2) % 2 == 1:
                    y1 = [round(y-0.5), round(y+0.5)]
                for x2 in x1:
                    for y2 in y1:
                        image[(x2,y2)] = 0
        else:
        #    image = self.make_stroke(pixels)
            image = self.make_stroke_filled(pixels)
        if self.debug_image != None:
            image = self.debug_image
            #print("here")
            image = [[image.get((x,y),1) for x in range(n)] for y in range(n)]
            return np.array(image)
        image = [[image.get((x,y),1) for x in range(n)] for y in range(n)]
        if debug:
            image = [[0.15 - sign(float(image[y][x]) - 0.5)*0.05 if not self.goal[y][x] else (image[y][x]/2 + 0.5) for x in range(n)] for y in range(n)]
            #0.1 or 0.2 or 1 or 0.5
        if debug:
            for pixel in pixels:
                x_raw = pixel.getX()
                y_raw = pixel.getY()
                x = round(x_raw)
                y = round(y_raw)
                if out_of_bound((x,y)):
                    continue
                if pixel.mark:
                    image[y][x] = 0.4
                else:
                    image[y][x] = 0.3
        return np.array(image)

    def transform_pixels(self, pixels):
        n = 64
        transform_pixels = []
        for pixel in pixels:
            x = pixel.getX()
            y = pixel.getY()
            y_diff = (y+0.5-n/2) * self.scaleY
            x_diff = (x+0.5-n/2) * self.scaleX
            radius = math.sqrt(y_diff**2+x_diff**2)
            degree = math.atan2(y_diff,x_diff) + self.rotation
            x_new_raw = n/2 + math.cos(degree) * radius - 0.5 + self.moveX
            y_new_raw = n/2 + math.sin(degree) * radius - 0.5 + self.moveY
            x_new = round(x_new_raw)
            y_new = round(y_new_raw)
            transform_pixel = pixel
            transform_pixel._x = x_new
            transform_pixel._y = y_new
            transform_pixel._offsetX += x_new_raw - x_new
            transform_pixel._offsetY += y_new_raw - y_new
            transform_pixels.append(transform_pixel)
        return transform_pixels

    def fill_gaps(self, pixels, fractional = True):
        transparency_dict = {} # Dict of black pixels, transparency determines transparent the black pixel is.
        # Dict is usefull for quickly counting black pixels at auto_stroke, could still be an array but whatever
        for pixel in pixels:
            if fractional:
                pos = pixel.getPos()
            else:
                pos = round(pixel.getX()), round(pixel.getY())
            x, y = pos
            if pos not in transparency_dict.keys() or transparency_dict[pos] > pixel.transparency:
                transparency_dict[pos] = pixel.stroke #was pixel.transparency
            for neighboor in pixel.get_neighboors(pixels):
                if not neighboor.out_of_bound(): # Necessary since it could be out of bound after transformations
                    if fractional:
                        x_2, y_2 = neighboor.getPos()
                    else:
                        x_2 = round(neighboor.getX())
                        y_2 = round(neighboor.getY())
                    t_2 = neighboor.transparency
                    s_2 = neighboor.stroke
                    x_0 = x
                    y_0 = y
                    t_0 = pixel.transparency
                    s_0 = pixel.stroke
                    self.fillGap(transparency_dict, x_0, y_0, t_0, s_0, x_2, y_2, t_2, s_2, fractional)
        return transparency_dict


    def make_stroke(self, pixels):
        transparency_dict = {} # Dict of black pixels, transparency determines transparent the black pixel is.
        for pixel in pixels:
            self.pixel_stroke(pixel, transparency_dict, ModifyMode.APPLY)
        return transparency_dict

    def make_stroke_filled(self, pixels):
        transparency_dict = self.fill_gaps(pixels)
        transparency_dict2 = {}
        for pos in transparency_dict: #FIX ignore transparency and pixel stroke
            self.do_stroke(pos, self.stroke + transparency_dict[pos], 0, transparency_dict2, ModifyMode.APPLY, False)
        return transparency_dict2

    def make_stroke_image(self, pixels):
        transparency_dict = self.make_stroke(pixels)
        n = 64
        image = np.array([[transparency_dict.get((x,y),1) for x in range(n)] for y in range(n)])
        return image 

    def make_stroke_image_layer(self, pixels):
        transparency_dict = {} # Dict of black pixels, transparency determines transparent the black pixel is.
        for pixel in pixels:
            self.pixel_stroke(pixel, transparency_dict, ModifyMode.COUNT)
        n = 64
        image = np.array([[transparency_dict.get((x,y),0) for x in range(n)] for y in range(n)])
        return image

    def pixel_stroke(self, pixel, image, modifymode):
        x = pixel.getX()
        y = pixel.getY()
        stroke = self.stroke + pixel.stroke #+ (2 if modifymode == ModifyMode.COUNT else 0)
        self.do_stroke((x,y), stroke, pixel.hits, image, modifymode)

    def do_stroke(self, pos, stroke, hits, image, modifymode, blurry = True):
        n = 64
        x_raw,y_raw = pos
        x,y = round(x_raw), round(y_raw)
        transparency_dict = image
        stroke_raw = stroke
        stroke = math.ceil(stroke_raw)
        for distance in range(0,stroke):
            for pos in self.distance_iterator[distance]:
                x_stroke = x + pos[0]
                y_stroke = y + pos[1]
                if x_stroke < 0 or x_stroke >= n or y_stroke < 0 or y_stroke >= n:
                    continue
                x_diff = abs(x_stroke - x_raw)
                y_diff = abs(y_stroke - y_raw)
                #print(x_raw)
                distance = math.sqrt(x_diff**2 + y_diff**2)
                if distance > stroke_raw:
                    continue
                transparency = 0
                if blurry:
                    transparency = 1 - min(stroke_raw - distance, 1)
                pos = (x_stroke, y_stroke)
                if modifymode == ModifyMode.APPLY:
                    if pos not in transparency_dict.keys() or transparency_dict[pos] > transparency:
                        transparency_dict[pos] = transparency
                elif modifymode == ModifyMode.REMOVE:
                    if pos in transparency_dict.keys():
                        del transparency_dict[pos]
                elif modifymode == ModifyMode.DISTANCE:
                    raise Exception("not implemented")
                    if pos not in transparency_dict.keys() or transparency_dict[pos] > distance:
                        transparency_dict[pos] = distance
                elif modifymode == ModifyMode.COUNT:
                    if not self.goal[y_stroke][x_stroke]:
                        if pos not in transparency_dict.keys():
                            transparency_dict[pos] = 0
                        transparency_dict[pos] += 1#hits
        
    def fillGap(self, transparency_dict, x_0, y_0, t_0, s_0, x_2, y_2, t_2, s_2, fractional):
        if abs(round(x_0) - round(x_2)) > 1 or abs(round(y_0) - round(y_2)) > 1:
            x_1 = (x_0 + x_2)/2
            y_1 = (y_0 + y_2)/2
            diff_0_1 = math.sqrt((round(x_0) - round(x_1))**2 + (round(y_0) - round(y_1))**2)
            diff_1_2 = math.sqrt((round(x_1) - round(x_2))**2 + (round(y_1) - round(y_2))**2)
            t_1 = t_0 + (t_2-t_0)*diff_0_1/(diff_0_1 + diff_1_2)
            s_1 = round(s_0 + (s_2-s_0)*diff_0_1/(diff_0_1 + diff_1_2))
            if fractional:
                pos = x_1, y_1
            else:
                pos = (round(x_1), round(y_1))        
            if pos not in transparency_dict.keys() or transparency_dict[pos] > t_1:
                transparency_dict[pos] = s_1 # was t_1
            self.fillGap(transparency_dict, x_0, y_0, t_0, s_0, x_1, y_1, t_1, s_1, fractional)
            self.fillGap(transparency_dict, x_1, y_1, t_1, s_1, x_2, y_2, t_2, s_2, fractional)
            
    def find_corners(self, image):
        n = 64
        x_min = n
        x_max = 0
        y_min = n
        y_max = 0
        for x in range(n):
            for y in range(n):
                if not image[y][x]:
                    x_min = min(x, x_min)
                    x_max = max(x, x_max)
                    y_min = min(y, y_min)
                    y_max = max(y, y_max)
        return x_min, x_max, y_min, y_max

    def diff_from_corner(self, corner):
        return corner[1] - corner[0] + 1, corner[3] - corner[2] + 1 #width, height [of black pixels], +1: if stroke = 0 then we dont want to get 0, we want 1
    
    def middle_from_corner(self, corner):
        return (corner[1] + corner[0])/2, (corner[3] + corner[2])/2 #middle [of black pixels]

    def resize(self, diff_mid):
        body_diff, base_diff, body_mid, base_mid = diff_mid

        self.scaleX = body_diff[0]/base_diff[0]
        self.scaleY = body_diff[1]/base_diff[1]
        self.moveX = body_mid[0] - base_mid[0]
        self.moveY = body_mid[1] - base_mid[1]

        pixels = self.clone_pixels()
        return self.transform_pixels(pixels)

    def calculate_diff_mid(self, image):
        corners = self.find_corners(image)
        diff = self.diff_from_corner(corners)
        mid = self.middle_from_corner(corners)
        return diff, mid

    def calculate_diff_mid_full(self): 
        body_diff, body_mid = self.calculate_diff_mid(self.goal)
        base_diff, base_mid = self.calculate_diff_mid(self.base)
        diff_mid = [body_diff, base_diff, body_mid, base_mid]
        self.update_diff_mid(diff_mid)
        return diff_mid

    def update_diff_mid(self, diff_mid):
        diff_mid[0] = (diff_mid[0][0] - self.stroke*2 + self.old_stroke*2, diff_mid[0][1] - self.stroke*2 + self.old_stroke * 2)
        if diff_mid[0][0] <= 0 or diff_mid[0][1] <= 0:
            if diff_mid[0][0] <= 0:
                diff_mid[0] = (diff_mid[1][0], diff_mid[0][1])
            else:
                diff_mid[0] = (diff_mid[0][0], diff_mid[1][1])
        self.old_stroke = self.stroke

    def count_black_pixels(self, transparency_dict):
        sum = 0
        for item in transparency_dict:
            sum += 1 - transparency_dict[item]
        return sum

    def auto_stroke(self, diff_mid):
        n = 64
        body_black_pixels_num = n**2 - sum(sum(self.goal))
        counter = 0
        while (True):
            counter += 1
            old_stroke = self.stroke
            self.update_diff_mid(diff_mid)
            pixels = self.resize(diff_mid)
            base_black_pixels_num = self.count_black_pixels(self.make_stroke(pixels))
            self.stroke = self.stroke + self.stroke * ((body_black_pixels_num/base_black_pixels_num)**(0.5) - 1) # One iteration of pewtons formula
            #print(abs(self.stroke - old_stroke))
            if abs(self.stroke - old_stroke) < min(0.25, counter * 0.01):
                return
            if counter > 30:
                print("auto_stroke is running a lot for this font:", self.name)
    def auto_morph(self):
        diff_mid = self.calculate_diff_mid_full()
        self.auto_stroke(diff_mid)
        self.pixels = self.resize(diff_mid)

    def get_neighboors(self, pixel, depth):
        visited = {}
        queue = []
        queue.append((pixel, depth))
        while queue:
            p, d = queue.pop(0)
            visited[p] = d
            if d == 1:
                continue
            for neighboor in p.get_neighboors(self.pixels):
                if neighboor not in visited:
                    queue.append((neighboor,d - 1))
        return visited

    # NB: This function drops overlapping pixels (because its faster)
    def make_dict(self):
        transparency_dict = {} # Dict of black pixels, transparency determines transparent the black pixel is.
        for pixel in self.pixels:
            x_raw, y_raw = pixel.getPos()
            x, y = round(x_raw), round(y_raw)
            transparency_dict[(x,y)] = pixel
        return transparency_dict

    #FIX: Consider optimizing this
    def find_closest_pixel(self, pos, pixel_dict):
        x, y = pos
        lowest_dist = float("inf")
        for dist in range(64):
            if lowest_dist <= dist**2: # <  if we really care about pythagoras
                break
            for x_2 in range(x - dist,x + dist + 1):
                if x_2 == x - dist or x_2 == x + dist:
                    area = range(y - dist,y + dist + 1)
                else:
                    area = [y - dist, y + dist]
                for y_2 in area:
                    if (x_2, y_2) in pixel_dict:
                        distance = distance_pow_2((x,y), (x_2,y_2))
                        if lowest_dist > distance:
                            lowest_dist = distance
                            lowest_pos = [(x_2, y_2)]
                        elif lowest_dist == distance:
                            lowest_pos.append((x_2, y_2))
        return np.array(random.choice(lowest_pos)) 

        #FIX: Consider optimizing this
    def find_closest_black_pixel(self, pos):
        x_raw, y_raw = pos
        x, y = round(x_raw), round(y_raw)
        limit = 1
        for distance in range(limit):
            for pos in self.distance_iterator[distance]:
                x2, y2 = x + pos[0], y + pos[1]
                if not out_of_bound((x2,y2)):
                    if not self.goal[y2][x2]:
                        return distance
        return limit

    def get_rubber_force_mat(self, pixel: Pixel, condition):
        force_total = np.array((0,0), dtype = float) 
        if len(pixel.neighboor_indexes) == 1 and condition:
            return force_total
        for neighboor in pixel.get_neighboors(self.pixels):
            toward_neighboor = np.array(neighboor.getPos()) - np.array(pixel.getPos())
            material_force = 0.5 #if not condition else 0.1
            original = pixel.neighboor_dist[neighboor] * pixel.neighboor_dir[neighboor]
            new = toward_neighboor
            force = (new - original) * material_force
            force_total += force
        return force_total

    def get_rubber_force_bend(self, pixel: Pixel, condition):
        force_total = np.array((0,0), dtype = float) 
        if len(pixel.neighboor_indexes) == 1 and condition:
            return force_total
        for neighboor in pixel.get_neighboors(self.pixels):
            toward_neighboor = np.array(neighboor.getPos()) - np.array(pixel.getPos())
            if magnitude(toward_neighboor) <= 0.5:
                toward_neighboor_normalized = pixel.neighboor_dir[neighboor]
            else:
                toward_neighboor_normalized = normalize(toward_neighboor)

            bend_force =  0.5
            old_dir = pixel.neighboor_dir[neighboor]
            new_dir = toward_neighboor_normalized
            toward_old_dir = new_dir - old_dir
            force = toward_old_dir * bend_force
            force_total += force

            self.splash_force(pixel, condition, force, 5, [neighboor])
            #if len(neighboor.neighboor_indexes) == 1 and condition:
            #    force *= 10

        #if len(pixel.neighboor_indexes) == 1 and condition:
        #    return force * 0.5
        return force_total

    # Vector that points toward best fitness increase
    def move_vector(self, pixel, count_image, distance, condition2):
        n = 64
        x = round(pixel.getX())
        y = round(pixel.getY())
        stroke_raw = distance
        stroke = math.ceil(stroke_raw)
        vector = np.array((0,0), dtype = float)
        strength_2 = 0.1
        lowest_dist = float("inf")
        highest_dist = 0
        for distance in range(1,stroke):
            if highest_dist > self.stroke and (highest_dist - lowest_dist) > self.stroke / 2:
                if distance > self.stroke + (2 if condition2 else 8):
                    break
            for pos in self.distance_iterator_half[distance]:
                x_stroke = x + pos[0]
                y_stroke = y + pos[1]
                x_stroke2 = x - pos[0]
                y_stroke2 = y - pos[1]
                no_move = x_stroke < 0 or x_stroke >= n or y_stroke < 0 or y_stroke >= n
                no_move2 = x_stroke2 < 0 or x_stroke2 >= n or y_stroke2 < 0 or y_stroke2 >= n
                if not no_move:
                    no_move = self.goal[y_stroke][x_stroke]
                if not no_move2:
                    no_move2 = self.goal[y_stroke2][x_stroke2]
                if no_move and no_move2:
                    continue
                if distance < lowest_dist:
                    lowest_dist = distance
                if distance > highest_dist:
                    highest_dist = distance
                strength = 0
                limit = 1 if distance < self.stroke else 0
                overlap_str = 0 #if condition2 else 0.5
                #how many overlap = how little they affect?
                if not no_move and no_move2:
                    strength += 1 if count_image[y_stroke][x_stroke] <= limit else overlap_str
                elif not no_move2 and no_move:
                    strength -= 1 if count_image[y_stroke2][x_stroke2] <= limit else overlap_str
                else:
                    if count_image[y_stroke][x_stroke] <= limit:
                        strength += 1 - overlap_str
                    if count_image[y_stroke2][x_stroke2] <= limit:
                        strength -= 1 - overlap_str
                strength *= 1 - distance/stroke_raw #FIX: Endre til 1?
                vector += normalize(np.array((x_stroke - x, y_stroke - y), dtype = float)) * strength * strength_2
        return vector

    def splash_force(self, start_pixel: Pixel, condition, force_total, amount, affected):
        queue = [(start_pixel, force_total / 2, amount)]
        while queue:
            pixel, force, amount = queue.pop(0)
            if amount == 0:
                continue
            if not(len(pixel.neighboor_indexes) == 1 and condition):
                pixel.next_moveX += force[0]
                pixel.next_moveY += force[1]
            for neighboor in pixel.get_neighboors(self.pixels):
                if neighboor not in affected:
                    affected.append(neighboor)
                    queue.append((neighboor, force*0.8, amount - 1))

    def transfer_vector(self, pixel: Pixel, condition):
        force = np.array((0,0), dtype = float) 
        if len(pixel.neighboor_indexes) == 1 and condition:
            return force
        highest_hits = 0
        for neighboor in pixel.get_neighboors(self.pixels):
            if neighboor.hits > highest_hits:
                highest_hits = neighboor.hits
        for neighboor in pixel.get_neighboors(self.pixels):
            toward_neighboor = np.array(neighboor.getPos()) - np.array(pixel.getPos())
            if magnitude(toward_neighboor) <= 0.5:
                toward_neighboor_normalized = pixel.neighboor_dir[neighboor]
            else:
                toward_neighboor_normalized = normalize(toward_neighboor)
            if neighboor.distance_to_nearest < pixel.distance_to_nearest:
                force += toward_neighboor_normalized * (pixel.distance_to_nearest - neighboor.distance_to_nearest) * 0.5
        return force

    def pixel_force(self, pixel: Pixel, count_image, condition, condition2, condition3):
        optimize_force = self.move_vector(pixel, count_image, 16, condition2) if condition else np.array((0,0), dtype = float)
        optimize_force *= 1
        if not condition2:
            optimize_force *= 0.5
            optimize_force *= 2
            if self.done_counter <= 2:
                optimize_force *= 2.5
        if condition2:
            optimize_force *= 1
            if self.done_counter >= 29:
                optimize_force /= 1
            else:
                optimize_force *= 2.5

        rubber_force = self.get_rubber_force_mat(pixel, condition3) * (2 / len(pixel.neighboor_indexes))

        if not projection_same_dir(rubber_force, pixel.rubber_force_last):
            rubber_force *= 0.25
        pixel.rubber_force_last = rubber_force

        
        self.get_rubber_force_bend(pixel, condition3) * (2 / len(pixel.neighboor_indexes))

        force_total = optimize_force + rubber_force

        forceX, forceY = force_total

        pixel.next_moveX += forceX
        pixel.next_moveY += forceY

    def find_adjacent_hits(self, pixel: Pixel, hits):
        if pixel.distance_to_nearest > 0:
            return [pixel]
        hits.append(pixel)
        for neighboor in pixel.get_neighboors(self.pixels): #could be infinite idk
            if neighboor not in hits:
                self.find_adjacent_hits(neighboor, hits)
        return hits

    def find_adjacent_miss(self, pixel: Pixel, hits):
        if pixel.distance_to_nearest == 0:
            return [pixel]
        hits.append(pixel)
        for neighboor in pixel.get_neighboors(self.pixels): #could be infinite idk
            if neighboor not in hits:
                self.find_adjacent_miss(neighboor, hits)
        return hits

    def transfer_work(self):
        miss_cluster = []
        miss_used = set()
        for pixel in self.pixels:
            if pixel in miss_used:
                continue
            hits = self.find_adjacent_miss(pixel, [])
            miss_used.update(hits)
            if len(hits) > 1:
                miss_cluster.append(hits)
        #print(miss_cluster)
        for miss in miss_cluster:
            edges = []
            for pixel in miss:
                for neighboor in pixel.get_neighboors(self.pixels):
                    if neighboor not in miss and neighboor not in edges:
                        edges.append(neighboor)
            for pixel in miss:
                for edge in edges:
                    toward_edge = np.array(edge.getPos()) - np.array(pixel.getPos())
                    if magnitude(toward_edge) < 0.01:
                        continue
                    force = 0.5*normalize(toward_edge)
                    pixel.next_moveX += force[0]
                    pixel.next_moveY += force[1]                    
                    pixel.mark = True
        #input()
    # Works well when its necessary, but all the other times it will ruin the skeleton, so that they dont overlap correctly
    def set_stroke(self, pixel):
        pos = pixel.getPos()
        x_raw, y_raw = pos
        x, y = round(x_raw), round(y_raw)
        for distance in range(16):
            balance = -1
            for pos in self.distance_iterator[distance]:
                x2, y2 = x + pos[0], y + pos[1]
                if not out_of_bound((x2,y2)):
                    balance += distance/self.stroke if self.goal[y2][x2] else -1
            if balance > 0:
                pixel.stroke = distance - 1 - self.stroke
                return
        pixel.stroke = 0


    def update(self):
        # todo: black hole force, gummi force

        condition1 = (self.done_counter % 5 == 0 or self.done_counter < 5) and self.done_counter < 25
        condition2 = self.done_counter >= 25 # and self.done_counter < 195
        condition3 = self.done_counter >= 5
        condition = condition1 or condition2
        count_image = None

        for pixel in self.pixels:
            pos = pixel.getPos()
            pixel.distance_to_nearest = self.find_closest_black_pixel(pos)
            pixel.mark = False
            pixel.next_moveX = 0
            pixel.next_moveY = 0

        if condition:
            count_image = self.make_stroke_image_layer(self.pixels)

        for pixel in self.pixels:
            self.pixel_force(pixel, count_image, condition, condition2, condition3)

        if not condition2:
            self.transfer_work()

        done = False
        for pixel in self.pixels:
            pixel.moveX += pixel.next_moveX
            pixel.moveY += pixel.next_moveY
            pixel.next_moveX = 0
            pixel.next_moveY = 0
        self.done_counter += 1 #* 2
        self.done = done
        if self.done_counter >= 32:
            self.done = True

            
    def auto_resize(self, pixels):
        pass #(x,y) = center + diff * scale + (diff X rot_vec) * scale

    # moveX and moveY must be 0
    def auto_move(self, pixels = None): #0.01s
        if self.stroke == 0:
            raise Exception("Not supported probably")
        if pixels is None:
            pixels = self.pixels
        for pixel in pixels:
            x = pixel.getX()
            y = pixel.getY()
            old_pos = np.array((x,y)) 
            lowest_dist = float("inf")
            lowest_pos = []
            target = not self.goal[y][x] # If allready black, look for white instead
            for dist in range(16):
                if lowest_dist <= dist**2: # <  if we really care about pythagoras
                    break
                for x_2 in range(x - dist,x + dist + 1):
                    if x_2 == x - dist or x_2 == x + dist:
                        area = range(y - dist,y + dist + 1)
                    else:
                        area = [y - dist, y + dist]
                    for y_2 in area:
                        if out_of_bound((x_2,y_2)):
                            continue
                        if self.goal[y_2][x_2] == target:
                            distance = distance_pow_2((x*1.5,y), (x_2*1.5,y_2))
                            if lowest_dist > distance:
                                lowest_dist = distance
                                lowest_pos = [(x_2, y_2)]
                            elif lowest_dist == distance:
                                lowest_pos.append((x_2, y_2))
            if len(lowest_pos) == 0:
                if not target:
                    print("Error: Middle of white area, should look at this")
                continue 
            lowest_pos = np.array(random.choice(lowest_pos)) 
            direction = normalize(lowest_pos - old_pos)
            if target: # If looked for white
                direction *= -1
            distance = 0
            current_pos = np.array(lowest_pos, dtype = float) # also acts as a deep copy
            while True:
                current_pos += direction
                x_2 = round(current_pos[0]) 
                y_2 = round(current_pos[1]) 
                if self.goal[y_2][x_2]:
                    break
                distance += 1
                if distance > self.stroke * 2:
                    break
            stroke = distance//2 - 1
            pixel.stroke = stroke - self.stroke
            end_pos = current_pos
            new_pos = (lowest_pos + end_pos)/2
            pixel.moveX += round(new_pos[0]) - pixel.getX()
            pixel.moveY += round(new_pos[1]) - pixel.getY()

        # moveX and moveY must be 0
    def get_all_condition(self, pos, condition, cluster_dict_pos, cluster_dict, index):
        cluster_dict_pos[pos] = index
        cluster_dict[index].append(pos)
        n = 64
        x, y = pos
        xMin = max(0, x - 1)
        xMax = min(n-1, x + 1)
        yMin = max(0, y - 1)
        yMax = min(n-1, y + 1)
        for x_2 in range(xMin, xMax+1):
            for y_2 in range(yMin, yMax+1):
                if condition((x_2,y_2)):
                    self.get_all_condition((x_2, y_2), condition, cluster_dict_pos, cluster_dict, index)

    def average_morph(self, pixels, edge_keep = False):
        pixel_dict = {}
        for pixel in pixels:
            neighboors = pixel.get_neighboors(pixels)
            #if len(neighboors) == 1: #This after optimize fitness thingy?
            #    pixel_dict[pixel] = (pixel.moveX, pixel.moveY)
            #    continue
            a = 0.5 if len(neighboors) > 1 else 0.05 # Edges stay still
            b = 0.5 if len(neighboors) > 1 else 0.95
            movement = np.array((0.0,0.0))
            for neighboor in neighboors:
                movement += np.array((neighboor.moveX, neighboor.moveY)) * a * 0.5 / len(neighboors)
                neighboors2 = neighboor.get_neighboors(pixels)
                for neighboor2 in neighboors2:
                    if neighboor2 == pixel:
                        continue
                    movement += np.array((neighboor2.moveX, neighboor2.moveY)) * a * 0.5 / (len(neighboors)*(len(neighboors2) - 1))
            movement += np.array((pixel.moveX, pixel.moveY)) * b
            pixel_dict[pixel] = movement
        for pixel in pixels:
            movement = pixel_dict[pixel]
            moveX, moveY = round(movement[0]), round(movement[1])
            x,y = pixel._x + moveX, pixel._y + moveY
            #if not self.goal[y][x]:
            pixel.moveX, pixel.moveY = moveX, moveY

                            






                
        



    
