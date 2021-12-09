#relevant imports
import random
from src.helper_functions import *
import copy

    
def surrounding_box(box0,box1):
    """
    surrounding box: a helper function to take two bounding boxes of objects and return a larger
                     box that surrounds both. Note that objects here can refer to individual objects,
                     lists of objects, whatever.
    param box0: the first boundiing box
    param box1: the second bounding box
    """
    small = vec3(min(box0.minimum[0],box1.minimum[0]),min(box0.minimum[1],box1.minimum[1]),min(box0.minimum[2],box1.minimum[2]))
    big = vec3(max(box0.maximum[0],box1.maximum[0]),max(box0.maximum[1],box1.maximum[1]),max(box0.maximum[2],box1.maximum[2]))
    return aabb(small,big)
    


class aabb:
    """
    This class is defined to set bounding boxes for a given object. It is initialized by taking
    two vectors, which act as the planes bounding a given object.
    The hit method checks whether a given ray hits the bounding box for t_min and t_max values
    and if it does, it returns True, else it returns False.
    """
    def __init__(self,a=None,b=None):
        #a and b are vectors
        self.minimum = a
        self.maximum = b
    
    
    def hit(self,r,t_min,t_max):
        for a in range(3):
            t0 = min((self.minimum[a] - r.origin[a]) / r.direction[a],(self.maximum[a]-r.origin[a]) / r.direction[a])
            t1 = max((self.minimum[a] - r.origin[a]) / r.direction[a],(self.maximum[a]-r.origin[a]) / r.direction[a])
 
            t_min = max(t0,t_min)
            t_max = min(t1,t_max)
            if t_max <= t_min:
                return False
        return True
            
    
    



class bvh_node:
    """
    This class is used in constructing a tree that organizes the objects in a hierarchy.
    The parent node always has a larger bounding box than its children.
    It is initialized with source objects, position 0 of the source objects, position end of the source objeccts
    t0, and t1. It recursively calls itself until all objects are in the tree.
    The bounding box for a given node is of course its own box.
    The final function is a hit function designed to test whether a node is hit or not. If it is hit, it
    needs to check its children to see if they're hit. If not, it can simply return False, which results
    in a significant speedup when the world has lots of objects.
    """
    def __init__(self,src_objects,start,end,time0,time1):
        objects = copy.deepcopy(src_objects)
        #axis = random.randint(0,2)
        axis = 1
        #print(axis)
        #print(end-start)
        if axis == 0:
             comparator = box_x_compare
        elif axis == 1:
            comparator = box_y_compare
        else:
            comparator = box_z_compare

        object_span = end - start
        if object_span == 1:
            self.left = self.right = objects[start]
        elif object_span == 2:
            if box_compare(objects[start],objects[start+1],axis):
                self.left = objects[start]
                self.right = objects[start+1]
            else:
                self.left = objects[start+1]
                self.right = objects[start]
        elif object_span == 3:
            self.left = bvh_node(objects,start,start+2,time0,time1)
            self.right = objects[start+2]

        else:

            sorted_objects = sorted(objects,key=comparator)

            mid = int(start + object_span / 2)
            self.left = bvh_node(sorted_objects,start,mid,time0,time1)
            self.right = bvh_node(sorted_objects,mid,end,time0,time1)

        box_left = aabb()
        box_right = aabb()

        bool_left,box_left = self.left.bounding_box(time0,time1,box_left)
        bool_right,box_right = self.right.bounding_box(time0,time1,box_right)
        #if self.left != None and self.right != None:
        if (not bool_left) or ( not bool_right):
                print('No bounding box in bvh node constructor')
        self.box = surrounding_box(box_left,box_right)

    def bounding_box(self,time0,time1,output_box):
        output_box = self.box
        return True,output_box
    #old hit function, left here in case of bugs in newer version
    # def hit(self,r,t_min,t_max,rec):
    #     left_rec = hit_record()
    #     right_rec = hit_record()
    #     if not self.box.hit(r,t_min,t_max):
    #         return False,rec
    #     hit_left,left_rec = self.left.hit(r,t_min,t_max,rec)
    #     if hit_left == True:
    #         temp = left_rec.t
    #     else:
    #         temp = t_max
    #     hit_right,right_rec = self.right.hit(r,t_min,temp,rec)
    #     if hit_left and  not hit_right:
    #         return hit_left,left_rec
    #     if hit_right and not hit_left:
    #         return hit_right,right_rec
    #     if hit_left and hit_right:
    #         test_left = left_rec.point - r.origin
    #         test_right = right_rec.point - r.origin
    #         length_left = np.linalg.norm(test_left)
    #         length_right = np.linalg.norm(test_right)
    #         #smallest difference is closer, return closer
    #         if length_left > length_right:
    #             return True,right_rec
    #         else:
    #             return True,left_rec
    #     if not hit_left and not hit_right:
    #         return False,rec

    def hit(self,r,t_min,t_max,rec):
        if not self.box.hit(r,t_min,t_max):
            return False,rec
        temp_rec = hit_record()
        hit_left,rec = self.left.hit(r,t_min,t_max,rec)
        if hit_left:
            temp = rec.t
            temp_rec = rec
        else:
            temp = t_max
        hit_right,rec = self.right.hit(r,t_min,temp,rec)
        if hit_right:
            temp_rec = rec
        hit_bool = hit_right or hit_left
        return hit_bool,temp_rec






    

def box_compare2(a,axis):
    temp_box = aabb()
    bool_temp,temp_box = a.bounding_box(0,0,temp_box)
    return temp_box.minimum[axis]



def box_compare(a,b,axis):
    box_a = aabb()
    box_b = aabb()
    bool_a,box_a = a.bounding_box(0,0,box_a)
    bool_b,box_b = b.bounding_box(0,0,box_b)

    if (not bool_a) or (not bool_b):
        print('no bounding box in bvh_node constructor')
    return box_a.minimum[axis] < box_b.minimum[axis]

def box_x_compare(a):
    return box_compare2(a,0)
def box_y_compare(a):
    return box_compare2(a,1)
def box_z_compare(a):
    return box_compare2(a,2)




