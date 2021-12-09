import math 
import numpy as np
from src.bvh import aabb
from src.helper_functions import *
import random
from src.hit_record import *
from src.ray import *
import copy
from src.materials import isotropic


#A Parent Class
#This is unused, I left it in here for organization
class hittable:
    def __init__(self):
        x = True

    def hit(self,r,t_min,t_max,rec):
        return False,rec
    def bounding_box(self,time0,time1,output_box):
        return False,output_box




class sphere(hittable):
    '''
    This class a sphere object
    '''
    def __init__(self,center,radius,material):
        '''
        This initialization takes a center (x,y,z), a radius, and a material and sets
        the properties of the sphere based on these inputs.
        '''
        self.center=center
        self.radius=radius
        self.material=material
        
     
    def get_sphere_uv(self,p):
        '''
        This function is used to determine direction if the sphere is classified as an object
        that emits light.
        '''
        theta = np.arccos(-1*p[1])
        phi = np.arctan2(-1*p[2],p[0]) + np.pi
        u = phi / (2*np.pi)
        v = theta / np.pi
        return u,v
        
    def hit(self,r, t_min,t_max,record):
        '''
        This function uses the quadratic formula to calculate the intersection of the ray
        with the boundaries of the sphere. It then updates the hit record accordingly
        with information about the point, the time, the normal, and the u-v unit vectors at the
        normal
        '''
        oc = r.origin - self.center
        a = (np.linalg.norm(r.direction))**2
        half_b = np.dot(oc,r.direction)
        c = np.linalg.norm(oc)**2 - self.radius**2
        discriminant = half_b**2 - a*c
        if discriminant < 0:
            #print('imag solution')
            return False,record
        else:
            sqrtd = np.sqrt(discriminant)
            root = (-half_b - sqrtd) / a
            if root < t_min or t_max < root:
                root = (-half_b + sqrtd) / a
                if root < t_min or t_max < root:
                    return False,record
            

        record.t = root
        record.point = r.at(record.t)
        record.normal = (record.point - self.center) / self.radius
        outward_normal = (record.point - self.center) / self.radius
        u,v = self.get_sphere_uv(outward_normal)
        record.u = u
        record.v = v
        record.set_face_normal(r,outward_normal)
        record.material = self.material
        return True,record
    
    
    def bounding_box(self,time0,time1,output_box):
        '''
        This calculates the box surrounding the sphere using its radius
        '''
        output_box = aabb(self.center - vec3(self.radius,self.radius,self.radius),self.center + vec3(self.radius,self.radius,self.radius)) 
        return True,output_box
   
    
    
class moving_sphere(hittable):
    '''
    This class is the same as the sphere above, only it allows for the sphere to move in time
    '''
    def __init__(self,center0,centerf,t0,tf,radius,material):
        self.center0=center0
        self.centerf = centerf
        self.t0 = t0
        self.tf = tf
        self.radius=radius
        self.material=material
        
    def get_sphere_uv(self,p):
        theta = np.arccos(-1*p[1])
        phi = np.arctan2(-1*p[2],p[0]) + np.pi
        u = phi / (2*np.pi)
        v = theta / np.pi
        return u,v
        
    def center(self,time):
        return self.center0 + ((time - self.t0) / (self.tf - self.t0)) * (self.centerf - self.center0)
    def hit(self,r, t_min,t_max,record):
        oc = r.origin - self.center(r.time)
        a = (np.linalg.norm(r.direction))**2
        half_b = np.dot(oc,r.direction)
        c = np.linalg.norm(oc)**2 - self.radius**2
        discriminant = half_b**2 - a*c
        if discriminant < 0:
            #print('imag solution')
            return False,record
        else:
            sqrtd = np.sqrt(discriminant)
            root = (-half_b - sqrtd) / a
            if root < t_min or t_max < root:
                root = (-half_b + sqrtd) / a
                if root < t_min or t_max < root:
                    return False,record
            
        record.t = root
        record.point = r.at(record.t)
        record.normal = (record.point - self.center(r.time)) / self.radius
        outward_normal = (record.point - self.center(r.time)) / self.radius
        u,v = self.get_sphere_uv(outward_normal)
        record.u = u
        record.v = v
        record.set_face_normal(r,outward_normal)
        record.material = self.material
        return True,record
    
    def bounding_box(self,time0,time1,output_box):
        #this bounding box is for the sphere's entire range of motion.
        box0 = aabb(self.center(time0) - vec3(self.radius,self.radius,self.radius),self.center(time0) + vec3(self.radius,self.radius,self.radius)) 
        box1 = aabb(self.center(time1) - vec3(self.radius,self.radius,self.radius),self.center(time1) + vec3(self.radius,self.radius,self.radius)) 
        output_box = surrounding_box(box0,box1)
        return True,output_box
    
    
    

class xy_rect(hittable):
    '''
    This is a axis aligned rectangle specified by 
    '''
    def __init__(self,x0,x1,y0,y1,k,material):
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        self.k = k
        self.material = material
        
        
    def hit(self,r,t_min,t_max,rec):

        t = (self.k-r.origin[2]) / r.direction[2]
        if t < t_min or t > t_max:
            return False,rec
        x = r.origin[0] + t*r.direction[0] 
        y = r.origin[1] + t*r.direction[1]
        if x < self.x0 or x > self.x1 or y< self.y0 or y > self.y1:
            return False,rec
        rec.u = (x-self.x0)/(self.x1-self.x0)
        rec.v = (y-self.y0)/(self.y1-self.y0)
        rec.t = t
        outward_normal = vec3(0,0,1)
        rec.set_face_normal(r,outward_normal)
        rec.material = self.material
        rec.point = r.at(t)
        return True,rec
        
        
        
        
    def bounding_box(self,time0,time1,output_box):
        output_box = aabb(vec3(self.x0,self.y0,self.k-0.0001),vec3(self.x1,self.y1,self.k+0.0001))
        return True,output_box
    
    
    
class xz_rect(hittable):

    def __init__(self,x0,x1,z0,z1,k,material):
        self.x0 = x0
        self.x1 = x1
        self.z0 = z0
        self.z1 = z1
        self.k = k
        self.material = material
        
        
    def hit(self,r,t_min,t_max,rec):

        t = (self.k-r.origin[1]) / r.direction[1]
        if t < t_min or t > t_max:
            return False,rec
        x = r.origin[0] + t*r.direction[0] 
        z = r.origin[2] + t*r.direction[2]
        if x < self.x0 or x > self.x1 or z < self.z0 or z > self.z1:
            return False,rec
        rec.u = (x-self.x0)/(self.x1-self.x0)
        rec.v = (z-self.z0)/(self.z1-self.z0)
        rec.t = t
        outward_normal = vec3(0,1,0)
        rec.set_face_normal(r,outward_normal)
        rec.material = self.material
        rec.point = r.at(t)
        return True,rec
        
        
        
        
    def bounding_box(self,time0,time1,output_box):
        output_box = aabb(vec3(self.x0,self.k-0.0001,self.z0),vec3(self.x1,self.k+0.0001,self.z1))
        return True,output_box
    
    
    
class yz_rect(hittable):

    def __init__(self,y0,y1,z0,z1,k,material):
        self.y0 = y0
        self.y1 = y1
        self.z0 = z0
        self.z1 = z1
        self.k = k
        self.material = material
        
        
    def hit(self,r,t_min,t_max,rec):
 
        t = (self.k-r.origin[0]) / r.direction[0]
        if t < t_min or t > t_max:
            return False,rec
        y = r.origin[1] + t*r.direction[1] 
        z = r.origin[2] + t*r.direction[2]
        if y < self.y0 or y > self.y1 or z < self.z0 or z > self.z1:
            return False,rec
        rec.v = (z-self.z0)/(self.z1-self.z0)
        rec.u = (y-self.y0)/(self.y1-self.y0)
        rec.t = t
        outward_normal = vec3(1,0,0)
        rec.set_face_normal(r,outward_normal)
        rec.material = self.material
        rec.point = r.at(t)
        return True,rec
        
        
        
        
    def bounding_box(self,time0,time1,output_box):
        output_box = aabb(vec3(self.k-0.0001,self.y0,self.z0),vec3(self.k+0.0001,self.y1,self.z1))
        return True,output_box
    
    
    
    
class box(hittable):
    def __init__(self,p0,p1,material,t0=0,t1=1):
        self.box_min = p0
        self.box_max = p1
        self.material = material
        sides = hittable_list()
        sides.add_object(xy_rect(p0[0],p1[0],p0[1],p1[1],p1[2],material))
        sides.add_object(xy_rect(p0[0],p1[0],p0[1],p1[1],p0[2],material))
        sides.add_object(xz_rect(p0[0],p1[0],p0[2],p1[2],p1[1],material))
        sides.add_object(xz_rect(p0[0],p1[0],p0[2],p1[2],p0[1],material))
        sides.add_object(yz_rect(p0[1],p1[1],p0[2],p1[2],p1[0],material))
        sides.add_object(yz_rect(p0[1],p1[1],p0[2],p1[2],p0[0],material))
        self.sides = sides
        #constructor(sides.data,0,len(sides.data),t0,t1)
        
    def hit(self,r,t_min,t_max,rec):
        out_rec = hit_record()
        hit_bool,out_rec = self.sides.hit(r,t_min,t_max,rec)
        if not hit_bool:
            return False,rec
        #out_rec.material = self.material
        return hit_bool,out_rec
    
    def bounding_box(self,time0,time1,output_box):
        output_box = aabb(self.box_min,self.box_max)
        return True,output_box
    
    
        
class translate(hittable):
    def __init__(self,obj,displacement):
        self.p = obj
        self.offset = displacement
        
    def hit(self,r,t_min,t_max,rec):
        temp_rec = hit_record()
        moved_r = ray(r.origin - self.offset,r.direction,r.time)
        hit_bool,temp_rec = self.p.hit(moved_r,t_min,t_max,rec)
        if not hit_bool:
            return False,rec
        temp_rec.point = temp_rec.point + self.offset
        temp_rec.set_face_normal(moved_r,temp_rec.normal)
        return True,temp_rec

    def bounding_box(self,time0,time1,output_box):
        temp_box = aabb()
        box_bool,temp_box = self.p.bounding_box(time0,time1,output_box)
        if not box_bool:
            return False,output_box
        out_box = aabb(temp_box.minimum + self.offset, temp_box.maximum + self.offset)
        return True,out_box
        
        
        
class rotate_y:
    def __init__(self,obj,angle):
        self.p = obj
        radians = math.radians(angle)
        self.sin_theta = np.sin(radians)
        self.cos_theta = np.cos(radians)
        temp_box = aabb()
        hasbox,temp_box = self.p.bounding_box(0,1,temp_box)
        minmum = vec3(np.inf,np.inf,np.inf)
        maxmum = vec3(-np.inf,-np.inf,-np.inf)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    x = i*temp_box.maximum[0] + (1-i)*temp_box.minimum[0]
                    y = j*temp_box.maximum[1] + (1-j)*temp_box.minimum[1]
                    z = k*temp_box.maximum[2] + (1-k)*temp_box.minimum[2]
                    newx = self.cos_theta*x +self.sin_theta*z
                    newz = -1*self.sin_theta*x + self.cos_theta*z
                    tester = vec3(newx,y,newz)
                    for c in range(3):
                        minmum[c] = min(minmum[c],tester[c])
                        maxmum[c] = max(maxmum[c],tester[c])
        self.bbox = aabb(minmum,maxmum)
        
    def bounding_box(self,time0,time1,output_box):
        return True,self.bbox
        
    def hit(self,r,t_min,t_max,rec):
        #Change ray from world space to object space
        origin = copy.deepcopy(r.origin)
        direction = copy.deepcopy(r.direction)
        
        origin[0] = self.cos_theta * r.origin[0] - self.sin_theta*r.origin[2]
        origin[2] = self.sin_theta * r.origin[0] + self.cos_theta*r.origin[2]
        
        direction[0] = self.cos_theta * r.direction[0] - self.sin_theta*r.direction[2]
        direction[2] = self.sin_theta * r.direction[0] + self.cos_theta*r.direction[2]
        
        rotated_r = ray(origin,direction,r.time)

        #determine whether new ray hits object
        temp_rec = hit_record()
        hit_bool,temp_rec = self.p.hit(rotated_r,t_min,t_max,rec)
        if not hit_bool:
            return False,rec
        #change intersection point from object space to world space
        p = copy.deepcopy(temp_rec.point)
        p[0] = self.cos_theta*temp_rec.point[0] + self.sin_theta*temp_rec.point[2]
        p[2] = -1*self.sin_theta*temp_rec.point[0] + self.cos_theta*temp_rec.point[2]

        #change normal from object space to world space

        temp_rec.point = p
        #temp_rec.set_face_normal(r,normal)
        temp_rec.set_face_normal(rotated_r,temp_rec.normal)
        return True,temp_rec
    


class flip_face(hittable):
    def __init__(self,obj):
        self.p = obj

    def hit(self,r,t_min,t_max,rec):
        temp_rec = hit_record()
        hit_bool,temp_rec = self.p.hit(r,t_min,t_max,rec)
        if not hit_bool:
            return False,rec

        temp_rec.front_face = not temp_rec.front_face
        return True,temp_rec

    def bounding_box(self,time0,time1,output_box):
        return self.p.bounding_box(time0,time1,output_box)






    
class constant_medium:
    def __init__(self,obj,d,texture):
        self.boundary = obj
        self.phase_function = isotropic(texture)
        self.neg_inv_density = (-1/d)
        
    def bounding_box(self,time0,time1,output_box):
        return self.boundary.bounding_box(time0,time1,output_box)
    
    def hit(self,r,t_min,t_max,rec):
        #print('called')
        
        rec1 = hit_record()
        #rec1.material = self.phase_function
        rec2 = hit_record()
        #rec2.material = self.phase_function
        hit_bool,rec1 = self.boundary.hit(r,-np.inf,np.inf,rec1)
        if not hit_bool:
            #print('first not hit bool')
            return False,rec
        hit_bool2,rec2 = self.boundary.hit(r,rec1.t+0.0001,np.inf,rec2)
        if not hit_bool2:
            #print('second not hit bool')
            return False,rec
        if rec1.t < t_min:
            rec1.t = t_min
        if rec2.t > t_max:
            rec2.t = t_max
        if rec1.t >= rec2.t:
            #print('third comparison')
            return False,rec
        if rec1.t <0:
            rec1.t = 0
            
        ray_length = np.linalg.norm(r.direction)
        distance_inside_boundary = (rec2.t - rec1.t) * ray_length
        hit_distance = self.neg_inv_density * np.log(np.random.random())
        
        if hit_distance > distance_inside_boundary:
            #print('fourth comparison')
            return False,rec
        
        rec.t = rec1.t + hit_distance / ray_length
        rec.point = r.at(rec.t)
        rec.normal = vec3(1,0,0)
        rec.front_face = True
        rec.material = self.phase_function
        rec.u = rec1.u
        rec.v = rec1.v
        #print('got here,material should be set')
        return True,rec

    
class hittable_list:
    def __init__(self):
        self.data = []
        
    def add_object(self,obj):
        self.data.append(obj)
    
    def clear(self):
        self.data = []
        
    def hit(self,r,t_min,t_max,record):
        temp_rec = hit_record()
        hit_anything = False
        closest_so_far = t_max
        for obj in self.data:
            obj_bool,temp_rec = obj.hit(r,t_min,closest_so_far,temp_rec)
            if obj_bool:
                hit_anything = True
                closest_so_far = temp_rec.t
                
        return hit_anything,temp_rec
    
    def bounding_box(self,tim0,time1):
        first_box = True
        temp_box = aabb()
        if len(self.data) == 0:
            return False
        else:
            for o in self.data:
                empty_box = aabb()
                box_bool,temp_box = o.bounding_box(tim0,time1,empty_box)
                if (not box_bool):
                    return False,temp_box
                output_box = temp_box if first_box == True else surrounding_box(ouput_box,temp_box)
                first_box = False
        return True,output_box
           
    
    