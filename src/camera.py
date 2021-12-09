import math
import numpy as np
from src.helper_functions import *
from src.ray import *



#Helper function
def random_in_unit_disk():
    '''
    This function is used to return a random number in a unit disk. This is used for the 
    lens approximation in the camera's depth of field calculation.
    '''
    while True:
        p = np.array([np.random.uniform(-1,1),np.random.uniform(-1,1),0])
        if np.linalg.norm(p)**2 <= 1:
            return p
            break
            



class camera:
    '''
    This class was created to simplify and organize the structure of the camera
    It is initialized with a point representing where the camera is looking from, a point
    representing where the camera should look, vup refers to the orientation of the camera, vfov 
    is the field of view in degrees, aspect_ratio is the aspect ratio of the image, aperture is the
    desired aperture of the camera, focus_dist is the distance to the plane that will be in focus
    t0 and tf are the time parameters used for creating motion blur with moving objects.

    This class also has a function get_ray, which returns a ray hitting the image at a random point in the specified 
    pixel. This information is passed in with s,t by the ray_color function
    '''
    def __init__(self,lookfrom,lookat,vup,vfov,aspect_ratio,aperture,focus_dist,t0,tf):
        
        self.lookfrom = lookfrom
        self.lookat = lookat
        self.vup = vup
        self.vfov = vfov #vertical field of view in degrees
        self.aspect_ratio = aspect_ratio
        self.aperture=aperture
        self.focus_dist = focus_dist
        self.t0 = t0
        self.tf = tf
        
        
        
        self.theta = math.radians(self.vfov)
        self.h = np.tan(self.theta/2)
        self.viewport_height = 2.0*self.h
        self.viewport_width = self.aspect_ratio * self.viewport_height
        
        #self.focal_length = 1
        
        self.w = unit_vector(self.lookfrom - self.lookat)
        self.u = unit_vector(np.cross(self.vup,self.w))
        self.v = np.cross(self.w,self.u)
        
        
        self.origin = self.lookfrom
        self.horizontal = self.focus_dist * self.viewport_width * self.u
        self.vertical = self.focus_dist * self.viewport_height * self.v
        self.lower_left_corner = self.origin-self.horizontal/2 -self.vertical/2 - self.focus_dist *self.w
        self.lens_radius = self.aperture /2
        
    def get_ray(self,s,t):
        rd = self.lens_radius * random_in_unit_disk()
        offset = self.u*rd[0] + self.v *rd[1]
        return ray(self.origin + offset, self.lower_left_corner+s*self.horizontal+t*self.vertical-self.origin-offset,np.random.uniform(self.t0,self.tf))

       