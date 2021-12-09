from PIL import Image
import numpy as np
from src.helper_functions import *
import math 

class solid_color:
    def __init__(self,color):
        self.color = color
        
    def value(self,u,v,p):
        return self.color
    
    
class checker_texture:
    def __init__(self,c1,c2):
        self.even = solid_color(c1)
        self.odd = solid_color(c2)
        
    def value(self,u,v,p):
        sines = np.sin(10*p[0])*np.sin(10*p[1])*np.sin(10*p[2])
        if sines < 0:
            return self.odd.value(u,v,p)
        else:
            return self.even.value(u,v,p)
        
        

    
    
class perlin:
    def __init__(self):
        self.point_count = 256
        self.ranfloat = np.random.random(self.point_count)
        self.ranvec = []
        for i in range(self.point_count):
            self.ranvec.append(unit_vector(np.random.uniform(-1,1,3)))
        
        self.perm_x = self.perlin_generate_perm()
        self.perm_y = self.perlin_generate_perm()
        self.perm_z = self.perlin_generate_perm()
        
        
    def perlin_generate_perm(self):
        p = np.arange(self.point_count)
        #p = self.permute(p,self.point_count)
        np.random.shuffle(p)
        return p
    
    def permute(self,p,n):
        for i in range(n-1,-1,-1):
            target = random.randint(0,i)
            tmp = p[i]
            p[i] = p[target]
            p[target] = tmp
            
        return p
    
    
    
        
    
    def turb(self,p,depth=7):
        accum = 0.0
        temp_p = p
        weight = 1.0
        for i in range(depth):
            accum+= weight*self.turb_noise(temp_p)
            weight = weight* 0.5
            temp_p = 2*temp_p
        return np.abs(accum)
        


    def simple_noise(self,p):
        i = int(4*p[0]) & 255
        j = int(4*p[1]) & 255
        k = int(4*p[2]) & 255
        return self.ranfloat[int(self.perm_x[i]) ^ int(self.perm_y[j]) ^ int(self.perm_z[k])]
        
    
    
    def tri_interp_noise(self,p):
        u = p[0] - math.floor(p[0])
        v = p[1] - math.floor(p[1])
        w = p[2] - math.floor(p[2])
      
        uu = u*u*(3-2*u)
        vv = v*v*(3-2*v)
        ww = w*w*(3-2*w)
        
        i = math.floor(p[0])
        j = math.floor(p[1])
        k = math.floor(p[2])
        accum = 0
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    val = self.ranfloat[int(self.perm_x[int(i+di) & 255]) ^ int(self.perm_y[int(j+dj) & 255]) ^ int(self.perm_z[int(k+dk) & 255])]
                    accum += (di*uu + (1-di)*(1-uu))* (dj*vv + (1-dj)*(1-vv))*(dk*ww + (1-dk)*(1-ww))*val
        return accum
        
    
    def noise(self,p):
        u = p[0] - math.floor(p[0])
        v = p[1] - math.floor(p[1])
        w = p[2] - math.floor(p[2])
        uu = u*u*(3-2*u)
        vv = v*v*(3-2*v)
        ww = w*w*(3-2*w)
        
        i = math.floor(p[0])
        j = math.floor(p[1])
        k = math.floor(p[2])
        accum = 0
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    weight_v = vec3(u-di,v-dj,w-dk)
                    vec = self.ranvec[int(self.perm_x[int(i+di) & 255]) ^ int(self.perm_y[int(j+dj) & 255]) ^ int(self.perm_z[int(k+dk) & 255])]
                    accum += (di*uu + (1-di)*(1-uu))* (dj*vv + (1-dj)*(1-vv))*(dk*ww + (1-dk)*(1-ww))*np.dot(vec,weight_v)
        
        return (1+accum)*0.5
    
    def turb_noise(self,p):
        u = p[0] - math.floor(p[0])
        v = p[1] - math.floor(p[1])
        w = p[2] - math.floor(p[2])
        uu = u*u*(3-2*u)
        vv = v*v*(3-2*v)
        ww = w*w*(3-2*w)
        
        i = math.floor(p[0])
        j = math.floor(p[1])
        k = math.floor(p[2])
        accum = 0
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    weight_v = vec3(u-di,v-dj,w-dk)
                    vec = self.ranvec[int(self.perm_x[int(i+di) & 255]) ^ int(self.perm_y[int(j+dj) & 255]) ^ int(self.perm_z[int(k+dk) & 255])]
                    accum += (di*uu + (1-di)*(1-uu))* (dj*vv + (1-dj)*(1-vv))*(dk*ww + (1-dk)*(1-ww))*np.dot(vec,weight_v)
        
        return accum


    
    
class noise_texture:
    def __init__(self,scale=1):
        self.scale = scale
        self.noise = perlin()
        
    def value(self,u,v,p):
        #return vec3(1,1,1)*0.5*(1+np.sin(self.scale*p[2]) + 10*self.noise.turb(p)) #marbled
        return vec3(1,1,1)*self.noise.turb(self.scale*p)
    
    
    
class image_texture:
    def __init__(self,filename):
        self.bytes_per_pixel = 3
        try:
            self.data = np.asarray(Image.open(filename))
        except FileNotFoundError:
            self.data == None
            print('ERROR: File not found')
            self.width = self.height = 0
        if type(self.data)== np.ndarray :
            self.width = self.data.shape[1]
            self.height = self.data.shape[0]
        self.bytes_per_scanline = self.bytes_per_pixel * self.width
        
        
    def value(self,u,v,p):
        if type(self.data) != np.ndarray :
            return vec3(0,1,1)
        u = clamp(u,0.0,1.0)
        v = 1.0-clamp(v,0.0,1.0)
        i = int(u*self.width)
        j = int(v*self.height)
        if i >= self.width:
            i = self.width -1
        if j >= self.height:
            j = self.height - 1
        color_scale = 1.0/255.0
        #print('width',self.width,i)
        #print('height',self.height,j)
        pixel = self.data[j,i] 
        return vec3(color_scale*pixel[0],color_scale*pixel[1],color_scale*pixel[2])
