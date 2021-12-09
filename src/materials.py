from src.helper_functions import *
from src.hit_record import *
from src.ray import *


#Useful Functions

def random_in_unit_sphere():
    '''
    this returns a random vector in a unit sphere, which is useful for calculating how rays 
    scatter 
    '''
    while True:
        p = np.random.uniform(-1,1,3)
        if np.linalg.norm(p)**2 <1:
            return p
            break

def random_in_unit_disk():
    '''
    this returns a random vector in a unit disk, as an alternative for calculating how rays 
    scatter 
    '''
    while True:
        p = np.array([np.random.uniform(-1,1),np.random.uniform(-1,1),0])
        if np.linalg.norm(p)**2 <= 1:
            return p
            break
            
    
def random_in_hemisphere(v):
    '''
    this returns a random vector in a unit hemisphere, as an alternative for calculating how rays 
    scatter. All of these model diffusion slightly differently and can be interchanged in the 
    lambertian material below
    '''
    in_unit_sphere = random_in_unit_sphere()
    if np.dot(in_unit_sphere,v) > 0.0:
        return in_unit_sphere
    else:
        return -1*in_unit_sphere
    
def random_unit_vector():
    #this is a unit_vector version of the random_in_unit_sphere
    return unit_vector(random_in_unit_sphere())

def reflect(v,n):
    #this is a function to calculate the direction of a ray when it reflects off of an object
    #It takes in an input vector and the normal vector of the point where the ray hit the object
    #and returns the reflected direction vector
    return v-2*np.dot(v,n)*n

def refract(uv,n,etai_over_etat):
    '''
    This function calculates refraction using Snell's Law
    '''
    cos_theta = min(np.dot(-1*uv,n),1.0)

    r_out_perp = etai_over_etat * (uv + cos_theta*n)
    r_out_parallel = -1*np.sqrt(np.abs(1.0-np.linalg.norm(r_out_perp)**2))*n
    return r_out_perp+r_out_parallel




#Parent Class
class material:
    '''
    A lot of these are empty initializations, but the useful one here is the emitted
    parameter. This parent class was defined so I could modify all with default parameters
    without difficulty. In this case, I wanted to make sure a material by default emitted no light,
    so here the emitted function returns a black pixel.
    '''
    def __init__(self):
        x = True
    def emitted(self,u,v,p):
        return vec3(0,0,0)
    def scatter(self,r_in,hit_rec):
        x= True



#Children





class lambertian(material):
    '''
    This material approximates diffuse objects, where rays scatter randomly when they hit this object.
    The initialization value takes in a texture. 
    The scatter function takes in a ray, a hit record, and uses them to calculate a scatter ray, 
    where the direction is randomized. It also calculates an attenuation vector, which is the color
    value of the given texture at a given pixel. It returns these for use in the ray_color function.
    '''
    def __init__(self,texture):
        self.albedo=texture
    
    def scatter(self,r_in,hit_rec):
        scatter_direction = hit_rec.normal + random_unit_vector()

        
        #catch degenerate scatter direction
        if near_zero(scatter_direction) == True:
            # print('hit degeneracy')
            scatter_direction = hit_rec.normal
            
                        
        scattered = ray(hit_rec.point,scatter_direction,r_in.time)
        attenuation = self.albedo.value(hit_rec.u,hit_rec.v,hit_rec.point)
        # print('after assigning to ray:',scattered.direction)
        return True,scattered,attenuation
    


        
class metal(material):
    '''
    This material approximates metal objects, where rays reflect when they hit this object.
    The initialization value takes in a color vector. 
    The scatter function takes in a ray, a hit record, and uses them to calculate a scatter ray, 
    where the direction is given by the reflect function above. 
    It also calculates an attenuation vector, which is the color given upon initialization
    It returns these for use in the ray_color function.
    The fuzz parameter here references how perfect the reflection is. A fuzz value of 1 means 
    the unit sphere in which the scatter direction is calculated is largest, and a fuzz value of
    zero corresponds to perfect reflections.
    '''
    def __init__(self,albedo,fuzz):
        self.albedo = albedo
        if fuzz > 1:
            self.fuzz = 1
        else:
            self.fuzz = fuzz
        
    def scatter(self,r_in,hit_rec):
        reflected = reflect(unit_vector(r_in.direction),hit_rec.normal)
        scattered = ray(hit_rec.point,reflected+self.fuzz*random_in_unit_sphere(),r_in.time)
        uv = unit_vector(r_in.direction)
        cos_theta = min(np.dot(-1*uv,hit_rec.normal),1.0)
        white = vec3(1,1,1)
        attenuation = self.albedo + (white - self.albedo) * (1-cos_theta)**5 #technically should use Schlick Approx here for reflections
        scat_bool = np.dot(scattered.direction, hit_rec.normal) >0
        return scat_bool,scattered,attenuation



    
class dielectric(material):
    '''
    This material approximates dielectric objects, where rays can reflect or refract given a
    particular index of refraction.
    The initialization value takes in this index of refraction.  
    The scatter function takes in a ray, a hit record, and uses them to calculate a scatter ray, 
    where the direction is given by testing whether the ray will reflect or refract.
    It also calculates an attenuation vector, which is assumed to be white (1,1,1)
    It returns these for use in the ray_color function.

    '''
    def __init__(self,index_of_refraction):
        self.ior = index_of_refraction
        
    def reflectance(self,cosine,ref_idx):
        r0 = (1-ref_idx) / (1+ref_idx)
        r0 = r0**2
        return r0 + (1-r0)*(1-cosine)**5
        
    def scatter(self,r_in,hit_rec):
        attenuation = vec3(1.0,1.0,1.0)
        refraction_ratio = (1.0)/self.ior if hit_rec.front_face == True else self.ior
        unit_direction = unit_vector(r_in.direction)
        cos_theta = min(np.dot(-1*unit_direction,hit_rec.normal),1.0)
        sin_theta = np.sqrt(1.0-cos_theta**2)
        cannot_refract = refraction_ratio *sin_theta > 1.0
        if (cannot_refract == True) or (self.reflectance(cos_theta,refraction_ratio) > np.random.rand()):
            direction = reflect(unit_direction,hit_rec.normal)
        else:
            direction = refract(unit_direction,hit_rec.normal,refraction_ratio)
            
        scattered = ray(hit_rec.point,direction,r_in.time)
        return True,scattered,attenuation
    


class diffuse_light(material):
    '''
    This material approximates objects, that emit light
    The initialization value takes in a texture.
    Here the scatter function returns False by default.
    The emitted function here emits the color of the texture rather than black, which
    is accounted for in the ray_color function.

    '''
    def __init__(self,texture):
        self.emit = texture
        
    def scatter(self,r_in,hit_rec):
        return False,vec3(0,0,0),vec3(0,0,0)
    
    def emitted(self,u,v,p):
        return self.emit.value(u,v,p)
        
class isotropic(material):
    '''
    This final material approxmiates isotropic materials, such as smoke. It takes in a 
    texture parameter, and then calculates the scattered and attenuation rays by allowing the 
    ray to bounce off at a random point, similar to the lambertian distribution.
    '''
    def __init__(self,texture):
        self.albedo = texture
        
    def scatter(self,r_in,rec):
        scattered= ray(rec.point,random_in_unit_sphere(),r_in.time)
        attenuation = self.albedo.value(rec.u,rec.v,rec.point)
        return True,scattered,attenuation

    
    