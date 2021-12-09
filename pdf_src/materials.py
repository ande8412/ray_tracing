from src.helper_functions import *
from pdf_src.hit_record import *
from src.ray import *
from pdf_src.pdf import *

#Useful Functions

def random_in_unit_sphere():
    while True:
        p = np.random.uniform(-1,1,3)
        if np.linalg.norm(p)**2 <1:
            return p
            break

def random_in_unit_disk():
    while True:
        p = np.array([np.random.uniform(-1,1),np.random.uniform(-1,1),0])
        if np.linalg.norm(p)**2 <= 1:
            return p
            break
            
    
def random_in_hemisphere(v):
    in_unit_sphere = random_in_unit_sphere()
    if np.dot(in_unit_sphere,v) > 0.0:
        return in_unit_sphere
    else:
        return -1*in_unit_sphere
    
def random_unit_vector():
    return unit_vector(random_in_unit_sphere())

def reflect(v,n):
    return v-2*np.dot(v,n)*n

def refract(uv,n,etai_over_etat):
    cos_theta = min(np.dot(-1*uv,n),1.0)

    r_out_perp = etai_over_etat * (uv + cos_theta*n)
    r_out_parallel = -1*np.sqrt(np.abs(1.0-np.linalg.norm(r_out_perp)**2))*n
    return r_out_perp+r_out_parallel


#Parent Class
class material:
    '''
    This empty initialization is actually used here in order to keep track of
    objects you want to sample.
    '''
    def __init__(self):
        x = True

    def emitted(self,rec,u,v,p):
        return vec3(0,0,0)
    def scatter(self,r_in,hit_rec):
        srec = scatter_record()
        return False,srec
    def scattering_pdf(self,r_in,hit_rec,scattered):
        return 0


#Children

class lambertian(material):
    '''
    Here the scattering method is fixed to account for a more accurate scattering of light
    as it depends on the cosine of the angle of the incoming ray
    '''
    def __init__(self,texture):
        self.albedo=texture
    
    def scatter(self,r_in,hit_rec):
        srec = scatter_record()
        srec.is_specular = False
        srec.attenuation = self.albedo.value(hit_rec.u,hit_rec.v,hit_rec.point)
        srec.pdf_ptr = cosine_pdf(hit_rec.normal)

        return True,srec
                        
    def scattering_pdf(self,r_in,rec,scattered):
        cosine = np.dot(rec.normal,unit_vector(scattered.direction))
        kale = 0 if cosine < 0 else cosine /np.pi
        return kale


    
        
class metal(material):
    '''
    Here the main changes are to keep track of the scattered ray in a cleaner way
    '''
    def __init__(self,albedo,fuzz):
        self.albedo = albedo
        if fuzz > 1:
            self.fuzz = 1
        else:
            self.fuzz = fuzz
        
    def scatter(self,r_in,hit_rec):
        srec = scatter_record()
        reflected = reflect(unit_vector(r_in.direction),hit_rec.normal)
        srec.specular_ray = ray(hit_rec.point,reflected+self.fuzz*random_in_unit_sphere())
        srec.attenuation = self.albedo
        srec.is_specular = True
        srec.pdf_ptr = None
        return True,srec


    
class dielectric(material):
    '''
    Here the main changes are to keep track of the scattered ray in a cleaner way
    '''
    def __init__(self,index_of_refraction):
        self.ior = index_of_refraction
        
    def reflectance(self,cosine,ref_idx):
        r0 = (1-ref_idx) / (1+ref_idx)
        r0 = r0**2
        return r0 + (1-r0)*(1-cosine)**5
        
    def scatter(self,r_in,hit_rec):
        srec = scatter_record()
        srec.is_specular = True
        srec.pdf_ptr= None
        srec.attenuation = vec3(1.0,1.0,1.0)
        refraction_ratio = (1.0)/self.ior if hit_rec.front_face == True else self.ior
        unit_direction = unit_vector(r_in.direction)
        cos_theta = min(np.dot(-1*unit_direction,hit_rec.normal),1.0)
        sin_theta = np.sqrt(1.0-cos_theta**2)
        cannot_refract = refraction_ratio *sin_theta > 1.0
        if (cannot_refract == True) or (self.reflectance(cos_theta,refraction_ratio) > np.random.rand()):
            direction = reflect(unit_direction,hit_rec.normal)
        else:
            direction = refract(unit_direction,hit_rec.normal,refraction_ratio)
            
        srec.specular_ray = ray(hit_rec.point,direction,r_in.time)
        return True,srec

        

class diffuse_light(material):
    '''
    This is the same as in the src. folder
    '''
    def __init__(self,texture):
        self.emit = texture

    def emitted(self,rec,u,v,p):
        if not rec.front_face:
            return vec3(0,0,0)
        return self.emit.value(u,v,p)



#not fixed for pdf methods        
# class isotropic(material):
#     def __init__(self,texture):
#         self.albedo = texture
        
#     def scatter(self,r_in,rec):
#         srec = scatter_record()
#         scattered = ray(rec.point,random_in_unit_sphere(),r_in.time)
#         alb = self.albedo.value(rec.u,rec.v,rec.point)
#         srec.is_specular = False
#         srec.attenuation = self.albedo.value(hit_rec.u,hit_rec.v,hit_rec.point)
#         srec.pdf_ptr = cosine_pdf(hit_rec.normal)

#         return True,srec
#         return True,scattered



    
    