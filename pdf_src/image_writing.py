from pdf_src.pdf import *
from src.helper_functions import *
from src.ray import *
from pdf_src.hit_record import *

def ray_color(r,background,world,lights,depth):
    '''
    Here the ray color function is fixed, which allows for a cleaner way to keep track
    of scattering, and to generate different probability distribution functions. It takes in a
    new parameter called 'lights' which is actually just empty objects that the user inputs if they want
    to sample. For example, if I wanted to sample a cube, I'd pass in a cube initialized with no material
    in the same location into the lights hittable list, which then this ray_color function uses to sample
    the particular object.
    '''

    temp_rec = hit_record()
    #if ray bounce limit exceeded, no more light is gathered
    if depth <= 0:
        return vec3(0,0,0)
    

    hit_bool,temp_rec = world.hit(r,0.001,np.inf,temp_rec)
    #if ray hits nothing return background color
    if hit_bool == False:
        return background
    elif hit_bool == True:
        srec = scatter_record()
        emitted = temp_rec.material.emitted(temp_rec,temp_rec.u,temp_rec.v,temp_rec.point)
        mat_bool,srec = temp_rec.material.scatter(r,temp_rec)
        #print(type(srec.pdf_ptr))
        if mat_bool == False:
            return emitted
        elif mat_bool == True:
            if srec.is_specular == True:
                return srec.attenuation * ray_color(srec.specular_ray,background,world,lights,depth-1)
            light_ptr = hittable_pdf(lights,temp_rec.point)
            #print(type(light_ptr))
            mix_pdf = mixture_pdf(light_ptr,srec.pdf_ptr)
            #print()
            drctin = mix_pdf.generate()
            #print("HELLO")
            #print(type(drctin))
            scattered = ray(temp_rec.point,drctin,r.time)
            pdf_val = mix_pdf.value(scattered.direction)

            return emitted + srec.attenuation*temp_rec.material.scattering_pdf(r,temp_rec,scattered)*ray_color(scattered,background,world,lights,depth-1) / pdf_val
  

