from src.helper_functions import *
from src.ray import *
from src.hit_record import *

def ray_color(r,background,world,depth):
    '''
    ray_color: this is the function that actually drives most of the code.
    param ray: a ray of light
    param background: an rgb vector of what this should return when the ray hits nothing
    param world: this is the list of objects in the world
    depth: the max depth to recur to. This is effectively how far to follow a ray of light before
           it returns no light.
    returns: this returns the color of the pixel, which is then passed to write color to write out
    '''
    rec = hit_record()
    if depth <= 0:
        return vec3(0,0,0)

    hit_bool,temp_rec = world.hit(r,0.001,np.inf,rec)
    if hit_bool == False:
        return background
    elif hit_bool == True:
        mat_bool,scattered,attenuation = temp_rec.material.scatter(r,temp_rec)
        emitted = temp_rec.material.emitted(temp_rec.u,temp_rec.v,temp_rec.point)
        #print(f'scattered: {scattered}',f'attenuation:{attenuation}',f'emitted: {emitted}')
        if mat_bool == False:
            #print('It is returning emitted')
            return emitted
        elif mat_bool == True:

            return emitted + attenuation*ray_color(scattered,background,world,depth-1)


    