import numpy as np
from src.hit_record import *



def vec3(a,b,c):
    '''
    this was just defined so i didn't have to write np.array all the time
    and to make it clear i am working with vectors
    '''
    return np.array([a,b,c])




def unit_vector(v):
    '''
    simple function to return a unit vector of a given vector
    '''
    return v / (np.linalg.norm(v))

def near_zero(v):
    '''
    simple function to check if a given vector is close to zero in all axes
    '''
    s = 1e-8
    return (np.abs(v[0]) < s) and (np.abs(v[1]) < s) and (np.abs(v[2]) < s) 





def clamp(x,min_,max_):
    '''
    simple function to clamp a number between a certain range
    '''
    return max(min_, min(x, max_))

def write_color(color,samples_per_pixel):
    '''
    this function takes in a vector of rgb values called color, and then 
    returns the string to write to the ppm file
    '''
    r = color[0]
    g = color[1]
    b = color[2]
    scale = 1.0/samples_per_pixel


    #catch Nans
    if r != r: 
        r = 0
    if g != g: 
        g = 0
    if b != b: 
        b = 0
    
    #apply gamma=2.0 correction
    r = np.sqrt(r*scale)
    g = np.sqrt(g*scale)
    b = np.sqrt(b*scale)
    #print(r,g,b)
    ir = int(256*clamp(r,0.0,0.999))
    ig = int(256*clamp(g,0.0,0.999))
    ib = int(256*clamp(b,0.0,0.999))
    line = ''.join([str(ir),' ',str(ig),' ',str(ib),'\n'])
    #print(line)
    return line





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


    