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

    #Catch Negative Values
    if r < 0:
        r = -r

    if g < 0:
        g = -g

    if b < 0:
        b = -b
    

    with np.errstate(invalid ='raise'):
        try:
        #apply gamma=2.0 correction
            r = np.sqrt(r*scale)
            g = np.sqrt(g*scale)
            b = np.sqrt(b*scale)
        except FloatingPointError:
            print('Floating Point Error, catching for diagnosis')
            print(f'r:{r},g: {g}, b: {b}')
            print('continuing with zeros')
            r = 0
            g = 0
            b = 0
    ir = int(256*clamp(r,0.0,0.999))
    ig = int(256*clamp(g,0.0,0.999))
    ib = int(256*clamp(b,0.0,0.999))
    line = ''.join([str(ir),' ',str(ig),' ',str(ib),'\n'])
    #print(line)
    return line


