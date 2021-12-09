import numpy as np

class hit_record:
    '''
    I use this class as a struct just to keep track of how and when I hit things
    Not initializing anything allows for a lot of flexibility in what I can store here
    For now it just has a function to set a normal vector based on an input ray and the outward
    normal of the object. This function also sets whether the ray hits the front face of the object
    or not, which is useful for keeping track of a lot of things such asreflections and refractions.
    '''
    def set_face_normal(self,r,outward_normal):
        self.front_face = np.dot(r.direction, outward_normal) < 0
        self.normal = outward_normal if self.front_face else -1*outward_normal
    

        
    