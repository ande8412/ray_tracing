import numpy as np

#this is largely the same as the one in the source directory,
#the main difference is a new struct to keep track of scatters

class hit_record:
    def set_face_normal(self,r,outward_normal):
        self.front_face = np.dot(r.direction, outward_normal) < 0
        self.normal = outward_normal if self.front_face == True else -1*outward_normal
    

        
    
class scatter_record:
    pass
