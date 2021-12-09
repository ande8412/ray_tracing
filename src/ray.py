class ray:
    '''
    This class represents a ray of light. It is given an origin, a direction, and a 'time' value
    which really just represents how far along a path the ray has traveled. 
    '''
    #requries numpy arrays to be passed in 
    def __init__(self,origin,vector,time =0):
        self.origin = origin
        self.direction = vector
        self.time = time
    def at(self,t):
        return self.origin+t*self.direction