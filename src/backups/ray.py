class ray:
    #requries numpy arrays to be passed in 
    def __init__(self,origin,vector,time =0):
        self.origin = origin
        self.direction = vector
        self.time = time
    def at(self,t):
        return self.origin+t*self.direction