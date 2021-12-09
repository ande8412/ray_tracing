from src.helper_functions import *
import numpy as np

class onb:
	'''
	This class is used to generate an orthornormal basis given a normal vector.
	This is used in the probability distrubtion functions.
	'''
	def __init__(self):
		self.axis = []

	def u(self):
		return self.axis[0]

	def v(self):
		return self.axis[1]

	def w(self):
		return self.axis[2]

	def local(self,a):
		return a[0] * self.u() + a[1] * self.v() + a[2]*self.w()

	def operator(self,i):
		return self.axis[i]
	def build_from(self,n):
		w = unit_vector(n)
		a = vec3(0,1,0) if np.abs(w[0]) > 0.9 else vec3(1,0,0)
		v = unit_vector(np.cross(w,a))
		u = np.cross(w,v)
		self.axis.append(u)
		self.axis.append(v)
		self.axis.append(w)
	