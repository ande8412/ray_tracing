from pdf_src.onb import *
from src.helper_functions import *
import numpy as np

#These functions are used for generating random directions
def random_cosine_direction():
	r1 = np.random.random()
	r2 = np.random.random()
	z = np.sqrt(1-r2)

	phi = 2*np.pi*r1
	x = np.cos(phi)*np.sqrt(r2)
	y = np.sin(phi)*np.sqrt(r2)

	return vec3(x,y,z)


def random_to_sphere(radius,distance_squared):
	r1 = np.random.random()
	r2 = np.random.random()
	z = 1 + r2*(np.sqrt(1-radius*radius/distance_squared) - 1)

	phi = 2*np.pi*r1
	x = np.cos(phi)*np.sqrt(1-z*z)
	y = np.sin(phi)*np.sqrt(1-z*z)
	return vec3(x,y,z)

class pdf:
	'''
	An empty parent class, mostly used for organization
	'''
	def __init__(self):
		x = True
	def value(self,direction):
		x = True
	def generate(self):
		x = True


class cosine_pdf(pdf):
	'''
	This is a probability distribution function based on the cosine of the incoming ray of light

	'''
	def __init__(self,w):
		uvw = onb()
		uvw.build_from(w)
		self.uvw = uvw

	def value(self,direction):
		cosine = np.dot(unit_vector(direction),self.uvw.w())
		kale = 0 if cosine <= 0 else cosine/np.pi
		return kale

	def generate(self):
		return self.uvw.local(random_cosine_direction())


class hittable_pdf(pdf):
	'''
	This uses an objects probability distribution function to figure out where rays go with greater
	accuracy.
	'''
	def __init__(self,obj,origin):
		self.o = origin
		self.ptr = obj


	def value(self,direction):
		return self.ptr.pdf_value(self.o,direction)

	def generate(self):
		return self.ptr.random(self.o)


class mixture_pdf(pdf):
	'''
	This combines the two pdfs above and creates a weighted average of the two.
	'''
	def __init__(self,p0,p1):
		self.p = []
		self.p.append(p0)
		self.p.append(p1)

	def value(self,direction):
		return 0.5*self.p[0].value(direction) + 0.5*self.p[1].value(direction)

	def generate(self):
		ran = np.random.random()
		if ran < 0.5:
			return self.p[0].generate()
		else:
			return self.p[1].generate()

