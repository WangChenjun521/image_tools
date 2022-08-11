from PIL import Image
import numpy as np



def read_image(path):
	image=Image.open(path)
	return image
