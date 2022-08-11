from tkinter import image_names
from PIL import Image
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def read_image(path):
	image=Image.open(path)
	return image
def read_image_to_np(path):
	image=Image.open(path)
	image=np.asarray(image,dtype=np.float32)
	return image
def plot_image(image):
	fig = plt.figure()
	plt.imshow(image)
	plt.show()	
def plot_matrix(matrix):
	fig = plt.figure()
	sns.heatmap(matrix, annot = True)
	plt.show()