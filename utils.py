from tkinter import image_names
from PIL import Image
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from shutil import copyfile

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
def sort_file_and_rename(old_dir,new_dir):
	# path_romp_pose='D:/论文_元宇宙/code/smpl/data/2022-6-26/test'
	path_list=os.listdir(old_dir)
	path_list.sort() #对读取的路径进行排序
	path_list_new=[]
	i=0
	for filename in path_list:
		# if filename.endswith(".pkl"):
		# 	path_list_new.append(filename)
		name,ext = os.path.splitext(filename)
		copyfile(os.path.join(old_dir,filename), os.path.join(new_dir,"{0:06d}".format(i)+ext))
		i=i+1
		# print(os.path.join(path_romp_output,filename))
	# print(path_list_new)