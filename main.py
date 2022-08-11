from utils import *


path="image/000000.png"

image=read_image_to_np(path)
# plot_image(image)

old_path="E:/code/image_tools/image/20220624_131403/depth"
new_path="E:\code\image_tools/image/20220624_131403_new/depth"

sort_file_and_rename(old_path,new_path)