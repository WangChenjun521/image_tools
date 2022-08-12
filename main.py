from utils import *


# path="image/seq337/depth/000375.png"

# image=read_image_to_np(path)
# plot_image(image,name="000373.png")

# old_path="E:/code/image_tools/image/20220624_131403/depth"
# new_path="E:\code\image_tools/image/20220624_131403_new/depth"

# sort_file_and_rename(old_path,new_path)


data_origin = read_json('data/test_350_400.json')
data_nnrt = read_json('data/test_nnrt_351_400.json')
#              live data | frame_id | R&T of C model to Live model

# print(np.array(data['data']['351']['rotation_pred']).reshape([-1,3,3]))

translation_pred_origin=np.array(data_origin['data']['368']['translation_pred'])
translation_pred_nnrt=np.array(data_nnrt['data']['368']['translation_pred'])

print(translation_pred_origin.shape)

# print(np.array(data_origin['graph_nodes']).shape)
# print(np.array(data_origin['graph_edges']))
# print("data_nnrt:")
# print(translation_pred_nnrt.shape)

print(np.mean(np.abs(translation_pred_origin-translation_pred_nnrt)))
# print(np.array(data_nnrt['graph_nodes']).shape)


draw_node_graph_2(np.array(data_origin['graph_nodes'])+translation_pred_origin,np.array(data_origin['graph_edges']),np.array(data_nnrt['graph_nodes'])+translation_pred_nnrt,np.array(data_nnrt['graph_edges']),False,num=2)
# print(np.array(data['graph_edges']))

