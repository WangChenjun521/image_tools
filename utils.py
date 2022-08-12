from tkinter import image_names
from PIL import Image
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from shutil import copyfile
import json
import open3d as o3d

import  line_mesh as line_mesh_utils

def read_image(path):
	image=Image.open(path)
	return image
def read_image_to_np(path):
	image=Image.open(path)
	image=np.asarray(image,dtype=np.float32)
	return image
def plot_image(image,name=""):
	fig = plt.figure()
	plt.imshow(image)
	plt.title(name)
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

def read_json(input_path):
    with open(input_path,'r')  as file:
        str = file.read()
        data = json.loads(str)
    return data
# results = data['results']
from typing import List

def make_z_aligned_image_plane(min_pt, max_pt, z, image):
    plane_vertices = [
        [min_pt[0], min_pt[1], z],
        [max_pt[0], min_pt[1], z],
        [max_pt[0], max_pt[1], z],
        [min_pt[0], max_pt[1], z]
    ]
    plane_triangles = [[2, 1, 0],
                       [0, 3, 2]]

    plane_mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(plane_vertices), o3d.utility.Vector3iVector(plane_triangles))
    plane_mesh.compute_vertex_normals()

    plane_texture_coordinates = [
        (1, 1), (1, 0), (0, 0),
        (0, 0), (0, 1), (1, 1)
    ]

    plane_mesh.triangle_uvs = o3d.utility.Vector2dVector(plane_texture_coordinates)
    plane_mesh.triangle_material_ids = o3d.utility.IntVector([0, 0])
    plane_mesh.textures = [image]
    return plane_mesh

def merge_meshes(meshes: List[o3d.geometry.TriangleMesh]) -> o3d.geometry.TriangleMesh:
    """
    Combine a list of meshes into a single mesh
    :param meshes: list of triangle meshes
    :return: a mesh comprising all of the meshes in the argument
    """
    # Compute total number of vertices and faces.
    num_vertices = 0
    num_triangles = 0
    num_vertex_colors = 0
    for i in range(len(meshes)):
        num_vertices += np.asarray(meshes[i].vertices).shape[0]
        num_triangles += np.asarray(meshes[i].triangles).shape[0]
        num_vertex_colors += np.asarray(meshes[i].vertex_colors).shape[0]

    # Merge vertices and faces.
    vertices = np.zeros((num_vertices, 3), dtype=np.float64)
    triangles = np.zeros((num_triangles, 3), dtype=np.int32)
    vertex_colors = np.zeros((num_vertex_colors, 3), dtype=np.float64)

    vertex_offset = 0
    triangle_offset = 0
    vertex_color_offset = 0
    for i in range(len(meshes)):
        current_vertices = np.asarray(meshes[i].vertices)
        current_triangles = np.asarray(meshes[i].triangles)
        current_vertex_colors = np.asarray(meshes[i].vertex_colors)

        vertices[vertex_offset:vertex_offset + current_vertices.shape[0]] = current_vertices
        triangles[triangle_offset:triangle_offset + current_triangles.shape[0]] = current_triangles + vertex_offset
        vertex_colors[vertex_color_offset:vertex_color_offset + current_vertex_colors.shape[0]] = current_vertex_colors

        vertex_offset += current_vertices.shape[0]
        triangle_offset += current_triangles.shape[0]
        vertex_color_offset += current_vertex_colors.shape[0]

    # Create a merged mesh object.
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles))
    mesh.paint_uniform_color([1, 0, 0])
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    return mesh

def draw_node_graph(graph_nodes, graph_edges):
	# Graph canonical_node_positions
	rendered_graph_nodes = []
	for node in graph_nodes:
		mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
		mesh_sphere.compute_vertex_normals()
		mesh_sphere.paint_uniform_color([1.0, 0.0, 0.0])
		mesh_sphere.translate(node)
		rendered_graph_nodes.append(mesh_sphere)

	# Merge all different sphere meshes
	rendered_graph_nodes = merge_meshes(rendered_graph_nodes)

	# Graph edges
	edges_pairs = []
	for node_id, edges in enumerate(graph_edges):
		for neighbor_id in edges:
			if neighbor_id == -1:
				break
			edges_pairs.append([node_id, neighbor_id])

	colors = [[0.2, 1.0, 0.2] for i in range(len(edges_pairs))]
	line_mesh = line_mesh_utils.LineMesh(graph_nodes, edges_pairs, colors, radius=0.003)
	line_mesh_geoms = line_mesh.cylinder_segments

	# Merge all different line meshes
	line_mesh_geoms = merge_meshes(line_mesh_geoms)

	o3d.visualization.draw_geometries([rendered_graph_nodes, line_mesh_geoms, source_object_pcd])
	# o3d.visualization.draw_geometries([rendered_graph_nodes, line_mesh_geoms])
	# Combined canonical_node_positions & edges
	rendered_graph = [rendered_graph_nodes, line_mesh_geoms]
	
	# vis = o3d.visualization.Visualizer()
	# vis.create_window(width=512, height=512)
	# vis.add_geometry(mesh)
	# vis.add_geometry(mesh)

	return rendered_graph

def draw_node_graph_2(graph_nodes, graph_edges,graph_nodes_2, graph_edges_2,only_one,num=1):
	# Graph canonical_node_positions
	rendered_graph_nodes = []
	for node in graph_nodes:
		mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
		mesh_sphere.compute_vertex_normals()
		mesh_sphere.paint_uniform_color([1.0, 0.0, 0.0])
		mesh_sphere.translate(node)
		rendered_graph_nodes.append(mesh_sphere)

	# Merge all different sphere meshes
	rendered_graph_nodes = merge_meshes(rendered_graph_nodes)

	# Graph edges
	edges_pairs = []
	for node_id, edges in enumerate(graph_edges):
		for neighbor_id in edges:
			if neighbor_id == -1:
				break
			edges_pairs.append([node_id, neighbor_id])

	colors = [[0.2, 1.0, 0.2] for i in range(len(edges_pairs))]
	line_mesh = line_mesh_utils.LineMesh(graph_nodes, edges_pairs, colors, radius=0.003)
	line_mesh_geoms = line_mesh.cylinder_segments

	# Merge all different line meshes
	line_mesh_geoms = merge_meshes(line_mesh_geoms)


####################################
	if only_one==False:
		rendered_graph_nodes_2 = []
		for node in graph_nodes_2:
			mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
			mesh_sphere.compute_vertex_normals()
			mesh_sphere.paint_uniform_color([0.0, 0.0, 1.0])
			mesh_sphere.translate(node)
			rendered_graph_nodes_2.append(mesh_sphere)

		# Merge all different sphere meshes
		rendered_graph_nodes_2 = merge_meshes(rendered_graph_nodes_2)

		# Graph edges
		edges_pairs_2 = []
		for node_id, edges in enumerate(graph_edges_2):
			for neighbor_id in edges:
				if neighbor_id == -1:
					break
				edges_pairs_2.append([node_id, neighbor_id])

		colors_2 = [[0.2, 0.1, 1.2] for i in range(len(edges_pairs_2))]
		line_mesh_2 = line_mesh_utils.LineMesh(graph_nodes_2, edges_pairs, colors, radius=0.003)
		line_mesh_geoms_2 = line_mesh_2.cylinder_segments

		# Merge all different line meshes
		line_mesh_geoms_2 = merge_meshes(line_mesh_geoms_2)
		o3d.visualization.draw_geometries([rendered_graph_nodes, line_mesh_geoms, rendered_graph_nodes_2,line_mesh_geoms_2])

	
	elif only_one==True:
		if num==1:
			o3d.visualization.draw_geometries([rendered_graph_nodes, line_mesh_geoms])
		if num==2:
			rendered_graph_nodes = []
			for node in graph_nodes_2:
				mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
				mesh_sphere.compute_vertex_normals()
				mesh_sphere.paint_uniform_color([0.0, 0.0, 1.0])
				mesh_sphere.translate(node)
				rendered_graph_nodes.append(mesh_sphere)

			# Merge all different sphere meshes
			rendered_graph_nodes = merge_meshes(rendered_graph_nodes)

			# Graph edges
			edges_pairs = []
			for node_id, edges in enumerate(graph_edges_2):
				for neighbor_id in edges:
					if neighbor_id == -1:
						break
					edges_pairs.append([node_id, neighbor_id])

			colors = [[0.2, 0.8, 0.1] for i in range(len(edges_pairs))]
			line_mesh = line_mesh_utils.LineMesh(graph_nodes_2, edges_pairs, colors, radius=0.003)
			line_mesh_geoms = line_mesh.cylinder_segments

			# Merge all different line meshes
			line_mesh_geoms = merge_meshes(line_mesh_geoms)
			o3d.visualization.draw_geometries([rendered_graph_nodes, line_mesh_geoms])

	# Combined canonical_node_positions & edges
	rendered_graph = [rendered_graph_nodes, line_mesh_geoms]
	
	# vis = o3d.visualization.Visualizer()
	# vis.create_window(width=512, height=512)
	# vis.add_geometry(mesh)
	# vis.add_geometry(mesh)

	return rendered_graph