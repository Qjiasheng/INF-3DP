import os
import logging
import numpy as np
import plyfile
import skimage.measure
import time
import torch
import torch.nn.functional as F
import diff_operators

import potpourri3d as pp3d
import scipy

from utils import field_query_with_grads, field_querying, general_text_writer
from vis import *

class PCD:
    def __init__(self, point_cloud_path, keep_aspect_ratio=True):
        point_cloud = np.genfromtxt(point_cloud_path)
        print('Point cloud loaded...')
        coords = point_cloud[:, :3]
        normals = point_cloud[:, 3:6]

        # process pcd
        self.normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        coords -= np.mean(coords, axis=0, keepdims=True)
        if keep_aspect_ratio:
            coord_max = np.amax(coords)
            coord_min = np.amin(coords)
        else:
            # normalize to bbx separately on each axis to [-1,1]
            coord_max = np.amax(coords, axis=0, keepdims=True)
            coord_min = np.amin(coords, axis=0, keepdims=True)

        self.coords = (coords - coord_min) / (coord_max - coord_min)
        self.coords -= 0.5
        # self.coords *= 2.
        self.coords *= 1.95        # surface not touches box buondary
        
    def get_point_cloud(self, down_ration=1):
        if down_ration > 1:
            down_ration = int(down_ration)
            return {'coords': self.coords[::down_ration], 'normals': self.normals[::down_ration]}
        return {'coords': self.coords, 'normals': self.normals}

def gen_voxle_queries(N=128, cube_size=1.0):
    """generate voxel queries in [-cube_size, cube_size]^3 cube"""

    voxel_origin = np.array([-cube_size, -cube_size, -cube_size])  # shape (3,)
    voxel_size = cube_size * 2.0 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)
    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N
    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    return samples, voxel_origin, voxel_size

def create_mesh_volume(decoder, save_path, N=256, max_batch=48 ** 3, offset=None, scale=None):
    """extract mesh from trained SDF, save volume data for visualization"""

    ply_fn = os.path.join(save_path, 'pcd.ply')
    sdf_volume_fn = os.path.join(save_path, 'sdf_volume')
    decoder.eval()

    # the voxel_origin is actually the (bottom, left, down) corner, not the middle
    samples, voxel_origin, voxel_size = gen_voxle_queries(N, cube_size=1.0)
    samples.requires_grad = False 
    fields = field_querying(decoder, samples, max_batch=max_batch, no_print=False, return_on_cuda=False)
    sdf_values = fields.reshape(N, N, N)

    # save to ply
    convert_sdf_samples_to_ply(
        sdf_values.data,
        voxel_origin,
        voxel_size,
        ply_fn,
        offset,
        scale,
    )
    print(f"mesh saved to {ply_fn}" )
 
    # save sdf volume data to npy file for visualization
    sdf_values_np = sdf_values.data.cpu().numpy()
    np.savez(sdf_volume_fn+'.npz', sdf=sdf_values_np, voxel_grid_origin=voxel_origin, voxel_size=voxel_size)
    print(f"sdf volume saved to {sdf_volume_fn+'.npz'}" )


def pcd_field_with_curv(decoder, data_paras, heat_field=False, max_batch=64 ** 3):
    """prepare dataset for further fields training. Not split batches. if memory issue, split it.
    -----------
    one line data augment: coords, normals, min_curv, max_curv, density, base_tags.
    NOTE coords here normalized to [-1, 1], same as SDF training"""

    # collect params in config
    point_cloud_path = data_paras.get('point_cloud_path')
    curv_step_size = data_paras.get('curv_step_size', 0.3)
    base_th = data_paras.get('base_threshold', 0.0)
    exp_data_path = data_paras.get('exp_data_path')

    pcd = PCD(point_cloud_path)
    pcd_data = pcd.get_point_cloud()

    samples = torch.from_numpy(pcd_data['coords']).float().cuda()
    samples.requires_grad = True 

    decoder.eval()
    model_output = decoder(samples)
    coords = model_output['model_in']
    fields = model_output['model_out']

    grads = diff_operators.gradient(fields, coords)
    normals = F.normalize(grads, dim=-1)

    # get curvature values and directions (as a sep module)
    kappa_batch, p_batch = diff_operators.fndiff_min_max_curvs_and_vectors(decoder, coords, h=curv_step_size)
    p_batch = F.normalize(p_batch, dim=-1)

    coords = coords.detach().cpu().numpy()
    fields = fields.detach().cpu().numpy()
    normals = normals.detach().cpu().numpy()
    kappa_batch = kappa_batch.detach().cpu().numpy()
    p_batch = p_batch.detach().cpu().numpy()
    # seems min/ max mix up, vis direction to check
    k_max, k_min = kappa_batch[:, 0], kappa_batch[:, 1]
    v_max, v_min = p_batch[:, 0, :], p_batch[:, 1, :]
    
    # show directions on pcd
    vis_grad(coords, normals, label='normals')
    vis_grad(coords, v_min, label='min_curv_dir')
    vis_grad(coords, v_max, label='max_curv_dir')

    # save pcd data
    base_tags = np.ones((coords.shape[0], 1), dtype=np.int8)
    base_tags[coords[..., 2] < base_th] = 0  # base pcd tag is 0
    # vis base tags
    vis_pcd_fields(coords, base_tags.squeeze())
    
    if heat_field:
        # generate heat field for coords
        heat_fields, heat_grads = gen_heat_grads(coords, base_tags, base_th=base_th)
        # projection to tangent plane
        heat_grads = heat_grads - (np.sum(heat_grads * normals, axis=1, keepdims=True)) * normals

        vis_pcd_fields(coords, heat_fields)
        vis_grad(coords, heat_grads, label='heat_grad')
    else:
        heat_grads = np.zeros_like(coords)

    write_data = np.hstack((coords, normals, v_min, v_max, heat_grads, base_tags.reshape(-1, 1)))
    fmt = '%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %d'
    general_text_writer(write_data, fmt, exp_data_path, chunk=None)



def gen_heat_grads(coords, tags, base_th=0.0):
    """generate heat field and get grads for pcd. 
    Here we set pcd below base_th as heat sources (multiple heat sources)"""
    x_dn, x_up = -1.0, 1.0
    y_dn, y_up = -1.0, 1.0
    z_dn, z_up = -1.0, base_th
    heat_src_idx = get_source_idx_bbx(coords, tags,  x_up, x_dn, y_up, y_dn, z_up, z_dn)
    heat_fields = heat_pcd(coords, heat_src_idx, t_coef=1.0)
    heat_grads = get_heat_grad(coords, heat_fields, k_neighbors=20)  # already normalized

    return heat_fields, heat_grads

def get_source_idx_bbx(coords, tags, x_up, x_dn, y_up, y_dn, z_up, z_dn):
    """get the source index of the point cloud, set points below base_th as sources. 
    you may also tune th get less base points."""
    condition = (
        (tags == 0) &  # from base points
        (coords[:, 0] >= x_dn) & (coords[:, 0] <= x_up) &  # x-axis bbx
        (coords[:, 1] >= y_dn) & (coords[:, 1] <= y_up) &  # y-axis 
        (coords[:, 2] >= z_dn) & (coords[:, 2] <= z_up)    # z-axis 
    )
    return np.where(condition)[0].tolist()

def heat_pcd(coords, source_idx, t_coef=1.):
    """heat method on pcd"""
    solver = pp3d.PointCloudHeatSolver(coords, t_coef)
   
    if len(source_idx) == 1:
        print('single source')
        distances = solver.compute_distance(source_idx)
    elif len(source_idx) > 1:
        print('multi sources')
        distances = solver.compute_distance_multisource(source_idx)
    else:
        raise ValueError('no source point')
    # completion points also have distance, filter them out further
    return distances

def get_heat_grad(points, distances, k_neighbors=20):
    """ 
    simple finite difference for pcd
    grad f = (f(x+ \delta x) - f(x) )/ \delta x
    for one point, solve it within knn points
    Use least squares to solve for the gradient: grad(f) ≈ (X^T X)^(-1) X^T \delta f
    where X is the matrix of position differences and \delta f is the vector of distance differences 
    """
    kdtree = scipy.spatial.cKDTree(points)
    gradients = np.zeros_like(points)
    for i in range(points.shape[0]):
        _, neighbor_indices = kdtree.query(points[i], k=k_neighbors)
        neighbor_points = points[neighbor_indices]
        neighbor_distances = distances[neighbor_indices]
        diff_positions = neighbor_points - points[i]
        diff_distances = neighbor_distances - distances[i]
        gradient = np.linalg.lstsq(diff_positions, diff_distances, rcond=None)[0]
        gradients[i] = gradient
    # large gradient values at heat intersection points as heat scalar changes rapidly
    # only gradient direction is used
    gradients = gradients / np.linalg.norm(gradients, axis=1)[:, np.newaxis]
    return gradients



def gen_density_field(sdf_decoder, dataset_paras, save_path, ray_len=1.0, ray_sample_num=200):
    """
    generate density field of model interior (clamped by sdf). 
    Everytime query samples on a ray along inversed gradient direction. point with the smallest sdf as next ray shooting point. 
    Final terminated samples lie on skeleton (sdf singularitis), use abs(sdf) as density value for the beginning query points.
    -----------
    ray_len: length of each ray, max is sqrt(3)*2 for [-1, 1]^3 cube. if model interior is large, use larger ray_len
    ray_sample_num: number of samples on each ray for computation
    """

    def clip_density(densities, min_clip_percentile=5.0, max_clip_percentile=95.0):
        """clip density to avoid extreme density values"""
        sorted_densities, _ = torch.sort(densities)
        min_threshold_index = int(len(sorted_densities) * (min_clip_percentile / 100.0)) + 1
        max_threshold_index = int(len(sorted_densities) * (max_clip_percentile / 100.0)) - 1
        min_threshold_value = sorted_densities[min_threshold_index]
        max_threshold_value = sorted_densities[max_threshold_index]
        return torch.clamp(densities, min=min_threshold_value, max=max_threshold_value)

    def field_smooth_filter(pcd, fields, k=20):
        """smooth fileds by averaging k nearest neighbors"""
        diff = pcd.unsqueeze(1) - pcd.unsqueeze(0)  
        distances = torch.norm(diff, dim=-1) 
        knn_distances, knn_indices = torch.topk(-distances, k=k + 1, dim=1) 
        knn_indices = knn_indices[:, 1:] 

        # Compute the mean field value of the neighbors for each point
        neighbor_fields = fields[knn_indices] 
        smoothed_fields = neighbor_fields.mean(dim=1)  
        return smoothed_fields

    density_fn = os.path.join(save_path, 'density_field.xyz')
    base_sample_num = dataset_paras.get('density_base_sample_num', 100000)  # uniformly sample in cube, filtered by sdf
    save_sample_num = dataset_paras.get('density_sample_num', 10000)   # target saved number of samples having densities

    coords = torch.empty((base_sample_num, 3), device='cuda').uniform_(-1.0, 1.0) # many base samples
    samples, sdf_fileds, sdf_grads = field_query_with_grads(sdf_decoder, coords, no_print=True, return_on_cuda=True)
    coords = coords.detach()
    samples, sdf_fileds, sdf_grads= samples.detach(), sdf_fileds.detach(), sdf_grads.detach()
    
    # interior samples
    samples = samples[sdf_fileds.squeeze() < 0.]
    coords = coords[sdf_fileds.squeeze() < 0.]
    sdf_grads = sdf_grads[sdf_fileds.squeeze() < 0.]
    normals = F.normalize(sdf_grads, dim=-1)

    # get ray sample points whose norm is min, iteratively
    for idx in range(10):
        ray_samples, ray_normals = find_next_point_along_rays(sdf_decoder, samples, normals, ray_len, ray_sample_num)
        samples = ray_samples.detach()
        normals = ray_normals.detach()
        print(f"ray marching step {idx} ...")

    del normals, sdf_grads, sdf_fileds
    torch.cuda.empty_cache()

    # use abs(sdf) of these samples as density
    ray_samples = ray_samples.detach()
    densities = field_querying(sdf_decoder, ray_samples, no_print=True, return_on_cuda=True).detach()

    # clip density to avoid extreme high density values
    densities = clip_density(densities, min_clip_percentile=5.0, max_clip_percentile=95.0)
    # smooth density field in original pcd space
    smoothed_density = field_smooth_filter(coords, densities, k=20)
    # NOTE fertility use min_density=0.2, bunny uses min_density=0.15
    norm_densities = normalize_density(smoothed_density, min_density=0.15, max_density=1.0, set_levels=False, use_non_linear=False)

    hist_item(densities.cpu().numpy())  
    vis_pcd_fields(samples.cpu().numpy(), smoothed_density.cpu().numpy())  # skeleton points
    vis_pcd_fields(coords.cpu().numpy(), norm_densities.cpu().numpy())   # original pcd density

    # write density field file 
    data = torch.cat([coords, norm_densities.unsqueeze(-1)], dim=-1).cpu().numpy()
    final_num = min(save_sample_num, data.shape[0])
    save_data = data[:final_num, :]  # all random
    fmt = "%.6f %.6f %.6f %.6f"
    general_text_writer(save_data, fmt, density_fn, chunk=None)


def normalize_density(densities, min_density=0.2, max_density=1.0, levels=10, set_levels=False, use_non_linear=False):
    """normalize density to [min_density, max_density]
    densities are negative sdf values, abs first. since density cannot be zero, set min_density.
    NOTE here density means grad norm for lattice field, thus no need to be 1/density
    you may use linear normalization or non-linear activation to control density distribution"""

    densities = 1.0 - torch.abs(densities) # NOTE here, linear or non-linear activation
    min_val = torch.min(densities)
    max_val = torch.max(densities)
    densities = (densities - min_val) / (max_val - min_val + 1e-6)  # Scale to [0, 1]
    
    if use_non_linear:
        # smooth_factor = 5.0
        # densities = torch.sigmoid(smooth_factor * (densities - 0.3))
        smooth_factor = 2.0  # Smoothness control
        densities = torch.nn.functional.softplus(smooth_factor * (densities - 0.5))  # Apply Softplus
        # Normalize the Softplus output back to [0, 1]
        min_val = torch.min(densities)
        max_val = torch.max(densities)
        densities = (densities - min_val) / (max_val - min_val + 1e-6)

    normalized_densities = densities * (max_density - min_density) + min_density  # Scale to [min_density, max_density]

    if set_levels:
        # Discretize into levels
        step = (max_density - min_density) / levels
        normalized_densities = ((normalized_densities - min_density) // step) * step + min_density
        # Ensure values stay within bounds
        normalized_densities = torch.clamp(normalized_densities, min_density, max_density)
    return normalized_densities


def find_next_point_along_rays(sdf_decoder, samples, normals, ray_len=1.0, ray_num=200):
    """
    Finds the next point along rays starting from given samples and extending in the direction of inverse gradients.
    """
    # get ray samples and gradients
    rays = -normals  
    ray_offsets = torch.linspace(0, ray_len, steps=ray_num, device='cuda').view(1, -1, 1)  
    samples_on_rays = samples.unsqueeze(1) + rays.unsqueeze(1) * ray_offsets  
    ray_samples = samples_on_rays.view(-1, 3) 
    ray_samples, ray_sdfs, ray_grads = field_query_with_grads(
        sdf_decoder, ray_samples, no_print=True, return_on_cuda=True
    )
    ray_samples, ray_sdfs, ray_grads = ray_samples.detach(), ray_sdfs.detach(), ray_grads.detach()

    ray_sdfs = ray_sdfs.view(samples.shape[0], ray_num)  # (N, ray_num)
    ray_grads = ray_grads.view(samples.shape[0], ray_num, 3)  # (N, ray_num, 3)
    ray_samples = ray_samples.view(samples.shape[0], ray_num, 3)  # (N, ray_num, 3)

    # only ray samples at fisrt inside volume interval are valid
    valid_mask = ray_sdfs < 0.0  # (N, ray_num)
    range_mask = valid_mask.cumsum(dim=1) == (torch.arange(ray_num, device='cuda').view(1, -1) + 1)
    valid_grad_norms = torch.where(range_mask, torch.norm(ray_grads, dim=-1), float('inf'))  # Set invalid values to infinity.
    min_grad_indices = torch.argmin(valid_grad_norms, dim=1)  # (N,)
    next_samples = ray_samples[torch.arange(samples.shape[0], device='cuda'), min_grad_indices]  # (N, 3)
    next_gradients = ray_grads[torch.arange(samples.shape[0], device='cuda'), min_grad_indices]  # (N, 3)
    next_normals = F.normalize(next_gradients, dim=-1)

    return next_samples, next_normals


def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
    level=0.0,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    verts, faces, normals, values = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    try:
        print("marching cubes ..............")
        # * marching_cubes_lewiner is deperecated now
        # verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
        #     numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
        # )
        verts, faces, normals, values = skimage.measure.marching_cubes(
            numpy_3d_sdf_tensor, level=level, spacing=[voxel_size] * 3
        )
    except:
        print("marching cubes failed !")
        pass

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    logging.debug("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)

    logging.debug(
        "converting to ply format and writing to file took {} s".format(
            time.time() - start_time
        )
    )


