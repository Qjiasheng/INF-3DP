# this script is util for slicing 

import numpy as np
import os
import time
import torch
import matplotlib.pyplot as plt
from plyfile import PlyData
from scipy.spatial import KDTree
from toolpath_vis import *
import torch.nn.functional as F
from sklearn.cluster import DBSCAN
import scipy.stats

# Custom CUDA func for isocontour extraction. if not installed and not needed, comment func use if raised error.
from qupkg.extract_isocontours import extract_isocontours, get_fields_intersection_inter_slice

from sdf_meshing import gen_voxle_queries, convert_sdf_samples_to_ply
from utils import field_query_with_grads, field_querying, general_text_writer
from vis import *


def cuda_extract_iso_contours(vertices, faces, fields, iso_values, len_lim=50):
    """extract iso-contours with marching triangles in cuda, then split into a dict in python.
    all are numpy arrays"""

    vertices = torch.from_numpy(vertices).float().cuda().contiguous()
    faces = torch.from_numpy(faces).int().cuda().contiguous() # int type
    fields = torch.from_numpy(fields).float().cuda().contiguous()
    iso_values = torch.from_numpy(iso_values).float().cuda().contiguous()
    
    # first get merged contours in cuda, then split them in python
    contour_merged_points, contour_nums = extract_isocontours(vertices, faces, fields, iso_values)
    contours = split_merged_contours(contour_merged_points, contour_nums, iso_values, len_lim=len_lim)
    
    return contours

def split_merged_contours(contour_points, contour_nums, iso_values, len_lim=50):
    """
    Split merged contours into multiple ordered sublists of subcontours,
    contour_points [num_isovalues, num_faces, 3], every 2nd dim contains ordered contour points.
    subcontours are separated by -1e9,  even single contour has a separator at the end.
    [subcontour1, -1e9, 00000...], [subcontour1, -1e9, subcontour2, -1e9, 00000...]
    -------
    contour_nums [num_isovalues] refs to how many subcontours at each iso-value
    -------
    return a dict, keys are iso-values and items are lists of ordered connection points.
    sublist refs to a subcontour
    """
    contour_points = contour_points.cpu().numpy()
    contour_nums = contour_nums.cpu().numpy()
    iso_values = iso_values.cpu().numpy()

    # minimum number of points for a subcontour, realated with mesh resolution
    contours_dict = {}

    for i, iso_value in enumerate(iso_values):
        num_contours = contour_nums[i]  # Number of contours for this iso-value
        if num_contours == 0:
            contours_dict[iso_value] = []
            continue
        
        # contours points, subcontours are separated by -1e9
        # even single contour has a separator at the end
        merged_points = contour_points[i] 
        contours = [] 
        separator_indices = np.where(merged_points < -1e2)[0]

        start_idx = 0
        broken_subcontour = []
        for separator in separator_indices:
            subcontour = merged_points[start_idx:separator]
            # Only append subcontours with at least lim_num
            if len(subcontour) > len_lim:  
                # discard unenclosed contours
                if np.linalg.norm(subcontour[0] - subcontour[-1]) < 1e-4:
                    contours.append(subcontour.tolist())
                else:
                    broken_subcontour.append(subcontour)

            start_idx = separator + 1

         # Connect broken pieces with their nearest starts and ends
        if broken_subcontour:
            while broken_subcontour:
                # Start with the first broken subcontour
                current_contour = broken_subcontour.pop(0) 
                while True:
                    min_distance = float('inf')
                    closest_idx = -1
                    reverse_candidate = False
                    connection_side = None  # To track which side of the current_contour is being connected

                    # Compare current_contour's endpoints with all remaining subcontours
                    for idx, candidate in enumerate(broken_subcontour):
                        # Compute distances between endpoints of current_contour and candidate
                        dist_end_to_start = np.linalg.norm(current_contour[-1] - candidate[0])  
                        dist_end_to_end = np.linalg.norm(current_contour[-1] - candidate[-1])    
                        dist_start_to_start = np.linalg.norm(current_contour[0] - candidate[0]) 
                        dist_start_to_end = np.linalg.norm(current_contour[0] - candidate[-1])  

                        # Check all possible valid connections
                        distances = [
                            (dist_end_to_start, False, 'end'),   
                            (dist_end_to_end, True, 'end'),      
                            (dist_start_to_start, True, 'start'),  
                            (dist_start_to_end, False, 'start')  
                        ]

                        for dist, reverse_cand, side in distances:
                            if dist < min_distance:
                                min_distance = dist
                                closest_idx = idx
                                reverse_candidate = reverse_cand
                                connection_side = side

                    # If no valid connection is found or the minimum distance is too large, break the loop
                    if closest_idx == -1: 
                        break
                    # Reverse the candidate subcontour if required
                    if reverse_candidate:
                        broken_subcontour[closest_idx] = broken_subcontour[closest_idx][::-1]
                    # Connect the candidate to the current_contour based on the connection side
                    if connection_side == 'end':  # Connect to the end of current_contour
                        current_contour = np.vstack((current_contour, broken_subcontour[closest_idx]))
                    elif connection_side == 'start':  # Connect to the start of current_contour
                        current_contour = np.vstack((broken_subcontour[closest_idx], current_contour))
                    # Remove the connected subcontour from the broken_subcontour list
                    broken_subcontour.pop(closest_idx)

                contours.append(current_contour.tolist())
        contours_dict[iso_value] = contours

    return contours_dict

def culster_sing_centers(samples, singularity_condition, prj_grad_norms, eps=0.05, min_samples=5):
    """cluster singular points and get cluster centers"""
    # sing candidates
    singularities = samples[singularity_condition]  
    singular_grad_norms = prj_grad_norms[singularity_condition]
  
    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(singularities)

    cluster_centers = []
    unique_labels = set(labels) - {-1}  

    for label in unique_labels:
        cluster_points = singularities[labels == label]
        cluster_grad_norms = singular_grad_norms[labels == label]
        # Select the point with the smallest prj_grad_norm
        best_idx = np.argmin(cluster_grad_norms)
        cluster_centers.append(cluster_points[best_idx])

    return np.array(cluster_centers)

def load_mesh_from_ply(ply_file):
    ply_data = PlyData.read(ply_file)
    vertices = np.vstack([ply_data['vertex']['x'], ply_data['vertex']['y'], ply_data['vertex']['z']]).T
    faces = np.vstack(ply_data['face'].data['vertex_indices'])
    return vertices, faces

def get_single_shell_contours(slice_decoder, sdf_decoder, slice_num, mesh_fn, around_sing=True, sing_th=0.3, len_lim=50):
    """extract iso-contours through SDF=0 and slice fields intersection. 
    mesh iso-contours extraction in cuda.
    -------------------- 
    around_sing: contour slice value using singuarities' slice values. 
    if True, set singualrity condition --- projected slice grad norm < sing_th (near vanlish)
    len_lim: minimum number of points for a subcontour, realated with mesh resolution"""

    vertices, faces = load_mesh_from_ply(mesh_fn)
    # query sdf and slice fields
    coords = vertices.copy()
    _, slice_fields, slice_grads = field_query_with_grads(slice_decoder, coords, no_print=True, return_on_cuda=True)

    if around_sing:
         # exclude base region for sing detection as not involve optimization
        base_margin, top_margin, sing_gradnorm_th = 0.1, 0.0, sing_th  
        _, _, sdf_grads = field_query_with_grads(sdf_decoder, coords, no_print=True, return_on_cuda=True)
        normals = F.normalize(sdf_grads, dim=-1)
        prj_slice_grad = slice_grads - torch.sum(slice_grads * normals, dim=-1, keepdim=True) * normals
        prj_slice_norms = torch.norm(prj_slice_grad, dim=-1).cpu().numpy()
        singularity_condition = ((prj_slice_norms < sing_gradnorm_th) & 
                                 (vertices[:, 2] > np.min(vertices[:, 2]) + base_margin) & 
                                 (vertices[:, 2] < np.max(vertices[:, 2]) - top_margin))
        
        if np.sum(singularity_condition) < 1:
            print("No singularities detected, use uniform slice values.")
            sing_values, sing_centers = None, None
        else:
            sing_centers = culster_sing_centers(vertices, singularity_condition, prj_slice_norms, eps=0.05, min_samples=1)
            sing_values_tmp = field_querying(slice_decoder, torch.from_numpy(sing_centers).float(), no_print=True, return_on_cuda=False).numpy()
            sing_values =  np.sort(sing_values_tmp)
            sing_centers = sing_centers[np.argsort(sing_values_tmp)]
            print(f"culster {len(sing_values)} singular points")
            vis_two_set_pcd(vertices, sing_centers)
    else:
        sing_values = None
        sing_centers = None
        
    
    # collect silice iso-values
    slice_fields_np = slice_fields.cpu().numpy()
    field_min, field_max = np.min(slice_fields_np), np.max(slice_fields_np)
    total_range = field_max - field_min

    if sing_values is not None and sing_centers is not None:
        # iso-values (include singular isovalues)
        interval_sizes = np.diff(np.concatenate(([field_min], sing_values, [field_max])))  
        num_slices = np.round((interval_sizes / total_range) * slice_num).astype(int)  
        slice_iso_values = np.concatenate([
            np.linspace(start, end, num, endpoint=False if i < len(num_slices) - 1 else True)
            for i, (start, end, num) in enumerate(zip(
                np.concatenate(([field_min], sing_values)), 
                np.concatenate((sing_values, [field_max])), 
                num_slices
            ))
        ])
        print(f'Generated {len(slice_iso_values)} slice iso-values including singularities.')
    else:
        # uniform iso-values
        slice_iso_values = np.linspace(field_min, field_max, slice_num)
        print(f'Generated {len(slice_iso_values)} uniform slice iso-values.')
    
    # parallel iso-contours extraction in cuda
    start = time.time()
    torch.cuda.synchronize()
    contours = cuda_extract_iso_contours(vertices, faces, slice_fields_np, slice_iso_values, len_lim=len_lim)
    torch.cuda.synchronize()
    print(f"CUDA iso-contours extraction takes: {time.time() - start:.6f} seconds")

    return contours, sing_values, sing_centers, slice_fields_np, vertices, slice_iso_values
    
def iter_project_contours_batch(contours, sdf_decoder, slice_decoder, save_path=None):
    """Batch project intersection points to the isosurface"""
    # Save all iteration errors
    start = time.time()
    err_path = os.path.join(save_path, 'iter_error') if save_path else None
    if err_path:
        os.makedirs(err_path, exist_ok=True)

    sdf_level = 0.0
    re_contours = {}
    batch_samples = []  # Collect all samples for batching
    batch_metadata = []  # Track metadata (iso_value, subcontour index)

    # Prepare batch data
    for iso_value, contour in contours.items():
        re_contours[iso_value] = []
        for sub_idx, subcontour in enumerate(contour):
            if len(subcontour) < 5:  # Skip too short contours
                continue
            batch_samples.append(np.array(subcontour))
            batch_metadata.append((iso_value, sdf_level, len(subcontour)))   # isovalue and zero sdf
    batch_samples_tensor = torch.from_numpy(np.vstack(batch_samples)).float().cuda()

    # Perform batch projection
    # NOTE not stable at singularity regions since gradients vanish
    # paper report error test on bunny model with [one] isovalue intersected with sdf level 0
    projected_samples = batch_isosurface_projection(
        batch_samples_tensor, sdf_decoder, slice_decoder, batch_metadata, save_path=err_path
    )
    projected_samples = projected_samples.detach().cpu().numpy() 
    # Reorganize projected samples into dictionary
    start_idx = 0
    for iso_value, sdf_value, subcontour_len in batch_metadata:
        projected_subcontour = projected_samples[start_idx:start_idx + subcontour_len]
        re_contours[iso_value].append(projected_subcontour)
        start_idx += subcontour_len

    print(f"Total projection time: {time.time() - start:.6f} seconds")
    return re_contours


def batch_isosurface_projection(batch_samples, sdf_decoder, slice_decoder, metadata, save_path=None):
    """
    Batch optimize for isosurface projection
    Handles multiple iso_values and contours in one batch.
    see paper Eq. 30 and Fig. 6
    """
    # ------------------------- Key parameters
    rho = 1.0
    max_iters = 30
    newton_iters = 3
    tol = 1e-6
    # -------------------------

    x = batch_samples.clone()
    z1 = x.clone()
    z2 = x.clone()
    _, _, g1 = field_query_with_grads(sdf_decoder, batch_samples, no_print=True, return_on_cuda=True)
    inf1_grad_norm_base = torch.norm(g1, dim=1)
    _, _, g2 = field_query_with_grads(slice_decoder, batch_samples, no_print=True, return_on_cuda=True)
    inf2_grad_norm_base = torch.norm(g2, dim=1)

    def inf1_value_and_grad(_x):
        _, f1, g1 = field_query_with_grads(sdf_decoder, _x, no_print=True, return_on_cuda=True)
        return f1 / inf1_grad_norm_base, g1 / inf1_grad_norm_base.unsqueeze(1)
        # return f1, g1
    def inf2_value_and_grad(_x):
        _, f2, g2 = field_query_with_grads(slice_decoder, _x, no_print=True, return_on_cuda=True)
        return f2 / inf2_grad_norm_base, g2 / inf2_grad_norm_base.unsqueeze(1)

    # Prepare iso_values and sdf_values from metadata
    c1_list = []
    c2_list = []

    for iso_value, sdf_value, subcontour_len in metadata:
        c1_list.extend([sdf_value] * subcontour_len)
        c2_list.extend([iso_value] * subcontour_len)

    # Convert to tensors
    c1 = torch.tensor(c1_list).float().cuda() / inf1_grad_norm_base
    c2 = torch.tensor(c2_list).float().cuda() / inf2_grad_norm_base

    # Dual variables
    u1 = torch.zeros_like(x)
    u2 = torch.zeros_like(x)

    for k in range(max_iters):
        # x-step: Minimize ||x - batch_samples||^2 + (\rho /2)||x - z1 + u1||^2 + (\rho /2)||x - z2 + u2||^2
        numerator = batch_samples + rho * (z1 - u1) + rho * (z2 - u2)
        x_next = numerator / (1.0 + 2.0 * rho)
        x_next = x_next.detach().requires_grad_(True)

        # z1-step: Constrain f1(z1) = c1
        v1 = x_next + u1
        z1_next = v1.clone().detach().requires_grad_(True)
        for _ in range(newton_iters):
            f1, g1 = inf1_value_and_grad(z1_next)
            denom = g1.pow(2).sum(-1, keepdim=True).clamp_min(1e-8)
            with torch.no_grad():
                z1_next = z1_next - ((f1 - c1).unsqueeze(1) * g1) / denom
            z1_next = z1_next.detach().requires_grad_(True)

        # z2-step: Constrain f2(z2) = c2
        v2 = x_next + u2
        z2_next = v2.clone().detach().requires_grad_(True)
        for _ in range(newton_iters):
            f2, g2 = inf2_value_and_grad(z2_next)
            denom = g2.pow(2).sum(-1, keepdim=True).clamp_min(1e-8)
            with torch.no_grad():
                z2_next = z2_next - ((f2 - c2).unsqueeze(1) * g2) / denom
            z2_next = z2_next.detach().requires_grad_(True)

        # Dual variable updates
        u1 = u1 + (x_next - z1_next)
        u2 = u2 + (x_next - z2_next)

        # Convergence check
        constraint_residual_1 = torch.max(torch.abs(f1 - c1))
        constraint_residual_2 = torch.max(torch.abs(f2 - c2))
        displacement_residual = max(
            torch.max((x_next - z1_next).norm(dim=1)),
            torch.max((x_next - z2_next).norm(dim=1))
        )
        if displacement_residual < tol and constraint_residual_1 < tol and constraint_residual_2 < tol:
            break

        # Save iteration error if required
        if save_path:
            f1_x, _ = inf1_value_and_grad(x_next)
            f2_x, _ = inf2_value_and_grad(x_next)
            r1 = f1_x - c1
            r2 = f2_x - c2
            mean_dis = torch.mean(torch.sqrt(r1**2 + r2**2)).cpu().numpy()
            # with open(os.path.join(save_path, f'iter_error_{k}.txt'), 'a') as f:
            with open(os.path.join(save_path, f'iter_error.txt'), 'a') as f:
                f.write(f'{k}, {mean_dis:.6f}\n')

    return x_next

def smooth_contours_resampling(contours, outer_down_ratio=10):
    """
    resample contours with unifrom spacing.
    this will cause a little deviation from original intersection points.
    """
    total_length = 0
    total_original_points = 0
    # to store lengths for each iso-value's subcontours
    subcontour_lengths = {}  
    for iso_value, subcontours in contours.items():
        subcontour_lengths[iso_value] = []
        for subcontour in subcontours:
            points = np.array(subcontour)
            cumulative_distances = calculate_cumulative_distances(points)
            length = cumulative_distances[-1]
            total_length += length
            total_original_points += len(points) 
            subcontour_lengths[iso_value].append(length)
    
    # determine the fixed spacing distance for downsampling
    spacing = total_length / total_original_points * outer_down_ratio
    # resample each subcontour
    resampled_contours = {}
    for iso_value, subcontours in contours.items():
        resampled_contours[iso_value] = []
        for subcontour, length in zip(subcontours, subcontour_lengths[iso_value]):
            points = np.array(subcontour)
            if length < spacing:
                # handle short subcontours: Keep the endpoints
                resampled_contours[iso_value].append(points)
            else:
                # Resample with fixed spacing
                resampled_points = resample_subcontour_fixed_spacing(points, spacing)
                resampled_contours[iso_value].append(resampled_points)
    
    return resampled_contours, spacing


def calculate_cumulative_distances(points):
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative_distances = np.concatenate(([0], np.cumsum(distances)))
    return cumulative_distances

def resample_subcontour_fixed_spacing(points, spacing):
    # Compute cumulative distances along the subcontour
    cumulative_distances = calculate_cumulative_distances(points)
    total_length = cumulative_distances[-1]
    # Generate target distances for resampling
    target_distances = np.arange(0, total_length, spacing)
    if total_length not in target_distances:
        target_distances = np.append(target_distances, total_length)  # Include the endpoint
    
    # resample points
    resampled_points = []
    for d in target_distances:
        # Find the segment containing the target distance
        i = np.searchsorted(cumulative_distances, d) - 1
        i = max(0, min(i, len(points) - 2)) 
        # Linear interpolation between points[i] and points[i+1]
        denom = cumulative_distances[i+1] - cumulative_distances[i]
        if abs(denom) < 1e-6: 
            resampled_point = points[i]  # Use the first point as fallback
        else:
            t = (d - cumulative_distances[i]) / denom
            resampled_point = (1 - t) * points[i] + t * points[i+1]
        resampled_points.append(resampled_point)
    
    return np.array(resampled_points)


def compute_subcontour_centers(contours):
    centers = {}
    for iso_value, subcontours in contours.items():
        centers[iso_value] = [np.mean(subcontour, axis=0) for subcontour in subcontours]
    return centers

def get_nonempty_lowest_isovalue(iso_values, unused_contours):
    """find the next lowest iso-value with non empty subcontours."""
    for iso in iso_values:
        if unused_contours[iso]:
            return iso
    return None


def build_graph_partition(contours, jump_threshold=0.1, continuous_cnt_lim=256):
    """
    plan the printing sequence by finding the most continuous contours.
    Always process remaining subcontours starting from the lowest iso-value after completing a branch.
    Final printing_order is list of subcontours, in continous printing order.
    --------- inputs
    jump_threshold: distance bw contours' center for switches. 
    continuous_cnt_lim: max number of points for a continous printing branch.
    --------- returns
    printing_order: list of subcontours in continous printing order.
    printing_isovalue: each subcontour's iso-value.
    printing_branch: this iso-value corresponds to how many branches.
    printing_seg_index: seg index, a jump will increase the index by 1.
    """

    # before partition, remove empty subcontours
    print(f'before empty removal, total contours: {len(contours)}')
    contours = {k: [np.array(sub) for sub in v if len(sub) > 0] for k, v in contours.items() if any(len(sub) > 0 for sub in v)}
    centers = compute_subcontour_centers(contours)
    print(f'after empty removal, total contours: {len(contours)}')


    # exe partition
    iso_values = list(contours.keys())  # all iso-values, not change during partition
    unused_contours = {iso: list(range(len(centers[iso]))) for iso in iso_values}  
    printing_order = []
    printing_isovalue = []
    printing_branch = []
    printing_seg_index = []

    initial_subcontours = {iso: len(centers[iso]) for iso in iso_values}  # amount of each iso-value subcontours
    fixed_initial_subcontours = initial_subcontours.copy()  # fixed amount of subcontours at each iso-value

    current_center = None
    current_seg_index = 0
    current_iso_value = get_nonempty_lowest_isovalue(iso_values, unused_contours)
    iterations = 0
    continuous_cnt = 0  # track how many subcontours added continuously

    mark = None # manual operation

    while any(unused_contours.values()):

        if current_iso_value is None:  # process None, including tempory stop situation
            # everytime current iso value is None, reset them to start new branch
            current_iso_value = get_nonempty_lowest_isovalue(iso_values, unused_contours)
            current_center = None
            continuous_cnt = 0
            iterations = 0
            current_seg_index += 1  # new seg once start a new branch

        available_indices = unused_contours[current_iso_value]

        # no next iso-value (highest), still start from the lowest
        if not available_indices: 
            current_iso_value = None
            continue

        # process subcontours at the current iso-value
        current_center, current_iso_value, current_seg_index, initial_subcontours, unused_contours, mark = \
            process_iso_value(current_iso_value, current_center, current_seg_index, centers, contours, 
                            unused_contours, printing_order, printing_isovalue, printing_branch, printing_seg_index,
                            jump_threshold, initial_subcontours, fixed_initial_subcontours, mark)
        
        # if beyond continuous count limit, start from a new lowest iso-value
        continuous_cnt += 1
        if continuous_cnt > continuous_cnt_lim:
            current_iso_value = None
            continue
        
        # still have jumps but beyond preset jump threshold
        # every 200, strat from a new lowest iso-value
        iterations += 1
        if iterations > 10 * len(iso_values):
            if iterations % 200 == 0:
                current_iso_value = None

    return printing_order, printing_isovalue, printing_branch, printing_seg_index


def process_iso_value(current_iso_value, current_center, current_seg_index, centers, contours, unused_contours, 
                      printing_order, printing_isovalue, printing_branch, printing_seg_index, 
                      jump_threshold, initial_subcontours, fixed_initial_subcontours, mark):
    """
    Process subcontours at given iso-value, adding closest subcontour in next iso-value
    and returning the updated current center. 
    """
    iso_values = list(contours.keys())
    available_indices = unused_contours[current_iso_value]
    if not available_indices: # highest iso-value, still start a new branch from lowest 
        return current_center, get_nonempty_lowest_isovalue(iso_values, current_iso_value), current_seg_index, initial_subcontours, unused_contours, mark

    # Ensure numpy array for efficient indexing
    iso_centers = np.array(centers[current_iso_value])
    selected_centers = iso_centers[available_indices]

    # in this case, start from a new lowest iso-value
    if current_center is not None:
        distance = np.linalg.norm(selected_centers - current_center, axis=1)
        if np.min(distance) > jump_threshold:
            return current_center, get_nonempty_lowest_isovalue(iso_values, unused_contours), current_seg_index, initial_subcontours, unused_contours, mark

    if initial_subcontours[current_iso_value] == 1:
        # Only one subcontour; add directly
        index = available_indices[0]
        printing_order.append(contours[current_iso_value][index])
        printing_isovalue.append(current_iso_value)
        printing_branch.append(fixed_initial_subcontours[current_iso_value])
        printing_seg_index.append(current_seg_index)
        # subcontour numbers should -1
        initial_subcontours[current_iso_value] -= 1

        unused_contours[current_iso_value].remove(index)
        return centers[current_iso_value][index], get_next_higher_isovalue(iso_values, current_iso_value), current_seg_index, initial_subcontours, unused_contours, mark
    else:
        # Multiple subcontours; find center closest one
        closest_index, closest_center = find_closest_subcontour(current_center, selected_centers)
        # closest_index, closest_center = find_closest_subcontour_with_z(current_center, selected_centers)
        real_index = available_indices[closest_index]
        printing_order.append(contours[current_iso_value][real_index])
        printing_isovalue.append(current_iso_value)
        printing_branch.append(fixed_initial_subcontours[current_iso_value])
        printing_seg_index.append(current_seg_index)
        # subcontour numbers should -1
        initial_subcontours[current_iso_value] -= 1

        unused_contours[current_iso_value].remove(real_index)
        return closest_center, get_next_higher_isovalue(iso_values, current_iso_value), current_seg_index, initial_subcontours, unused_contours, mark
    

def get_next_higher_isovalue(iso_values, current_iso_value):
    """next higher iso-values with remaining subcontours.    """
    higher_iso_values = [iso for iso in iso_values if iso > current_iso_value]
    return higher_iso_values[0] if higher_iso_values else None

def find_closest_subcontour(current_center, candidates):
    """Find the closest subcontour center to the current center."""
    if current_center is None:
        return 0, candidates[0]  # Pick the first candidate if no current center
    distances = np.linalg.norm(candidates - current_center, axis=1)
    closest_idx = np.argmin(distances)
    return closest_idx, candidates[closest_idx]

def find_closest_subcontour_with_z(current_center, candidates):
    """Find the closest subcontour center to the current center."""
    if current_center is None:
        return 0, candidates[0]  # Pick the first candidate if no current center
    # idx with smallest z value of centers
    closest_idx = np.argmin(candidates[:, 2])
    return closest_idx, candidates[closest_idx]


def get_same_order_contours(contours):
    """alter all contours with same rounding orientation"""
    ordered_contours = []
    for contour in contours:
        signed_area = calculate_signed_area_2d(np.array(contour))
        if signed_area < 0:
            ordered_contours.append(contour[::-1])
        else:
            ordered_contours.append(contour)
    return ordered_contours

def calculate_signed_area_2d(contour):
    """determine sign of area by shoelace formula, 
    to be same clockwise or counter-clockwise.
    NOTE for contours near singularity, their intersections may yield arbitrary signs."""
    if contour.shape[1] != 3:
        raise ValueError("Each point in the contour must have three coordinates (x, y, z).")
    # Extract the X and Y coordinates
    x_coords = contour[:, 0]
    y_coords = contour[:, 1]
    # Use the shoelace formula to calculate the signed area
    area = 0.5 * np.sum(x_coords[:-1] * y_coords[1:] - y_coords[:-1] * x_coords[1:])
    return area


def simple_arrange_subcontour_starts(contours):
    """simple arrange subcontour starts, use shortest distance to current start"""
    if not contours:
        return []
    start_points = []
    start_indices = []

    # Manually select the first point from the first subcontour
    start_index = 0
    current_start = np.array(contours[0][start_index])
    start_points.append(current_start)
    start_indices.append(start_index)
    remaining_contours = contours[1:] 

    for next_contour in remaining_contours:
        next_contour_np = np.array(next_contour)
        distances = np.linalg.norm(next_contour_np - current_start, axis=1)
        closest_idx = np.argmin(distances)
        closest_point = next_contour[closest_idx]
        start_points.append(closest_point)
        start_indices.append(closest_idx)
        current_start = np.array(closest_point)

    return start_points, start_indices

def generate_waypoints_with_levels(printing_order_contours, start_indices, print_isovalues):
    """
    based on start_indices, generate waypoints for printing, 
    for each waypt, set partition levels (isovalue jump refs to next level).
    """
    if len(printing_order_contours) != len(start_indices):
        raise ValueError("subcontours and start indices must have the same length")
    
    ordered_waypoints = []
    level_tags = []

    # build contour levels (isovalue jump refs to next level)
    level = 0
    contour_levels = []
    for i, isovalue in enumerate(print_isovalues):
        if i>0 and isovalue < print_isovalues[i-1]:
            level += 1
        contour_levels.append(level)
    
    print(f'contour levels {max(contour_levels)+1}')
            
    for contour_loop, start_index, con_level in zip(printing_order_contours, start_indices, contour_levels):
        if not 0 <= start_index < len(contour_loop):
            raise ValueError(f"Start index {start_index} out of range for contour with length {len(contour_loop)}")
        
        if start_index == len(contour_loop) - 1:
            start_index = 0  # contour start and end are same

        contour = contour_loop[:-1] 
        # after removal, contour may be empty
        if len(contour) < 2:
            print(f"Contour with length {len(contour)} is too short, skipping.")
            continue

        ordered_contour = list(contour[start_index:]) + list(contour[:start_index]) + [contour[start_index]]
        ordered_waypoints.extend(ordered_contour)
        # level tags
        level_tags.extend([con_level] * len(ordered_contour))

    return np.array(ordered_waypoints), np.array(level_tags)


def build_shell_level_dataset(ori_print_contours, ori_print_isovalues, level_fn, interpolate=False, inter_num=5):
    """Build print level dataset, only shell.
    since original contours may be space-downsampled, interpolation is needed to densify points."""   

    # Remove empty contours
    print_contours = []
    print_isovalues = []
    for contour, isovalue in zip(ori_print_contours, ori_print_isovalues):
        if len(contour) > 0:
            print_contours.append(contour)
            print_isovalues.append(isovalue)  
        else:
            print(f"Warning: Empty contour found for isovalue {isovalue}, skipping this contour.")
    
    dataset = []  # To store datapairs: (slice_isovalue, xyz) --> level label
    level = 0  # Start with level 0
    contour_levels = []  # To track the level for each slice

    # An isovalue jump indicates the next level
    for i, isovalue in enumerate(print_isovalues):
        if i > 0 and isovalue < print_isovalues[i - 1]:
            level += 1 
        contour_levels.append(level)

    print(f"Number of levels: {level + 1}")

    for layer_idx, (contour, level_label) in enumerate(zip(print_contours, contour_levels)):
        for point_idx, point in enumerate(contour):
            if len(point) == 3:
                dataset.append((print_isovalues[layer_idx], point.tolist(), level_label))

            # Interpolation logic
            if interpolate and point_idx > 0:
                # Interpolate between the current point and the previous one
                prev_point = contour[point_idx - 1]
                num_interpolation_points = inter_num  # Number of intermediate points
                interpolated_points = np.linspace(prev_point, point, num=num_interpolation_points, endpoint=False)

                for interp_point in interpolated_points:
                    dataset.append((print_isovalues[layer_idx], interp_point.tolist(), level_label))

    # Each element in the dataset is (slice_isovalue, xyz, level_label)
    dataset_dict = {
        "xyz": torch.tensor([entry[1] for entry in dataset], dtype=torch.float32),
        "slice_isovalues": torch.tensor([entry[0] for entry in dataset], dtype=torch.float32),
        "level_labels": torch.tensor([entry[2] for entry in dataset], dtype=torch.long)
    }

    torch.save(dataset_dict, level_fn)
    print(f"{dataset_dict['xyz'].shape[0]} points in total, Dataset saved to {level_fn}")


def build_calibration_thickness(print_contours, sdf_decoder, slice_decoder, object_scale=100, fit_down_ratio=200, dist_threshold=60):
    """build relationship between slice grad norm and layer thickness. 
    layer thickness is the Geodesic distance between two consecutive slice contours
    with high resolution voxel grid, use Euclidean distance as approximation

    unit is mm    
    print_contours: [contour1(arr), contour2(arr), ...]"""

    distances, print_grad_norms, all_points = get_one_level_distances(print_contours, sdf_decoder, slice_decoder, object_scale)

    # everytime before fitting, check distances distribution
    ratio = np.sum(distances <= 2.0) / len(distances)
    print(f'ratio of thickness <= 2.0 mm: {ratio * 100} %')
    plt.hist(distances, bins=200, range=(0, 3.0))
    plt.show()
    
    # filter outliers manually if needed
    mask = (distances < dist_threshold) 
    distances = distances[mask]
    print_grad_norms = print_grad_norms[mask]

    # --- fitting as a function
    fit_function = fitting(print_grad_norms, distances, down_ratio=fit_down_ratio)
    print(f"Calibration function: {fit_function}")
    # vis fitting results with original data
    vis_fitting_results(print_grad_norms, distances, fit_function, down_ratio=1)

    return fit_function

def get_prj_grad_len(coords, slice_decoder, sdf_decoder):
    """helper function to get projected slice grads length along normals
    for further fitting to waypts thickness
    coords: N by 3 numpy array"""
    _, _, sdf_grads = field_query_with_grads(sdf_decoder, coords, no_print=True, return_on_cuda=False)
    sdf_grads = sdf_grads.numpy()
    normals = sdf_grads / np.linalg.norm(sdf_grads, axis=1, keepdims=True)
    _, _, slice_grads = field_query_with_grads(slice_decoder, coords, no_print=True, return_on_cuda=False)
    slice_grads = slice_grads.numpy()
    prj_slice_grads = slice_grads - np.sum(slice_grads * normals, axis=1, keepdims=True) * normals
    print_grad_norms = np.linalg.norm(prj_slice_grads, axis=1)

    return print_grad_norms

def get_one_level_distances(print_contours, sdf_decoder, slice_decoder, object_scale=100):
    """get distances and slice grad norms for one level contours"""

    if len(print_contours) < 2:
        raise ValueError("At least two contours are required to build calibration data.")
    
    distances = []
    for i in range(1, len(print_contours)):
        current_contour = print_contours[i]
        previous_contour = print_contours[i - 1]
        kd_tree = KDTree(previous_contour)
        dist, _ = kd_tree.query(current_contour)
        distances.extend(dist)  # scale to mm

    distances = np.array(distances) * object_scale  # scale to mm
    # get projected slice grads along normals, except first contour points
    all_points = np.concatenate(print_contours[1:], axis=0)
    print_grad_norms = get_prj_grad_len(all_points, slice_decoder, sdf_decoder)

    vis_pcd_fields(all_points, print_grad_norms, label='Projected slice grad norms')
    return distances, print_grad_norms, all_points

def fitting(norms, dists, down_ratio=200):
    """fitting function cubic polynomial, dists = f(norms)
    downsampling is crucial, avoiding more data in high density region"""
    dists = dists[::down_ratio]
    norms = norms[::down_ratio]

    coefficients = np.polyfit(norms, dists, deg=3)
    fit_function = np.poly1d(coefficients)
    return fit_function


def write_waypoints_single_shell(waypoints, fit_function, slice_decoder, sdf_decoder, object_scale, wayfn, 
                                 level_labels=None, for_collision=True):
    """write waypoints for single shell, fit_function is used to get thickness (already object-scaled).
    waypoints not yet scaled, in range [-1, 1]"""
    coords = np.array(waypoints)

    # get (slice grad) print directions
    _, _, slice_grads = field_query_with_grads(slice_decoder, coords, no_print=True, return_on_cuda=True)
    print_dirs = F.normalize(slice_grads, p=2, dim=1).detach().cpu().numpy()

    # get thickness through fit function
    slice_grad_len = get_prj_grad_len(coords, slice_decoder, sdf_decoder)
    thickness = fit_function(slice_grad_len) # already scaled to mm
    
    # physical coords
    phy_coords = coords * object_scale   # scale to mm, move to positive
    phy_coords[:, 2] = phy_coords[:, 2] - np.min(phy_coords[:, 2]) 

    # visualize thickness on pcd
    vis_pcd_fields(phy_coords, thickness, label='Waypoints thickness (mm)')

    # if for further coll, still in [-1, 1] cube
    write_coords = coords if for_collision else phy_coords
    data = np.hstack((write_coords, print_dirs, thickness.reshape(-1, 1), level_labels.reshape(-1, 1)))
    fmt = '% .6f ' * (data.shape[1]-1) + '%d'
    general_text_writer(data=data, fmt=fmt, filename=wayfn, chunk=None)
    print(f"Waypoints saved to {wayfn}, total {data.shape[0]} points.")


def fit_GS_distribution(source_data, target_data):
    source_mean, source_std = np.mean(source_data), np.std(source_data)
    target_mean, target_std = np.mean(target_data), np.std(target_data)

    def map_to_target(data):
        cdf_values = scipy.stats.norm.cdf(data, loc=source_mean, scale=source_std)
        mapped_data = scipy.stats.norm.ppf(cdf_values, loc=target_mean, scale=target_std)
        return mapped_data
    
    return map_to_target

def write_waypoints_lattice(waypoints, fit_function, slice_decoder, sdf_decoder, object_scale, 
                            indicators, wayfn, spacing, level_labels, for_collision=True):
    """"write waypoints for lattice, fit_function is used to get outer shell thickness (already scaled).
    inner points thickness is set by Gaussian distribution mapping of outer shell thickness"""

    print(f"inner points: {sum(indicators)}, outer points: {len(indicators) - sum(indicators)}")

    coords = np.array(waypoints)
    indicators = np.array(indicators)
    level_labels = np.array(level_labels)

    # get printing directions for all waypoints
    _, _, slice_grads = field_query_with_grads(slice_decoder, coords, no_print=True, return_on_cuda=True)
    print_dirs = F.normalize(slice_grads, p=2, dim=1).detach().cpu().numpy()

    thickness = np.zeros_like(indicators, dtype=float)
    # outer waypoints use fitted thickness
    outer_waypts = coords[indicators == 0]
    outer_slice_grad_len = get_prj_grad_len(outer_waypts, slice_decoder, sdf_decoder)
    outer_thickness = fit_function(outer_slice_grad_len) # already scaled to mm
    thickness[indicators == 0] = outer_thickness

    # inner waypoints map slice grad norm to shell thickness (both are treated as Gaussians)
    # inner_waypts = coords[indicators == 1]
    inner_slice_grad_norm = slice_grads[indicators == 1].detach().norm(p=2, dim=-1).cpu().numpy()
    inv_inner_slice_norm  = 1.0 / (inner_slice_grad_norm + 1e-6)  
    map_fun = fit_GS_distribution(inv_inner_slice_norm, outer_thickness)
    inner_thickness = map_fun(inv_inner_slice_norm)
    thickness[indicators == 1] = inner_thickness * 1.5 # lattice infill at one layer by a factor
    thickness = thickness.clip(max=4.0)  # clamp thickness

    # vis all thickness
    hist_item(thickness)

    # physical coords
    phy_coords = coords * object_scale   # scale to mm, move to positive
    phy_coords[:, 2] = phy_coords[:, 2] - np.min(phy_coords[:, 2]) 

    # visualize thickness on pcd
    vis_pcd_fields(phy_coords, thickness, label='Waypoints thickness (mm)')

    # if for further coll, still in [-1, 1] cube
    # have extra indicator column for collision data sampling
    write_coords = coords if for_collision else phy_coords
    data = np.hstack((write_coords, print_dirs, thickness.reshape(-1, 1), level_labels.reshape(-1, 1), indicators.reshape(-1, 1)))
    fmt = '% .6f ' * (data.shape[1]-2) + '%d' + '%d'
    general_text_writer(data=data, fmt=fmt, filename=wayfn, chunk=None)
    print(f"Waypoints saved to {wayfn}, total {data.shape[0]} points.")
    



def gen_sdf_level_surface(sdf_fields, ply_filename, N=256, max_batch=32 ** 3, level=0.0, voxel_origin=[-1, -1, -1], voxel_size=2.0/255):
    """MC extracts triangle mesh from SDF"""
    # voxel_origin = [-1, -1, -1]
    # voxel_size = 2.0 / (N - 1)
    # save to ply
    convert_sdf_samples_to_ply(
        sdf_fields.data.cpu(),
        voxel_origin,
        voxel_size,
        ply_filename,
        offset=None,
        scale=None,
        level=level,
    ) 


def get_wall_shell_contours(slice_decoder, sdf_decoder, slice_num, 
                            sdf_level_num, sdf_level_margin, save_path, N=256, len_lim=50):
    """
    multiply shell contours extraction.
    first generate sdf level mesh. then extract iso-contours for each level.
    """
    # spatial grid sdf fields
    grid_queries, voxel_origin, voxel_size = gen_voxle_queries(N=N, cube_size=1.0)
    sdf_grid_fields = field_querying(sdf_decoder, grid_queries, no_print=True, return_on_cuda=False)
    sdf_grid_fields = sdf_grid_fields.reshape((N, N, N))

    # save sdf level meshes 
    for i in range(sdf_level_num):
        ply_fn = os.path.join(save_path, f'sdf_{i}.ply')
        level = 0 - i * sdf_level_margin
        gen_sdf_level_surface(sdf_grid_fields, ply_fn, N=N, level=level,
                              voxel_origin=voxel_origin, voxel_size=voxel_size)
        
    # use 0-level to determine slice iso-values
    level0_fn = os.path.join(save_path, f'sdf_0.ply')
    contours_level0, _, _, _, _, slice_iso_values = get_single_shell_contours(slice_decoder, sdf_decoder, slice_num, mesh_fn=level0_fn,
                                                                            around_sing=True, sing_th=0.3, len_lim=50)
    
    # extract contours for each level
    contours_total = []
    contours_total.append(contours_level0)
    for i in range(1, sdf_level_num):
        vertices, faces = load_mesh_from_ply(os.path.join(save_path, f'sdf_{i}.ply'))
        fields = field_querying(slice_decoder, vertices, no_print=True, return_on_cuda=False).cpu().numpy()
        contours = cuda_extract_iso_contours(vertices, faces, fields, slice_iso_values, len_lim=len_lim)
        contours_total.append(contours)
    print(f'Extracted {len(contours_total)} shells.')

    return contours_total, slice_iso_values


def wall_smooth_contours_resampling(contours, outer_down_ratio=10):
    """"since contours at different levels use same parameters, resolution, down ratio... the spacing is almost the same
    no need to resample with total same spacing, do it for each level"""
    smooth_contours = []
    spacings = []
    for contour in contours:
        smooth_contours.append(smooth_contours_resampling(contour, outer_down_ratio=outer_down_ratio)[0])
        spacings.append(smooth_contours_resampling(contour, outer_down_ratio=outer_down_ratio)[1])    
    return smooth_contours, spacings


def get_grid_fields(decoder_list, N=256):
    """get grid fields for multiple decoders"""
    grid_queries, _, _ = gen_voxle_queries(N=N, cube_size=1.0)
    all_fields = []
    for decoder in decoder_list:
        fields = field_querying(decoder, grid_queries, no_print=True, return_on_cuda=False)
        fields = fields.reshape((N, N, N))
        all_fields.append(fields)
    return all_fields


def extract_corner_fields(fields):
    """convert fields (N x N x N) to corner fields (voxel format N-1 x N-1 x N-1 x 8)"""
    N = fields.shape[0]

    # Define corner offsets as a (8, 3) tensor
    corner_offsets = torch.tensor([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
        [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]
    ], dtype=torch.long)

    # Create a grid of voxel indices
    grid_size = N - 1
    grid_x, grid_y, grid_z = torch.meshgrid(
        torch.arange(grid_size, dtype=torch.long),
        torch.arange(grid_size, dtype=torch.long),
        torch.arange(grid_size, dtype=torch.long),
        indexing="ij"
    )
    # Stack grid indices into (grid_size, grid_size, grid_size, 3)
    grid_indices = torch.stack([grid_x, grid_y, grid_z], dim=-1)  # Shape: (N-1, N-1, N-1, 3)
    # Add corner offsets to the grid indices to get the 8 corners for each voxel
    voxel_corner_indices = grid_indices.unsqueeze(-2) + corner_offsets  # Shape: (N-1, N-1, N-1, 8, 3)
    flat_corner_indices = voxel_corner_indices.reshape(-1, 3)  # Shape: (num_voxels * 8, 3)
    # Use advanced indexing to gather the corner values for each field
    corner_fields = fields[flat_corner_indices[:, 0], flat_corner_indices[:, 1], flat_corner_indices[:, 2]]
    # Reshape the results back to (N-1, N-1, N-1, 8)
    corner_fields = corner_fields.view(grid_size, grid_size, grid_size, 8)

    return corner_fields

def get_voxel_centers(N):
    """"get all voxel centers"""
    voxel_size = 2.0 / (N - 1)
    voxel_origin = torch.tensor([-1.0, -1.0, -1.0])
    # Precompute the voxel center coordinates
    grid_indices = torch.arange(0, N - 1, dtype=torch.float32)
    x, y, z = torch.meshgrid(grid_indices, grid_indices, grid_indices, indexing="ij")
    voxel_centers = torch.stack([x, y, z], dim=-1) + 0.5  # Center offset
    voxel_centers = voxel_centers * voxel_size + voxel_origin  # Scale to world coordinates
    # voxel_centers = voxel_centers.reshape(-1, 3)  # Flatten into (M x 3)

    return voxel_centers

def compute_iso_values(field, num_isovalues):
    valid_values = field[~torch.isnan(field)]  # Extract only valid (non-NaN) values
    min_iso_value = valid_values.min().item()
    max_iso_value = valid_values.max().item()
    return torch.linspace(min_iso_value, max_iso_value, num_isovalues)

def get_intersection_points_cuda(fields_list, num_lattice1, num_lattice2, sdf_level=0.0, isovalues_slice=None):
    """ all fileds are N x N x N, convert into voxels format,
    fields_list: [sdf_field, slice_field, lattice_field1, lattice_field2]
    then perfrom intersections in CUDA along voxles.
    NOTE all cube is default set as [-1, 1]^3 with origin [-1, -1, -1] and voxel size 2/(N-1).
    if any issue, pls check"""

    sdf_field, slice_field, lattice_field1, lattice_field2 = fields_list

    N = slice_field.shape[0]
    assert slice_field.shape == (N, N, N), "slice_field must have the same shape."
    assert lattice_field1.shape == (N, N, N), "lattice_field1 must have the same shape."
    assert lattice_field2.shape == (N, N, N), "lattice_field2 must have the same shape."
    assert sdf_field.shape == (N, N, N), "SDF field must have the same shape as the other fields."
    
    # convert voxels format (N-1) x (N-1) x (N-1) x 8, origianl is N x N x N
    corner_fields_slice = extract_corner_fields(slice_field)
    corner_fields_lattice1 = extract_corner_fields(lattice_field1)
    corner_fields_lattice2 = extract_corner_fields(lattice_field2)
    corner_fields_sdf = extract_corner_fields(sdf_field)

    # voxel centers (N-1) x (N-1) x (N-1) x 3
    voxel_centers = get_voxel_centers(N)

    # prepare isovlaues for different fields
    valid_mask = (
        ~torch.isnan(corner_fields_slice).any(dim=-1) &
        ~torch.isnan(corner_fields_lattice1).any(dim=-1) &
        ~torch.isnan(corner_fields_lattice2).any(dim=-1) &
        ~torch.isnan(corner_fields_sdf).any(dim=-1)
    )

    valid_mask &= corner_fields_sdf.min(dim=-1).values < sdf_level  # sdf_level, we use inner level (<0)

    # filter out invalid corner fields
    corner_fields_sdf = torch.where(valid_mask.unsqueeze(-1), corner_fields_sdf, torch.tensor(float('nan')))
    corner_fields_slice = torch.where(valid_mask.unsqueeze(-1), corner_fields_slice, torch.tensor(float('nan')))
    corner_fields_lattice1 = torch.where(valid_mask.unsqueeze(-1), corner_fields_lattice1, torch.tensor(float('nan')))
    corner_fields_lattice2 = torch.where(valid_mask.unsqueeze(-1), corner_fields_lattice2, torch.tensor(float('nan')))

    # iso-values in three fields, combine them for intersections
    iso_values_slice = torch.from_numpy(isovalues_slice)
    iso_values_lattice1 = compute_iso_values(corner_fields_lattice1, num_lattice1)
    iso_values_lattice2 = compute_iso_values(corner_fields_lattice2, num_lattice2)

    torch.cuda.synchronize()
    start = time.time()
    # slice, lattice1, and lattice2 fields intersections
    intersection_dict = get_fields_intersection_wrapper(voxel_centers, corner_fields_slice, corner_fields_lattice1, 
                                                        corner_fields_lattice2, iso_values_slice, iso_values_lattice1, 
                                                        iso_values_lattice2)   

    torch.cuda.synchronize()
    print(f"CUDA Lattice Intersections computation took: {time.time() - start:.2f} seconds")

    return intersection_dict


def get_fields_intersection_wrapper(voxel_centers, corner_fields_slice, corner_fields_lattice1, corner_fields_lattice2, 
                                    iso_values_slice, iso_values_lattice1, iso_values_lattice2):
    """CUDA wrapper for intersection computation,
    intersection_dict: [key: tuple(iso_vlues combination), value: averaged intersection voxel centers]
    isovlaue combination is (slice, lattice1, lattice2)"""

    # to cuda 
    corner_fields_slice = corner_fields_slice.float().cuda().contiguous()
    corner_fields_lattice1 = corner_fields_lattice1.float().cuda().contiguous()
    corner_fields_lattice2 = corner_fields_lattice2.float().cuda().contiguous()
    iso_values_slice = iso_values_slice.float().cuda().contiguous()
    iso_values_lattice1 = iso_values_lattice1.float().cuda().contiguous()
    iso_values_lattice2 = iso_values_lattice2.float().cuda().contiguous()
    voxel_centers = voxel_centers.float().cuda().contiguous()

    # get intersection masks and iso-values index with cuda
    # intersection_masks ---> N_slice x N_lattice1 x N_lattice2; iso_indices, intersections ---> N_slice x N_lattice1 x N_lattice2 x 3
    intersection_masks, iso_indices, intersections = get_fields_intersection_inter_slice(corner_fields_slice, corner_fields_lattice1, 
                                                              corner_fields_lattice2, iso_values_slice, 
                                                              iso_values_lattice1, iso_values_lattice2, voxel_centers)

    # all in numpy
    intersection_masks = intersection_masks.cpu().numpy()
    iso_indices = iso_indices.cpu().numpy()  
    intersections = intersections.cpu().numpy()
    iso_values_slice = iso_values_slice.cpu().numpy()
    iso_values_lattice1 = iso_values_lattice1.cpu().numpy()
    iso_values_lattice2 = iso_values_lattice2.cpu().numpy()

    # empty cuda memory
    torch.cuda.empty_cache()  

    # extract valid intersections
    intersection_dict = {}
    intersecting_iso_indices = iso_indices[intersection_masks]         # (num_intersections, 3)
    intersectings = intersections[intersection_masks]                  # (num_intersections, 3)

    # build dict for intersections
    sum_dict = {}
    count_dict = {}

    for iso_idx, inter in zip(intersecting_iso_indices, intersectings):
        iso_tuple = (iso_values_slice[iso_idx[0]], iso_values_lattice1[iso_idx[1]], iso_values_lattice2[iso_idx[2]])
        if iso_tuple not in sum_dict:
            sum_dict[iso_tuple] = np.zeros(3) 
            count_dict[iso_tuple] = 0 

        sum_dict[iso_tuple] += inter
        count_dict[iso_tuple] += 1

    # average at same iso-values combination
    for iso_tuple in sum_dict:
        intersection_dict[iso_tuple] = sum_dict[iso_tuple] / count_dict[iso_tuple]

    torch.cuda.empty_cache()

    return intersection_dict    



def build_linked_dict(intersection_dict, front='lattice1'):
    """build a dict(tuple, 3d point) for linked dict. 
    {slice: {lattice1: {lattice2: 3d point}}}"""

    if front not in ['lattice1', 'lattice2']:
        raise ValueError("Argument 'front' must be either 'lattice1' or 'lattice2'.")
    # Determine the order of keys 
    second_index, third_index = (1, 2) if front == 'lattice1' else (2, 1)
    linked_structure = {}
    for key, intersection_point in intersection_dict.items():
        slice_iso, second_iso, third_iso = key[0], key[second_index], key[third_index]

        if slice_iso not in linked_structure:
            linked_structure[slice_iso] = {}
        if second_iso not in linked_structure[slice_iso]:
            linked_structure[slice_iso][second_iso] = {}
        linked_structure[slice_iso][second_iso][third_iso] = intersection_point

    return linked_structure

def downsample_lattice_contours(inter_s12, inter_s21, lattice1_num, lattice2_num, sdf_decoder, 
                                till_slice_layers=50, ratio=1.0, split_sdf_level=0.0):
    """
    Downsample lattice contours with targeted lattice numbers.
    split subcontours (outside sdf at concave regions) based on `interval centers` inside or outside sdf.
    For each slice lattice subcontours, always keep the two endpoints (near outer contours).
    Downsampled lattice isovalues are determined by both lattice 1 and 2 isovalues.
    """
    torch.cuda.empty_cache()

    down_subcontour_s1 = {}
    down_subcontour_s2 = {}

    all_lattice2_isos = set()
    all_lattice1_isos = set()

    # Collect all lattice isovalues
    for slice_iso, data_12 in inter_s12.items():
        for lattice2 in data_12.values():
            all_lattice2_isos.update(lattice2.keys())

    for slice_iso, data_21 in inter_s21.items():
        for lattice1 in data_21.values():
            all_lattice1_isos.update(lattice1.keys())

    all_lattice2_isos = sorted(all_lattice2_isos)
    all_lattice1_isos = sorted(all_lattice1_isos)

    # get all interval centers through sdf for splitting flags
    s12_flags = get_split_flags(inter_s12, sdf_decoder, split_sdf_level)  # already sorted along lattice2 isovalues --- for centers
    s21_flags = get_split_flags(inter_s21, sdf_decoder, split_sdf_level)

    # get global statistics threshold distance for filtering
    # NOTE this is only used in uniform lattice infill whose interval are unifrom
    s12_threshold = get_global_distance_threshold(inter_s12, till_slice_layers, ratio)
    s21_threshold = get_global_distance_threshold(inter_s21, till_slice_layers, ratio)

    # each slice
    for (slice_iso, slice_data_s12), (_, slice_data_s21), (_, s12_flag), (_, s21_flag) in \
          zip(inter_s12.items(), inter_s21.items(), s12_flags.items(), s21_flags.items()):
        # Filter and split subcontours using global interval threshold
        filtered_slice_s12 = filter_and_split_contours_with_flags_dis(slice_data_s12, s12_flag, s12_threshold)
        filtered_slice_s21 = filter_and_split_contours_with_flags_dis(slice_data_s21, s21_flag, s21_threshold)

        # Downsample lattice isovalues for intersection points selection
        reference_lattice2_isos = downsample_lattice_isos(all_lattice2_isos, lattice2_num)
        reference_lattice1_isos = downsample_lattice_isos(all_lattice1_isos, lattice1_num)

        # Downsample filtered segments
        down_subcontour_s1[slice_iso] = downsample_filtered_segments(
            filtered_slice_s12, reference_lattice1_isos, reference_lattice2_isos
        )
        down_subcontour_s2[slice_iso] = downsample_filtered_segments(
            filtered_slice_s21, reference_lattice2_isos, reference_lattice1_isos
        )

    # {slice: {lattice1: [arr1, arr2, ...]}}, arr is N by 3 as a segment
    return down_subcontour_s1, down_subcontour_s2

def downsample_filtered_segments(slice_data, reference_lattice1_isos, reference_lattice2_isos):
    """
    Downsample filtered segments for a given slice by both lattice1 and lattice2 isovalues.
    Always save the endpoints of each segment.
    """
    downsampled_slice = {}
    for lattice1_iso, subcontour_segments in slice_data.items():
        if lattice1_iso not in reference_lattice1_isos:
            continue  # Skip this lattice1_iso if it's not in the reference list

        downsampled_segments = []
        for segment in subcontour_segments:
            # Downsample the segment while preserving endpoints
            if len(segment) < 2:
                continue
           
            seg = [point[2] for point in segment]
            downsampled_segments.append(np.array(seg))

        downsampled_slice[lattice1_iso] = downsampled_segments

    return downsampled_slice

def downsample_lattice_isos(available_isos, target_num):
    """ Downsample lattice isovalues to target number. """
    if len(available_isos) <= target_num:
        return available_isos
    # Downsample available isovalues uniformly
    sampled_indices = np.linspace(0, len(available_isos) - 1, target_num, dtype=int)
    sampled_isos = [available_isos[i] for i in sampled_indices]
    return sampled_isos

def get_split_flags(inter_s12, sdf_decoder, split_sdf_level=0.0):
    """ 
    inter_s12: {slice_iso: {lattice1: {lattice2: 3d point}}}
    get split flags based on sdf values at interval centers
    returned flags is a dict {slice_iso: {lattice1: [segment_flag(list of bool)] ...]}}
    each segment_flag is already sorted along lattice2 isovalues, True inside, False outside (need split)
    """

    all_midpoints = []  # To store all midpoints for batch SDF querying
    segment_counts = {slice_iso: {lattice1: 0 for lattice1 in lattice1_data.keys()} for slice_iso, lattice1_data in inter_s12.items()}

    # collect all midpoints and track segment counts
    for slice_iso, lattice1_data in inter_s12.items():
        for lattice1, lattice2_data in lattice1_data.items():
            # Sort lattice2 keys (isovalues) to ensure proper order
            sorted_lattice2 = sorted(lattice2_data.keys())
            midpoints = []
            for i in range(len(sorted_lattice2) - 1):
                if not len(lattice2_data[sorted_lattice2[i]]) or not len(lattice2_data[sorted_lattice2[i + 1]]):
                    continue
                midpoints.append((lattice2_data[sorted_lattice2[i]] + lattice2_data[sorted_lattice2[i + 1]]) / 2.0)

            segment_counts[slice_iso][lattice1] = len(midpoints)
            all_midpoints.extend(midpoints)   

    #  Batch SDF computation
    all_midpoints = np.array(all_midpoints)
    all_midpoints = torch.from_numpy(all_midpoints).float().cuda() 
    sdf_decoder.eval() 
    with torch.no_grad():
        sdf_output = sdf_decoder(all_midpoints)
    sdf_values = sdf_output['model_out'].squeeze().detach().cpu().numpy() 

    #  Map SDF values back to split_flags structure
    split_flags = {}
    sdf_idx = 0 

    for slice_iso, lattice1_data in inter_s12.items():
        slice_flags = {}
        for lattice1 in lattice1_data.keys():
            # Get the number of segments for this lattice1
            num_segments = segment_counts[slice_iso][lattice1]
            # Extract the corresponding SDF values and compute flags
            segment_flags = sdf_values[sdf_idx:sdf_idx + num_segments] <= split_sdf_level  # True if inside, False if outside
            slice_flags[lattice1] = segment_flags.tolist()
            sdf_idx += num_segments  # Move the index forward

        split_flags[slice_iso] = slice_flags

    return split_flags


def get_global_distance_threshold(inter_s12, till_slice_layers=50, ratio=1.0):
    """iterate subcontours within one slice layer to get mean and std, then average them"""
    global_distances = []
    selected_keys = sorted(list(inter_s12.keys()))[:till_slice_layers]
    for slice_isovalue in selected_keys:
        slice_data = inter_s12[slice_isovalue]
        global_mean, global_std = compute_global_distance_statistics(slice_data)
        global_distances.append((global_mean, global_std))

    # Aggregate global statistics
    global_mean = np.mean([mean for mean, _ in global_distances])
    global_std = np.mean([std for _, std in global_distances])

    return global_mean + ratio * global_std

def compute_global_distance_statistics(slice_data):
    """ statistics (mean and std) of invterval distances across all subcontours within one slice layer.
        slice_data (dict): {lattice1_iso: {lattice2_iso: point}}
        tuple: (global_mean, global_std) of distances
    """
    all_distances = []
    
    for lattice1_iso, lattice2_data in slice_data.items():
        # Sort lattice2 isovalues and extract points
        sorted_lattice2 = sorted(lattice2_data.keys())
        points = [lattice2_data[lattice2_iso] for lattice2_iso in sorted_lattice2]
        
        if len(points) < 2:
            continue  # Skip if fewer than 2 points
        # Compute distances between consecutive points
        coordinates = np.array(points)
        distances = np.linalg.norm(np.diff(coordinates, axis=0), axis=1)
        all_distances.extend(distances)
    
    # Calculate global mean and standard deviation
    global_mean = np.mean(all_distances)
    global_std = np.std(all_distances)
    return global_mean, global_std

def filter_and_split_contours_with_flags_dis(slice_data, split_flags, threshold):
    """split segments using interval centers flags, ---- within one slice layer
    thus the num of split flags = num of points - 1 
    slice_data: {lattice1_iso: {lattice2_iso: point}}
    split_flags: {lattice1: [segment_flag(list of bool)] ...]}"""
    subcontours = reconstruct_subcontours(slice_data)  # {lattice1_iso: [tuple(lattice1_iso, lattice2_iso, point)]}

    # Filter and split the subcontours
    filtered_slice = {}
    for (lattice1_iso, subcontour_points), (_, subcontour_flag) in zip(subcontours.items(), split_flags.items()):
        # apply flag for splitting, with moving average smoothing
        filtered_slice[lattice1_iso] = filter_segments_by_sdf_flags_and_dis(subcontour_points, subcontour_flag, threshold, window_size=5)
    return filtered_slice


def filter_segments_by_sdf_flags_and_dis(subcontour_points, subcontour_flag, threshold, window_size=5):
    """
    Split a subcontour into segments based on interval center flags inside or outside sdf.
    Smooth each segment using the moving average method.
        subcontour_points (list): List of points, where each point is a tuple 
                                  (lattice1_iso, lattice2_iso, point).
        subcontour_flag (list): in/ out flags bool
        window_size (int): Window size for moving average smoothing (default: 3).
        
    Returns:
        list: A list of smoothed segments (each segment is a list of points).
    """
    if len(subcontour_points) < 2:
        return []
    
    coordinates = np.array([point[2] for point in subcontour_points])  # Extract the 3D coordinates
    distances = np.linalg.norm(np.diff(coordinates, axis=0), axis=1)  

    segments = []
    current_segment = [subcontour_points[0]]
    for i in range(1, len(subcontour_points)):
        if (not subcontour_flag[i - 1]) or (distances[i - 1] > threshold):
            # Smooth the current segment before appending
            if len(current_segment) < 1:
                continue

            if len(current_segment) > 3: # > 3 points, do smooth
                smoothed_segment = smooth_segment_with_moving_average(current_segment, window_size)
                segments.append(smoothed_segment)
            else:
                segments.append(current_segment) #too short, skip smoothing

            current_segment = []
        current_segment.append(subcontour_points[i])

    # Smooth the last non-empty segment
    if len(current_segment) > 1:
        if len(current_segment) > 3:
            smoothed_segment = smooth_segment_with_moving_average(current_segment, window_size)
            segments.append(smoothed_segment)
        else:
            segments.append(current_segment)

    return segments

def moving_average_smooth(points, window_size=3):
    """
    Smooth a set of 3D points using a moving average filter.
    padding by edge points, thus hold same size as input points
    """
    points = np.asarray(points)
    pad_size = (window_size - 1) // 2
    # Pad the data by replicating edge points
    padded_points = np.pad(points, ((pad_size, pad_size), (0, 0)), mode='edge')
    # Apply moving average along each coordinate axis
    smoothed_points = np.column_stack([
        np.convolve(padded_points[:, i], np.ones(window_size) / window_size, mode='valid')
        for i in range(3)  # Smooth x, y, z independently
    ])
    
    return smoothed_points

def smooth_segment_with_moving_average(segment, window_size):
    coordinates = np.array([point[2] for point in segment])
    smoothed_coordinates = moving_average_smooth(coordinates, window_size=window_size)
    smoothed_segment = [
        (segment[i][0], segment[i][1], smoothed_coordinates[i])
        for i in range(len(smoothed_coordinates))
    ]
    return smoothed_segment

def reconstruct_subcontours(slice_data):
    """
    Reconstruct subcontours from nested slice_data by connecting points sorted by lattice2 isovalues.
    Each point is stored as (lattice1_iso, lattice2_iso, point).
    Input:
        slice_data: {lattice1_iso: {lattice2_iso: point}}
    Output:
        subcontours: {lattice1_iso: [(lattice1_iso, lattice2_iso, point)]}
    """
    subcontours = {}
    for lattice1_iso, lattice2_data in slice_data.items():
        # Sort lattice2 isovalues and reconstruct the subcontour
        sorted_lattice2 = sorted(lattice2_data.keys())
        # with argumented lattice isovalues
        subcontour = [(lattice1_iso, lattice2_iso, lattice2_data[lattice2_iso]) for lattice2_iso in sorted_lattice2]
        subcontours[lattice1_iso] = subcontour
    return subcontours


def lattice_build_graph_partition_separate_pca_rect(
    contours_total, inter_s1, inter_s2,
    jump_threshold=0.1, loop_order=False, layer_loop_order=True, outer_to_inner=False, continuous_cnt_lim=256
):
    """
    Build lattice or volume infill with one-direction lattice at a layer, alternating directions for each layer.
    inter_s1: {slice: {lattice1: [arr1, arr2, ...]}}, arr is N by 3 as a segment. sgements order is along lattice2 isovalues
    still use outer partitioned contours to guide inner lattice,
    --------------
    when encontering >1 branches, use centers to get near segments from inter_s1 or inter_s2.
    use pca to get OBB for projected bounding box to filter infill segments.
    --------------
    loop_order: True -- odd layers reverse order    
    outer_to_inner: True -- the most outer to inner shell
    layer_loop_order: infill lattice 1 or 2 switching for layer by layer
    """
    start = time.time()
    # use level-0 contours to guide graph partition
    level0_print_contours, level0_print_isovalues, level0_print_branch, _ = \
        build_graph_partition(contours_total[0], jump_threshold, continuous_cnt_lim=continuous_cnt_lim)
    visualize_printing_contours(level0_print_contours)

    outer_printing_order = []
    inner_printing_order = []
    outer_contour_0_flags = []  # 0 means the most outer, 1 means other inner contours

    for iso_idx, (isovalue, isobranch) in enumerate(zip(level0_print_isovalues, level0_print_branch)):
        # ---- outer contours
        iso_printing_order = [] # [layer outer contours [subcontour1(arr), subcontour2], ...]
        iso_contour_flags = []
        level0_center = np.mean(level0_print_contours[iso_idx], axis=0)
        prj_bbx = get_projected_bounding_box(level0_print_contours[iso_idx], scale_factors=np.array([1.0, 1.0]))

        if len(contours_total) > 1:
            for level_idx, level_contours_dict in enumerate(reversed(contours_total[1:])):  # from inner to second outer
                level_contour = level_contours_dict.get(isovalue, [])
                if not level_contour:  # Skip empty subcontours
                    continue
                if len(level_contour) == 1:  # One contour, add directly
                    iso_printing_order.append(level_contour[0])
                    iso_contour_flags.append(1)
                else:  # >1 subcontours, find closest to level-0 center
                    closest_subcontour = find_closest_subcontour_infill(level_contour, level0_center)
                    iso_printing_order.append(closest_subcontour)
                    iso_contour_flags.append(1)

        # Add level-0 contour and reverse order if needed
        iso_printing_order.append(level0_print_contours[iso_idx])
        iso_contour_flags.append(0)

        if loop_order and iso_idx % 2 == 1:
            iso_printing_order.reverse()
            iso_contour_flags.reverse()
        
        if outer_to_inner:
            iso_printing_order.reverse()
            iso_contour_flags.reverse()

        outer_printing_order.append(iso_printing_order)
        outer_contour_0_flags.append(iso_contour_flags)  # not extend

        # ---- Lattice infill
        used_segments = set() # NOTE to avoid duplicate segments used in different branches, KEEP SAFE
        iso_inter = inter_s2.get(isovalue, {}) if (layer_loop_order and iso_idx % 2 == 1) else inter_s1.get(isovalue, {})
        if not iso_inter: # keep same len with outer
            inner_printing_order.append([])
            continue

        infill_contour = []  # [layer inner contours [subcontour1, seg1(arr), seg2], [subcontour2], ...]
        if isobranch == 1:  #  all lattice subcontours (their segments)
            for lattice1_iso, subcontour in sorted(iso_inter.items()):
                sub_list = []
                for segment in subcontour:
                    # Convert the segment to a tuple of tuples (hashable) to track it in the set
                    segment_tuple = tuple(map(tuple, segment))
                    # segment only has one point, skip
                    if segment.shape[0] == 1:
                        continue
                    if segment_tuple not in used_segments:  # Only add unused segments
                        sub_list.append(segment)
                        used_segments.add(segment_tuple)  # Mark segment as used
                if sub_list:
                    infill_contour.append(sub_list)
        else:
            for lattice1_iso, subcontour in sorted(iso_inter.items()):
                sub_list = []
                for segment in subcontour:
                    segment_tuple = tuple(map(tuple, segment))
                    # segment only has one point, skip
                    if segment.shape[0] == 1:
                        continue

                    # Compute the segment center
                    segment_center = np.mean(segment, axis=0)
                    endpoint1, endpoint2 = segment[0], segment[-1]                    
                    if segment_tuple not in used_segments and in_prj_bbx(prj_bbx, segment_center) and \
                        in_prj_bbx(prj_bbx, endpoint1) and in_prj_bbx(prj_bbx, endpoint2):

                        sub_list.append(segment)  # Add segment to sub_list
                        used_segments.add(segment_tuple)  # Mark segment as used
                if sub_list:
                    infill_contour.append(sub_list)

        inner_printing_order.append(infill_contour)

    print(f'printing order time: {time.time() - start}')
    print(f'Printing order contours, outer: {len(outer_printing_order)}; inner: {len(inner_printing_order)}')

    # outer_printing_order, all slice layers[ layer1[subcontour1 (arr), subcontour2], layer2[...], ...]
    # outer_contour_0_flags, all slice layers[ layer1[0, 1, 1], layer2[...], ...]     amount of 0 and 1 refs to how many subcontours at each layer
    # inner_printing_order, all slice layers[ layer1[subcontour1[seg1(arr), seg2] ], layer2[...], ...]
    return outer_printing_order, outer_contour_0_flags, inner_printing_order, level0_print_isovalues


def find_closest_subcontour_infill(subcontours, reference_center):
    """Find the subcontour closest to the reference center."""
    centers = [np.mean(subcontour, axis=0) for subcontour in subcontours]
    closest_idx = np.argmin([np.linalg.norm(reference_center - center) for center in centers])
    return subcontours[closest_idx]

def get_projected_bounding_box(points, scale_factors=np.array([1.0, 1.0])):
    """PCA-based bounding box in 2D projected plane, 
    with scaling factors applied to the bounding box size."""

    centroid = np.mean(points, axis=0)

    centered_points = points - centroid
    covariance_matrix = np.cov(centered_points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    plane_axes = eigenvectors[:, :2].T  
    projected_points = centered_points @ plane_axes.T  

    min_2d = np.min(projected_points, axis=0)
    max_2d = np.max(projected_points, axis=0)

    # Apply scaling factors to the bounding box
    center_2d = (min_2d + max_2d) / 2 
    half_range_2d = (max_2d - min_2d) / 2 
    scaled_half_range_2d = half_range_2d * scale_factors  
    min_2d_scaled = center_2d - scaled_half_range_2d 
    max_2d_scaled = center_2d + scaled_half_range_2d  

    return centroid, plane_axes, min_2d_scaled, max_2d_scaled

def in_prj_bbx(projected_bounding_box, point):
    """check if a 3d point is in the projected bounding box"""
    plane_point, plane_axes, min_2d_scaled, max_2d_scaled = projected_bounding_box
    local_point = point - plane_point  
    projected_point = local_point @ plane_axes.T 
    return np.all(min_2d_scaled <= projected_point) and np.all(projected_point <= max_2d_scaled)


def lattice_arrange_printing_contours_starts(outer_contours, inner_contours, isovalues):
    """
    outer_printing_order, all slice layers[ layer1[subcontour1 (arr), subcontour2], layer2[...], ...]
    inner_printing_order, all slice layers[ layer1[subcontour1[seg1(arr), seg2] ], layer2[...], ...]
    total length are same. equal to num of slice layers
    there may exist empty subcontours, skip them
    --------------------
    combine outer and inner contours at same layer to return a final list of subcontours (sublists)
    inner contours printed in ZIG-ZAG-shape order
    also return outer or inner indicators
    """

    printing_contours = []
    starts = []
    start_indices = []
    outer_inner_indicators = []
    level_tags = []

    # build printing contours levels
    level = 0
    contour_levels = []
    for i, isovalue in enumerate(isovalues):
        if i>0 and isovalue < isovalues[i-1]:
            level += 1
        contour_levels.append(level)
    print(f'Number of levels: {level + 1}')


    # Iterate over each layer
    for outer_layer, inner_layer, con_level in zip(outer_contours, inner_contours, contour_levels):
        # Skip empty layers
        if not outer_layer and not inner_layer:
            continue

        layer_printing_order = []  
        layer_starts = []  
        layer_start_indices = []  
        layer_indicators = []  
        level_labels = []  

        # Determine the starting point for the first outer subcontour
        if inner_layer and inner_layer[0]:  # Check if inner_layer has segments
            reference_point = np.array(inner_layer[0][0][0]) # First point of the first inner segment
        else:
            reference_point = np.array(outer_layer[0][0])    # First point of the first outer subcontour

        # Process outer contours
        if outer_layer:
            # Find the first outer subcontour and the exact nearest point within it
            first_outer_index, first_outer_start_index = min(
                (
                    (i, j)  # i = subcontour index, j = point index within the subcontour
                    for i, outer_subcontour in enumerate(outer_layer)
                    for j, point in enumerate(outer_subcontour)
                ),
                key=lambda pair: np.linalg.norm(np.array(outer_layer[pair[0]][pair[1]]) - reference_point)
            )
            first_outer = outer_layer.pop(first_outer_index)
            first_outer_start = first_outer[first_outer_start_index]

            # Add the first outer subcontour to the printing order
            layer_printing_order.append(first_outer)
            layer_starts.append(first_outer_start)
            layer_start_indices.append(first_outer_start_index)
            layer_indicators.append(0)
            level_labels.append(con_level)

            # Process the remaining outer subcontours based on proximity to the previous subcontour's start point
            current_start = first_outer_start
            while outer_layer:
                nearest_index, nearest_start_index = min(
                    (
                        (i, j)  # i = subcontour index, j = point index within the subcontour
                        for i, outer_subcontour in enumerate(outer_layer)
                        for j, point in enumerate(outer_subcontour)
                    ),
                    key=lambda pair: np.linalg.norm(np.array(outer_layer[pair[0]][pair[1]]) - np.array(current_start))
                )
                nearest_subcontour = outer_layer.pop(nearest_index)
                current_start = nearest_subcontour[nearest_start_index]

                # Add the nearest subcontour to the printing order
                layer_printing_order.append(nearest_subcontour)
                layer_starts.append(current_start)
                layer_start_indices.append(nearest_start_index)
                layer_indicators.append(0)
                level_labels.append(con_level)

        # Process inner contours in Z-shape order (treat each segment individually)
        if inner_layer:
            for i, inner_subcontour in enumerate(inner_layer):
                if not inner_subcontour:
                    continue

                # Handle Z-shape order for segments. 
                # Typically, inner_subcontour only has one segment. 
                # BUT, broken subsegments are possible in one inner_subcontour
                if i % 2 == 0:  # Even index: process segments in order
                    for segment_index, segment in enumerate(inner_subcontour):
                        if len(segment) < 2:  # Skip segments with less than 2 points
                            continue
                        start_point = segment[0]  
                        layer_printing_order.append(segment)
                        layer_starts.append(start_point)
                        layer_start_indices.append(0)  
                        layer_indicators.append(1)
                        level_labels.append(con_level)
                else:  # Odd index: process segments in reverse order
                    for segment_index, segment in enumerate(reversed(inner_subcontour)):
                        if len(segment) < 2:
                            continue
                        reversed_segment = segment[::-1]  # Reverse the points within the segment
                        start_point = reversed_segment[0]  # First point of the reversed segment
                        layer_printing_order.append(reversed_segment)
                        layer_starts.append(start_point)
                        layer_start_indices.append(0)  # First point index
                        layer_indicators.append(1)
                        level_labels.append(con_level)

        # Append the layer results to the final results
        printing_contours.extend(layer_printing_order)
        starts.extend(layer_starts)
        start_indices.extend(layer_start_indices)
        outer_inner_indicators.extend(layer_indicators)
        level_tags.extend(level_labels)

    print(f'Total length of printing order: {len(printing_contours)}')
    # printing_contours: list[contour1 (arr), contour2, ...]
    return printing_contours, starts, start_indices, outer_inner_indicators, level_tags


def build_print_level_dataset(outer_contours, inner_contours, isovalues, output_fn):
    """use both outer and inner contours (final array of 3d points) to build print level dataset across sdf inside space
    jumps in isovlaues indicate next level, 
    final datapair is (slice isovalue, xyz) --- level label for classification"""
    
    dataset = []  # To store datapairs: (slice_isovalue, xyz) --> level label
    level = 0  # Start with level 0
    contour_levels = []  # To track the level for each slice

    # an isovalue jump indicates next level
    for i, isovalue in enumerate(isovalues):
        if i > 0 and isovalue < isovalues[i - 1]:
            level += 1 
        contour_levels.append(level)

    print(f"Number of levels: {level + 1}")

    # add outer and inner contours points to dataset
    for layer_idx, (outer_layer, inner_layer, level_label) in enumerate(zip(outer_contours, inner_contours, contour_levels)):
        for subcontour in outer_layer:
            for point in subcontour: 
                if len(point) == 3:
                    dataset.append((isovalues[layer_idx], point.tolist(), level_label))

        for subcontour in inner_layer:  # multiple segments
            for segment in subcontour:
                for point in segment: 
                    if len(point) == 3: # there may exist empty points
                        dataset.append((isovalues[layer_idx], point.tolist(), level_label))

    # Each element in the dataset is (slice_isovalue, xyz, level_label)
    dataset_dict = {
        "xyz": torch.tensor([entry[1] for entry in dataset], dtype=torch.float32),
        "slice_isovalues": torch.tensor([entry[0] for entry in dataset], dtype=torch.float32),
        "level_labels": torch.tensor([entry[2] for entry in dataset], dtype=torch.long)
    }
    torch.save(dataset_dict, output_fn)
    print(f"{dataset_dict['xyz'].shape[0]} points in total, Dataset saved to {output_fn}")


def build_calibration_thickness_wall(level_contours, sdf_decoder, slice_decoder, object_scale=100, 
                                     fit_down_ratio=200, dist_threshold=60, jump_threshold=0.5, continuous_cnt_lim=256):
    """
    similar to single shell fitting for each level contours.
    first partition contours at different levels. For each, get their distances and grad norms.
    Then merge them together for fitting.
    level_contours: list[dict1{slice_iso: [contour1(arr), contour2(arr), ...]}, dict2{...}, ]
    """

    distances = []
    grad_lens = []
    # partition each level contours just for continuous contours. this partition means nothing
    # so that collect distances and project grad norms
    for level_dict in level_contours:
        # centers = compute_subcontour_centers(level_dict)
        print_contours, _, _, _= build_graph_partition(level_dict, jump_threshold, continuous_cnt_lim)
        distance, grad_len, _ = get_one_level_distances(print_contours, sdf_decoder, slice_decoder, object_scale) 
        distances.extend(distance)
        grad_lens.extend(grad_len)


    distances = np.array(distances)
    grad_lens = np.array(grad_lens)

    # everytime before fitting, check distances distribution
    plt.hist(distances, bins=200, range=(0, 3.0))
    plt.show()

    # filter outliers manually, norm < 0.4 are singularities
    mask = (distances < dist_threshold)
    distances = distances[mask]
    grad_lens = grad_lens[mask]

    # fitting
    fit_function = fitting(grad_lens, distances, down_ratio=fit_down_ratio * len(level_contours))
    print(f"Calibration function: {fit_function}")
    # vis fitting results with original data
    vis_fitting_results(grad_lens, distances, fit_function, down_ratio=1)

    return fit_function


def uniform_resampling_inner_contours(contours, spacing):
    """uniform resamping for inner contours with fixed spacing from uniform-space outer contours.    
    contours: all slice layers[ layer1[subcontour1[seg1(arr), seg2] ], layer2[...], ...]"""

    def resample_segment(segment, spacing):
        """ Resample a single curved segment with uniform spacing. """
        # Compute the cumulative distance along the curve
        diff = np.diff(segment, axis=0) 
        dist = np.linalg.norm(diff, axis=1)
        cumulative_dist = np.concatenate(([0], np.cumsum(dist))) 

        # Generate new resampled distances
        new_distances = np.arange(0, cumulative_dist[-1], spacing)
        if cumulative_dist[-1] not in new_distances:
            new_distances = np.append(new_distances, cumulative_dist[-1])

        # Interpolate to find new points for each coordinate (x, y, z)
        resampled_points = np.empty((len(new_distances), segment.shape[1]))
        for i in range(segment.shape[1]): 
            resampled_points[:, i] = np.interp(new_distances, cumulative_dist, segment[:, i])
        return resampled_points

    # Resample all segments in all contours
    resampled_contours = []
    for layer in contours:
        resampled_layer = []
        for subcontour in layer:
            resampled_subcontour = []
            for segment in subcontour:
                resampled_segment = resample_segment(segment, spacing)
                resampled_subcontour.append(resampled_segment)
            resampled_layer.append(resampled_subcontour)
        resampled_contours.append(resampled_layer)

    return resampled_contours


def generate_waypoints_lattice(printing_order_contours, start_indices, indicators, z_rm_th, printing_levels):
    """
    based on start_indices, generate waypoints for printing
    the mess base should be removed, thus affecting printing order
    all waypoints z below z_rm_th will be removed, requiring rearrangement of printing order

    printing_order_contours: list of contours, each is an arr( x 3) of outer/inner contours
    """
    ordered_waypoints = []
    ordered_indicators = []
    level_tags = []

    # arrange waypoints of both outer shell and inner infill
    for contour, start_index, indicator, level in zip(printing_order_contours, start_indices, indicators, printing_levels):
        if not 0 <= start_index < len(contour):
            raise ValueError(f"Start index {start_index} out of range for contour with length {len(contour)}")
        
        # Reorder the contour starting from the specified start index
        if indicator == 0:   # outer contour
            if start_index == len(contour) - 1:
                start_index = 0   # outer contour start and end are same
            contour = contour[:-1]  # remove last repeated point
            ordered_contour = list(contour[start_index:]) + list(contour[:start_index]) + [contour[start_index]]

            order_contour = np.array(ordered_contour)
            if len(order_contour) < 2: # only one point
                continue
            # skip if all z below threshold
            if (np.all(order_contour[:, 2] < z_rm_th)):
                continue
            # directly keep if all z above threshold
            if np.all(order_contour[:, 2] >= z_rm_th):
                ordered_waypoints.extend(list(order_contour)) 
                ordered_indicators.extend([indicator] * len(order_contour))
                level_tags.extend([level] * len(order_contour))            
                continue

            # split into segments, and connect them directly, 
            # Jumps may insert based on intervals
            above_th = order_contour[:, 2] >= z_rm_th
            segments = []
            current_segment = []
            for i, is_above in enumerate(above_th):
                if is_above:
                    current_segment.append(order_contour[i])
                elif current_segment:
                    # Finish the current segment and start a new one
                    segments.append(np.array(current_segment))
                    current_segment = []
            if current_segment: # add last segment if exists
                segments.append(np.array(current_segment))
            if not segments:
                continue
            # Concatenate all segments into a single continuous segment
            connected_segment = np.concatenate(segments, axis=0)
            ordered_waypoints.extend(list(connected_segment))
            ordered_indicators.extend([indicator] * len(connected_segment))
            level_tags.extend([level] * len(connected_segment))

        else: # inner contour typically from endpoint, segment has been reversed
              # direct discard inner contour waypoints whose z below threshold
            contour = contour[contour[:, 2] >= z_rm_th]
            if len(contour) < 2: # only one point
                continue
            ordered_contour = list(contour)
            ordered_waypoints.extend(ordered_contour)
            ordered_indicators.extend([indicator] * len(ordered_contour))
            level_tags.extend([level] * len(ordered_contour))

    print(f'number of waypoints after removing mess base: {len(ordered_waypoints)}')
    return ordered_waypoints, ordered_indicators, level_tags




