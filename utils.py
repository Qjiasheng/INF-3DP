import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import random
import yaml
import time
import diff_operators
import skimage.measure
import torch.nn.functional as F
import trimesh

def set_seed(seed):
    # Set the seed for PyTorch (both CPU and CUDA)
    torch.manual_seed(seed)
    # If you are using CUDA, make the GPU deterministic
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If using multi-GPU
        # Ensure that CUDA operations are deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set the seed for numpy (if you're using it)
    np.random.seed(seed)
    # Set the seed for Python's built-in random module
    random.seed(seed)
    print(f"Seed set to {seed}")

def load_configs(config_path, exp_name):
    # load training parameters
    with open(config_path, 'r') as f:
        configs = yaml.safe_load(f)
    # dataset, optim, loss parameters
    # return configs[exp_name]['dataset'], configs[exp_name]['optim'], configs[exp_name]['parameters']
    return configs[exp_name]

def put_configs(config_path, paras):
    with open(config_path, 'w') as f:
        yaml.dump(paras, f)
    print(f"Configs saved to {config_path}")

def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords


def lin2img(tensor, image_resolution=None):
    batch_size, num_samples, channels = tensor.shape
    if image_resolution is None:
        width = np.sqrt(num_samples).astype(int)
        height = width
    else:
        height = image_resolution[0]
        width = image_resolution[1]

    return tensor.permute(0, 2, 1).view(batch_size, channels, height, width)


def make_scatter_plot(array_2d):
    Y, X = np.meshgrid(np.arange(array_2d.shape[0]), np.arange(array_2d.shape[1]))
    X_flat, Y_flat = X.flatten(), Y.flatten()
    intensities = array_2d.flatten()
    fig, ax = plt.subplots(figsize=(2.75, 2.75), dpi=300)
    scatter = ax.scatter(X_flat, Y_flat, c=intensities, cmap='Spectral')
    contour = ax.contour(X, Y, array_2d, levels=np.arange(-2.0, 0, 0.05), colors='b', linewidths=0.4)
    contour = ax.contour(X, Y, array_2d, levels=np.arange(0, 2.0, 0.05), colors='r', linewidths=0.4)
    contour = ax.contour(X, Y, array_2d, levels=[0], colors='k', linewidths=1)
    cbar = fig.colorbar(scatter, ax=ax, label='SDF Value')
    return fig


def write_sdf_summary(model, model_input, gt, model_output, writer, total_steps, prefix='train_'):
    slice_coords_2d = get_mgrid(512)

    with torch.no_grad():
        x_cross, y_cross, z_cross = 0.0, 0.0, 0.0 
        yz_slice_coords = torch.cat((torch.ones_like(slice_coords_2d[:, :1]) * x_cross, slice_coords_2d), dim=-1)
        yz_slice_model_input = {'coords': yz_slice_coords.cuda()[None, ...]}

        yz_model_out = model(yz_slice_model_input)
        sdf_values = yz_model_out['model_out']
        sdf_values = lin2img(sdf_values).squeeze().cpu().numpy()
        # fig = make_contour_plot(sdf_values)
        fig = make_scatter_plot(sdf_values)
        writer.add_figure(prefix + 'yz_sdf_slice', fig, global_step=total_steps)

        xz_slice_coords = torch.cat((slice_coords_2d[:,:1],
                                     torch.ones_like(slice_coords_2d[:, :1]) * y_cross,
                                     slice_coords_2d[:,-1:]), dim=-1)
        xz_slice_model_input = {'coords': xz_slice_coords.cuda()[None, ...]}

        xz_model_out = model(xz_slice_model_input)
        sdf_values = xz_model_out['model_out']
        sdf_values = lin2img(sdf_values).squeeze().cpu().numpy()
        # fig = make_contour_plot(sdf_values)
        fig = make_scatter_plot(sdf_values)
        writer.add_figure(prefix + 'xz_sdf_slice', fig, global_step=total_steps)

        xy_slice_coords = torch.cat((slice_coords_2d[:,:2],
                                     z_cross*torch.ones_like(slice_coords_2d[:, :1])), dim=-1)
        xy_slice_model_input = {'coords': xy_slice_coords.cuda()[None, ...]}

        xy_model_out = model(xy_slice_model_input)
        sdf_values = xy_model_out['model_out']
        sdf_values = lin2img(sdf_values).squeeze().cpu().numpy()
        # fig = make_contour_plot(sdf_values)
        fig = make_scatter_plot(sdf_values)
        writer.add_figure(prefix + 'xy_sdf_slice', fig, global_step=total_steps)


def write_collision_summary(model, model_input, gt, model_output, writer, total_steps, prefix='train_', out_index = 0):
    pass



def field_querying(decoder, samples, max_batch=32 ** 3, no_print=False, return_on_cuda=False, set_grad_flase=True):
    """querying field with given samples
    samples: N x 3 torch tensor
    return fields: N x 1 torch tensor"""

    start = time.time()
    decoder.eval()

    if not isinstance(samples, torch.Tensor):
        samples = torch.from_numpy(samples).float()
    else:
        samples = samples.float()
    
    num_samples = samples.shape[0]
    fields = torch.zeros(num_samples)

    if set_grad_flase:
        samples.requires_grad = False

    head = 0
    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples), :].cuda()

        sub_output = decoder(sample_subset)
        sub_coords = sub_output['model_in']
        sub_fields = sub_output['model_out']

        fields[head : min(head + max_batch, num_samples)] = sub_fields.squeeze().detach().cpu()
        head += max_batch

    end = time.time()
    if not no_print:
        print("field querying takes: %f" % (end - start))

    if return_on_cuda:
        return fields.cuda()
    return fields

def quat_field_querying(decoder, samples, max_batch=32 ** 3, no_print=False, return_on_cuda=False, set_grad_flase=True):
    """querying field with given samples
    samples: N x 3 torch tensor
    return fields: N x 4 torch tensor"""

    start = time.time()
    decoder.eval()

    if not isinstance(samples, torch.Tensor):
        samples = torch.from_numpy(samples).float()
    else:
        samples = samples.float()
    
    num_samples = samples.shape[0]
    fields = torch.zeros(num_samples, 4)

    if set_grad_flase:
        samples.requires_grad = False

    head = 0
    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples), :].cuda()

        sub_output = decoder(sample_subset)
        sub_coords = sub_output['model_in']
        sub_fields = sub_output['model_out']

        fields[head : min(head + max_batch, num_samples), :] = sub_fields.squeeze().detach().cpu()
        head += max_batch

    end = time.time()
    if not no_print:
        print("field querying takes: %f" % (end - start))

    if return_on_cuda:
        return fields.cuda()
    return fields


def field_query_with_grads(decoder, coords, max_batch=32 ** 3, no_print=False, return_on_cuda=False):
    """seperate func to query field and grads"""

    start = time.time()
    # check coords on cuda tensor
    if not isinstance(coords, torch.Tensor):
        samples = torch.from_numpy(coords).float()
    else:
        samples = coords.float()
    samples.requires_grad = True

    decoder.eval()

    head = 0
    num_samples = samples.shape[0]
    fields = torch.zeros(num_samples)
    grads = torch.zeros(num_samples, 3)

    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples), :].cuda()
        
        with torch.enable_grad():   # must enable for custom autograd function
            sub_output = decoder(sample_subset)
            sub_coords = sub_output['model_in']
            sub_fields = sub_output['model_out']

            fields[head : min(head + max_batch, num_samples)] = (
                sub_fields
                .squeeze()
                .detach()
            )

            grads[head : min(head + max_batch, num_samples)] = (
                diff_operators.gradient(sub_fields, sub_coords) 
                .squeeze()
                .detach()
            )

        head += max_batch

    if not no_print:
        print('Elapsed time: ', time.time() - start)

    if return_on_cuda:
        return samples.cuda(), fields.cuda(), grads.cuda()

    samples = samples.detach().cpu()
    fields = fields.detach().cpu()
    grads = grads.detach().cpu()
    return samples, fields, grads


def level_querying_with_confidence(decoder, samples, num_classes, max_batch=32 ** 3, 
                                   no_print=False, return_on_cuda=False, set_grad_flase=True):
    """
    querying field with given samples
    samples: N x 4 torch tensor, [x, y, z, slice_isovalue]
    return labels: N x 1 torch tensor
    return confidence: N x 1 torch tensor (softmax probability)
    """

    start = time.time()
    decoder.eval()
    
    num_samples = samples.shape[0]
    levels = torch.zeros(num_samples, num_classes)
    if set_grad_flase:
        samples.requires_grad = False 
    head = 0

    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples), :].cuda()
        sub_levels = decoder(sample_subset)

        levels[head : min(head + max_batch, num_samples), :] = sub_levels.squeeze().detach().cpu()
        head += max_batch

    probs = F.softmax(levels, dim=-1)  # Convert logits to probabilities
    confidence, predicted_label = torch.max(probs, dim=-1)

    end = time.time()
    if not no_print:
        print("level querying takes: %f" % (end - start))
    if return_on_cuda:
        return predicted_label.cuda(), confidence.cuda()

    return predicted_label, confidence



def gen_iso_layers(fields, num_iso_layers, mask, N=128):
    """generate iso-layers from volume fields, mask is to get only interior layers"""

    masked_fields = torch.where(mask, fields, torch.tensor(float('nan')))
    valid_values = masked_fields[~torch.isnan(masked_fields)]  # Extract only valid (non-NaN) values
    min_iso_value = valid_values.min().item()
    max_iso_value = valid_values.max().item()

    # Generate a range of iso-values evenly 
    iso_values = np.linspace(min_iso_value, max_iso_value, num_iso_layers)

    # extract iso-layers
    iso_layers = {}
    for iso_value in iso_values:
        try:
            verts, faces, _, _ = skimage.measure.marching_cubes(
                masked_fields.numpy(), level=iso_value, spacing=(2.0 / N, 2.0 / N, 2.0 / N) # follow our voxel size
            )
            iso_layers[iso_value] = (verts, faces)  
            print(f"Extracted iso-layer for iso-value: {iso_value:.6f}")
        except Exception as e:
            print(f"Failed to extract iso-layer for iso-value: {iso_value:.6f}, Error: {e}")

    return iso_layers

def general_text_writer(data, fmt, filename, chunk=None):
    """Write data to a text file, optionally in chunks."""

    base, ext = os.path.splitext(filename)
    if chunk is None:
        output_filename = f"{filename}"
        np.savetxt(output_filename, data, fmt=fmt, delimiter=" ", comments="")
        print(f"{data.shape[0]} lines written to {output_filename}")
    else:
        num_points = data.shape[0]
        num_chunks = (num_points + chunk - 1) // chunk  # Ceiling division 

        for i in range(num_chunks):
            start_idx = i * chunk
            end_idx = min(start_idx + chunk, num_points)
            chunk_data = data[start_idx:end_idx]

            # Write the current chunk to a file with an appropriate suffix.
            output_filename = f"{base}_{i}{ext}"
            np.savetxt(output_filename, chunk_data, fmt=fmt, delimiter=" ", comments="")
            print(f"{chunk_data.shape[0]} lines written to {output_filename}")



def load_waypoints(waypt_path):
    """load waypoints from already saved .xyz file.
    typical fmt is -- xyz, dirs, layer_thickness, levels"""
    pcd = np.genfromtxt(waypt_path)
    waypoints = pcd[:, :3]
    directions = pcd[:, 3:6]
    layer_thickness = pcd[:, 6]
    levels = pcd[:, 7].astype(int)
    print(f'Loaded {waypoints.shape[0]} waypoints from {waypt_path}')

    return waypoints, directions, layer_thickness, levels

def load_nozzle_pcd(nozzle_pcd_path, object_scale=1.0, down_ratio=1):
    """load nozzle pcd (physical scale in mm) with origin at tooltip.
    scale into unit cube based on object scale.
    ---------------------
    you may prepare ur nozzle pcd with higher density of tip part
    """
    pcd = np.genfromtxt(nozzle_pcd_path)
    pcd /= object_scale  # scale into unit cube
    print(f'Loaded {pcd.shape[0]} nozzle points from {nozzle_pcd_path}')

    if down_ratio > 1:
        down_ratio = int(down_ratio)
        pcd = pcd[::down_ratio, :]
        print(f'Downsampled nozzle pcd by ratio {down_ratio}, new num points: {pcd.shape[0]}')

    return pcd

def load_nozzle_mesh(nozzle_path, object_scale):
    mesh = trimesh.load(nozzle_path)
    # mesh.apply_scale(10.0 / object_scale)  # cm
    mesh.apply_scale(1.0 / object_scale) # mm
    return mesh

def build_level_slice_dict(slice_fields, levels):
    """build level slice dict {'0': max0_slice_value, '1': max1_slice_value, ...}
    from all waypts slice fields and levels
    -------
    find top slice values for each level. higher level may have smaller slice value.
    but monotonic increasing along one level.
    """
    unique_levels = torch.unique(levels)
    level_slice_dict = {}
    for level in unique_levels:
        mask = (levels == level).squeeze()
        level_slice = slice_fields[mask]
        max_slice = torch.max(level_slice)
        level_slice_dict[level.item()] = max_slice.item()
    
    return level_slice_dict


def build_current_level_slice(level_slice_dict, current_level, current_slice):
    """for levels below current level, use max slice in current_max_slices,
    replace current level's slice value for its level.
    only <= current_level is valid for TVSDF construction.
     keep regular data shape """
    
    current_max_slices = torch.zeros((current_level.shape[0], len(level_slice_dict)), device=current_level.device)
    for level, max_slice in level_slice_dict.items():
        max_slice_tensor = torch.tensor(max_slice, device=current_slice.device)
        current_max_slices[:, int(level)] = max_slice_tensor  # init as max slice for each level
        mask = (current_level == level)
        max_slice_tmp = torch.min(current_slice, max_slice_tensor)
        current_max_slices[mask, int(level)] = max_slice_tmp[mask]

    return current_max_slices


def batch_query_tvsdf(queries, sdf_decoder, slice_decoder, level_decoder, num_classes, current_level_maxslices, current_levels):
    """
    tvsdf values for queries with batched waypts and their current levels.
    --------
    queries: (N, M, 3), N is batched waypts num, M is num of query points per waypt (driven by quaternion at different waypts)
    !!! one batched queries in corresponding to one waypt. But can stack. 
    --------
    current_level_maxslices: (N, num_levels), max slice values for each level. 
    N is batched waypts num.
    current_levels: (N,), current level for waypts.
    !! only <= current_level is valid for TVSDF construction.

    all set on cuda tensors
    """
    N, M, _ = queries.shape                        # batch_size, num_queries
    queries_flat = queries.view(-1, 3)             # (N*M, 3)
    coords = queries_flat.clone()

    # collect all fields
    original_sdfs = field_querying(sdf_decoder, coords, no_print=True, return_on_cuda=True) 
    samples, slice_fields, slice_grads = field_query_with_grads(slice_decoder, coords, no_print=True, return_on_cuda=True)
    level_inputs = torch.cat((samples.detach(), slice_fields.unsqueeze(-1)), dim=1) # (x, y, z, isovalue)
    coord_levels, _ = level_querying_with_confidence(level_decoder, level_inputs, num_classes, no_print=True, return_on_cuda=True)

    # build tvsdf
    # sdf interpolation are different for each level
    level_tvsdf_dict = {}  # {level: tvsdf_values (N* M, ), ...}
    level_maxslices_dict = {}  # {level: max_slice_value (N*M, ), ...}
    max_level = torch.max(current_levels).item()  # max level in batched all waypts
    waypt_levels = current_levels.unsqueeze(-1).expand(-1, M).reshape(-1)  # (N*M, ) batch waypt max level
    
    # traverse all levels to build tvsdf at each level. then merge them as final.
    for level in range(max_level + 1):
        tvsdf_temp = original_sdfs.clone() # init sdf values
        level_mask  = (coord_levels == level)  # (N*M, )

        if level_mask.sum() == 0:
            # no points predicted at this level
            level_tvsdf_dict[level] = tvsdf_temp
            continue

        # get max slice value at this level (top surface)
        max_slices = current_level_maxslices[:, level]  # (N,
        max_slices_expanded = max_slices.unsqueeze(-1).expand(-1, M).reshape(-1)  # (N*M, )

        # for push check use waypt level's max slice value
        waypt_max_slices = current_level_maxslices.gather(1, current_levels.unsqueeze(-1))  # (N, 1)
        waypt_max_slices_expanded = waypt_max_slices.expand(-1, M).reshape(-1)  # (N*M, )

        level_maxslices_dict[level] = max_slices_expanded

        # first mask to get current level things. less number for further query
        queries_level = queries_flat[level_mask]          
        original_sdfs_level = original_sdfs[level_mask]
        slice_fields_level = slice_fields[level_mask]
        slice_grads_level = slice_grads[level_mask]
        max_slices_level = max_slices_expanded[level_mask]
        waypt_max_slices_level = waypt_max_slices_expanded[level_mask]
        waypt_levels_level = waypt_levels[level_mask]
        coord_levels_level = coord_levels[level_mask]

        # field to distance. Eq. 27
        slice_dists_level = (slice_fields_level - max_slices_level) / torch.norm(slice_grads_level, dim=-1)
        pseudo_sdfs_level = torch.max(original_sdfs_level, slice_dists_level)

        # current level recheck, may be already covered by higher level
        # need recheck to use pseudo_sdf or original_sdf. 
        recheck_mask  = (slice_dists_level > original_sdfs_level) & (original_sdfs_level < 0.0)  # only inside regions
        if recheck_mask.sum() == 0:
            # no need recheck
            tvsdf_temp[level_mask] = pseudo_sdfs_level
            level_tvsdf_dict[level] = tvsdf_temp
            continue

        # recheck points (further mask)
        queries_recheck = queries_level[recheck_mask]
        original_sdfs_recheck = original_sdfs_level[recheck_mask]
        slice_fields_recheck = slice_fields_level[recheck_mask]
        slice_grads_recheck = slice_grads_level[recheck_mask]
        max_slices_recheck = max_slices_level[recheck_mask]
        waypt_max_slices_recheck = waypt_max_slices_level[recheck_mask]
        slice_dists_recheck = slice_dists_level[recheck_mask]
        waypt_levels_recheck = waypt_levels_level[recheck_mask]
        coord_levels_recheck = coord_levels_level[recheck_mask]

        # push queries along slice_grad direction a little bit. line 15 in Alg. 1 
        # here set two steps push. first with 1.0 to hit level top surface, then further 0.5 to outside !use new slice grad
        # since the slice field is regular (Eq. 4) of interior, not very different
        # NOTE errors mainly from level classification (bw adjacent levels), and second push step distance
        # first hit current level's top surface
        push_queries_recheck_step1 = queries_recheck + torch.abs(slice_dists_recheck).unsqueeze(-1) * F.normalize(slice_grads_recheck.detach(), dim=-1) * 1.0
        push_queries_recheck_step1 = torch.clamp(push_queries_recheck_step1, -1.0, 1.0) 
        # small step second push 
        _, _, push_slice_grads_step1 = field_query_with_grads(slice_decoder, push_queries_recheck_step1, no_print=True, return_on_cuda=True)
        push_queries_recheck = push_queries_recheck_step1 + (torch.max(torch.abs(slice_dists_recheck).unsqueeze(-1) * 0.5,  # at least push 0.04
                                                                       torch.tensor(0.04).cuda()) * F.normalize(push_slice_grads_step1.detach(), dim=-1))  
        push_queries_recheck = torch.clamp(push_queries_recheck, -1.0, 1.0)  

        # query levels again 
        push_slice_fields = field_querying(slice_decoder, push_queries_recheck.detach(), no_print=True, return_on_cuda=True)
        push_level_inputs = torch.cat((push_queries_recheck.detach(), push_slice_fields.unsqueeze(-1)), dim=1)
        push_queries_levels, _ = level_querying_with_confidence(level_decoder, push_level_inputs, num_classes, no_print=True, return_on_cuda=True)

        # decide final tvsdf values
        # condition of not use new sdf: (1), (2) -- pushed regions are alreay printed. not real exploded top surface
        alter_mask = ((push_queries_levels < waypt_levels_recheck) |  # smaller level after push
                      ((push_slice_fields < waypt_max_slices_recheck) & (push_queries_levels == waypt_levels_recheck)))  # same level but not outside top surface
        
        if alter_mask.sum() > 0:
            print(f"Level {level}: {alter_mask.sum().item()} points recheck altered tvsdf values.")
            # indices that really have to revert to original_sdf
            final_alter_mask = torch.zeros_like(recheck_mask)
            final_alter_mask[recheck_mask] = alter_mask        # combine the two masks
            pseudo_sdfs_level[final_alter_mask] = original_sdfs_level[final_alter_mask]

        tvsdf_temp[level_mask] = pseudo_sdfs_level
        level_tvsdf_dict[level] = tvsdf_temp

    # merge all level tvsdf as final tvsdf
    sorted_levels = sorted(level_tvsdf_dict.keys(), reverse=True) 
    tvsdf = level_tvsdf_dict[sorted_levels[0]].clone()  # init as highest level tvsdf
    coords_maxslices = level_maxslices_dict[sorted_levels[0]].clone() 
    for level in sorted_levels[1:]:
        level_tvsdf = level_tvsdf_dict[level]
        tvsdf = torch.where(coord_levels <= level, level_tvsdf, tvsdf)

        level_max_slices = level_maxslices_dict[level]
        coords_maxslices = torch.where(coord_levels <= level, level_max_slices, coords_maxslices)

    # valid mask (N*M,)
    valid_mask = ((original_sdfs < 0.0) &
                  ((coord_levels < waypt_levels) |
                   ((coord_levels == waypt_levels) & (slice_fields < coords_maxslices))))
    
    tvsdf = tvsdf.view(N, M)
    coords_maxslices = coords_maxslices.view(N, M)
    valid_mask = valid_mask.view(N, M)

    return tvsdf, coords_maxslices, valid_mask


def transform_nozzle_pcd_with_frame(nozzle_pcd, waypts, print_dirs, frame_dirs):
    """"
    transform nozzle pcd at each waypoint given frames, by converting quaternions.
    print_dirs ---- nozzle z+
    frame_dirs ---- nozzle x+
    """
    # Create reference vectors of shape (M, 3) as nozzle axis
    x_ref = torch.tensor([1., 0., 0.], device=frame_dirs.device).expand_as(frame_dirs) 
    z_ref = torch.tensor([0., 0., 1.], device=print_dirs.device).expand_as(print_dirs) 

    q_align_x = compute_alignment_quaternion(x_ref, frame_dirs)  
    aligned_z_dirs = quaternion_rotate(q_align_x, z_ref)  
    q_align_z = compute_alignment_quaternion(aligned_z_dirs, print_dirs)  
    q_final = F.normalize(quaternion_multiply(q_align_z, q_align_x), dim=-1) 

    transed_nozzle_pcd = quaternion_rotate(q_final[:, None, :], nozzle_pcd[None, :, :]) 
    transed_nozzle_pcd += waypts[:, None, :]  # (M, N, 3)

    return transed_nozzle_pcd

def quaternion_rotate(q, v):
    """Rotate vector v using quaternion q (supports broadcasting)."""
    q_vec = q[..., 1:]  # Extract vector part (x, y, z) -> shape (..., 3)
    q_scalar = q[..., :1]  # Extract scalar part (w) -> shape (..., 1)
    # Ensure q and v have the same batch dimensions
    t = 2 * torch.cross(q_vec, v, dim=-1)
    v_rot = v + q_scalar * t + torch.cross(q_vec, t, dim=-1)
    return v_rot  # (..., 3)

def compute_alignment_quaternion(src_vec, target_vec):
    """Compute quaternion rotating src_vec to target_vec."""
    v = torch.cross(src_vec, target_vec, dim=-1)  # (M, 3)
    dot = torch.sum(src_vec * target_vec, dim=-1, keepdim=True)  # (M, 1)
    q = torch.cat([dot + 1, v], dim=-1)  # (M, 4) -> (w, x, y, z)
    return F.normalize(q, dim=-1)

def quaternion_multiply(q1, q2):
    """Multiply two quaternions q1 * q2."""
    w1, xyz1 = q1[:, :1], q1[:, 1:]
    w2, xyz2 = q2[:, :1], q2[:, 1:]
    w = w1 * w2 - torch.sum(xyz1 * xyz2, dim=-1, keepdim=True)
    xyz = w1 * xyz2 + w2 * xyz1 + torch.cross(xyz1, xyz2, dim=-1)
    return torch.cat([w, xyz], dim=-1)  # (M, 4)


def transform_nozzle_pcd_with_quaternion(nozzle_pcd, waypts, quaternions):
    """
    transform nozzle pcd at each waypoint given quaternions (M, 4).
    quaternions as rotation for local frame to each waypoint.
    """
    # normalize quaternions
    quaternions = F.normalize(quaternions, dim=-1)
    transed_nozzle_pcd = quaternion_rotate(quaternions[:, None, :], nozzle_pcd[None, :, :]) 
    transed_nozzle_pcd += waypts[:, None, :]  # (M, N, 3)
    return transed_nozzle_pcd


def quaternion_para_axes(quaternions):
    """get print directions and frame axes from quaternions
    quaternions as rotation for local frame to each waypoint.
    print_dirs ---- nozzle z+
    frame_dirs ---- nozzle x+
    """
    N = quaternions.shape[0]
    x_ref = torch.tensor([1., 0., 0.], device=quaternions.device).unsqueeze(0).expand(N, -1)
    z_ref = torch.tensor([0., 0., 1.], device=quaternions.device).unsqueeze(0).expand(N, -1)

    # normalize quaternions
    quaternions = F.normalize(quaternions, dim=-1)
    # get print directions and frame axes from local frames
    waypt_dirs = quaternion_rotate(quaternions, z_ref) 
    frame_dirs = quaternion_rotate(quaternions, x_ref) 
    waypt_dirs = F.normalize(waypt_dirs, dim=-1) 
    frame_dirs = F.normalize(frame_dirs, dim=-1)

    return waypt_dirs, frame_dirs









