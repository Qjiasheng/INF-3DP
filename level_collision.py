import numpy as np
import torch
import torch.nn.functional as F
import os

from sdf_meshing import gen_voxle_queries
from utils import *
from vis import *


def test_level_accuracy(level_decoder, dataset, num_classes, save_path):
    """
    level classification accuracy on training pcd. 
    show err predications.
    """
    xyz = dataset['xyz']
    isovalues = dataset['slice_isovalues']
    levels = dataset['level_labels']

    level_decoder.eval()
    level_inputs = torch.cat((xyz, isovalues.unsqueeze(-1)), dim=1) # (x, y, z, isovalue)
    levels_pred, confidence = level_querying_with_confidence(level_decoder, level_inputs, num_classes, no_print=True)

    levels_pred = levels_pred.squeeze().numpy()
    confidence = confidence.squeeze().numpy()
    xyz = xyz.numpy()
    levels = levels.numpy()

    # abs accuracy
    accuracy = (levels_pred == levels).sum().item() / len(levels)
    print(f"Level accuracy: {accuracy}")

    # acc with confidence 
    conf = 0.8
    high_confidence_mask = confidence > conf
    high_confidence_accuracy = (levels_pred[high_confidence_mask] == levels[high_confidence_mask]).sum().item() / len(levels[high_confidence_mask])
    print(f"confidence higher than {conf},  level accuracy: {high_confidence_accuracy}")

    # vis err points
    err_mask = levels_pred != levels
    vis_pcd_fields(xyz, err_mask.astype(np.float32))


def query_volume_levels(slice_decoder, level_decoder, num_classes, N=256, save_path=None, max_batch=32 ** 3):
    """volume levels, partition the space into different levels"""

    level_volume_fn = os.path.join(save_path, 'level_volume')
    slice_decoder.eval()
    level_decoder.eval()

    samples, voxel_origin, voxel_size = gen_voxle_queries(N, cube_size=1.0)
    samples.requires_grad = False 
    slice_fields = field_querying(slice_decoder, samples, max_batch=max_batch, no_print=False, return_on_cuda=True)

    # concatenate slice field values to samples
    samples = samples.cuda()
    level_inputs = torch.cat((samples, slice_fields.unsqueeze(-1)), dim=1)
    levels_pred, _ = level_querying_with_confidence(level_decoder, level_inputs, num_classes, no_print=True)
    levels_pred = levels_pred.squeeze().cpu().numpy()

    # save levels volume
    levels_volume = levels_pred.reshape(N, N, N)
    np.savez(level_volume_fn+'.npz', sdf=levels_volume, voxel_grid_origin=voxel_origin, voxel_size=voxel_size)
    print(f"sdf volume saved to {level_volume_fn+'.npz'}" )



class CollTVSDF(torch.autograd.Function):
    """Custom autograd function for collision evaluation with TVSDF and gradient computation.
    algorithm details in paper sec. 5.1"""

    @staticmethod
    def forward(ctx, nozzle_pcd, sdf_decoder, slice_decoder, level_decoder, num_classes, current_level_maxslices, current_levels, base_th):
        """
        fw collision depth  
        first exact collision checking with sequence field (partition + guaidance).
        Second, collision points for tvsdf values and gradients. 
        NO need care much about collision depth values (since \partial loss / \partial depth = 1 is used),
        but the gradients of queries in collision.
        --------------
        also consider queries under base region (collision with base platform), gradients set z-up
        """
        N, M, _ = nozzle_pcd.shape  # N waypts, M nozzle pts each
        coords = nozzle_pcd.view(-1, 3)  # (N*M, 3)

        waypt_levels = current_levels.repeat_interleave(M, dim=0)  # (N*M, )
        current_level_maxslices = current_level_maxslices.repeat_interleave(M, dim=0)  # (N*M, num_levels)

        coll_depths = torch.zeros(coords.shape[0], device=coords.device)  # (N*M, )
        coll_grads = torch.zeros_like(coords)
        coll_mask = torch.zeros(coords.shape[0], dtype=torch.bool, device=coords.device)

        # valid nozzle pcd should in cube [-1, 1]
        cube_mask = ((coords >= -1.0) & (coords <= 1.0)).all(dim=-1)
        if cube_mask.sum() == 0:
            # all points out of cube, no collision
            ctx.save_for_backward(coll_grads.view(N, M, 3))
            coll_depths = F.relu(-coll_depths)
            return coll_depths.view(N, M), coll_mask.view(N, M)

        # get all field, reduce query 
        cube_coords = coords[cube_mask, :]
        original_sdfs = torch.ones(coords.shape[0], device=cube_coords.device)
        sdf_grads = torch.zeros_like(coords)
        _, cube_sdfs, cube_sdf_grads = field_query_with_grads(sdf_decoder, cube_coords, no_print=True, return_on_cuda=True, max_batch=64**3) 
        original_sdfs[cube_mask] = cube_sdfs
        sdf_grads[cube_mask, :] = cube_sdf_grads

        del cube_coords, cube_sdfs, cube_sdf_grads, cube_mask
        torch.cuda.empty_cache()

        # base collision handling
        base_coll_mask = coords[:, 2] < base_th
        if base_coll_mask.sum() > 0:
            coll_depths[base_coll_mask] = -0.1 # arbitrary small neg value
            coll_grads[base_coll_mask, :] = torch.tensor([0.0, 0.0, 1.0], device=coords.device)
            coll_mask[base_coll_mask] = True


        inside_mask = original_sdfs < 0.0  # inside has potential collision
        if inside_mask.sum() == 0:
            # no collision at all
            ctx.save_for_backward(coll_grads.view(N, M, 3))
            coll_depths = F.relu(-coll_depths)
            return coll_depths.view(N, M), coll_mask.view(N, M)
        
        # check inside points collision, save memory
        inside_coords = coords[inside_mask, :]  
        inside_waypt_levels = waypt_levels[inside_mask]  
        inside_current_level_maxslices = current_level_maxslices[inside_mask, :]  
        inside_sdfs = original_sdfs[inside_mask]
        inside_sdf_grads = sdf_grads[inside_mask, :]
        inside_waypt_maxslices = inside_current_level_maxslices.gather(1, inside_waypt_levels.unsqueeze(-1)).squeeze(-1)

        del coords, waypt_levels, current_level_maxslices, original_sdfs, sdf_grads
        torch.cuda.empty_cache()

        # further querying
        _, inside_slice_fields, inside_slice_grads = field_query_with_grads(slice_decoder, inside_coords, no_print=True, 
                                                                                         return_on_cuda=True, max_batch=64**3)
        level_inputs = torch.cat((inside_coords.detach(), inside_slice_fields.unsqueeze(-1)), dim=1)
        inside_coord_levels, _ = level_querying_with_confidence(level_decoder, level_inputs, num_classes, no_print=True, return_on_cuda=True, max_batch=64**3)

        # determine collision
        inside_coll_mask = ((inside_coord_levels < inside_waypt_levels) |
                            ((inside_coord_levels == inside_waypt_levels) &
                             (inside_slice_fields < inside_waypt_maxslices))
                           )
        if inside_coll_mask.sum() == 0:
            # no collision at all
            ctx.save_for_backward(coll_grads.view(N, M, 3))
            coll_depths = F.relu(-coll_depths)
            return coll_depths.view(N, M), coll_mask.view(N, M)

        # different from a complete tvsdf construction. no need traverse along levels
        # but sel original sdf or intepolated sdf and their grads
        coll_coords = inside_coords[inside_coll_mask, :]
        coll_coords_levels = inside_coord_levels[inside_coll_mask]
        coll_sdfs = inside_sdfs[inside_coll_mask]
        coll_sdf_grads = inside_sdf_grads[inside_coll_mask]
        coll_slice_fields = inside_slice_fields[inside_coll_mask]
        coll_slice_grads = inside_slice_grads[inside_coll_mask, :]
        coll_current_level_maxslices = inside_current_level_maxslices[inside_coll_mask, :]      # (num_coll_pts, num_levels)
        coll_waypt_levels = inside_waypt_levels[inside_coll_mask]
        coll_waypt_maxslices = coll_current_level_maxslices.gather(1, coll_waypt_levels.unsqueeze(-1)).squeeze(-1)

        # assemble each qurey coord's current level maxslice depending on its level
        coll_current_level_maxslices_single = coll_current_level_maxslices.gather(1, coll_coords_levels.unsqueeze(-1)).squeeze(-1)  # (num_coll_pts, )

        # field to dist 
        coll_slices_dist = (coll_slice_fields - coll_current_level_maxslices_single) / torch.norm(coll_slice_grads, dim=-1)  # (num_coll_pts, )
        coll_tvsdfs = torch.max(coll_sdfs, coll_slices_dist)

        # no need update sdfs for depth, only care gradients
        recheck_mask = coll_slices_dist > coll_sdfs
        if recheck_mask.sum() == 0:
            # no need recheck, combine two masks. use all original sdf grads, all in collision
            combined_mask = torch.zeros_like(inside_mask)
            combined_mask[inside_mask] = inside_coll_mask
            coll_depths[combined_mask] = coll_tvsdfs
            coll_grads[combined_mask, :] = coll_sdf_grads
            coll_mask |= combined_mask  # not overwrite previous base coll mask
            ctx.save_for_backward(coll_grads.view(N, M, 3))
            coll_depths = F.relu(-coll_depths)
            return coll_depths.view(N, M), coll_mask.view(N, M)
        
        # recheck with interpolated tvsdf grads
        recheck_coords = coll_coords[recheck_mask, :]
        recheck_coords_levels = coll_coords_levels[recheck_mask]
        recheck_sdfs = coll_sdfs[recheck_mask]
        recheck_sdf_grads = coll_sdf_grads[recheck_mask, :]
        recheck_slice_fields = coll_slice_fields[recheck_mask]
        recheck_slice_grads = coll_slice_grads[recheck_mask, :]
        recheck_slice_dist = coll_slices_dist[recheck_mask]
        recheck_current_level_maxslices = coll_current_level_maxslices[recheck_mask, :]
        recheck_waypt_levels = coll_waypt_levels[recheck_mask]
        recheck_waypt_maxslices = coll_waypt_maxslices[recheck_mask]

        # two steps push
        # hit top surface
        recheck_push_coords_step1 = recheck_coords + torch.abs(recheck_slice_dist).unsqueeze(-1) * F.normalize(recheck_slice_grads, dim=-1)
        recheck_push_coords_step1 = torch.clamp(recheck_push_coords_step1, min=-1.0, max=1.0)
        # outside a step
        _, _, recheck_push_slice_grads_step1 = field_query_with_grads(slice_decoder, recheck_push_coords_step1, no_print=True, return_on_cuda=True, max_batch=64**3)
        recheck_push_coords = recheck_push_coords_step1 + (torch.max(torch.abs(recheck_slice_dist).unsqueeze(-1) * 0.5, torch.tensor(0.04).cuda()) * 
                                                           F.normalize(recheck_push_slice_grads_step1.detach(), dim=-1))
        recheck_push_coords = torch.clamp(recheck_push_coords, min=-1.0, max=1.0)

        # query levels
        recheck_push_slice_fields = field_querying(slice_decoder, recheck_push_coords, max_batch=64**3, no_print=True, return_on_cuda=True)
        recheck_level_inputs = torch.cat((recheck_push_coords.detach(), recheck_push_slice_fields.unsqueeze(-1)), dim=1)
        recheck_push_coord_levels, _ = level_querying_with_confidence(level_decoder, recheck_level_inputs, num_classes, no_print=True, return_on_cuda=True, max_batch=64**3)

        # decide alter origianl sdf or interp slice sdf
        alter_mask = ((recheck_push_coord_levels < recheck_waypt_levels) | 
                      ((recheck_push_slice_fields < recheck_waypt_maxslices) & (recheck_push_coord_levels == recheck_waypt_levels)))
        final_recheck_sdfs = torch.where(alter_mask, recheck_sdfs, recheck_slice_dist)  # if True use original sdf, else use interp slice sdf
        final_recheck_sdf_grads = torch.where(alter_mask.unsqueeze(-1), recheck_sdf_grads, recheck_slice_grads)

        # assemble back
        final_coll_tvsdfs = coll_tvsdfs.clone()
        final_coll_tvsdf_grads = coll_sdf_grads.clone()
        final_coll_tvsdfs[recheck_mask] = final_recheck_sdfs
        final_coll_tvsdf_grads[recheck_mask, :] = final_recheck_sdf_grads

        combined_mask = torch.zeros_like(inside_mask)
        combined_mask[inside_mask] = inside_coll_mask
        coll_depths[combined_mask] = final_coll_tvsdfs
        coll_grads[combined_mask, :] = final_coll_tvsdf_grads
        coll_mask |= combined_mask
        ctx.save_for_backward(coll_grads.view(N, M, 3))
        coll_depths = F.relu(-coll_depths)  # all positive, not influence grad directions

        del inside_coords, inside_waypt_levels, inside_current_level_maxslices, inside_sdfs, inside_sdf_grads  
        torch.cuda.empty_cache()

        return coll_depths.view(N, M), coll_mask.view(N, M)

    @staticmethod
    def backward(ctx, dl_dcoll_depths, dl_dcoll_mask=None):
        """"
        return gradients for nozzle pcd in collision
        """

        coll_grads, = ctx.saved_tensors # (N, M, 3)
        N, M, _ = coll_grads.shape
        if not ctx.needs_input_grad[0]:
            return None, None, None, None, None, None, None, None
        
        coll_grads = F.normalize(coll_grads, dim=-1, eps=1e-8)  # only direction matters

        # chain rule
        coll_grads = -coll_grads  # NOTE inverse direction is right gradients
        dl_dpcd = coll_grads.mul_(dl_dcoll_depths.view(N, -1, 1))  # (N, M, 3)
        return dl_dpcd, None, None, None, None, None, None, None
       
    

def collision_evaluation(nozzle_pcd, sdf_decoder, slice_decoder, level_decoder, num_classes, current_level_maxslice, waypt_levels, base_th):
    """
    differentiable collision evaluation function wrapper.
    waypt_levels: (num_waypts, )
    nozzle_pcd: (num_waypts, num_nozzle_pts, 3). alreay transformed at each waypoint.
    current_level_maxslice: (num_waypts, num_levels) -- for TVSDF querying
    --------------
    return:
    collision_depths: (num_waypts, num_nozzle_pts)
    collision_mask: (num_waypts, num_nozzle_pts)
    waypt_collision_mask: (num_waypts, ), .any() indicates whether this waypoint has collision.
    --------------
    all on cuda 
    """

    # differentiable collision function
    coll_depths, coll_mask = CollTVSDF.apply(
        nozzle_pcd, sdf_decoder, slice_decoder, level_decoder, num_classes, 
        current_level_maxslice, waypt_levels, base_th) # (num_waypts, num_nozzle_pts)
    
    waypt_coll_mask = coll_mask.any(dim=-1)  # (num_waypts, )

    return coll_depths, coll_mask, waypt_coll_mask



def collision_test(sdf_decoder, slice_decoder, level_decoder, num_classes, dataset_paras, loss_paras, save_path, waypt_down_ratio=10):
    """
    differentiable collision response. collision evluation is wrapped in autograd function.
    a smaple nozzle (origin is set at tooltip) with physical scale, need to be rescale to [-1, 1] with object_scale.
    -------------- 
    1. batched waypts collision checking,
    2. get collided points' gradients. \partial coll_depth / \partial x, to vis push directions.
    """

    waypt_path = dataset_paras.get('waypoint_path')
    nozzle_shell_pcd_path = dataset_paras.get('nozzle_shell_pcd_path')
    object_scale = loss_paras.get('object_scale')

    # load waypoints and nozzle shell pcd
    # downsample, batch to test all waypts
    ori_waypts, ori_print_dirs, _, ori_waypt_levels = load_waypoints(waypt_path)
    nozzle_pcd = load_nozzle_pcd(nozzle_shell_pcd_path, object_scale=object_scale, down_ratio=10)

    # save original index for full waypts vis
    waypt_down_ratio = waypt_down_ratio
    waypt_down_index = np.arange(len(ori_waypts))[::waypt_down_ratio]
    if waypt_down_ratio > 1:
        waypts = ori_waypts[::waypt_down_ratio, :]
        print_dirs = ori_print_dirs[::waypt_down_ratio, :]
        waypt_levels = ori_waypt_levels[::waypt_down_ratio]

    # vis check waypts and print dirs
    vis_grad(waypts, print_dirs, label='print_dirs')

    # use original waypts to build raw level max slice dict
    waypts_full = torch.from_numpy(ori_waypts).float().cuda()
    slice_fields_full = field_querying(slice_decoder, waypts_full, no_print=True, return_on_cuda=False)
    level_slice_dict = build_level_slice_dict(slice_fields_full, torch.from_numpy(ori_waypt_levels).long())  
    # build current level max slice for downsampled waypts
    waypt_levels = torch.from_numpy(waypt_levels).long()
    slice_fields = field_querying(slice_decoder, torch.from_numpy(waypts).float().cuda(), no_print=True, return_on_cuda=False)
    current_level_maxslice = build_current_level_slice(level_slice_dict, waypt_levels, slice_fields)  # (num_waypts, num_levels)

    # on cuda
    nozzle_pcd = torch.from_numpy(nozzle_pcd).float().cuda() 
    waypts = torch.from_numpy(waypts).float().cuda()
    print_dirs = F.normalize(torch.from_numpy(print_dirs).float().cuda(), dim=-1)
    waypt_levels = waypt_levels.cuda()
    current_level_maxslice = current_level_maxslice.cuda()

    # set ref dir to determine frames with print dirs
    ref_dirs = torch.zeros_like(print_dirs, device=print_dirs.device)
    ref_dirs[:, 0] = -1.0  # x-

    # transform nozzle pcd at each waypoint.
    frame_dirs = torch.cross(ref_dirs, print_dirs, dim=-1)
    transed_nozzle_pcd = transform_nozzle_pcd_with_frame(nozzle_pcd, waypts, print_dirs, frame_dirs) # (num_waypts, num_nozzle_pts, 3)

    # vis check transed nozzle pcd 
    # see_index = 15000
    # transed_nozzle_pcd_np = transed_nozzle_pcd.cpu().numpy()
    # vis_print_nozzle_pcd_comparison(raw_pcd=transed_nozzle_pcd_np[see_index], waypoints=ori_waypts[:waypt_down_index[see_index], :])

    # collision evalution
    base_coll_th = ori_waypts[:, 2].min().item()  - 2.0 / object_scale  # 2 mm below the lowest waypt as collision with base
    transed_nozzle_pcd.requires_grad = True
    nozzle_pcd_coll_depths, nozzle_pcd_coll_mask, waypt_coll_mask = collision_evaluation(
        transed_nozzle_pcd, sdf_decoder, slice_decoder, level_decoder, num_classes, current_level_maxslice, waypt_levels, base_th=base_coll_th)
    
    # hook nozzle pcd gradients at collided waypts
    coll_loss = nozzle_pcd_coll_depths.sum()
    coll_loss.backward(retain_graph=False)
    transed_nozzle_pcd_grads = transed_nozzle_pcd.grad  # leaf, no need to hook

    # # NOTE when vis push directions, inverse grads
    transed_nozzle_pcd_grads = -transed_nozzle_pcd_grads
    
    # vis collided waypts
    waypts = waypts.cpu().numpy()
    waypt_coll_mask = waypt_coll_mask.cpu().numpy()
    print(f'collision ratio: {waypt_coll_mask.sum()/len(waypt_coll_mask)*100:.2f}%, {waypt_coll_mask.sum()} / {len(waypt_coll_mask)}')
    vis_pcd_fields(waypts, waypt_coll_mask.astype(np.float32))

    # vis check one collided waypoint's nozzle push directions
    transed_nozzle_pcd_np = transed_nozzle_pcd.detach().cpu().numpy()
    transed_nozzle_pcd_grads_np = transed_nozzle_pcd_grads.detach().cpu().numpy()
    nozzle_pcd_coll_mask_np = nozzle_pcd_coll_mask.cpu().numpy()

    see_index = 100  # coll index to see
    see_coll_index = np.where(waypt_coll_mask)[0][see_index]
    vis_print_nozzle_pcd_comparison(raw_pcd=transed_nozzle_pcd_np[see_coll_index],
                                    raw_grads=transed_nozzle_pcd_grads_np[see_coll_index],
                                    waypoints=ori_waypts[:waypt_down_index[see_coll_index], :],
                                    add_cube=True)



def query_waypt_tvsdf(sdf_decoder, slice_decoder, level_decoder, num_classes, dataset_paras, save_path):
    """construct TVSDF till a given waypoint for visualization.
    [optional] masked outside regions outside current evloved level.
    collision repsonse only cares about inside regions
    -----------------------
    algorithm details in paper sec. 5.1
    """

    # collect parameters
    waypoint_path = dataset_paras.get('waypoint_path')
    N = dataset_paras.get('resolution')    

    # load waypoints and query slice fields
    waypts, _, _, waypt_levels = load_waypoints(waypoint_path)
    waypts_np = waypts.copy()  # for saving later
    waypts = torch.from_numpy(waypts).float().cuda()
    waypt_levels = torch.from_numpy(waypt_levels).long().cuda()

    # linear get a batch of waypoints to test
    num = 20
    batch_waypt_idx = torch.arange(1000, waypts.shape[0], waypts.shape[0]//num).long().cuda() 
    # batch_waypt_idx = torch.tensor(400000).long().cuda()  # single waypoint

    # build each level's max slice field value
    waypt_slice_fields = field_querying(slice_decoder, waypts, no_print=True, return_on_cuda=True)
    level_slice_dict = build_level_slice_dict(waypt_slice_fields, waypt_levels)

    current_waypt = waypts[batch_waypt_idx]
    current_waypt_level = waypt_levels[batch_waypt_idx]
    current_waypt_slice = waypt_slice_fields[batch_waypt_idx]
    
    # single test
    # current_waypt = waypts[batch_waypt_idx].unsqueeze(0)  # (1, 3)
    # current_waypt_level = waypt_levels[batch_waypt_idx].unsqueeze(0)  # (1, )
    # current_waypt_slice = waypt_slice_fields[batch_waypt_idx].unsqueeze(0)  # (1, )

    current_level_maxslice  = build_current_level_slice(level_slice_dict, current_waypt_level, current_waypt_slice)

    # prepare volume queries
    voxel_queries, voxel_origin, voxel_size = gen_voxle_queries(N=N, cube_size=1.0)

    # repeat volume queries for each waypoint, but nozzle samples at different positions in collision checking
    queries = voxel_queries.unsqueeze(0).repeat(current_waypt.shape[0], 1, 1)  # (num_waypts, N^3, 3)

    queries = queries.cuda()
    current_level_maxslice = current_level_maxslice.cuda()
    current_waypt_level = current_waypt_level.cuda()
    tvsdf, maxslices, valid_mask = batch_query_tvsdf(queries, sdf_decoder, slice_decoder, level_decoder, num_classes, 
                                                      current_level_maxslice, current_waypt_level)  # all (num_waypts * N^3, )
    
    # reshape to volume data
    tvsdf_volume = tvsdf.view(-1, N, N, N).cpu().numpy()         
    maxslices_volume = maxslices.view(-1, N, N, N).cpu().numpy()  
    valid_mask_volume = valid_mask.view(-1, N, N, N).cpu().numpy()  

    tvsdf_volume *= valid_mask_volume
    # save volume data and current pcd
    see_idx = -2
    till_waypts = waypts_np[:batch_waypt_idx.cpu().numpy()[see_idx], ...]
    # till_waypts = waypts_np[:batch_waypt_idx.cpu().numpy(), ...]

    np.savetxt(os.path.join(save_path, 'till_waypts.xyz'), till_waypts, fmt='%.6f')

    tvsdf_volume_fn = os.path.join(save_path, 'tvsdf_volume')
    np.savez(tvsdf_volume_fn+'.npz', sdf=tvsdf_volume[see_idx, ...], voxel_grid_origin=voxel_origin, voxel_size=voxel_size)

    maxslice_volume_fn = os.path.join(save_path, 'maxslice_volume')
    np.savez(maxslice_volume_fn+'.npz', sdf=maxslices_volume[see_idx, ...], voxel_grid_origin=voxel_origin, voxel_size=voxel_size)

    mask_volume_fn = os.path.join(save_path, 'mask_volume')
    np.savez(mask_volume_fn+'.npz', sdf=valid_mask_volume[see_idx, ...], voxel_grid_origin=voxel_origin, voxel_size=voxel_size)



def quat_collision_test(sdf_decoder, slice_decoder, level_decoder, quat_decoder, num_classes, dataset_paras, loss_paras, save_path, waypt_down_ratio=10):
    """
    collision test after quaternion field optimization. 
    similar process as collision_test function but without grads enabled.
    """   

    waypt_path = dataset_paras.get('waypoint_path')
    nozzle_shell_pcd_path = dataset_paras.get('nozzle_shell_pcd_path')
    object_scale = loss_paras.get('object_scale')

    # load waypoints and nozzle shell pcd
    ori_waypts, ori_print_dirs, _, ori_waypt_levels = load_waypoints(waypt_path)
    nozzle_pcd = load_nozzle_pcd(nozzle_shell_pcd_path, object_scale=object_scale, down_ratio=10)

    # save original index for full waypts vis
    waypt_down_ratio = waypt_down_ratio
    waypt_down_index = np.arange(len(ori_waypts))[::waypt_down_ratio]
    if waypt_down_ratio > 1:
        waypts = ori_waypts[::waypt_down_ratio, :]
        print_dirs = ori_print_dirs[::waypt_down_ratio, :]
        waypt_levels = ori_waypt_levels[::waypt_down_ratio]

    # use original waypts to build raw level max slice dict
    waypts_full = torch.from_numpy(ori_waypts).float().cuda()
    slice_fields_full = field_querying(slice_decoder, waypts_full, no_print=True, return_on_cuda=False)
    level_slice_dict = build_level_slice_dict(slice_fields_full, torch.from_numpy(ori_waypt_levels).long())  
    # build current level max slice for downsampled waypts
    waypt_levels = torch.from_numpy(waypt_levels).long()
    slice_fields = field_querying(slice_decoder, torch.from_numpy(waypts).float().cuda(), no_print=True, return_on_cuda=False)
    current_level_maxslice = build_current_level_slice(level_slice_dict, waypt_levels, slice_fields)  # (num_waypts, num_levels)

    # on cuda
    nozzle_pcd = torch.from_numpy(nozzle_pcd).float().cuda() 
    waypts = torch.from_numpy(waypts).float().cuda()
    print_dirs = F.normalize(torch.from_numpy(print_dirs).float().cuda(), dim=-1)
    waypt_levels = waypt_levels.cuda()
    current_level_maxslice = current_level_maxslice.cuda()

    # Compare original and optimized frames. 
    # set arbitrary ref dir to construct original frames
    base_coll_th = ori_waypts[:, 2].min().item()  - 2.0 / object_scale
    # original sets
    ref_dirs = torch.zeros_like(print_dirs, device=print_dirs.device)
    ref_dirs[:, 0] = -1.0  # x-
    frame_dirs = torch.cross(ref_dirs, print_dirs, dim=-1) 
    transed_nozzle_pcd = transform_nozzle_pcd_with_frame(nozzle_pcd, waypts, print_dirs, frame_dirs)
    with torch.no_grad():
        orig_coll_depths, orig_coll_mask, orig_waypt_coll_mask = collision_evaluation(
            transed_nozzle_pcd, sdf_decoder, slice_decoder, level_decoder, num_classes, current_level_maxslice, waypt_levels, base_th=base_coll_th)
    
    # optimized sets
    quaternions = quat_field_querying(quat_decoder, waypts, no_print=True, return_on_cuda=True)
    quat_transed_nozzle_pcd = transform_nozzle_pcd_with_quaternion(nozzle_pcd, waypts, quaternions)
    quat_print_dirs, quat_frame_dirs = quaternion_para_axes(quaternions) 
    with torch.no_grad():
        quat_coll_depths, quat_coll_mask, quat_waypt_coll_mask = collision_evaluation(
            quat_transed_nozzle_pcd, sdf_decoder, slice_decoder, level_decoder, num_classes, current_level_maxslice, waypt_levels, base_th=base_coll_th)    
        
    # report and vis collisions
    waypts = waypts.cpu().numpy()
    orig_waypt_coll_mask = orig_waypt_coll_mask.cpu().numpy()
    quat_waypt_coll_mask = quat_waypt_coll_mask.cpu().numpy()
    print(f'Original collision ratio: {orig_waypt_coll_mask.sum()/len(orig_waypt_coll_mask)*100:.2f}%, {orig_waypt_coll_mask.sum()} / {len(orig_waypt_coll_mask)}')
    print(f'Optimized collision ratio: {quat_waypt_coll_mask.sum()/len(quat_waypt_coll_mask)*100:.2f}%, {quat_waypt_coll_mask.sum()} / {len(quat_waypt_coll_mask)}')
    vis_pcd_fields(waypts, orig_waypt_coll_mask.astype(np.float32), label='Original Collision')
    vis_pcd_fields(waypts, quat_waypt_coll_mask.astype(np.float32), label='Optimized Collision')

    # vis frame dirs see if smooth. set larger loss to get smoother results
    vis_grad(waypts, quat_print_dirs.detach().cpu().numpy(), label='Optimized Print Dirs')
    vis_grad(waypts, quat_frame_dirs.detach().cpu().numpy(), label='Optimized Frame Dirs')

    # vis check one collided waypoint's nozzle push directions
    # original in collision while optimized not at same waypt
    see_index = 200 
    coll_index = np.where(orig_waypt_coll_mask)[0][see_index]

    orig_transed_nozzle_pcd_np = transed_nozzle_pcd.detach().cpu().numpy()
    quat_transed_nozzle_pcd_np = quat_transed_nozzle_pcd.detach().cpu().numpy()
    orig_nozzle_pcd_coll_mask_np = orig_coll_mask.cpu().numpy()
    quat_nozzle_pcd_coll_mask_np = quat_coll_mask.cpu().numpy()
    vis_print_nozzle_pcd_comparison(raw_pcd=orig_transed_nozzle_pcd_np[coll_index],
                                    raw_grads=None,
                                    raw_mask=orig_nozzle_pcd_coll_mask_np[coll_index],
                                    pcd=quat_transed_nozzle_pcd_np[coll_index],
                                    grads=None,
                                    mask=quat_nozzle_pcd_coll_mask_np[coll_index],
                                    waypoints=ori_waypts[:waypt_down_index[coll_index], :],
                                    add_cube=True,)
    
    # transform mesh to vis better
    see_index = 200 
    coll_index = np.where(orig_waypt_coll_mask)[0][see_index]
    nozzle_mesh_path = dataset_paras.get('nozzle_mesh_path')
    nozzle_mesh = load_nozzle_mesh(nozzle_mesh_path, object_scale=object_scale)
    vertices = torch.from_numpy(nozzle_mesh.vertices).float().cuda()
    faces = nozzle_mesh.faces
    # select one pose for mesh transformation and vis
    orig_print_dir = print_dirs[coll_index, :].unsqueeze(0)
    orig_frame_dir = frame_dirs[coll_index, :].unsqueeze(0)
    quat_print_dir = quat_print_dirs[coll_index, :].unsqueeze(0)
    quat_frame_dir = quat_frame_dirs[coll_index, :].unsqueeze(0)
    see_waypt = torch.from_numpy(waypts[coll_index, :]).unsqueeze(0).cuda()

    orig_transed_vertices = transform_nozzle_pcd_with_frame(vertices, see_waypt, orig_print_dir, orig_frame_dir).cpu().numpy()
    quat_transed_vertices = transform_nozzle_pcd_with_frame(vertices, see_waypt, quat_print_dir, quat_frame_dir).cpu().numpy()

    orig_transed_mesh = [trimesh.Trimesh(vertices=v, faces=faces) for v in orig_transed_vertices]
    quat_transed_mesh = [trimesh.Trimesh(vertices=v, faces=faces) for v in quat_transed_vertices]

    vis_print_nozzle_mesh_comparison(raw_mesh=orig_transed_mesh[0],
                                     mesh=quat_transed_mesh[0],
                                     waypoints=ori_waypts[:waypt_down_index[coll_index], :],
                                     add_cube=True,)
    


def write_cf_waypoints(quat_decoder, dataset_paras, loss_paras, save_path):
    """
    write waypoints with collision-free frames with optimized quaternion.
    for further motion planning use.
    scale waypts in physical space.
    """
    waypt_path = dataset_paras.get('waypoint_path')
    object_scale = loss_paras.get('object_scale')

    # load original waypoints
    waypts, _, layer_thickness, _ = load_waypoints(waypt_path)
    waypts_tensor = torch.from_numpy(waypts).float().cuda()

    # query optimized quaternions
    quaternions = quat_field_querying(quat_decoder, waypts_tensor, no_print=True, return_on_cuda=True, max_batch=64**3)
    quaternions = F.normalize(quaternions, dim=-1, eps=1e-8) 
    # quat_print_dirs, quat_frame_dirs = quaternion_para_axes(quaternions)  # if save frame axes

    # save cf waypoints
    phy_waypts = waypts * object_scale  # scale to physical space
    phy_waypts[:, 2] = phy_waypts[:, 2] - np.min(phy_waypts[:, 2]) 
    quaternions = quaternions.cpu().numpy()

    cf_waypt_path = os.path.join(save_path, 'cf_waypoints.xyz')
    data = np.hstack((phy_waypts, quaternions, layer_thickness.reshape(-1, 1)))
    fmt = '% .6f' * data.shape[1]
    general_text_writer(data=data, fmt=fmt, filename=cf_waypt_path, chunk=None)
    print(f"Total {len(waypts)} cf waypoints saved to {cf_waypt_path}")
    



    
