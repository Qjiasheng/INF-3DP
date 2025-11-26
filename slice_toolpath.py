from toolpath_utils import *
from toolpath_vis import *

import os

# set seed
torch.manual_seed(0)
np.random.seed(0)



def single_shell_toolpath_gen(sdf_decoder, slice_decoder, data_paras, toolpath_paras, save_path):
    """SDF=0, and slice field intersection to get single shell surface toolpath.
    we first use off-shelf mesh to [parallel] extract iso-contours, then [optional] iterative projection (see Sec. 5.2).
    --------------------
    then plan printing sequence with graph-based method for continulity.
    NOTE graph refinments during collison optimization is not included. Users may adjust assigned printing order manually."""

    N = toolpath_paras.get('resolution')
    sdf_level_num = toolpath_paras.get('sdf_level_num')  
    slice_num = toolpath_paras.get('slice_num')
    jump_threshold = toolpath_paras.get('jump_threshold')
    outer_down_ratio = toolpath_paras.get('outer_down_ratio')
    waypt_fn = data_paras.get('waypoint_filename')
    continuous_cnt_lim = toolpath_paras.get('continuous_cnt_lim')
    object_scale = data_paras.get('object_scale')
    mesh_fn = data_paras.get('mesh_path')

    assert sdf_level_num == 1, 'single shell, sdf level num must be 1'

    # iso-contours extraction, consider slice field singualrities optinally
    init_contours, sing_values, sing_centers, ver_fields, vertices, _ = get_single_shell_contours(slice_decoder, sdf_decoder, slice_num, mesh_fn,
                                                                                                around_sing=True, sing_th=0.3, len_lim=50)
    
    # [optional] iterative projection. optimizer parameters inside function. also with prj err output during iterations
    # init_contours = iter_project_contours_batch(init_contours, sdf_decoder, slice_decoder, save_path=save_path)

    # visualize_merged_iso_contours_mul(init_contours, start=None, end=None, line_width=7, random_color=True)

    # [optional] downsample contours with uniform spacing. then iterative projection again to keep on surface
    contours, spacings = smooth_contours_resampling(init_contours, outer_down_ratio=outer_down_ratio)
    contours = iter_project_contours_batch(contours, sdf_decoder, slice_decoder, save_path=save_path)  # time consuming
    hist_spacing_distance(init_contours, contours)
    visualize_merged_iso_contours_mul(init_contours, start=None, end=None, line_width=7, random_color=True)

    # plan printing sequence with graph-based method for continulity
    # print_contours here in most cases already can be used for printing
    # manual adjust is needed when partions are not desired
    print_contours, print_isovalues, print_branches, print_seg_index = build_graph_partition(contours, jump_threshold=jump_threshold,
                                                                                             continuous_cnt_lim=continuous_cnt_lim)
    visualize_printing_contours(print_contours)


    # pick start point of each contour after loops follow same loop direction
    print_contours = get_same_order_contours(print_contours)
    starts, start_indices = simple_arrange_subcontour_starts(print_contours)
    visualize_subcontour_starts(starts, mesh_fn=mesh_fn)

    # [optional] a seperate function to generate partition levels as training dataset 
    # for partition field training, used for further collision optimization
    level_fn = os.path.join(save_path, 'print_level_dataset.pth')
    build_shell_level_dataset(print_contours, print_isovalues, level_fn, interpolate=False, inter_num=4)

    # determine thickness of waypoints -- a fitting bw prj_slice_grad_norm and physical thickness
    # NOTE we here just show a simple example without carefully choosing how many slice layers with specific nozzle thickness capability
    # e.g., we here use 2.5 mm nozzle, 1000 slice for fertility model is too many. we just show the capability of our method.
    # Negative thickness are from cubic polynomial fitting, usrs can set a clip(min=0) to avoid negative values
    fit_function = build_calibration_thickness(print_contours, sdf_decoder, slice_decoder, fit_down_ratio=1000, 
                                               object_scale=object_scale, dist_threshold=3.0)
    # fit_function = np.poly1d([-2.507, 7.146, -7.048, 2.749])  # cubic polynomial fitting. use results after calibration

    # generate and write waypoints file
    waypoints, level_labels = generate_waypoints_with_levels(print_contours, start_indices, print_isovalues)
    wayfn = os.path.join(save_path, waypt_fn)
    write_waypoints_single_shell(waypoints, fit_function, slice_decoder, sdf_decoder, object_scale, wayfn, 
                                 level_labels=level_labels, for_collision=True)


def wall_shell_toolpath_gen(sdf_decoder, slice_decoder, data_paras, toolpath_paras, save_path):
    """similar to single shell, but with multiple sdf levels.
    part of infill functions. users can refactor for wall shell use."""
    pass
    

def lattice_toolpath_gen(sdf_decoder, slice_decoder, lattice1_decoder, lattice2_decoder, data_paras, 
                         toolpath_paras, save_path, resample_infills=False):
    """
    infill toolpath generation with two lattice fields intersection with slice field.
    outer shell is similar to single shell toolpath generation.
    --------------------
    field intersection is supposed to follow two field intersection as implemented in `single_shell`, 
    which needs each isolayers triangle mesh for init isocontours extraction (with parallel cuda marching cubes, 
    one implementation is [cumcubes](https://github.com/lzhnb/CuMCubes)), then iterative projection.
    Since our lattice fields propogates [monotonically], we use three fields (two lattice + slice) intersection with linear interpolation in 
    parallel voxels, pls see our cuda implementation. Users desire accuracy can still do interative projection after that.
    --------------------
    """
    N = toolpath_paras.get('resolution')
    sdf_level_num = toolpath_paras.get('sdf_level_num')
    sdf_level_margin = toolpath_paras.get('sdf_level_margin')
    slice_num = toolpath_paras.get('slice_num')
    jump_threshold = toolpath_paras.get('jump_threshold')
    till_slice_layers = toolpath_paras.get('till_slice_layers')
    lattice1_num = toolpath_paras.get('lattice1_num')
    lattice2_num = toolpath_paras.get('lattice2_num')
    outer_down_ratio = toolpath_paras.get('outer_down_ratio')
    continuous_cnt_lim = toolpath_paras.get('continuous_cnt_lim')
    waypt_fn = data_paras.get('waypoint_filename')
    object_scale = data_paras.get('object_scale')

    # generate multiply sdf levels for outer shells, as similar to single shell
    # contours -- list of dict (isovalues as keys, contours as values), each dict is one shell.
    # more inside levels may have empty contours at an iso-value
    outer_contours, slice_isovalues = get_wall_shell_contours(slice_decoder, sdf_decoder, slice_num, sdf_level_num, sdf_level_margin, save_path, N=N)
    
    # [optional] downsample outer contours with uniform spacing, then iterative projection again to keep on surface
    outer_contours, spacings = wall_smooth_contours_resampling(outer_contours, outer_down_ratio=outer_down_ratio)
    visualize_merged_iso_contours_mul(outer_contours[0])


    # resolve intersections bw lattice fields and slice field for infill toolpaths
    # we use dense grid intersection first, then downsample for usages
    # users can also use CUDA marching cubes then get two fields intersections as done for outer shells
    fields_list = get_grid_fields(decoder_list=[sdf_decoder, slice_decoder, lattice1_decoder, lattice2_decoder], N=N)
    fixed_lattice1_num, fixed_lattice2_num = 128, 128 
    intersection_dict = get_intersection_points_cuda(fields_list, fixed_lattice1_num, fixed_lattice2_num, 
                                                     sdf_level=0.0, isovalues_slice=slice_isovalues)
    vis_intersections_dict(intersection_dict)

    # build a nested dict for easy access
    inter_linked_s12 = build_linked_dict(intersection_dict, front='lattice1')
    inter_linked_s21 = build_linked_dict(intersection_dict, front='lattice2')

    # downsample lattice contours with targeted lattice numbers
    subcontour_s1, subcontour_s2 = downsample_lattice_contours(inter_linked_s12, inter_linked_s21, lattice1_num, 
                                                                lattice2_num, sdf_decoder, till_slice_layers=till_slice_layers, 
                                                                ratio=20.0, split_sdf_level=-0.002)
    
    # visualize_lattice_subcontours(subcontour_s1, start_layer=50, end_layer=101)  # time consuming

    # build graph partition for printing sequence
    # printing sequence is same as done in single shell, two sets of innner infill contours (share same slice isovalues) follow the same order
    # just treat each outer contour as an isolayer for planning. 
    # each isolayer follows outer-inward shells, lattice segments follow zig-zag order (next func). each layer only cotains one lattice segments.
    # alter lattice type layer by layer.
    outer_print_order, outer_contour_flags, inner_print_order, print_isovalues = \
        lattice_build_graph_partition_separate_pca_rect(outer_contours, subcontour_s1, subcontour_s2, jump_threshold, 
                                                        loop_order=False, layer_loop_order=True, continuous_cnt_lim=continuous_cnt_lim)
    
    # [optional] uniform resample infill contours with same spacing
    if resample_infills:
        spacing = np.array(spacings).mean()
        inner_print_order = uniform_resampling_inner_contours(inner_print_order, spacing)

    # arrange shell and infill contours starts (infill in zig-zag order)
    print_contours, starts, start_indices, outer_inner_indicators, print_levels = \
        lattice_arrange_printing_contours_starts(outer_print_order, inner_print_order, print_isovalues)
    
    visualize_printing_contours(print_contours, start_num=1000, end_num=4000)  # vis contours in printing order
    # visualize_subcontour_starts(starts, mesh_fn=data_paras.get('mesh_path'))  # time consuming

    # [optional] a seperate function to generate partition levels as training dataset
    level_fn = os.path.join(save_path, 'print_level_dataset.pth')
    build_print_level_dataset(outer_print_order, inner_print_order, print_isovalues, level_fn)

    # build layer thickness of waypoints
    # shell thickness should be geodesic ditance (approximated with closest point distance here)
    # infill thickness can be direct euclidean distance 
    fit_function = build_calibration_thickness_wall(outer_contours, sdf_decoder, slice_decoder, fit_down_ratio=300,
                                                    object_scale=object_scale, dist_threshold=3.0, jump_threshold=jump_threshold,
                                                    continuous_cnt_lim=continuous_cnt_lim)
    # fit_function = np.poly1d([-2.33, 6.668, -6.62, 2.621])  # used result after fitting

    # generate and write waypoints file (here we remove waypoints base 3 mm)
    margin = 3.0 / object_scale
    z_rm_th = margin + np.min(np.vstack(print_contours)[:, 2])
    waypoints, indicators, level_tags = generate_waypoints_lattice(print_contours, start_indices, outer_inner_indicators, z_rm_th, print_levels)
    wayfn = os.path.join(save_path, waypt_fn)
    write_waypoints_lattice(waypoints, fit_function, slice_decoder, sdf_decoder, object_scale, 
                            indicators, wayfn, spacing, level_tags, for_collision=True)
    









