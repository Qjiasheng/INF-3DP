import numpy as np
import torch
import diff_operators
import torch.nn.functional as F

from vis import *
import os

from dataio import load_train_pcd
from sdf_meshing import gen_voxle_queries
from utils import gen_iso_layers, field_querying
from toolpath_vis import vis_isolayers


def query_pcd_field(decoder, point_cloud_path=None, max_batch=64 ** 3, offset=None, scale=None):
    """query pcd in slice field, use data pcd (already normalized to unit cube)"""

    # load pcd 
    coords, normals, _, _, _, _ = load_train_pcd(point_cloud_path)

    # querying slice field 
    decoder.eval()
    samples = torch.from_numpy(coords).float().cuda()
    samples.requires_grad = True  
    
    model_output = decoder(samples)
    coords = model_output['model_in']
    fields = model_output['model_out']
    grads = diff_operators.gradient(fields, coords)
   
    coords = coords.detach().cpu().numpy()
    fields = fields.detach().cpu().numpy()
    grads = grads.detach().cpu().numpy()
    grads_tangent = grads - np.sum(grads * normals, axis=1)[:, np.newaxis] * normals

    # vis slice field and gradients
    vis_pcd_fields(coords, fields)
    vis_grad(coords, grads, label='slice_grads')
    vis_grad(coords, grads_tangent, label='slice_grads_tangent')

    # show tangent norms (for uniform layer thickness)
    grads_tangent_norm = np.linalg.norm(grads_tangent, axis=-1)
    hist_item(grads_tangent_norm)
    vis_pcd_fields(coords, grads_tangent_norm)



def show_inside_isolayers(sdf_decoder, slice_decoder, num_layers=60, N=256, max_batch=32 ** 3):

    sdf_decoder.eval()
    slice_decoder.eval()

    samples, voxel_origin, voxel_size = gen_voxle_queries(N=N, cube_size=1.0)
    samples.requires_grad = False 
    sdf_fields = field_querying(sdf_decoder, samples, max_batch=max_batch, no_print=False, return_on_cuda=False)
    slice_fields = field_querying(slice_decoder, samples, max_batch=max_batch, no_print=False, return_on_cuda=False)

    sdf_values = sdf_fields.reshape(N, N, N)
    slice_values = slice_fields.reshape(N, N, N)

    # generate iso-layers interior 
    inside_mask = sdf_values < 0
    slice_iso_layers = gen_iso_layers(slice_values, num_layers, inside_mask, N)
    vis_isolayers(slice_iso_layers)


def query_volume_field(decoder, save_path, N=256, max_batch=32**3):
    """query scalar field on volume grid, save for vis volume"""

    slice_volume_fn = os.path.join(save_path, 'slice_volume')
    decoder.eval()
    # the voxel_origin is actually the (bottom, left, down) corner, not the middle
    samples, voxel_origin, voxel_size = gen_voxle_queries(N, cube_size=1.0)
    samples.requires_grad = False 
    fields = field_querying(decoder, samples, max_batch=max_batch, no_print=False, return_on_cuda=False)
    slice_values = fields.reshape(N, N, N)

    slice_values_np = slice_values.data.cpu().numpy()
    np.savez(slice_volume_fn+'.npz', sdf=slice_values_np, voxel_grid_origin=voxel_origin, voxel_size=voxel_size)
    print(f"sdf volume saved to {slice_volume_fn+'.npz'}" )


def streamline_on_surface(decoder, dataset_paras, num_seeds=2000, show_mesh=True):
    """tracing streamlines on surface, base regions not cared"""

    mesh_path = dataset_paras.get('mesh_path')  # already extracted during sdf meshing
    pv_mesh = pv.read(mesh_path)

    verts = pv_mesh.points
    normals = pv_mesh.point_normals
    samples = torch.from_numpy(verts).float().cuda()
    normals = F.normalize(torch.from_numpy(normals).float().cuda(), dim=1)
    samples.requires_grad = True
    decoder.eval()
    model_output = decoder(samples)
    coords = model_output['model_in']
    fields = model_output['model_out']
    grads = diff_operators.gradient(fields, coords).detach()

    grads_prj = grads - torch.sum(grads * normals, dim=1, keepdim=True) * normals
    grads_prj = F.normalize(grads_prj, dim=1).cpu().numpy()

    # tracing streamlines
    seed_indices = np.random.choice(verts.shape[0], num_seeds, replace=False)
    seed_points = pv_mesh.points[seed_indices]

    pv_mesh["gradients"] = grads_prj
    streamlines = pv_mesh.streamlines_from_source(
        pv.PolyData(seed_points),       # Seed points for the streamlines
        vectors="gradients",            # Use the gradient field as the vector field
        integrator_type=2,              # Runge-Kutta 4/5 integrator
        integration_direction="both",   # Streamlines flow in both forward and backward directions
        surface_streamlines=True,       # Constrain streamlines to the surface
        max_time=.15,                   # Maximum length of streamlines (adjust as needed)
        initial_step_length=0.5,        # Initial step size for streamline integration
        terminal_speed=1e-6,            # Stop integration when the vector magnitude is very small
        compute_vorticity=False,        # Compute vorticity (optional)
    )

    n_streamlines = streamlines.n_cells  
    random_colors = np.random.rand(n_streamlines, 3)  
    random_colors = (random_colors * 255).astype(np.uint8)  
    streamlines.cell_data["colors"] = random_colors

    plotter = pv.Plotter()
    tube_streamlines = streamlines.tube(radius=0.002, n_sides=0.5) 
    plotter.add_mesh(
        tube_streamlines,
        scalars="colors", 
        rgb=True,         
        # line_width=4.0,
        opacity=1.0,      
        render=True,
        # render_lines_as_tubes=True, 
        lighting=True,
        specular=0.5,
        specular_power=50,
    )
    if show_mesh:
        plotter.add_mesh(pv_mesh, color='lightgrey', opacity=1.0, show_edges=False)

    plotter.enable_anti_aliasing('msaa')
    plotter.show_axes()
    plotter.show()








