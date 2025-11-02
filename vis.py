
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv


def vis_grad(coords, grads, arrow_scale=0.05, label=None):
    """all np arrays.
    segments show dir, length shows its magnitude"""

    grad_norms = np.linalg.norm(grads, axis=1)

    normalized_grads = np.where(grad_norms[:, np.newaxis] > 0, grads / grad_norms[:, np.newaxis], 0)
    scaled_grads = normalized_grads * grad_norms[:, np.newaxis] * arrow_scale
    
    end_points = coords + scaled_grads
    points = np.vstack([coords, end_points])  # Combine start and end points into one array
    lines = np.hstack([[2, i, i + len(coords)] for i in range(len(coords))])  # Connectivity for line segments
    
    poly_lines = pv.PolyData()
    poly_lines.points = points
    poly_lines.lines = lines  
    plotter = pv.Plotter()
    plotter.add_points(coords, color="grey", point_size=7, render_points_as_spheres=True)
    plotter.add_mesh(poly_lines, color="blue", line_width=2)
    if label is not None:
        plotter.add_text(label, position='upper_left', font_size=18, color='black')  
    plotter.show_axes()
    plotter.show()


def hist_item(arr):
    plt.hist(arr, bins=100)
    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.title("Histogram of Values")
    plt.grid()
    plt.show()


def vis_pcd_fields(coords, fields, label=None, cmap="seismic"):
    point_cloud = pv.PolyData(coords)
    point_cloud["intensity"] = fields
    plotter = pv.Plotter()
    plotter.add_mesh(point_cloud, scalars="intensity", point_size=10, render_points_as_spheres=True, cmap=cmap)
    if label is not None:
        plotter.add_text(label, position='upper_left', font_size=18, color='black')
    plotter.show_axes()
    plotter.show()  



def vis_print_nozzle_pcd_comparison(raw_pcd=None, raw_grads=None, waypoints=None, add_cube=False, pcd=None, grads=None,
                                    raw_mask=None, mask=None, mag=0.1):
    """
    vis raw nozzle pcd and transformed nozzle pcd at waypoints.
    """
    
    plotter = pv.Plotter()
    raw_cloud = pv.PolyData(raw_pcd)
    plotter.add_mesh(raw_cloud, color="blue", point_size=5, render_points_as_spheres=True)
    if pcd is not None:
        cloud = pv.PolyData(pcd)
        plotter.add_mesh(cloud, color="green", point_size=5, render_points_as_spheres=True)

    if waypoints is not None:
        waypoints_cloud = pv.PolyData(waypoints)
        plotter.add_mesh(waypoints_cloud, color="gray", point_size=5, render_points_as_spheres=True)
        # num_waypoints = waypoints.shape[0]
        # waypoint_scalars = np.linspace(0, 1, num_waypoints) 
        # waypoints_cloud = pv.PolyData(waypoints)
        # waypoints_cloud["scalars"] = waypoint_scalars 
        # plotter.add_mesh(
        #     waypoints_cloud, cmap="coolwarm", scalars="scalars",
        #     point_size=8, render_points_as_spheres=True, label="Waypoints"
        # )

    if add_cube:
        cube = pv.Cube(center=(0, 0, 0), x_length=2, y_length=2, z_length=2)
        plotter.add_mesh(cube, color="grey", opacity=0.1, show_edges=True)

    # for collision vis, use grads for mask
    if raw_pcd is not None and raw_grads is not None:
        grad_norms = np.linalg.norm(raw_grads, axis=1)
        coll_mask = grad_norms > 1e-6
        coll_raw_nozzle_pcd = raw_pcd[coll_mask]
        coll_raw_nozzle_grads = raw_grads[coll_mask]
        noozle_pcd = pv.PolyData(coll_raw_nozzle_pcd)
        plotter.add_mesh(noozle_pcd, color="red", point_size=8, render_points_as_spheres=True)
        plotter.add_arrows(coll_raw_nozzle_pcd, coll_raw_nozzle_grads, color="orange", mag=mag, show_scalar_bar=False)

    if pcd is not None and grads is not None:
        grad_norms = np.linalg.norm(grads, axis=1)
        coll_mask = grad_norms > 1e-6
        coll_nozzle_pcd = pcd[coll_mask]
        coll_nozzle_grads = grads[coll_mask]
        noozle_pcd = pv.PolyData(coll_nozzle_pcd)
        plotter.add_mesh(noozle_pcd, color="red", point_size=8, render_points_as_spheres=True)
        plotter.add_arrows(coll_nozzle_pcd, coll_nozzle_grads, color="orange", mag=mag, show_scalar_bar=False)
    
    if raw_pcd is not None and raw_mask is not None:
        point_cloud = pv.PolyData(raw_pcd)
        colors = np.zeros((raw_pcd.shape[0], 3)) 
        colors[raw_mask] = [1, 0, 0]  # in collision
        colors[~raw_mask] = [0, 0, 1]
        point_cloud["colors"] = colors
        plotter.add_mesh(point_cloud, scalars="colors", rgb=True, point_size=8, render_points_as_spheres=True, show_scalar_bar=False)
    
    if pcd is not None and mask is not None:
        point_cloud = pv.PolyData(pcd)
        colors = np.zeros((pcd.shape[0], 3)) 
        colors[mask] = [1, 0, 0]
        colors[~mask] = [0, 1, 0]
        point_cloud["colors"] = colors
        plotter.add_mesh(point_cloud, scalars="colors", rgb=True, point_size=8, render_points_as_spheres=True, show_scalar_bar=False)

    plotter.show_axes()
    plotter.show()


def vis_print_nozzle_mesh_comparison(raw_mesh=None, mesh=None, waypoints=None, add_cube=False):
    """
    vis raw nozzle mesh and transformed nozzle mesh at waypoints.
    """
    
    plotter = pv.Plotter()
    if raw_mesh is not None:
        plotter.add_mesh(raw_mesh, color='whitesmoke', opacity=0.5, 
        show_edges=False, metallic=1.0, roughness=0.3, specular=0.5, specular_power=50)
    if mesh is not None:
        plotter.add_mesh(mesh, color='whitesmoke', opacity=1.0, 
        show_edges=False, metallic=1.0, roughness=0.3, specular=0.5, specular_power=50)

    if waypoints is not None:
        # waypoints_cloud = pv.PolyData(waypoints)
        # plotter.add_mesh(waypoints_cloud, color="gray", point_size=5, render_points_as_spheres=True)
        num_waypoints = waypoints.shape[0]
        waypoint_scalars = np.linspace(0, 1, num_waypoints) 
        waypoints_cloud = pv.PolyData(waypoints)
        waypoints_cloud["scalars"] = waypoint_scalars 
        plotter.add_mesh(
            waypoints_cloud, cmap="coolwarm", scalars="scalars",
            point_size=8, render_points_as_spheres=True, label="Waypoints", show_scalar_bar=False
        )

    if add_cube:
        cube = pv.Cube(center=(0, 0, 0), x_length=2, y_length=2, z_length=2)
        plotter.add_mesh(cube, color="grey", opacity=0.1, show_edges=True)

    plotter.show_axes()
    plotter.show()
