# this script contains all visualization utils for slicing and toolpath generation
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import random
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.spatial.transform import Rotation as R

def random_color():
    return [random.random(), random.random(), random.random()]

def vis_isolayers(iso_layers):
    """visualize iso-layers"""
    plotter = pv.Plotter()
    for iso_value, (verts, faces) in iso_layers.items():
        # Convert vertices and faces to a PyVista-compatible mesh
        mesh = pv.PolyData(verts, np.c_[np.full(len(faces), 3), faces])
        plotter.add_mesh(mesh, color=random_color(), opacity=0.5, show_edges=False)
    plotter.show()


def visualize_merged_iso_contours_mul(contours, start=None, end=None, line_width=7, random_color=True):
    """ Visualize merged iso-contours.
    Handles multiple contours (>1 sublists) for each iso-value. """

    def get_random_color(index, seed=0):
        random.seed(seed + index)  # combine seed with index for reproducibility
        return (random.random(), random.random(), random.random())

    sorted_keys = sorted(contours.keys())  
    if start is None:
        start = 0  # Default first index
    if end is None:
        end = len(sorted_keys)  # Default last index
    color_range = np.linspace(1, len(sorted_keys) , len(sorted_keys)  - 1) 

    plotter = pv.Plotter()
    # for iso_value in sliced_keys:
    for idx, iso_value in enumerate(sorted_keys):
        points = contours[iso_value]  
        if not points:
            continue
        if random_color:
            color = get_random_color(index=idx)  
        else: 
            color = get_value_color(color_range, idx, colormap='seismic')
        for contour in points:
            contour = np.array(contour) if isinstance(contour, list) else contour
            if len(contour) < 2:
                continue
            lines = pv.lines_from_points(contour)
            if idx >= start and idx <= end:
                plotter.add_mesh(lines, color=color, line_width=line_width)
            else:
                plotter.add_mesh(lines, color='lightgray', line_width=5, opacity=0.2)
    plotter.add_axes()
    # set_view(plotter)
    plotter.show()


def hist_spacing_distance(contours, smooth_contours):

    original_distances = calculate_spacing_distances(contours)
    smooth_distances = calculate_spacing_distances(smooth_contours)

    plt.figure(figsize=(12, 6))

    # Histogram of original distances
    plt.subplot(1, 2, 1)
    plt.hist(original_distances, bins=30, color='blue', alpha=0.7, edgecolor='black')
    plt.title('Histogram of Spacing Distances (Original)')
    plt.xlabel('Spacing Distance')
    plt.ylabel('Frequency')

    # Histogram of resampled distances
    plt.subplot(1, 2, 2)
    plt.hist(smooth_distances, bins=30, color='green', alpha=0.7, edgecolor='black')
    plt.title('Histogram of Spacing Distances (Resampled)')
    plt.xlabel('Spacing Distance')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()


def calculate_spacing_distances(contours):
    distances = []
    for iso_value, subcontours in contours.items():
        for subcontour in subcontours:
            points = np.array(subcontour)
            if len(points) > 1:  # Only compute distances if there are at least two points
                pairwise_distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
                distances.extend(pairwise_distances)
    return distances

def get_value_color(values, value, colormap='seismic'):
    norm = Normalize(vmin=min(values), vmax=max(values))
    cmap = cm.get_cmap(colormap)
    color = cmap(norm(value))[:3]  # Extract RGB values
    return color


def visualize_printing_contours(contours, start_num=None, end_num=None):
    """ Visualize iso-contours for printing order.
    contours: list of subcontours (in the expected printing order)."""
    if start_num is None:
        start_num = 0
    if end_num is None:
        end_num = len(contours)
    color_range = np.linspace(start_num+1, end_num, end_num-start_num)
    plotter = pv.Plotter()
    for i, contour in enumerate(contours[start_num:end_num]):
        if len(contour) < 2:
            continue
        points_np = np.array(contour)
        lines = pv.lines_from_points(points_np)
        # color = random_color()
        color = get_value_color(color_range, i+start_num, colormap='seismic')
        plotter.add_mesh(lines, color=color, line_width=5)
    set_view(plotter)
    plotter.add_axes()
    plotter.show()


def visualize_subcontour_starts(starts, mesh_fn=None):
    """starts are start point within each subcontour"""
    num_points = len(starts)
    order_normalized = np.linspace(0, 1, num_points)

    # Map normalized order to colors using a colormap
    colormap = cm.get_cmap("seismic")  
    colors = (colormap(order_normalized)[:, :3]) 
    plotter = pv.Plotter()
    
    for i, point in enumerate(starts):
        point_cloud = pv.PolyData(np.array(point))
        plotter.add_mesh(
            point_cloud,
            color=colors[i],
            point_size=7,
            render_points_as_spheres=True,
            # label=f"Iso {iso_value:.2f}"
        )
    if mesh_fn is not None:
        mesh = pv.read(mesh_fn)
        plotter.add_mesh(mesh, color='lightgrey', opacity=0.6, show_edges=False)

    plotter.add_axes()
    plotter.show()


def vis_two_set_pcd(pcd1, pcd2):

    cloud1 = pv.PolyData(pcd1)  
    cloud2 = pv.PolyData(pcd2)  
    plotter = pv.Plotter()

    plotter.add_mesh(cloud1, color='gray', point_size=10, render_points_as_spheres=True)
    plotter.add_mesh(cloud2, color='red', point_size=40, render_points_as_spheres=True)
    plotter.add_axes()  
    plotter.show()


def vis_fitting_results(norms, dists, fit_function, down_ratio=10):
    dists = dists[::down_ratio]
    norms = norms[::down_ratio]

    plt.figure(figsize=(6, 6))
    plt.scatter(norms, dists, alpha=0.5, s=10, label='thickness X gradient norm', color='royalblue')

    # Fit a curve to the data
    x = np.linspace(min(norms), max(norms), 1000)
    y = fit_function(x)
    
    plt.plot(x, y, 'r--', linewidth=3.0)
    plt.xticks(fontsize=18)
    plt.ylim(0.0, 3.0)
    plt.yticks(fontsize=18)
    # plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(fontsize=18)
    ax = plt.gca()  # Get the current axes
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    ax.tick_params(axis='both', width=3, length=8)

    plt.show()

def vis_intersections_dict(intersection_dict):
    """Visualize intersection points as a point cloud."""
    plotter = pv.Plotter()
    intersections = np.array(list(intersection_dict.values()))  # Shape: (num_points, 3)
    plotter.add_points(intersections, color='blue', point_size=5.0, render_points_as_spheres=True)
    plotter.show()


def visualize_lattice_subcontours(subcontours_s1, start_layer=None, end_layer=None, colormap='seismic', line_width=5):
    plotter = pv.Plotter()

    slices = sorted(subcontours_s1.keys())
    start_layer = 0 if start_layer is None else start_layer
    end_layer = len(slices) if end_layer is None else end_layer
    slices_to_visualize = slices[start_layer:end_layer]

    color_range = np.linspace(0, len(slices_to_visualize), len(slices_to_visualize))

    for slice_idx, slice_iso in enumerate(slices_to_visualize):
        lattice1_data = subcontours_s1[slice_iso]
        for lattice1_iso, segments in lattice1_data.items():
            for segment in segments:
                if segment.shape[0] < 2:  # Skip segments with fewer than 2 points
                    continue
                lines = pv.lines_from_points(segment)
                color = get_value_color(color_range, slice_idx, colormap=colormap)
                plotter.add_mesh(lines, color=color, line_width=line_width)
    plotter.add_axes()
    plotter.show()


def set_view(plotter):
    """
    Set the camera view for a PyVista plotter.
    """
    # # Calculate the camera position based on azimuth and elevation
    azimuth, elevation, radius = 20, 60, 4.0  # bunny, fertility
    x = radius * np.sin(np.radians(azimuth)) * np.cos(np.radians(elevation))
    y = radius * np.cos(np.radians(azimuth)) * np.cos(np.radians(elevation))
    z = radius * np.sin(np.radians(elevation))
    plotter.camera.parallel_projection = False
    plotter.camera_position = [(x, y, z), (0, 0, 0), (0, 0, 1)]  # [camera position, focus point, view-up]


