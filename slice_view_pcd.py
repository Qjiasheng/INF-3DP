import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt5 import QtWidgets, QtCore
import sys

# Enable global theme to allow empty meshes (prevents crashes)
pv.global_theme.allow_empty_mesh = True

# Function to update the slice and contour plot
def update_plot(self, axis, slice_position):
    slice_position = float(slice_position)
    
    # Remove previous slice and contour for the current axis
    self.plotter.remove_actor(f'{axis}_slice', render=False)  # Avoid re-rendering each removal
    self.plotter.remove_actor(f'{axis}_contour', render=False)
    self.plotter.remove_actor(f'{axis}_zero_contour', render=False)
    
    # Choose the axis for slicing
    if axis == 'x':
        slice = grid.slice(normal=(1, 0, 0), origin=(slice_position, 0, 0))
    elif axis == 'y':
        slice = grid.slice(normal=(0, 1, 0), origin=(0, slice_position, 0))
    elif axis == 'z':
        slice = grid.slice(normal=(0, 0, 1), origin=(0, 0, slice_position))
    
    # Add the new slice to the plot
    if slice.n_points > 0:  # Safeguard against empty slices
        # self.plotter.add_mesh(slice, name=f'{axis}_slice', cmap="bwr", n_colors=256)
        self.plotter.add_mesh(slice, name=f'{axis}_slice', cmap="Spectral", n_colors=24)

    # Generate iso-contours for the slice
    contour_values = np.arange(-1.5, 1.5, 0.03)  # Reduced number of contours

    # Generate iso-contours for the slice and ensure it has valid points
    contours = slice.contour(isosurfaces=contour_values)
    zero_contour = slice.contour(isosurfaces=[0.0])  # show zero level contour on slice

    if contours.n_points > 0:  # Check if the contour has valid points before plotting
        # self.plotter.add_mesh(contours, name=f'{axis}_contour', color="black", line_width=3)
        pass
    if zero_contour.n_points > 0:
        # self.plotter.add_mesh(zero_contour, name=f'{axis}_zero_contour', color="black", line_width=10)
        pass
    
    self.plotter.render()

# PyQt Application setup
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Volume Cross-section Viewer")

        # Create the PyVista plotter
        self.plotter = QtInteractor(self)
        self.setCentralWidget(self.plotter.interactor)

        # Create a dock widget for the sidebar
        dock = QtWidgets.QDockWidget("Slice & Bounding Box Controls", self)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)

        # Create a widget to house the controls
        self.controls = QtWidgets.QWidget()
        dock.setWidget(self.controls)
        layout = QtWidgets.QVBoxLayout()

        # Add group box for slice controls
        slice_group_box = QtWidgets.QGroupBox("Slice Controls")
        slice_layout = QtWidgets.QVBoxLayout()

        # Create slice sliders for X, Y, Z cross-section adjustment
        self.x_slice_label = QtWidgets.QLabel("X Slice: 0.00")
        self.x_slider = self.create_slider('X Slice', N, slice_layout)
        self.x_slider.valueChanged.connect(self.on_x_slider_moved)

        self.y_slice_label = QtWidgets.QLabel("Y Slice: 0.00")
        self.y_slider = self.create_slider('Y Slice', N, slice_layout)
        self.y_slider.valueChanged.connect(self.on_y_slider_moved)

        self.z_slice_label = QtWidgets.QLabel("Z Slice: 0.00")
        self.z_slider = self.create_slider('Z Slice', N, slice_layout)
        self.z_slider.valueChanged.connect(self.on_z_slider_moved)

        slice_group_box.setLayout(slice_layout)
        layout.addWidget(slice_group_box)

        # Add group box for bounding box controls
        bbx_group_box = QtWidgets.QGroupBox("Bounding Box Controls")
        bbx_layout = QtWidgets.QVBoxLayout()

        # Create bounding box sliders for X, Y, Z axes
        self.x1_slider, self.x1_label = self.create_bbx_slider('X1', N, bbx_layout)
        self.x1_slider.valueChanged.connect(self.on_bbx_slider_moved)

        self.x2_slider, self.x2_label = self.create_bbx_slider('X2', N, bbx_layout)
        self.x2_slider.valueChanged.connect(self.on_bbx_slider_moved)

        self.y1_slider, self.y1_label = self.create_bbx_slider('Y1', N, bbx_layout)
        self.y1_slider.valueChanged.connect(self.on_bbx_slider_moved)

        self.y2_slider, self.y2_label = self.create_bbx_slider('Y2', N, bbx_layout)
        self.y2_slider.valueChanged.connect(self.on_bbx_slider_moved)

        self.z1_slider, self.z1_label = self.create_bbx_slider('Z1', N, bbx_layout)
        self.z1_slider.valueChanged.connect(self.on_bbx_slider_moved)

        self.z2_slider, self.z2_label = self.create_bbx_slider('Z2', N, bbx_layout)
        self.z2_slider.valueChanged.connect(self.on_bbx_slider_moved)

        bbx_group_box.setLayout(bbx_layout)
        layout.addWidget(bbx_group_box)

        # Add a toggle button for point cloud visibility
        self.toggle_pcd_button = QtWidgets.QPushButton("Toggle Point Cloud")
        layout.addWidget(self.toggle_pcd_button)
        self.toggle_pcd_button.clicked.connect(self.toggle_pcd_visibility)

        self.controls.setLayout(layout)

        # Show the first slice
        self.update_x_slice(N // 2)

        # Show the coordinate axes
        self.plotter.show_axes()

        # Load and add the point cloud (PCD)
        self.pcd_visible = True
        self.load_and_add_pcd()

    def create_slider(self, label_text, max_value, layout):
        """Helper to create a slider with a label."""
        label = QtWidgets.QLabel(label_text)
        layout.addWidget(label)
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(max_value - 1)
        slider.setValue(max_value // 2)
        layout.addWidget(slider)
        return slider

    def create_bbx_slider(self, label_text, max_value, layout):
        """Helper to create a bounding box slider with a label and a value display."""
        label = QtWidgets.QLabel(f'{label_text}: 0.00')
        layout.addWidget(label)
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(max_value - 1)
        slider.setValue(max_value // 2)
        layout.addWidget(slider)
        return slider, label


    def on_x_slider_moved(self):
        """Update the X cross-section when the slider is moved."""
        x_value = self.x_slider.value()
        slice_position = voxel_grid_origin[0] + voxel_size * x_value
        self.x_slice_label.setText(f'X Slice: {slice_position:.2f}')
        self.update_x_slice(x_value)

    def on_y_slider_moved(self):
        """Update the Y cross-section when the slider is moved."""
        y_value = self.y_slider.value()
        slice_position = voxel_grid_origin[1] + voxel_size * y_value
        self.y_slice_label.setText(f'Y Slice: {slice_position:.2f}')
        self.update_y_slice(y_value)

    def on_z_slider_moved(self):
        """Update the Z cross-section when the slider is moved."""
        z_value = self.z_slider.value()
        slice_position = voxel_grid_origin[2] + voxel_size * z_value
        self.z_slice_label.setText(f'Z Slice: {slice_position:.2f}')
        self.update_z_slice(z_value)

    def update_x_slice(self, value):
        """Update the plot with the X slice."""
        slice_position = voxel_grid_origin[0] + voxel_size * value
        update_plot(self, 'x', slice_position)

    def update_y_slice(self, value):
        """Update the plot with the Y slice."""
        slice_position = voxel_grid_origin[1] + voxel_size * value
        update_plot(self, 'y', slice_position)

    def update_z_slice(self, value):
        """Update the plot with the Z slice."""
        slice_position = voxel_grid_origin[2] + voxel_size * value
        update_plot(self, 'z', slice_position)

    def load_and_add_pcd(self):
        """Load the point cloud from a .xyz file and add it to the plot."""
        # NOTE following same normalize params as in training
        pcd_data = np.loadtxt(pcd_fn)  # Assuming .xyz format: [x, y, z]
        coords = pcd_data[:, :3]

        if scale_coords:
            coords -= np.mean(coords, axis=0, keepdims=True)
            coord_max = np.amax(coords)
            coord_min = np.amin(coords)

            coords = (coords - coord_min) / (coord_max - coord_min)
            coords -= 0.5
            coords *= 1.95  # Scale so that the surface doesn't touch the box boundary

        self.coords = coords  # Store the coordinates for easy access
        # Initial update of the point cloud color based on the bounding box
        self.update_pcd_color()

    def update_pcd_color(self):
        """Update the point cloud color based on the bounding box."""
        # Get bounding box limits from sliders
        x1 = voxel_grid_origin[0] + voxel_size * self.x1_slider.value()
        x2 = voxel_grid_origin[0] + voxel_size * self.x2_slider.value()
        y1 = voxel_grid_origin[1] + voxel_size * self.y1_slider.value()
        y2 = voxel_grid_origin[1] + voxel_size * self.y2_slider.value()
        z1 = voxel_grid_origin[2] + voxel_size * self.z1_slider.value()
        z2 = voxel_grid_origin[2] + voxel_size * self.z2_slider.value()

        # Update slider labels with the current positions (formatted to 2 decimal places)
        self.x1_label.setText(f'X1: {x1:.2f}')
        self.x2_label.setText(f'X2: {x2:.2f}')
        self.y1_label.setText(f'Y1: {y1:.2f}')
        self.y2_label.setText(f'Y2: {y2:.2f}')
        self.z1_label.setText(f'Z1: {z1:.2f}')
        self.z2_label.setText(f'Z2: {z2:.2f}')

        # Update point cloud colors based on bounding box
        coords = self.coords
        colors = np.zeros((coords.shape[0], 3))  # Prepare an array for RGB colors

        # Points within the bounding box will be red, others will be gray
        inside_bbx = (
            (coords[:, 0] >= x1) & (coords[:, 0] <= x2) &
            (coords[:, 1] >= y1) & (coords[:, 1] <= y2) &
            (coords[:, 2] >= z1) & (coords[:, 2] <= z2)
        )
        colors[inside_bbx] = [1, 0, 0]  # Red for points inside the bounding box
        colors[~inside_bbx] = [0.5, 0.5, 0.5]  # Gray for points outside the bounding box

        # Update the point cloud with the new colors
        self.pcd = pv.PolyData(coords)
        self.last_point = pv.PolyData(coords[-1, ...].reshape(1, 3))
        self.pcd['colors'] = colors  # Add the color array to the PolyData

        # Re-add the point cloud to the plot with updated colors
        # self.plotter.add_mesh(self.pcd, scalars="colors", rgb=True, point_size=5.0, name="point_cloud", reset_camera=False)
        # last point is red with larger size
        self.plotter.add_mesh(self.pcd, scalars="colors", rgb=True, point_size=8.0, name="point_cloud", reset_camera=False,
                              render_points_as_spheres=True)
        self.plotter.add_mesh(self.last_point, color='red', point_size=50.0, name="last", reset_camera=False, 
                              render_points_as_spheres=True)
        self.plotter.render()

    def on_bbx_slider_moved(self):
        """Update the bounding box and point cloud colors when a bounding box slider is moved."""
        self.update_pcd_color()

    def toggle_pcd_visibility(self):
        """Toggle the visibility of the point cloud."""
        self.pcd_visible = not self.pcd_visible
        if self.pcd_visible:
            self.plotter.add_mesh(self.pcd, color="gray", point_size=5.0, name="point_cloud")
        else:
            self.plotter.remove_actor("point_cloud")
        self.plotter.render()

# PyQt Application Main Loop
if __name__ == "__main__":
    # Load the SDF data from a .npz file
    global pcd_fn
    global scale_coords

    # -----------------------  bunny
    scale_coords = True
    sdf_volume_fn = './logs/sdf/bunnyz/sdf_volume.npz'  # sdf
    # sdf_volume_fn = './logs/levels/bunny_single_shell/level_volume.npz'  # single shell levels
    # sdf_volume_fn = './logs/levels/bunny_infill/level_volume.npz'  # infill
    pcd_fn = './sdf_data/bunnyz.xyz'

    # ------ tvsdf
    # scale_coords = False
    # sdf_volume_fn = './logs/collision/bunny_single_shell/tvsdf_volume.npz'  # tvsdf
    # pcd_fn = './logs/collision/bunny_single_shell/till_waypts.xyz'

    # ----------------------- fertility
    # scale_coords = True
    # sdf_volume_fn = './logs/sdf/fertilityz/sdf_volume.npz'
    # sdf_volume_fn = './logs/levels/fer_infill/level_volume.npz' 
    # pcd_fn = './data/fertilityz.xyz'

    # ------ tvsdf
    # scale_coords = False
    # sdf_volume_fn = './logs/collision/fer_single_shell/tvsdf_volume.npz'  # tvsdf
    # pcd_fn = './logs/collision/fer_single_shell/till_waypts.xyz'

    data = np.load(sdf_volume_fn)
    sdf_values = data['sdf']
    voxel_grid_origin = data['voxel_grid_origin']
    voxel_size = data['voxel_size']
    N = sdf_values.shape[0]

    # Create a 3D grid of voxel positions
    x = np.linspace(voxel_grid_origin[0], voxel_grid_origin[0] + voxel_size * (N - 1), N)
    y = np.linspace(voxel_grid_origin[1], voxel_grid_origin[1] + voxel_size * (N - 1), N)
    z = np.linspace(voxel_grid_origin[2], voxel_grid_origin[2] + voxel_size * (N - 1), N)
    Z, Y, X = np.meshgrid(x, y, z, indexing="ij")
    points = np.c_[X.flatten(), Y.flatten(), Z.flatten()]

    # Create structured grid for PyVista
    grid = pv.StructuredGrid()
    grid.points = points
    grid.dimensions = [N, N, N]
    grid["SDF"] = sdf_values.flatten(order="F")  # Fortran-style flattening

    # Setup the PyQt application
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()

    # Start the event loop
    sys.exit(app.exec_())