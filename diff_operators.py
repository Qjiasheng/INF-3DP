import torch
from torch.autograd import grad
import torch.nn.functional as F


def hessian(y, x):
    ''' hessian of y wrt x
    y: shape (meta_batch_size, num_observations, channels)
    x: shape (meta_batch_size, num_observations, 2)
    '''
    meta_batch_size, num_observations = y.shape[:2]
    grad_y = torch.ones_like(y[..., 0]).to(y.device)
    h = torch.zeros(meta_batch_size, num_observations, y.shape[-1], x.shape[-1], x.shape[-1]).to(y.device)
    for i in range(y.shape[-1]):
        # calculate dydx over batches for each feature value of y
        dydx = grad(y[..., i], x, grad_y, create_graph=True)[0]

        # calculate hessian on y for each x value
        for j in range(x.shape[-1]):
            h[..., i, j, :] = grad(dydx[..., j], x, grad_y, create_graph=True)[0][..., :]

    status = 0
    if torch.any(torch.isnan(h)):
        status = -1
    return h, status

def hessian3d(y, x):
    '''
    Compute the Hessian of y with respect to x (3D coordinates).
    y: shape (num_observations, channels), channels is 1
    x: shape (num_observations, 3)
    '''
    num_observations = y.shape[0]
    channels = y.shape[1]
    grad_y = torch.ones_like(y[..., 0]).to(y.device)

    h = torch.zeros(num_observations, channels, x.shape[-1], x.shape[-1]).to(y.device)
    for i in range(channels):  # Loop over channels of y
        dydx = grad(y[..., i], x, grad_y, create_graph=True)[0]
    
        for j in range(x.shape[-1]):  # Loop over the 3 coordinates (x1, x2, x3)
            # Compute the second derivative of dydx[..., j] w.r.t. the input x
            h[..., i, j, :] = grad(dydx[..., j], x, grad_y, create_graph=True)[0][..., :]
    return h  # (num_observations, channels, 3, 3)


def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)

def vector_laplace(y, x, norm=False):
    """y is a vector field, x is the coordinates batch data.
    vector laplace on scalar field gradients"""
    if norm:
        grad = gradient(y, x)
        grad_norm = torch.norm(grad, dim=1, keepdim=True)
        grad = grad / (grad_norm + 1e-6)
    else:
        grad = gradient(y, x)
    x_div = divergence(grad[..., 0:1], x)
    y_div = divergence(grad[..., 1:2], x)
    z_div = divergence(grad[..., 2:3], x)
    laplacian = torch.cat([x_div, y_div, z_div], dim=-1)
    return laplacian

def dir_vector_laplace(v, x, norm=False):
    """v is a vector field, x is the coordinates batch data"""
    if norm:
        v = F.normalize(v, p=2, dim=-1)
    x_div = divergence(v[..., 0:1], x)
    y_div = divergence(v[..., 1:2], x)
    z_div = divergence(v[..., 2:3], x)
    laplacian = torch.cat([x_div, y_div, z_div], dim=-1)
    return laplacian


def surface_vector_laplace(v, x, normals, norm=False):
    """
    \Delta F = \nabla( \nabla \cdot F) - \nabla \times (\nabla \times F) 
    approximate surface vector laplace by projection noto tangent plane. 
    rescale vector norms to total length. Direct set all vectors as unit length causing inf energy at singularities.
    see paper --- Globally Optimal Direction Fields, Felix Knöppel et al., TOG 2013.
    https://www.cs.cmu.edu/~kmcrane/Projects/GloballyOptimalDirectionFields/
    """

    def project_onto_tangent(v, normals):
        dot = torch.sum(v * normals, dim=-1, keepdim=True)  
        v_proj = v - dot * normals  
        return v_proj
    
    def surface_divergence(v, x, normals):
        v_proj = project_onto_tangent(v, normals)  # Ensure the vector field is tangential
        div = divergence(v_proj, x)  # Compute the divergence in 3D
        return div
    
    def surface_gradient(y, x, normals):
        grad = gradient(y, x)  # Full 3D gradient
        dot = torch.sum(grad * normals, dim=-1, keepdim=True)  # Component along normal
        grad_surf = grad - dot * normals  # Project onto tangent plane
        return grad_surf
    
    def normalize_proj(v):
        norms = torch.norm(v, dim=-1, keepdim=True).clamp(min=0.5)  # Compute the original norms
        total_norm = torch.sum(norms)  # Total sum of original norms
        scale = v.shape[1] / (total_norm + 1e-6)  # Compute the scaling factor to ensure the sum of lengths equals N
        v_normalized = v * scale  # Scale the vectors proportionally
        return v_normalized

    if norm:
        v = F.normalize(v, p=2, dim=-1)
   
    # \Delta F = \nabla( \nabla \cdot F) - \nabla \times (\nabla \times F) 
    # gradient, thus curl is zero. D
    v_proj = project_onto_tangent(v, normals) 
    v_proj = normalize_proj(v_proj)  # Normalize the projected vector field
    div_v = surface_divergence(v_proj, x, normals)
    laplacian = surface_gradient(div_v, x, normals)

    return laplacian.squeeze(-1), v_proj

def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def jacobian(y, x):
    ''' jacobian of y wrt x '''
    meta_batch_size, num_observations = y.shape[:2]
    jac = torch.zeros(meta_batch_size, num_observations, y.shape[-1], x.shape[-1]).to(y.device) # (meta_batch_size*num_points, 2, 2)
    for i in range(y.shape[-1]):
        # calculate dydx over batches for each feature value of y
        y_flat = y[...,i].view(-1, 1)
        jac[:, :, i, :] = grad(y_flat, x, torch.ones_like(y_flat), create_graph=True)[0]

    status = 0
    if torch.any(torch.isnan(jac)):
        status = -1

    return jac, status

def curl(y, x):
    """curl of y wrt x
    curl of gradient is always zero"""
    pass


def curvature(y, x):
    ''' mean curvature of y wrt x '''
    grad_y = gradient(y, x)
    grad_norm = torch.norm(grad_y, dim=1, keepdim=True) # shape [N, 1]
    curvature = - .5 * divergence(grad_y / (grad_norm+1e-6), x).squeeze()
    status = 0
    if torch.any(torch.isnan(curvature)):
        status = -1

    return curvature, status

def min_max_curvs_and_vectors(f, x_temp):
    '''
    Compute the minimal and maximal curvature (principal curvatures) and their direction vectors for batched inputs.
    f: function mapping x (shape: [N, 3]) to y (SDF values, shape: [N])
    x_temp: coordinates, shape: [N, 3], requires_grad=True
    Returns:
        kappa_batch: shape (N, 2), principal curvatures (kappa_min, kappa_max)
        p_batch: shape (N, 2, 3), principal direction vectors
    '''
    N = x_temp.shape[0]
    x_temp = x_temp.clone().detach().requires_grad_(True)
    output = f(x_temp)  # output is a dict 
    x_batch = output['model_in']
    y_batch = output['model_out']

    grad_y = gradient(y_batch, x_batch) # normals (N, 3)
    grad_norm = grad_y.norm(dim=1, keepdim=True)
    n_batch = grad_y / grad_norm  
    Hessian_batch = hessian3d(y_batch, x_batch).squeeze() # shape: (N, 3, 3)

    kappa_batch = torch.zeros(N, 2).to(x_batch.device)  
    p_batch = torch.zeros(N, 2, 3).to(x_batch.device)

    for i in range(N):
        n = n_batch[i]  
        Hf = Hessian_batch[i]  

        min_idx = torch.argmin(torch.abs(n))
        # Create a vector that is 1 in the dimension of the smallest component and 0 elsewhere
        a = torch.zeros_like(n)
        a[min_idx] = 1.0
        # tangent plane basis
        e1 = a - torch.dot(a, n) * n
        e1 = e1 / e1.norm()
        e2 = torch.cross(n, e1)
        # Compute Shape Operator (Weingarten map)
        W11 = -(e1 @ Hf @ e1) / grad_norm[i]
        W12 = -(e1 @ Hf @ e2) / grad_norm[i]
        W22 = -(e2 @ Hf @ e2) / grad_norm[i]
        W = torch.tensor([[W11, W12],
                          [W12, W22]], dtype=x_batch.dtype, device=x_batch.device)
        eigvals, eigvecs = torch.linalg.eigh(W)
        kappa_min, kappa_max = eigvals 
        v_min, v_max = eigvecs[:, 0], eigvecs[:, 1]

        # principal directions wrt min and max curvatures
        p_min = v_min[0] * e1 + v_min[1] * e2
        p_max = v_max[0] * e1 + v_max[1] * e2

        kappa_batch[i] = torch.stack((kappa_min, kappa_max))
        p_batch[i, 0] = p_min
        p_batch[i, 1] = p_max

    return kappa_batch, p_batch


def fndiff_min_max_curvs_and_vectors(f, x_temp, h=3e-1):
    '''
    Compute the principal curvatures and their directions for batched inputs using finite differences.
    f: function mapping x_batch (N, 3) to y_batch (N,)
    x_batch: tensor of shape (N, 3)
    h: finite difference step size, in [-1, 1] cube
    Returns:
        kappa_batch: tensor of shape (N, 2), principal curvatures (kappa_min, kappa_max)
        p_batch: tensor of shape (N, 2, 3), principal direction vectors
    '''
    N = x_temp.shape[0]
    device = x_temp.device
    dtype = x_temp.dtype

    # Compute SDF values and gradients
    output = f(x_temp)
    x_batch = output['model_in']
    y_batch = output['model_out']

    grad_y =  gradient(y_batch, x_batch)  # (N, 3)
    grad_norm = grad_y.norm(dim=1, keepdim=True)
    n_batch = grad_y / grad_norm  

    # Compute Hessians using finite differences
    H_batch = hessian_fd_batch(f, x_batch.detach(), h=h)  # Shape: (N, 3, 3)

    kappa_batch = torch.zeros(N, 2, device=device, dtype=dtype)
    p_batch = torch.zeros(N, 2, 3, device=device, dtype=dtype)

    # Compute curvatures and principal directions for each point
    for i in range(N):
        n = n_batch[i] 
        H = H_batch[i] 
        # Construct tangent plane basis
        e1, e2 = construct_tangent_basis(n)
        # Compute Shape Operator components
        W11 = -(e1 @ H @ e1) / grad_norm[i]
        W12 = -(e1 @ H @ e2) / grad_norm[i]
        W22 = -(e2 @ H @ e2) / grad_norm[i]
        # Shape Operator matrix
        W = torch.tensor([[W11, W12],
                          [W12, W22]], device=device, dtype=dtype)

        eigvals, eigvecs = torch.linalg.eigh(W)
        kappa_min, kappa_max = eigvals
        v_min, v_max = eigvecs[:, 0], eigvecs[:, 1]

        # Convert to 3D principal directions
        p_min = v_min[0] * e1 + v_min[1] * e2
        p_max = v_max[0] * e1 + v_max[1] * e2

        kappa_batch[i] = torch.stack([kappa_min, kappa_max])
        p_batch[i, 0] = p_min
        p_batch[i, 1] = p_max

    return kappa_batch, p_batch

def construct_tangent_basis(n):
    '''
    Construct an orthonormal basis (e1, e2) for the tangent plane at a point with normal vector n.
    n: normal vector of shape (3,)
    Returns:
        e1, e2: orthonormal vectors of shape (3,)
    '''
    device = n.device
    dtype = n.dtype
    # Choose the smallest component to avoid near-zero divisions
    min_idx = torch.argmin(torch.abs(n))
    a = torch.zeros(3, device=device, dtype=dtype)
    a[min_idx] = 1.0

    e1 = a - (n @ a) * n
    e1 = e1 / e1.norm()
    e2 = torch.linalg.cross(n, e1)

    return e1, e2

def hessian_fd_batch(f, x, h=3e-1):
    '''
    Compute the Hessian matrices of function f at points x using finite differences in batch.
    f: function that accepts inputs of shape (N, 3) and returns outputs of shape (N,)
    x: tensor of shape (N, 3)
    h: finite difference step size
    Returns:
        H: tensor of shape (N, 3, 3)
    '''
    N = x.shape[0]
    device = x.device
    dtype = x.dtype
    x = x.clone().detach().requires_grad_(False).to(dtype)

    # Prepare shift vectors, f(x+h), f(x-h), ... 
    shifts = []
    # Diagonal shifts (for second partial derivatives)
    for i in range(3):
        shift = torch.zeros(3, device=device, dtype=dtype)
        shift[i] = h
        shifts.append(shift)
        shift_neg = torch.zeros(3, device=device, dtype=dtype)
        shift_neg[i] = -h
        shifts.append(shift_neg)
    # Off-diagonal shifts (for mixed partial derivatives)
    for i in range(3):
        for j in range(i+1, 3):
            for s_i in [h, -h]:
                for s_j in [h, -h]:
                    shift = torch.zeros(3, device=device, dtype=dtype)
                    shift[i] = s_i
                    shift[j] = s_j
                    shifts.append(shift)
    
    shifts = torch.stack(shifts)  # Shape: (num_shifts, 3)
    num_shifts = shifts.shape[0]
    x_expanded = x.unsqueeze(1).expand(N, num_shifts, 3)  # Shape: (N, num_shifts, 3)
    shifts_expanded = shifts.unsqueeze(0).expand(N, num_shifts, 3)  # Shape: (N, num_shifts, 3)
    x_shifted = x_expanded + shifts_expanded
    x_shifted_flat = x_shifted.reshape(N * num_shifts, 3)  # Flatten to (N * num_shifts, 3)
    # Compute SDF at all shifted positions
    f_shifted_flat = f(x_shifted_flat)['model_out'].to(dtype)  # Shape: (N * num_shifts,)
    f_shifted = f_shifted_flat.reshape(N, num_shifts)  # Reshape back to (N, num_shifts)
    # Compute f at center positions
    f_center = f(x)['model_out'].to(dtype)  # Shape: (N,)
    f_center = f_center.squeeze()  # Shape: (N, 1)

    H = torch.zeros(N, 3, 3, device=device, dtype=dtype)

    # Compute diagonal elements (second partial derivatives)
    for i in range(3):
        f_forward = f_shifted[:, 2 * i]  # f(x + h * e_i)
        f_backward = f_shifted[:, 2 * i + 1]  # f(x - h * e_i)
        H[:, i, i] = (f_forward - 2 * f_center + f_backward) / (h ** 2)

    # Compute off-diagonal elements (mixed partial derivatives)
    idx = 6  # Start index for off-diagonal shifts in f_shifted
    for i in range(3):
        for j in range(i + 1, 3):
            f_pp = f_shifted[:, idx]     # f(x + h * e_i + h * e_j)
            f_pm = f_shifted[:, idx + 1] # f(x + h * e_i - h * e_j)
            f_mp = f_shifted[:, idx + 2] # f(x - h * e_i + h * e_j)
            f_mm = f_shifted[:, idx + 3] # f(x - h * e_i - h * e_j)

            H_ij = (f_pp - f_pm - f_mp + f_mm) / (4 * h ** 2)
            H[:, i, j] = H_ij
            H[:, j, i] = H_ij  # Hessian is symmetric

            idx += 4
    return H