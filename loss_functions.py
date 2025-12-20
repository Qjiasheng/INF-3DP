import torch
import torch.nn.functional as F
import diff_operators
from utils import transform_nozzle_pcd_with_quaternion, quaternion_para_axes, field_query_with_grads
from level_collision import collision_evaluation
from vis import vis_print_nozzle_pcd_comparison

def height(model_output, gt, epoch, epochs, params):

    gt_sdf = gt['sdf']  # not use gt sdf values
    gt_normals = gt['normals']

    coords = model_output['model_in']
    pred_sdf = model_output['model_out']

    # field gradient align directions
    gradient = diff_operators.gradient(pred_sdf, coords)
    normal_constraint = 1 - F.cosine_similarity(gradient, gt_normals, dim=-1)[..., None]
    grad_constraint = torch.abs(gradient.norm(dim=-1) - 1)

    # add harmonic energy
    laplace = diff_operators.laplace(pred_sdf, coords)

    # values not outside [-1, 1]
    delta_lower, delta_upper = torch.clamp(-1 - pred_sdf, min=0), torch.clamp(pred_sdf - 1, min=0)
    sdf_pen = delta_lower + delta_upper

    return {'sdf_pen': sdf_pen.mean() * params.get('sdf_pen'), 
            'normal': normal_constraint.mean() * params.get('normal'), 
            'grad': grad_constraint.mean() * params.get('grad'),  
            'laplace': torch.abs(laplace).mean() * params.get('laplace')} 


def sdf(model_output, gt, epoch, epochs, params):

    gt_sdf = gt['sdf']
    gt_normals = gt['normals']

    coords = model_output['model_in']
    pred_sdf = model_output['model_out']

    gradient = diff_operators.gradient(pred_sdf, coords)

    # Wherever boundary_values is not equal to zero, we interpret it as a boundary constraint.
    sdf_constraint = torch.where(gt_sdf != -1, pred_sdf, torch.zeros_like(pred_sdf))
    inter_constraint = torch.where(gt_sdf != -1, torch.zeros_like(pred_sdf), torch.exp(-1e2 * torch.abs(pred_sdf)))
    normal_constraint = torch.where(gt_sdf != -1, 1 - F.cosine_similarity(gradient, gt_normals, dim=-1)[..., None],
                                    torch.zeros_like(gradient[..., :1]))
    grad_constraint = torch.abs(gradient.norm(dim=-1) - 1)

    if epoch >= epochs // 2.5:
        grad_wight = params.get('grad_init')
    else:
        grad_wight = params.get('grad_decay')

    # add harmonic energy, with decay weights
    if epoch >= epochs // 20 and epoch < epochs // 2:
        laplace_constraint = diff_operators.laplace(pred_sdf, coords)
        laplace = torch.abs(laplace_constraint).mean()
        laplace_weight = params.get('laplace_init') - (epoch // (epochs // 10)) * 0.1
    else:
        laplace = torch.tensor(0.0)
        laplace_weight = 0.0

    return {'sdf': torch.abs(sdf_constraint).mean() * params.get('sdf'), 
            'inter': inter_constraint.mean() * params.get('inter'), 
            'normal': normal_constraint.mean() * params.get('normal'),  
            'grad': grad_constraint.mean() * grad_wight, 
            'laplace': laplace * laplace_weight} 


def init_slice_field(model_output, gt, epoch, epochs, params):
    """first stage for guidance field training. vec field on surface"""

    use_heat_grad = params.get('use_heat_grad', False)
    normals = gt['normals']
    min_dirs = gt['min_dirs']
    heat_dirs = gt['heat_dirs']
    tags = gt['tags']  # 0-base, 1-surface

    coords = model_output['model_in']
    slice_fields = model_output['model_out']

    # vector laplace (care about directions not norms)
    slice_grad = diff_operators.gradient(slice_fields, coords)
    slice_vec_laplace, _ = diff_operators.surface_vector_laplace(slice_grad, coords, normals=normals, norm=True)
    laplace_constraint = torch.abs(slice_vec_laplace.norm(dim=-1)) ** 2

    # manifold grad norm 1
    slice_grad_prj = slice_grad - torch.sum(slice_grad * normals, dim=-1, keepdim=True) * normals
    grad_constraint = torch.abs(slice_grad_prj.norm(dim=-1)-1) ** 2

    # base pcd gradient up to z
    base_grad = slice_grad[tags == 0]
    base_grad = F.normalize(base_grad, dim=-1)
    base_grad_constraint = torch.abs(base_grad[..., 2] - 1)

    # set guidance direction (min curv dirs) aligned
    slice_prj_dir = F.normalize(slice_grad_prj, dim=-1)
    align_loss_temp = (torch.norm(torch.cross(slice_prj_dir, min_dirs, dim=-1), dim=-1)) **2 
    align_loss = torch.where(tags == 1, align_loss_temp, torch.zeros_like(align_loss_temp))

    # align with heat dir (optional)
    if use_heat_grad:
        heat_align_loss_temp = F.cosine_similarity(slice_prj_dir, heat_dirs, dim=-1) ** 2
        heat_align_loss = torch.where(tags == 1, heat_align_loss_temp, torch.zeros_like(heat_align_loss_temp))
    else:
        heat_align_loss = torch.tensor(0.0).cuda()

    # lie on tangent plane
    print_dir = F.normalize(slice_grad, dim=-1)
    tangent_loss_temp = torch.abs(torch.sum(print_dir * normals, dim=-1)) ** 2
    tangent_loss = torch.where(tags == 1, tangent_loss_temp, torch.zeros_like(tangent_loss_temp))

    return {'grad': grad_constraint.mean() * params.get('grad'),  
            'base_grad': base_grad_constraint.mean() * params.get('base_grad'), 
            'laplace': laplace_constraint.mean() * params.get('laplace'),
            'curv_align_loss': align_loss.mean() * params.get('curv_align'), 
            'heat_align_loss': heat_align_loss.mean() * params.get('heat_align'),
            'tangent': tangent_loss.mean() * params.get('tangent')}



def final_slice_field(model_output, gt, epoch, epochs, paras):
    """
    second stage for final guidance feild training, for surface, base, inside points.
    vec laplace lift to 3D space, interior is singularity-free.
    NOTE not necessary to set Support-free loss on surface points, since further collision-stage will set it.
    """
    gt_normals = gt['normals']
    gt_dirs = gt['dirs']  # surf dir already projected to tangent plane
    gt_tags = gt['tags']  # 0-base, 1-surface, 2-inside
    
    coords = model_output['model_in']
    slice_fields = model_output['model_out']

    slice_grad = diff_operators.gradient(slice_fields, coords)
    slice_veclap = diff_operators.vector_laplace(slice_fields, coords).squeeze(-1) 
    
    # vec lap in 3d space, not surface manifold. For all pcd
    veclap_loss = torch.abs(slice_veclap.norm(dim=-1)) ** 2
 
    # Surface grad_len close to 1. MUST on surface -- project to tangent plane
    slice_grad_prj = slice_grad - torch.sum(slice_grad * gt_normals, dim=-1, keepdim=True) * gt_normals
    surf_grad_len = torch.abs(slice_grad_prj.norm(dim=-1)-1) ** 2
    surf_grad_len_loss = torch.where(gt_tags == 1, surf_grad_len, torch.zeros_like(surf_grad_len))

    # inside grad_len close to 1. no projection
    inside_grad_len = torch.abs(slice_grad.norm(dim=-1)-1) ** 2
    inside_grad_len_loss = torch.where(gt_tags == 2, inside_grad_len, torch.zeros_like(inside_grad_len))

    # surface dirs align with gt dirs
    slice_grad_dir = F.normalize(slice_grad, dim=-1)
    grad_dir_loss = (1 - F.cosine_similarity(slice_grad_dir, gt_dirs, dim=-1)) ** 2
    surf_dir_loss = torch.where(gt_tags == 1, grad_dir_loss, torch.zeros_like(grad_dir_loss))

    # whatever base fields not be curved
    base_grad = slice_grad[gt_tags == 0]
    base_grad_dir = F.normalize(base_grad, dim=-1)
    base_dir_loss = torch.abs(base_grad_dir[..., 2] - 1.0)

    # inside hessian F norm zero
    hessian_weight = paras.get('hessian', 0.0)
    if hessian_weight > 0.0:
        slice_hessian = diff_operators.hessian(slice_fields, coords)[0].squeeze(2)
        hessian = slice_hessian[gt_tags != 0]
        hessian_loss = torch.sum(hessian ** 2, dim=(1, 2))
    else:
        hessian_loss = torch.tensor(0.0).cuda()

    return {'surf_grad_len': surf_grad_len_loss.mean() * paras.get('surf_grad_len'), 
            'inside_grad_len': inside_grad_len_loss.mean() * paras.get('inside_grad_len'),
            'surf_dir': surf_dir_loss.mean() * paras.get('surf_dir'), 
            'base_dir': base_dir_loss.mean() * paras.get('base_dir'), 
            'veclap': veclap_loss.mean() * paras.get('veclap'),
            'hessian': hessian_loss.mean() * paras.get('hessian')}  


def infill_field(model_output, gt, epoch, epochs, paras):
    """loss for infill field training, with density if needed.
    direction controls field gradient flow, density controls gradient lengths
    """
    infill_dirs = gt['dirs']
    densities = gt['densities']

    coords = model_output['model_in']
    fields = model_output['model_out']

    grads = diff_operators.gradient(fields, coords)
    scalar_laplace = diff_operators.laplace(fields, coords)
    vec_laplace = diff_operators.vector_laplace(fields, coords, norm=True).squeeze(-1)

    grad_dir = F.normalize(grads, dim=-1)
    dir_loss = 1 - F.cosine_similarity(grad_dir, infill_dirs, dim=-1)[..., None]

    density_constraint = torch.abs(grads.norm(dim=-1) - densities)
    len_loss = density_constraint ** 2 / (densities + 1e-6)  # small density -> larger penalty

    scalar_laplace_loss = torch.abs(scalar_laplace)
    veclap_loss = torch.abs(vec_laplace.norm(dim=-1)) 

    return {'dir': dir_loss.mean() * paras.get('dir'),
            'len': len_loss.mean() * paras.get('len'),
            'scalar_lap': scalar_laplace_loss.mean() * paras.get('scalar_lap'),
            'vec_lap': veclap_loss.mean() * paras.get('vec_lap')}



def cal_sf_loss_(normal, print_dir, alpha_degree=45, expansion=3e1):
    """inner/ outer surface are sf"""
    sin_alpha = torch.sin(torch.deg2rad(torch.tensor(alpha_degree)))
    inner_product = torch.sum(normal * print_dir, dim=-1)
    sf_term1 = torch.abs(inner_product) - sin_alpha
    sf_loss_term = torch.sigmoid(expansion * sf_term1)

    # compute sf ratio, [45, 135] is sf
    angles = torch.acos(torch.clamp(inner_product, -1.0, 1.0))
    sf_mask = (angles >= torch.deg2rad(torch.tensor(alpha_degree))) & (angles <= torch.deg2rad(torch.tensor(180 - alpha_degree)))

    return sf_loss_term, sf_mask

def collision_loss(model_output, gt, epoch, epochs, paras):
    """
    collision loss for quaternion field. 
    quaternion as rotation transform local base frame to frame at waypoint.
    smoothness added on its paramterized print and frame axes.
    with SF constraint on [surface] non-base waypoints.
    --------------
    infill waypts not has this SF loss, usrs can set tags in dataio.
    [NOTE] usrs may adjust coll, sf ... weights along training.
    """
    # collect all decoders
    decoders = paras.get('decoders')
    sdf_decoder = decoders.get('sdf_decoder')
    slice_decoder = decoders.get('slice_decoder')
    level_decoder = decoders.get('level_decoder')

    num_classes = paras.get('num_classes')
    base_threshold = paras.get('base_threshold') # base waypts no sf loss

    # collect gts. all squeeze first dim for my use
    waypt_dirs = gt['waypt_dirs'].squeeze()
    waypt_levels = gt['waypt_levels'].squeeze()
    current_level_maxslice = gt['current_level_maxslice'].squeeze()
    nozzle_pcd = gt['nozzle_pcd'].squeeze() # (m, 3)
    base_coll_th = gt['base_coll_th'].squeeze() 

    # model output
    quaternions = model_output['model_out']  # (1, n, 4)
    waypts = model_output['model_in']  # (1, n, 3)

    # quaternion must be normalized, smoothness on para axes
    quaternions = F.normalize(quaternions, dim=-1).squeeze() 
    print_dirs, frame_dirs = quaternion_para_axes(quaternions)  
    print_dir_vec_lap = diff_operators.dir_vector_laplace(print_dirs, waypts, norm=True).squeeze()
    frame_dir_vec_lap = diff_operators.dir_vector_laplace(frame_dirs, waypts, norm=True).squeeze()

    if epoch < epochs // 10:
        # warm-up stage, set print dirs align with original waypt dirs. and smoothness
        # [optional] also set frame dirs align with a ref dir during warmup, e.g., z axis \cross print dir
        dir_align_loss = (1 - F.cosine_similarity(print_dirs, waypt_dirs, dim=-1)) ** 2
        losses = {
            'dir_align': dir_align_loss.mean() * paras.get('dir_align'),
            'print_vec_lap': print_dir_vec_lap.norm(dim=-1).mean() * paras.get('print_vec_lap_warm'), 
            'frame_vec_lap': frame_dir_vec_lap.norm(dim=-1).mean() * paras.get('print_vec_lap_warm')
        }
        metrics = {'collision_rate': 100.0, 'sf_rate': 100.0}   # 100 means during warm-up no collision check
        return losses, metrics

    # collision evaluation, gradients to quaternions, detach waypts
    waypts_cp = waypts.clone().detach().squeeze() 
    transed_nozzle_pcd = transform_nozzle_pcd_with_quaternion(nozzle_pcd, waypts_cp, quaternions)  # (n, m, 3)

    # vis check transed nozzle pcd after warm up
    # see_index = 0  # random pick one
    # vis_print_nozzle_pcd_comparison(raw_pcd=transed_nozzle_pcd.detach().cpu().numpy()[see_index, ...], 
    #                                 waypoints=waypts_cp.detach().cpu().numpy(), add_cube=True)
    
    nozzle_pcd_coll_depths, _, waypt_coll_mask = collision_evaluation(
        transed_nozzle_pcd, sdf_decoder, slice_decoder, level_decoder, num_classes, current_level_maxslice, waypt_levels, base_th=base_coll_th)
    coll_loss = nozzle_pcd_coll_depths.sum()  # total number as a weight

    # support-free loss on surface waypts (not base)
    _, _, waypt_normals = field_query_with_grads(sdf_decoder, waypts_cp, no_print=True, return_on_cuda=True, max_batch=64 ** 3)
    waypt_normals = F.normalize(waypt_normals, dim=-1)
    sf_loss_term, sf_mask = cal_sf_loss_(waypt_normals, print_dirs, alpha_degree=45, expansion=3e1)

    non_base_mask = waypts_cp[..., 2] > base_threshold
    sf_loss = torch.where(non_base_mask, sf_loss_term, torch.zeros_like(sf_loss_term))
    sf_mask = sf_mask | (~non_base_mask) # base no sf 

    losses = {'coll': coll_loss * paras.get('coll'), 
        'sf': sf_loss.mean() * paras.get('sf'), 
        'print_vec_lap': print_dir_vec_lap.norm(dim=-1).mean() * paras.get('print_vec_lap'), 
        'frame_vec_lap': frame_dir_vec_lap.norm(dim=-1).mean() * paras.get('frame_vec_lap')
    }

    # collision ratio for metrics
    coll_ratio = waypt_coll_mask.float().mean().item() * 100.0
    sf_ratio = sf_mask.float().mean().item() * 100.0
    metrics = {'collision_rate': coll_ratio, 'sf_rate': sf_ratio}

    return losses, metrics





