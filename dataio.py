import numpy as np
import torch
from torch.utils.data import Dataset
import diff_operators
import torch.nn.functional as F
from utils import load_waypoints, load_nozzle_pcd
from utils import field_querying, build_level_slice_dict, build_current_level_slice


class OffSurfaceSampler:
    """Sample off-surface points with a minimum distance away from on-surface points."""

    def __init__(self, on_surface_points, off_surface_samples_ratio=10, min_distance=0.02):
        self.on_surface_points = on_surface_points
        self.off_surface_samples_ratio = off_surface_samples_ratio
        self.min_distance = min_distance

    def sample_off_surface_points(self):
        device = self.on_surface_points.device
        num_on_surface_points = self.on_surface_points.shape[0]
        num_off_surface_samples = num_on_surface_points * self.off_surface_samples_ratio

        # random uniform off-surface points within the bounding box
        off_surface_candidates = torch.empty((num_off_surface_samples, 3), device=device).uniform_(-1, 1)
        expanded_on_surface_points = self.on_surface_points.unsqueeze(1) 
        expanded_candidates = off_surface_candidates.unsqueeze(0) 
        distances = torch.cdist(expanded_on_surface_points, expanded_candidates, p=2) 
        # filter candidates that are too close 
        min_distances, _ = distances.min(dim=0)  
        valid_off_surface_points = off_surface_candidates[(min_distances > self.min_distance).squeeze()]

        return valid_off_surface_points


class PointCloud(Dataset):
    def __init__(self, pointcloud_path, on_surface_points, keep_aspect_ratio=True):
        super().__init__()

        point_cloud = np.genfromtxt(pointcloud_path)
        print(f"Finished loading {point_cloud.shape[0]} point clouds")

        coords = point_cloud[:, :3]
        self.normals = point_cloud[:, 3:6] 
        self.normals /= np.linalg.norm(self.normals, axis=1, keepdims=True)

        # Reshape point cloud such that it lies in bounding box of (-1, 1)
        coords -= np.mean(coords, axis=0, keepdims=True)
        if keep_aspect_ratio:
            coord_max = np.amax(coords)
            coord_min = np.amin(coords)
        else:
            # normalize to bbx separately on each axis to [-1,1]
            coord_max = np.amax(coords, axis=0, keepdims=True)
            coord_min = np.amin(coords, axis=0, keepdims=True)

        self.coords = (coords - coord_min) / (coord_max - coord_min)
        self.coords -= 0.5
        # self.coords *= 2.
        self.coords *= 1.95       # surface not touches box buondary

        self.on_surface_points = on_surface_points

    def __len__(self):
        return self.coords.shape[0] // self.on_surface_points

    def __getitem__(self, idx):
        point_cloud_size = self.coords.shape[0]
        # Random choose batch_size points
        rand_idcs = np.random.choice(point_cloud_size, size=self.on_surface_points)

        on_surface_coords = self.coords[rand_idcs, :]
        on_surface_normals = self.normals[rand_idcs, :]

        on_surface_coords = torch.from_numpy(on_surface_coords).float().to('cuda')
        on_surface_normals = torch.from_numpy(on_surface_normals).float().to('cuda')
        
        # off surface points. their normals not used 
        sampler = OffSurfaceSampler(on_surface_points=on_surface_coords, off_surface_samples_ratio=10, min_distance=0.02)
        off_surface_coords = sampler.sample_off_surface_points()

        off_surface_normals = torch.ones((off_surface_coords.shape[0], 3), device=on_surface_coords.device) * -1
        total_samples = on_surface_coords.shape[0] + off_surface_coords.shape[0]

        sdf = np.zeros((total_samples, 1))  # on-surface = 0
        sdf[self.on_surface_points:, :] = -1  # off-surface = -1

        coords = torch.concatenate((on_surface_coords, off_surface_coords), axis=0)
        normals = torch.concatenate((on_surface_normals, off_surface_normals), axis=0)
        # all on GPU
        return {'coords': coords}, {'sdf': torch.from_numpy(sdf).float().cuda(), 'normals': normals}



class HeightPCD(Dataset):
    """Unifotm height filed as scalar field initialization, training not uses sdf values but only normals"""
    def __init__(self, sample_number=1000, num_batches=10, ref_dir='z'):
        super().__init__()
        self.sample_number = sample_number
        self.num_batches = num_batches

        self.ref_dir = ref_dir  # 'x', 'y', 'z'

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        coords = torch.empty((self.sample_number, 3), device='cuda').uniform_(-1, 1)

        if self.ref_dir == 'z':
            # height field
            sdf = coords[:, 2].reshape(-1, 1)
            normals = torch.zeros_like(coords)
            normals[:, 2] = 1
        elif self.ref_dir == 'x':
            # x fields
            sdf = coords[:, 0].reshape(-1, 1)
            normals = torch.zeros_like(coords)
            normals[:, 0] = 1
        else:
            # y fields
            sdf = coords[:, 1].reshape(-1, 1)
            normals = torch.zeros_like(coords)
            normals[:, 1] = 1

        # model input and gt
        return {'coords': coords}, {'sdf': sdf, 'normals': normals}
    

def load_train_pcd(pcd_path, down_ratio=1):
    """load prepared pcd for training.
    coords already in [-1, 1] cube.
    all directions are unit vectors.
    NOTE heat_dirs could be all zeros if not set.
    """
    data = np.genfromtxt(pcd_path)
    coords = data[:, :3]
    normals = data[:, 3:6]
    min_dirs = data[:, 6:9]
    max_dirs = data[:, 9:12]
    heat_dirs = data[:, 12:15]
    base_tags = data[:, 15]  # 1--surface points, 0--base points

    if down_ratio > 1:
        down_ratio = int(down_ratio)
        coords = coords[::down_ratio]
        normals = normals[::down_ratio]
        min_dirs = min_dirs[::down_ratio]
        max_dirs = max_dirs[::down_ratio]
        heat_dirs = heat_dirs[::down_ratio]
        base_tags = base_tags[::down_ratio]

    return coords, normals, min_dirs, max_dirs, heat_dirs, base_tags

def load_density_pcd(pcd_path, down_ratio=1):
    """load prepared pcd for density field.
    coords already in [-1, 1] cube, clamped by model's SDF
    density values are in [0, 1].
    """
    data = np.genfromtxt(pcd_path)
    coords = data[:, :3]
    density = data[:, 3] 

    if down_ratio > 1:
        down_ratio = int(down_ratio)
        coords = coords[::down_ratio]
        density = density[::down_ratio]

    return coords, density

    
class SliceSurfPCDHeat(Dataset):
    """train surafce vector field (grad of a scalar field). using heat grad is optional.
    training data pcd has already been prepared in `test_sdf.py`, in [-1, 1] cube.
    if goes errors, pls check any part missed in data preparation.
    """
    def __init__(self, pointcloud_path, sample_number=2000, num_batches=10):
        super().__init__()
    
        self.sample_number = sample_number
        self.num_batches = num_batches
        self.device = 'cuda'

        self.coords, self.normals, self.min_dirs, self.max_dirs, self.heat_dirs, self.base_tags = load_train_pcd(pointcloud_path)
        print(f"Loaded point cloud {self.coords.shape[0]}; base points {np.sum(self.base_tags == 0)}")

        # you may adjust for less base points
        self.base_number = int((self.base_tags == 0).sum() / (self.base_tags == 1).sum() * self.sample_number * 1.5)  
        print(f'sample non-base points {self.sample_number}, base points {self.base_number}')

        self.non_base_indices = np.where(self.base_tags == 1)[0]
        self.base_indices = np.where(self.base_tags == 0)[0]
    

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        # everytime shuffle indices
        np.random.shuffle(self.non_base_indices)
        np.random.shuffle(self.base_indices)
        non_base_idx = self.non_base_indices[:self.sample_number]
        base_idx = self.base_indices[:min(self.base_number, len(self.base_indices))]  # not exceed base points

        # collect data
        total_idx = np.concatenate((non_base_idx, base_idx), axis=0)
        coords = torch.from_numpy(self.coords[total_idx]).float().to(self.device)
        normals = torch.from_numpy(self.normals[total_idx]).float().to(self.device)
        min_dirs = torch.from_numpy(self.min_dirs[total_idx]).float().to(self.device)
        heat_dirs = torch.from_numpy(self.heat_dirs[total_idx]).float().to(self.device) # may be all zeros if not set
        tags = torch.from_numpy(self.base_tags[total_idx]).float().to(self.device)  # 1--surface, 0--base

        # vis check
        # from vis import vis_pcd_fields, vis_grad
        # vis_pcd_fields(coords.cpu().numpy(), tags.cpu().numpy())
        # vis_grad(coords.cpu().numpy(), normals.cpu().numpy(), label='normals')
        # vis_grad(coords.cpu().numpy(), min_dirs.cpu().numpy(), label='min_curv_dir')
        # vis_grad(coords.cpu().numpy(), heat_dirs.cpu().numpy(), label='heat_dir')

        # All on GPU
        return {'coords': coords}, {'normals': normals, 'min_dirs': min_dirs, 'heat_dirs': heat_dirs, 'tags': tags.squeeze()}
    
    
class FinalSlicePCD(Dataset):
    """second stage to train final guidance field, pcd contains surface, inside, base pcd.
    interior is clamped by sdf eps=-0.005."""

    def __init__(self, sdf_decoder, preslice_decoder, pointcloud_path, base_infill_number=100000, sample_number=2000, num_batches=10):
        super().__init__()

        self.preslice_decoder = preslice_decoder
        self.sample_number = sample_number
        self.num_batches = num_batches
        self.device = 'cuda'

        # load surface/base pcd
        self.coords, self.normals, _, _, _, self.base_tags = load_train_pcd(pointcloud_path)
        print(f"Loaded point cloud {self.coords.shape[0]}; base points {np.sum(self.base_tags == 0)}")

        # prepare a large set of infill pcd
        base_infill_coords = torch.empty((base_infill_number, 3), device=self.device).uniform_(-1, 1)
        base_infill_coords.requires_grad = False
        base_infill_sdf = sdf_decoder(base_infill_coords)
        base_infill_sdf = base_infill_sdf['model_out']
        self.infill_coords = base_infill_coords[base_infill_sdf.squeeze() < -0.005].squeeze().cpu().numpy()  # inside points
        del base_infill_coords, base_infill_sdf  # save memory
        print(f'surface/base points {self.coords.shape[0]}, infill points {self.infill_coords.shape[0]}')


        self.surf_indices = np.where(self.base_tags == 1)[0]
        self.base_indices = np.where(self.base_tags == 0)[0]
        self.infill_indices = np.arange(self.infill_coords.shape[0], dtype=np.int32)

        # adjust sample numbers
        self.surf_number = int(self.sample_number)
        self.base_number = int((self.base_tags == 0).sum() / (self.base_tags == 1).sum() * self.surf_number * 1.5)
        self.infill_number = int(self.surf_number * 1.5)  # more inside points

    
    def __len__(self):
        return self.num_batches 

    def __getitem__(self, idx):
        """ sample from on-surface points, also set a base field"""
        # collect coords
        np.random.shuffle(self.surf_indices)
        np.random.shuffle(self.base_indices)
        np.random.shuffle(self.infill_indices)

        surf_coords = self.coords[self.surf_indices[:self.surf_number]]
        base_coords = self.coords[self.base_indices[:min(self.base_number, len(self.base_indices))]]
        infill_coords = self.infill_coords[self.infill_indices[:self.infill_number]]
        
        surf_coords = torch.from_numpy(surf_coords).float().to(self.device)
        base_coords = torch.from_numpy(base_coords).float().to(self.device)
        infill_coords = torch.from_numpy(infill_coords).float().to(self.device)

        # normals
        surf_normals = self.normals[self.surf_indices[:self.surf_number]]
        surf_normals = torch.from_numpy(surf_normals).float().to(self.device)
        base_normals = F.normalize(torch.ones_like(base_coords), dim=-1) # not used
        infill_normals = F.normalize(torch.ones_like(infill_coords), dim=-1)  # not used

        # tags. 1--surface, 0--base, 2--inside
        surf_tags = torch.ones((surf_coords.shape[0]), device=self.device)
        base_tags = torch.zeros((base_coords.shape[0]), device=self.device)
        inside_tags = torch.ones((infill_coords.shape[0]), device=self.device) * 2  

        # get surf slice grads
        samples = surf_coords.clone()
        samples.requires_grad = True
        slice_model_output = self.preslice_decoder(samples)
        model_coords = slice_model_output['model_in']
        model_slice_value = slice_model_output['model_out']
        slice_grads = diff_operators.gradient(model_slice_value, model_coords).detach()
        slice_grads_prj = slice_grads - (slice_grads * surf_normals).sum(dim=1, keepdim=True) * surf_normals # project to tangent plane
        surf_dirs = F.normalize(slice_grads_prj, dim=1, p=2)
        base_dirs = F.normalize(torch.ones_like(base_coords), dim=-1)  # not used
        infill_dirs = F.normalize(torch.ones_like(infill_coords), dim=-1)  # not used
        
        # cat all
        total_coords = torch.cat((surf_coords, base_coords, infill_coords), dim=0)
        total_normals = torch.cat((surf_normals, base_normals, infill_normals), dim=0)
        total_dirs = torch.cat((surf_dirs, base_dirs, infill_dirs), dim=0)
        total_tags = torch.cat((surf_tags, base_tags, inside_tags), dim=0)

        # vis check
        # from vis import vis_pcd_fields, vis_grad
        # vis_pcd_fields(total_coords.cpu().numpy(), total_tags.cpu().numpy())
        # vis_grad(surf_coords.cpu().numpy(), surf_dirs.cpu().numpy(), label='slice_dirs')
        
        return {'coords': total_coords}, {'normals': total_normals.squeeze().detach(),'dirs':total_dirs.squeeze().detach(), 
                                          'tags': total_tags.squeeze()}
    

class BetaInfill(Dataset):
    """infill field dataset, angle beta to control direction, see Sec. 4.3 of paper.
    infill field density (related to model volume size) is optional. if not used, set uniform density.
    We only consider uniform/ density-varying properties, not value ranges among different fields.
    -------------
    Here only example lattice infill, other infill types such porous, TPMS ... can be extended with field designs."""
    
    def __init__(self, sdf_decoder, slice_decoder, beta_degree=0, enable_density=False, 
                 density_pcd_path=None, base_infll_num=100000, sample_number=2000, num_batches=10):
        super().__init__()

        self.sdf_decoder = sdf_decoder
        self.slice_decoder = slice_decoder
        self.sample_number = sample_number
        self.num_batches = num_batches
        self.beta = torch.deg2rad(torch.tensor(beta_degree)).to('cuda')  # convert to radians

        #  prepare interior pcd for infill
        if enable_density:
            infill_coords, densities = load_density_pcd(density_pcd_path)
            print(f"Loaded pre density pcd {infill_coords.shape[0]} ....")
        else:
            base_infill_coords = torch.empty((base_infll_num, 3), device='cuda').uniform_(-1, 1)
            base_infill_coords.requires_grad = False
            base_infill_sdf = sdf_decoder(base_infill_coords)
            base_infill_sdf = base_infill_sdf['model_out']
            infill_coords = base_infill_coords[base_infill_sdf.squeeze() < -0.005].squeeze().cpu().numpy()  # inside points
            del base_infill_coords, base_infill_sdf  # save memory
            densities = np.ones((infill_coords.shape[0],), dtype=np.float32) # uniform density
            print(f'Prepared infill pcd {infill_coords.shape[0]} ....')

        self.infill_coords = infill_coords
        self.densities = densities
        self.infill_indices = np.arange(self.infill_coords.shape[0], dtype=np.int32)
       

    def __len__(self):
        return self.num_batches
    
    def __getitem__(self, idx):
        # collect coords and densities
        np.random.shuffle(self.infill_indices)
        infill_idx = self.infill_indices[:self.sample_number]
        infill_samples = torch.from_numpy(self.infill_coords[infill_idx]).float().to('cuda')
        infill_densities = torch.from_numpy(self.densities[infill_idx]).float().to('cuda')

        # set infill field direction
        samples = infill_samples.clone()
        samples.requires_grad = True
        slice_model_output = self.slice_decoder(samples)
        model_coords = slice_model_output['model_in']   
        model_slice_value = slice_model_output['model_out']
        slice_grads = diff_operators.gradient(model_slice_value, model_coords).detach()
        slice_dirs = F.normalize(slice_grads, dim=-1, p=2)
        

        # set a ref dir for orthogonal directions wrt slice_dir
        # NOTE you may adjust cross order
        ref_dir = torch.zeros_like(slice_dirs)
        ref_dir[:, 0] = 1 # (1, 0, 0)
        base_dir1 = torch.cross(slice_dirs, ref_dir, dim=1) 
        base_dir2 = torch.cross(slice_dirs, base_dir1, dim=1) 
        infill_dirs = torch.cos(self.beta) * base_dir1 + torch.sin(self.beta) * base_dir2
        infill_dirs = F.normalize(infill_dirs, dim=-1, p=2)

        # vis check
        # from vis import vis_grad, vis_pcd_fields
        # vis_pcd_fields(infill_samples.cpu().numpy(), infill_densities.cpu().numpy())
        # vis_grad(infill_samples.cpu().numpy(), infill_dirs.cpu().numpy(), label='infill_dirs')

        return {'coords': infill_samples.detach()}, {'dirs': infill_dirs.detach(), 'densities': infill_densities.detach()}


class CollisionDataset(Dataset):
    """
    collision dataset for quaternion field training.
    loading generated waypoints and nozzle pcd.
    ------------------------------------------- 
    quaternion as a rotation to transform base frame at each waypoint.
    """
    def __init__(self, slice_decoder, nozzle_pcd_path, waypoint_path, object_scale, 
                 num_classes, batch_size=30000, nozzle_downsample=1):
        super().__init__()

        self.slice_decoder = slice_decoder
        self.batch_size = batch_size
        self.num_classes = num_classes

        # load waypts and nozzle pcd
        # waypt dirs is fn_slice_grad, u can project onto tangent plane, use gt level labels
        self.waypts, self.waypt_dirs, _, self.waypt_levels = load_waypoints(waypoint_path)
        self.nozzle_pcd = load_nozzle_pcd(nozzle_pcd_path, object_scale=object_scale, down_ratio=nozzle_downsample)
        # 2.0 mm below as collision of base platform
        self.base_coll_th = self.waypts[:, 2].min().item()  - 2.0 / object_scale 

        # prepare max slices for all levels using full waypts
        waypts_full = self.waypts.copy()
        waypt_levels_full = self.waypt_levels.copy()
        slice_fields_full = field_querying(slice_decoder, waypts_full, no_print=True, return_on_cuda=False)
        self.level_slice_dict = build_level_slice_dict(slice_fields_full, torch.from_numpy(waypt_levels_full).long())  

        # random waypts as a batch
        self.num_waypts = self.waypts.shape[0]
        self.random_indices = self._generate_random_indices()

    def _generate_random_indices(self):
        indices = np.arange(self.num_waypts)
        np.random.shuffle(indices)
        return indices
    
    def __len__(self):
        return (self.num_waypts + self.batch_size - 1) // self.batch_size
    

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.num_waypts)
        batch_indices = self.random_indices[start_idx:end_idx]

        # collect batch waypts data
        waypts = torch.from_numpy(self.waypts[batch_indices]).float().to('cuda')
        waypt_dirs = torch.from_numpy(self.waypt_dirs[batch_indices]).float().to('cuda')
        waypt_levels = torch.from_numpy(self.waypt_levels[batch_indices]).long().to('cuda')
        nozzle_pcd = torch.from_numpy(self.nozzle_pcd).float().to('cuda')

        # build current level max slices
        slice_fields = field_querying(self.slice_decoder, waypts, no_print=True, return_on_cuda=True)
        current_level_maxslice = build_current_level_slice(self.level_slice_dict, waypt_levels, slice_fields) # (num_waypts, num_levels) in current waypts batch

        # vis check of waypt batch
        # from vis import vis_pcd_fields, vis_grad
        # vis_pcd_fields(nozzle_pcd.cpu().numpy(), np.zeros((nozzle_pcd.shape[0],)))
        # vis_pcd_fields(waypts.cpu().numpy(), waypt_levels.cpu().numpy())
        # vis_grad(waypts.cpu().numpy(), waypt_dirs.cpu().numpy(), label='waypt_dirs')
        
        return {'coords': waypts}, {'waypt_dirs': waypt_dirs, 'waypt_levels': waypt_levels,
                                    'current_level_maxslice': current_level_maxslice,
                                    'nozzle_pcd': nozzle_pcd,
                                    'base_coll_th': self.base_coll_th}






