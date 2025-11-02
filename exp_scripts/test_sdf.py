import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import modules, utils
import sdf_meshing
import configargparse

p = configargparse.ArgumentParser()
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')
p.add_argument('--config_path', default=None, help='config file for training')
opt = p.parse_args()

# General training options in config file
configs = utils.load_configs(opt.config_path, opt.experiment_name)
dataset_paras, optim_paras, loss_paras = configs['dataset'], configs['optim'], configs['parameters']


class SDFDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define the model.
        self.model = modules.SingleBVPNet(type=optim_paras.get('model_type'), final_layer_factor=1, in_features=3)
        self.model.load_state_dict(torch.load(dataset_paras.get('sdf_checkpoint_path')))
        self.model.cuda()

    def forward(self, coords):
        model_in = {'coords': coords}
        # return self.model(model_in)['model_out']
        return self.model(model_in)

sdf_decoder = SDFDecoder()

save_path = os.path.join(dataset_paras.get('save_path'), opt.experiment_name)
utils.cond_mkdir(save_path)

# extract mesh from SDF and save SDF volume for vis
sdf_meshing.create_mesh_volume(sdf_decoder, save_path=save_path, N=dataset_paras.get('resolution'))

# prepare datasets for futher fields training. 
# Typically, curv directions are used to train init guidance field.
# [optional] heat field here is optional, also provided as a separate .xyz
sdf_meshing.pcd_field_with_curv(sdf_decoder, dataset_paras, heat_field=True)

# generate density field with infill pcd (clamped by sdf), only use for varying-density infill
sdf_meshing.gen_density_field(sdf_decoder, dataset_paras, save_path)