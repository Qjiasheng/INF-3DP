import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import modules, utils
import configargparse
import shutil
import slice_toolpath

p = configargparse.ArgumentParser()
p.add_argument('--experiment_name', type=str, required=True,
                help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')
p.add_argument('--config_path', default=None, help='config file for training')
opt = p.parse_args()

# General options in config file
configs = utils.load_configs(opt.config_path, opt.experiment_name)
data_paras, optim_paras, toolpath_paras, toolpath_type = configs['dataset'], configs['optim'], configs['parameters'], configs['type']

class NNDecoder(torch.nn.Module):
    def __init__(self, checkpont_path):
        super().__init__()
        self.model = modules.SingleBVPNet(type=optim_paras.get('model_type'), final_layer_factor=1, in_features=3)
        self.model.load_state_dict(torch.load(checkpont_path))
        self.model.cuda()

    def forward(self, coords):
        model_in = {'coords': coords}
        return self.model(model_in)
    
sdf_decoder = NNDecoder(data_paras.get('sdf_checkpoint_path'))
slice_decoder = NNDecoder(data_paras.get('slice_checkpoint_path'))


save_path = os.path.join(data_paras.get('save_path'), opt.experiment_name)
if os.path.exists(save_path):
    val = input("The model directory %s exists. Overwrite? (y/n)"%save_path)
    if val == 'y':
        shutil.rmtree(save_path)
    else:
        sys.exit()

if not os.path.exists(save_path):
    os.makedirs(save_path)
utils.put_configs(os.path.join(data_paras.get('save_path'), opt.experiment_name, 'vol_toolpath_configs.yaml'), configs) 


# toolpath generation including three applications 

# ------------------- 1 single shell surface toolpath generation
if toolpath_type == 'single_shell':
    slice_toolpath.single_shell_toolpath_gen(sdf_decoder, slice_decoder, data_paras, toolpath_paras, save_path)

# ------------------- 2 multiple shell surface by sdf levels
if toolpath_type == 'wall_shell':
    slice_toolpath.wall_shell_toolpath_gen(sdf_decoder, slice_decoder, data_paras, toolpath_paras, save_path)

# ------------------- 3 lattice infill or full solid volume 
if toolpath_type == 'lattice':
    lattice1_decoder = NNDecoder(data_paras.get('lattice1_checkpoint_path'))
    lattice2_decoder = NNDecoder(data_paras.get('lattice2_checkpoint_path'))
    slice_toolpath.lattice_toolpath_gen(sdf_decoder, slice_decoder, lattice1_decoder, lattice2_decoder, 
                                        data_paras, toolpath_paras, save_path, resample_infills=True)
    


