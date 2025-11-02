# train slice lattice field (three orthogonal fields) for infill space

import torch
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, meta_modules, utils, training, loss_functions, modules

from torch.utils.data import DataLoader
import configargparse
import shutil

# set seed
# utils.set_seed(124)
utils.set_seed(1024)

p = configargparse.ArgumentParser()
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')
p.add_argument('--config_path', default=None, help='config file for training.')
opt = p.parse_args()

# General training options in config file
configs = utils.load_configs(opt.config_path, opt.experiment_name)
data_paras, optim_paras, loss_paras = configs['dataset'], configs['optim'], configs['parameters']

class NNDecoder(torch.nn.Module):
    def __init__(self, checkpont_path):
        super().__init__()
        # Define the model.
        self.model = modules.SingleBVPNet(type=optim_paras.get('model_type'), final_layer_factor=1, in_features=3)
        self.model.load_state_dict(torch.load(checkpont_path))
        self.model.cuda()

    def forward(self, coords):
        model_in = {'coords': coords}
        return self.model(model_in)
    
sdf_decoder = NNDecoder(data_paras.get('sdf_checkpoint_path'))
slice_decoder = NNDecoder(data_paras.get('slice_checkpoint_path'))

infill_dataset = dataio.BetaInfill(sdf_decoder=sdf_decoder, slice_decoder=slice_decoder, 
                                   beta_degree=data_paras.get('beta_degree'),
                                   enable_density=data_paras.get('enable_density'),
                                   density_pcd_path=data_paras.get('infill_density_pcd_path'),
                                   sample_number=data_paras.get('batch_size'), 
                                   num_batches=data_paras.get('num_batches'))

dataloader = DataLoader(infill_dataset, shuffle=True, batch_size=1, pin_memory=False, num_workers=0)

# next(iter(dataloader))

# lattice fields still use x, y fields as initialization
if optim_paras.get('model_type') == 'nerf':
    lattice_model = modules.SingleBVPNet(type='relu', mode='nerf', in_features=3)
else:
    lattice_model = modules.SingleBVPNet(type=optim_paras.get('model_type'), in_features=3)

lattice_model.load_state_dict(torch.load(data_paras.get('init_checkpoint_path')))
lattice_model.cuda()
lattice_model.train()


loss_fn = loss_functions.infill_field
summary_fn = utils.write_sdf_summary

save_path = os.path.join(data_paras.get('save_path'), opt.experiment_name)
if os.path.exists(save_path):
    val = input("The model directory %s exists. Overwrite? (y/n)"%save_path)
    if val == 'y':
        shutil.rmtree(save_path)

os.makedirs(save_path)
utils.put_configs(os.path.join(data_paras.get('save_path'), opt.experiment_name, 'train_configs.yaml'), configs) # save to prj folder

# training
training.train(model=lattice_model, train_dataloader=dataloader, epochs=optim_paras.get('num_epochs'), lr=optim_paras.get('lr'),
               steps_til_summary=optim_paras.get('steps_til_summary'), epochs_til_checkpoint=optim_paras.get('epochs_til_ckpt'),
               model_dir=save_path, loss_fn=loss_fn, summary_fn=summary_fn, double_precision=False,
               clip_grad=True, loss_paras=loss_paras)