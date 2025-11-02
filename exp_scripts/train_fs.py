# script to use pre-trained SDF and heightfield as initialisation for a scalar field training
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import modules, utils
import configargparse
import dataio, training
from torch.utils.data import DataLoader
import loss_functions
import shutil

p = configargparse.ArgumentParser()


p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')
p.add_argument('--config_path', default=None, help='config file for training.')
opt = p.parse_args()

# General training options in config file
configs = utils.load_configs(opt.config_path, opt.experiment_name)
dataset_paras, optim_paras, loss_paras = configs['dataset'], configs['optim'], configs['parameters']

# trained SDF 
class SliceDecoder(torch.nn.Module):
    def __init__(self, checkpont_path):
        super().__init__()
        # Define the model.
        self.model = modules.SingleBVPNet(type=optim_paras.get('model_type'), final_layer_factor=1, in_features=3)
        self.model.load_state_dict(torch.load(checkpont_path))
        self.model.cuda()

    def forward(self, coords):
        model_in = {'coords': coords}
        return self.model(model_in)

slice_decoder = SliceDecoder(dataset_paras.get('init_checkpoint_path'))

# dataset contains surface, inside, base regions
sdf_decoder = SliceDecoder(dataset_paras.get('sdf_checkpoint_path'))
slice_dataset = dataio.FinalSlicePCD(sdf_decoder=sdf_decoder, preslice_decoder=slice_decoder, 
                                             pointcloud_path=dataset_paras.get('point_cloud_path'),
                                             sample_number=dataset_paras.get('batch_size'), 
                                             num_batches=dataset_paras.get('num_batches'))

dataloader = DataLoader(slice_dataset, shuffle=True, batch_size=1, pin_memory=False, num_workers=0)

# batch_data = next(iter(dataloader))  # for testing, return list, batch_data[0]--coords, batch_data[1]--tags

# slice network using height field initialisation
if optim_paras.get('model_type') == 'nerf':
    slice_model = modules.SingleBVPNet(type='relu', mode='nerf', in_features=3)
else:
    slice_model = modules.SingleBVPNet(type=optim_paras.get('model_type'), in_features=3)
slice_model.load_state_dict(torch.load(dataset_paras.get('init_checkpoint_path')))
slice_model.cuda()
slice_model.train()


# define loss function
loss_fn = loss_functions.final_slice_field
summary_fn = utils.write_sdf_summary

save_path = os.path.join(dataset_paras.get('save_path'), opt.experiment_name)
if os.path.exists(save_path):
    val = input("The model directory %s exists. Overwrite? (y/n)"%save_path)
    if val == 'y':
        shutil.rmtree(save_path)

os.makedirs(save_path)
utils.put_configs(os.path.join(dataset_paras.get('save_path'), opt.experiment_name, 'train_configs.yaml'), configs) # save to prj folder
#  training
training.train(model=slice_model, train_dataloader=dataloader, epochs=optim_paras.get('num_epochs'), lr=optim_paras.get('lr'),
               steps_til_summary=optim_paras.get('steps_til_summary'), epochs_til_checkpoint=optim_paras.get('epochs_til_ckpt'),
               model_dir=save_path, loss_fn=loss_fn, summary_fn=summary_fn, double_precision=False,
               clip_grad=True, loss_paras=loss_paras)


