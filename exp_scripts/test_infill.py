import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import modules, utils
import configargparse
import slice_meshing


p = configargparse.ArgumentParser()

p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')
p.add_argument('--config_path', default=None, help='config file for training')
opt = p.parse_args()

configs = utils.load_configs(opt.config_path, opt.experiment_name)
dataset_paras, optim_paras, loss_paras = configs['dataset'], configs['optim'], configs['parameters']

class NNDecoder(torch.nn.Module):
    def __init__(self, checkpont_path):
        super().__init__()
        # Define the model.
        self.model = modules.SingleBVPNet(type=optim_paras.get('model_type'), final_layer_factor=1, in_features=3)
        self.model.load_state_dict(torch.load(checkpont_path))
        self.model.cuda()

    def forward(self, coords):
        model_in = {'coords': coords}
        return self.model(model_in) # dict

sdf_decoder = NNDecoder(dataset_paras.get('sdf_checkpoint_path'))

# slice field decoder
infill_decoder = NNDecoder(dataset_paras.get('infill_checkpoint_path'))
save_path = os.path.join(dataset_paras.get('save_path'), opt.experiment_name)

# slice field volume query 
slice_meshing.query_volume_field(infill_decoder, save_path=save_path, N=dataset_paras.get('resolution'))

# show interior iso-layers clamped by sdf
slice_meshing.show_inside_isolayers(sdf_decoder, infill_decoder, num_layers=30, N=dataset_paras.get('resolution'))

