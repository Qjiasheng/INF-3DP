import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import modules, utils
import configargparse
from train_levels import MLPClassifier
import level_collision

p = configargparse.ArgumentParser()
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')
p.add_argument('--config_path', default=None, help='config file for training.')
opt = p.parse_args()

# General training options in config file
configs = utils.load_configs(opt.config_path, opt.experiment_name)
dataset_paras, optim_paras, loss_paras = configs['dataset'], configs['optim'], configs['parameters']

class NNDecoder(torch.nn.Module):
    def __init__(self, checkpont_path):
        super().__init__()
        # Define the model.
        self.model = modules.SingleBVPNet(type=optim_paras.get('model_type'), out_features=1, in_features=3)
        self.model.load_state_dict(torch.load(checkpont_path))
        self.model.cuda()

    def forward(self, coords):
        model_in = {'coords': coords}
        return self.model(model_in)
    
class LevelDecoder(torch.nn.Module):
    def __init__(self, checkpont_path, num_classes=3):
        super().__init__()
        self.model = MLPClassifier(num_classes=num_classes)
        self.model.load_state_dict(torch.load(checkpont_path))
        self.model.cuda()
        print("---- Level classifier loaded.")

    def forward(self, x):
        return self.model(x)

class QuatDecoder(torch.nn.Module):
    def __init__(self, checkpont_path):
        super().__init__()
        # Define the model.
        self.model = modules.SingleBVPNet(type=optim_paras.get('model_type'), out_features=4, in_features=3)
        self.model.load_state_dict(torch.load(checkpont_path))
        self.model.cuda()

    def forward(self, coords):
        model_in = {'coords': coords}
        return self.model(model_in)

# prepare decoders
sdf_decoder = NNDecoder(dataset_paras.get('sdf_checkpoint_path'))
slice_decoder = NNDecoder(dataset_paras.get('slice_checkpoint_path'))
quat_decoder = QuatDecoder(dataset_paras.get('collision_checkpoint_path'))

dataset = torch.load(dataset_paras.get('level_dataset_path'))
num_classes = dataset["level_labels"].max().item() + 1  # get number of classes of original dataset
level_decoder = LevelDecoder(dataset_paras.get('level_checkpoint_path'), num_classes=num_classes)

save_path = os.path.join(dataset_paras.get('save_path'), opt.experiment_name)

# collision test applications
# waypts collision test after optimization
level_collision.quat_collision_test(sdf_decoder, slice_decoder, level_decoder, quat_decoder, num_classes, dataset_paras, loss_paras, save_path, waypt_down_ratio=2)

# write collision-free waypoints in physical scale
level_collision.write_cf_waypoints(quat_decoder, dataset_paras, loss_paras, save_path)
