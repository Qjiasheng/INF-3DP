import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import modules, utils
import configargparse
import dataio, training
from torch.utils.data import DataLoader
from train_levels import MLPClassifier
import loss_functions
import shutil


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
    

# prepare decoders
sdf_decoder = NNDecoder(data_paras.get('sdf_checkpoint_path'))
slice_decoder = NNDecoder(data_paras.get('slice_checkpoint_path'))

level_dataset = torch.load(data_paras.get('level_dataset_path'))
num_classes = level_dataset["level_labels"].max().item() + 1  # get number of classes of original dataset
level_decoder = LevelDecoder(data_paras.get('level_checkpoint_path'), num_classes=num_classes)

# dataset
collision_dataset = dataio.CollisionDataset(slice_decoder=slice_decoder,
                                            nozzle_pcd_path=data_paras.get('nozzle_pcd_path'),
                                            waypoint_path=data_paras.get('waypoint_path'),
                                            object_scale=loss_paras.get('object_scale'),
                                            num_classes=num_classes,
                                            batch_size=data_paras.get('batch_size'),
                                            nozzle_downsample=30)
dataloader = DataLoader(collision_dataset, shuffle=True, batch_size=1, pin_memory=False, num_workers=0)

# next(iter(dataloader))

# Quaternion field for collision, out_features = 4 
if optim_paras.get('model_type') == 'nerf':
    quater_model = modules.SingleBVPNet(type='relu', mode='nerf', in_features=3, out_features=4)
else:
    quater_model = modules.SingleBVPNet(type=optim_paras.get('model_type'), in_features=3, out_features=4)

quater_model.cuda()
quater_model.train()


# Define the loss
loss_fn = loss_functions.collision_loss
summary_fn = utils.write_collision_summary


save_path = os.path.join(data_paras.get('save_path'), opt.experiment_name)
if os.path.exists(save_path):
    val = input("The model directory %s exists. Overwrite? (y/n)"%save_path)
    if val == 'y':
        shutil.rmtree(save_path)

os.makedirs(save_path)
utils.put_configs(os.path.join(data_paras.get('save_path'), opt.experiment_name, 'trainconfigs.yaml'), configs) 

# put decoders into loss paras passing
# waypts under base th not in SF loss
loss_paras['decoders'] = {'sdf_decoder': sdf_decoder, 'slice_decoder': slice_decoder, 'level_decoder': level_decoder}
loss_paras.update({'num_classes': num_classes, 'base_threshold': data_paras.get('base_threshold')})

training.train(model=quater_model, train_dataloader=dataloader, epochs=optim_paras.get('num_epochs'), lr=optim_paras.get('lr'),
               steps_til_summary=optim_paras.get('steps_til_summary'), epochs_til_checkpoint=optim_paras.get('epochs_til_ckpt'),
               model_dir=save_path, loss_fn=loss_fn, summary_fn=summary_fn, double_precision=False,
               clip_grad=True, loss_paras=loss_paras, collision_training=True)


