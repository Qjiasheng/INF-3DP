# train a height filed for initialization
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
dataset_paras, optim_paras, loss_paras = configs['dataset'], configs['optim'], configs['parameters']

height_dataset = dataio.HeightPCD(sample_number=dataset_paras.get('batch_size'), 
                                  num_batches=dataset_paras.get('num_batches'),
                                  ref_dir=dataset_paras.get('ref_dir'))
dataloader = DataLoader(height_dataset, shuffle=True, batch_size=1, pin_memory=False, num_workers=0)

# Define the model.
if optim_paras.get('model_type') == 'nerf':
    model = modules.SingleBVPNet(type='relu', mode='nerf', in_features=3)
else:
    model = modules.SingleBVPNet(type=optim_paras.get('model_type'), in_features=3)
model.cuda()

# Define the loss
loss_fn = loss_functions.height
summary_fn = utils.write_sdf_summary

save_path = os.path.join(dataset_paras.get('save_path'), opt.experiment_name)
if os.path.exists(save_path):
    val = input("The model directory %s exists. Overwrite? (y/n)"%save_path)
    if val == 'y':
        shutil.rmtree(save_path)

os.makedirs(save_path)
utils.put_configs(os.path.join(dataset_paras.get('save_path'), opt.experiment_name, 'train_configs.yaml'), configs) # save to prj folder

training.train(model=model, train_dataloader=dataloader, epochs=optim_paras.get('num_epochs'), lr=optim_paras.get('lr'),
               steps_til_summary=optim_paras.get('steps_til_summary'), epochs_til_checkpoint=optim_paras.get('epochs_til_ckpt'),
               model_dir=save_path, loss_fn=loss_fn, summary_fn=summary_fn, double_precision=False,
               clip_grad=True, loss_paras=loss_paras)
