#  this script use model print level dataset as classification task
# to partition the space, where original contour near two levels are hyperplanes for separation.

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import utils
import configargparse
from torch.utils.data import DataLoader
import shutil
from torch.utils.data import Dataset
from tqdm.autonotebook import tqdm


class LevelDataset(Dataset):
    def __init__(self, preloaded_data):
        self.slice_isovalues = preloaded_data["slice_isovalues"]
        self.xyz = preloaded_data["xyz"]
        self.level_labels = preloaded_data["level_labels"]

    def __len__(self):
        return len(self.level_labels)

    def __getitem__(self, idx):
        input_data = torch.cat([self.xyz[idx], self.slice_isovalues[idx].unsqueeze(0)])  # (x, y, z, isovalue)
        label = self.level_labels[idx]
        return {"input": input_data, "label": label}
    

class MLPClassifier(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=256, num_classes=3):
        """
        Multi-Layer Perceptron for classification.
        input (xyz + slice_isovalue).
        """
        super(MLPClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.network(x)
    

def train_model(dataset, num_classes, batch_size=2000, epochs=20, learning_rate=1e-3):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = MLPClassifier(input_dim=4, hidden_dim=256, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training loop
    with tqdm(total=epochs) as pbar:
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0

            for batch in dataloader:
                inputs = batch["input"].to(device)  # (batch_size, 4)
                labels = batch["label"].to(device)  # (batch_size)

                # Forward pass
                outputs = model(inputs)  # (batch_size, num_classes)
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            # Print epoch loss
            # print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(dataloader):.4f}")
            pbar.set_postfix(loss=running_loss / len(dataloader))
            pbar.update(1)

    print("Training complete.")
    return model



if __name__ == '__main__':
    p = configargparse.ArgumentParser()
    p.add_argument('--experiment_name', type=str, required=True,
                help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')
    p.add_argument('--config_path', default=None, help='config file for training.')
    opt = p.parse_args()

    # General training options in config file
    configs = utils.load_configs(opt.config_path, opt.experiment_name)
    dataset_paras, optim_paras, loss_paras = configs['dataset'], configs['optim'], configs['parameters']


    save_path = os.path.join(dataset_paras.get('save_path'), opt.experiment_name)
    if os.path.exists(save_path):
        val = input("The model directory %s exists. Overwrite? (y/n)"%save_path)
        if val == 'y':
            shutil.rmtree(save_path)

    os.makedirs(save_path)
    utils.put_configs(os.path.join(dataset_paras.get('save_path'), opt.experiment_name, 'train_configs.yaml'), configs) # save to prj folder


    # do classification task
    dataset = torch.load(dataset_paras.get('level_dataset_path'))
    num_classes = dataset["level_labels"].max().item() + 1
    print(f"Number of classes: {num_classes}; total data: {len(dataset['level_labels'])}")

    level_dataset = LevelDataset(dataset)
    model = train_model(level_dataset, num_classes=num_classes, batch_size=optim_paras.get('batch_size'), 
                        epochs=optim_paras.get('epochs'), learning_rate=optim_paras.get('lr'))
    torch.save(model.state_dict(), os.path.join(save_path, 'level_classifier.pth'))



