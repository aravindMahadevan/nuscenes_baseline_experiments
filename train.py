from nuscenes.prediction.models.backbone import ResNetBackbone
from nuscenes.prediction.models.mtp import MTP, MTPLoss
from nuscenes.prediction.models.covernet import CoverNet
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from NuscenesDataset import NuscenesDataset
import numpy as np 
import time
import os 
import matplotlib
import matplotlib.pyplot as plt 

class MTP_Experiment():
    def __init__(self, 
                 M = 20, 
                 num_epochs = 50, 
                 num_workers = 4,
                 output_dir = 'mtp_experiment_data',
                 lr = 3e-4,
                 batch_size=8, 
                 data_root = '../full_data/sets/nuscenes',
                 backbone_architecture='resnet50'):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        #init experiment parameters
        self.output_dir = output_dir
        self.checkpoint_path = os.path.join(self.output_dir, "checkpoint.pth.tar")
        self.config_path = os.path.join(self.output_dir, "config.txt")
        
        self.num_epochs = num_epochs
        self.batch_size = batch_size 
        self.lr = lr 
        self.backbone_architecture = 'resnet50'
        
        #init network 
        self.num_modes = M
        self.backbone = ResNetBackbone(backbone_architecture)
        self.mtp = MTP(self.backbone, num_modes=self.num_modes).to(self.device) 
        
        #init data set and data loader 
        self.dataset = NuscenesDataset(data_root)
        self.dataloader = DataLoader(self.dataset, 
                                     batch_size=self.batch_size, 
                                     shuffle = True, 
                                     pin_memory = True,
                                     num_workers = num_workers)
        
        #init loss function and optimizers
        self.criterion = MTPLoss(self.num_modes)
        self.optimizer = optim.Adam(self.mtp.parameters(), lr=self.lr)    
        
        self.train_loss = []
        
        #check if already created experiment
        if os.path.isfile(self.config_path):
            print("Reloading previous experiment")
            self.load()
        else: 
            #create directory
            os.makedirs(output_dir, exist_ok=True)
            self.save()
                    
    @property
    def epoch(self):
        return len(self.train_loss)

    def setting(self):
        return {'MTP' : self.mtp,
                'Optimizer' : self.optimizer,
                'Number of Modes' : self.num_modes,
                'Backbone architecture' : self.backbone_architecture, 
                'Batch Size' : self.batch_size,
                'Learning Rate' : self.lr, 
                'Number of Epochs' : self.num_epochs}

    def __repr__(self):
        string = ''
        for key, val in self.setting().items():
            string += '{} : {}\n'.format(key, val)
        return string

    def state_dict(self):
        return {'MTP' : self.mtp.state_dict(),
                'Optimizer' : self.optimizer.state_dict(),
                'TrainLoss' : self.train_loss}
        
    def save_fig(self):
        fig, ax = plt.subplots(1)
        ax.plot(range(len(self.train_loss)), self.train_loss, 'r', label='Train Loss')
        ax.legend()
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        fig.savefig(os.path.join(self.output_dir, 'mtp_loss_epoch'))
        plt.close()

    def save(self):
        self.save_fig()
        torch.save(self.state_dict(), self.checkpoint_path)
        with open(self.config_path, 'w') as f:
            print(self, file=f)

    def load_state_dict(self, checkpoint):
        self.mtp.load_state_dict(checkpoint['MTP'])
        self.optimizer.load_state_dict(checkpoint['Optimizer'])
        self.train_loss = checkpoint['TrainLoss']
        
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
         
    def load(self):
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.load_state_dict(checkpoint)
        del checkpoint
    
    def run(self):
        print("Starting/Continuing Training!")
        for epoch in range(self.num_epochs): 
            self.train(epoch)
            
            if epoch % 10 == 0: 
                print('saving weights')
                self.save()
                
        self.save()
        print("Done training!")
          
    def train(self, epoch): 
        epoch_loss = 0
        count=0
        
        print("Starting Epoch {0} at {1}".format(epoch, str(time.strftime("%H:%M:%S", time.localtime()))))
        for image_tensor, agent_vec, ground_truth in self.dataloader:
            output = self.mtp(image_tensor.to(self.device), agent_vec.to(self.device))
            loss = self.criterion(output, ground_truth.to(self.device))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            count+=1
            print("{0}/{1} batches finished".format(count, len(self.dataloader)))
            epoch_loss += loss
            
        print("Epoch {0}/{1} complete. Loss at epoch: {2}".format(epoch, self.num_epochs - 1, epoch_loss))
        print("Ending Epoch {0} at {1}".format(epoch, str(time.strftime("%H:%M:%S", time.localtime()))))
        self.train_loss.append(epoch_loss)

if __name__ == "__main__":
    exp = MTP_Experiment(output_dir = 'exp4')
    exp.run()
