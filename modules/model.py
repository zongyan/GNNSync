import os
import torch

class Model:    
    def __init__(self,
                 # Architecture (nn.Module)
                 architecture,
                 # Loss Function (nn.modules.loss._Loss)
                 loss,
                 # Optimization Algorithm (nn.optim)
                 optimizer,
                 # Training Algorithm (Modules.training)
                 trainer,
                 # Evaluating Algorithm (Modules.evaluation)
                 evaluator,
                 # Other
                 device, name, saveDir):
        
        #\\\ ARCHITECTURE
        # Store
        self.archit = architecture
        # Move it to device
        self.archit.to(device)
        # Count parameters (doesn't work for EdgeVarying)
        self.nParameters = 0
        for param in list(self.archit.parameters()):
            if len(param.shape)>0:
                thisNParam = 1
                for p in range(len(param.shape)):
                    thisNParam *= param.shape[p]
                self.nParameters += thisNParam
            else:
                pass
        #\\\ LOSS FUNCTION
        self.loss = loss
        #\\\ OPTIMIZATION ALGORITHM
        self.optim = optimizer
        #\\\ TRAINING ALGORITHM
        self.trainer = trainer
        #\\\ EVALUATING ALGORITHM
        self.evaluator = evaluator
        #\\\ OTHER
        # Device
        self.device = device
        # Model name
        self.name = name
        # Saving directory
        self.saveDir = saveDir
        
    def train(self, data, nEpochs, batchSize, **kwargs):
        
        self.trainer = self.trainer(self, data, nEpochs, batchSize, **kwargs)
        
        return self.trainer.train()
    
    def evaluate(self, data, **kwargs):
        
        return self.evaluator(self, data, **kwargs)
    
    def save(self, label = '', **kwargs):
        if 'saveDir' in kwargs.keys():
            saveDir = kwargs['saveDir']
        else:
            saveDir = self.saveDir
        saveModelDir = os.path.join(saveDir,'savedModels')
        # Create directory savedModels if it doesn't exist yet:
        if not os.path.exists(saveModelDir):
            os.makedirs(saveModelDir)
        saveFile = os.path.join(saveModelDir, self.name)
        torch.save(self.archit.state_dict(), saveFile+'Archit'+ label+'.ckpt')
        torch.save(self.optim.state_dict(), saveFile+'Optim'+label+'.ckpt')

    def load(self, label = '', **kwargs):
        if 'loadFiles' in kwargs.keys():
            (architLoadFile, optimLoadFile) = kwargs['loadFiles']
        else:
            saveModelDir = os.path.join(self.saveDir,'savedModels')
            architLoadFile = os.path.join(saveModelDir,
                                          self.name + 'Archit' + label +'.ckpt')
            optimLoadFile = os.path.join(saveModelDir,
                                         self.name + 'Optim' + label + '.ckpt')
        self.archit.load_state_dict(torch.load(architLoadFile))
        self.optim.load_state_dict(torch.load(optimLoadFile))
