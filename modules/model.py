import os
import torch

class Model:    
    def __init__(self,
                 architecture, # architecture (nn.Module)
                 loss, # loss function (nn.modules.loss._Loss)                 
                 optimizer, # optimisation algorithm (nn.optim)
                 trainer, # training algorithm (Modules.training)
                 evaluator, # evaluating algorithm (Modules.evaluation)
                 device, name, saveDir):
        
        self.archit = architecture
        self.archit.to(device)
        self.nParameters = 0
        
        for param in list(self.archit.parameters()):
            if len(param.shape)>0:
                thisNParam = 1
                for p in range(len(param.shape)):
                    thisNParam *= param.shape[p]
                self.nParameters += thisNParam
            else:
                pass
              
        self.loss = loss
        self.optim = optimizer
        self.trainer = trainer
        self.evaluator = evaluator
        self.device = device
        self.name = name
        self.saveDir = saveDir
 
    def train(self, data, nEpochs, batchSize, nDAggers, expertProb, aggregationSize, nLayers, hParamsbaseGNN, paramsLayerWiseTrain, **kwargs):        
        self.trainer = self.trainer(self, data, nEpochs, batchSize, nDAggers, expertProb, aggregationSize, nLayers, hParamsbaseGNN, paramsLayerWiseTrain, **kwargs)        
        return self.trainer.train()
    
    def evaluate(self, data, **kwargs):        
        return self.evaluator(self, data, **kwargs)
    
    def save(self, label = '', **kwargs):
        if 'saveDir' in kwargs.keys():
            saveDir = kwargs['saveDir']
        else:
            saveDir = self.saveDir
        saveModelDir = os.path.join(saveDir,'savedModels')
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
