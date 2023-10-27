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
 
    def train(self, data, nEpochs, batchSize, nDAggers, expertProb, aggregationSize, paramsLayerWiseTrain, layerWiseTraining, endToEndTraining, **kwargs):        
        self.trainer = self.trainer(self, data, nEpochs, batchSize, nDAggers, expertProb, aggregationSize, paramsLayerWiseTrain, layerWiseTraining, endToEndTraining, **kwargs)        
        return self.trainer.train()
    
    def evaluate(self, data, **kwargs):        
        return self.evaluator(self, data, **kwargs)
    
    def save(self, layerWiseTraining, endToEndTraining, l, iteration, epoch, batch, label = '', **kwargs):        
        assert layerWiseTraining == (not endToEndTraining)
        
        if 'saveDir' in kwargs.keys():
            saveDir = kwargs['saveDir']
        else:
            saveDir = self.saveDir

        saveModelDir = os.path.join(saveDir,'savedModels')
        
        if layerWiseTraining == True:
            saveModelDir = os.path.join(saveModelDir,'layerWiseTraining')
        elif endToEndTraining == True:
            saveModelDir = os.path.join(saveModelDir,'endToEndTraining')

        if not os.path.exists(saveModelDir):
            os.makedirs(saveModelDir)

        if layerWiseTraining == True:
            saveFile = os.path.join(saveModelDir, self.name + '-LayerWise-' + str(l) + '-DAgger-' + str(iteration) + '-Epoch-' + str(epoch) + '-Batch-' + str(batch))
        elif endToEndTraining == True:
            saveFile = os.path.join(saveModelDir, self.name + '-EndToEnd-' + str(l) + '-DAgger-' + str(iteration) + '-Epoch-' + str(epoch) + '-Batch-' + str(batch))
                    
        torch.save(self.archit.state_dict(), saveFile+'-Archit-'+ label+'.ckpt')
        torch.save(self.optim.state_dict(), saveFile+'-Optim-'+label+'.ckpt')

    def load(self, layerWiseTraining, endToEndTraining, l, iteration, epoch, batch, label = '', **kwargs):
        assert layerWiseTraining == (not endToEndTraining)
        
        if 'loadFiles' in kwargs.keys():
            (architLoadFile, optimLoadFile) = kwargs['loadFiles']
        else:
            saveModelDir = os.path.join(self.saveDir,'savedModels')
            
            if layerWiseTraining == True:
                saveModelDir = os.path.join(saveModelDir,'layerWiseTraining')
            elif endToEndTraining == True:
                saveModelDir = os.path.join(saveModelDir,'endToEndTraining')

            if layerWiseTraining == True:
                saveFile = os.path.join(saveModelDir, self.name + '-LayerWise-' + str(l) + '-DAgger-' + str(iteration) + '-Epoch-' + str(epoch) + '-Batch-' + str(batch))
            elif endToEndTraining == True:
                saveFile = os.path.join(saveModelDir, self.name + '-EndToEnd-' + str(l) + '-DAgger-' + str(iteration) + '-Epoch-' + str(epoch) + '-Batch-' + str(batch))
            
            architLoadFile = saveFile + '-Archit-' + label +'.ckpt'
            optimLoadFile = saveFile + '-Optim-' + label + '.ckpt'
        self.archit.load_state_dict(torch.load(architLoadFile))
        self.optim.load_state_dict(torch.load(optimLoadFile))
