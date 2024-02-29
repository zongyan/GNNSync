import os
import torch
import copy

class Model:    
    def __init__(self,
                 architecture, # architecture (nn.Module)
                 loss, # loss function (nn.modules.loss._Loss)                 
                 optimizer, # optimisation algorithm (nn.optim)
                 trainer, # training algorithm (Modules.training)
                 evaluator, # evaluating algorithm (Modules.evaluation)
                 device, name, nDAggersValues, layerWise, saveDir):
        
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
        self.trainer = []
        self.nDAggersValues = nDAggersValues
        self.layerWise = layerWise 
                    
        self.trainer = [copy.deepcopy([copy.deepcopy(trainer) for k in range(len(nDAggersValues))]) for j in range(len(layerWise))]            
        self.config = [copy.deepcopy([copy.deepcopy(trainer) for k in range(len(nDAggersValues))]) for j in range(len(layerWise))]                    

        self.evalModel = False
        self.heatKenel = False        
        self.useNonlinearity = False
        
        self.evaluator = evaluator
        self.device = device
        self.name = name
        self.saveDir = saveDir
 
    def train(self, data, nEpochs, batchSize, \
              nDAggers, expertProb, aggregationSize, \
                  paramsLayerWiseTrain, layerWiseTraining, \
                      lossFunction, learningRate, beta1, beta2, useNonlinearity, **kwargs):
        
        self.trainer[self.layerWise.index(layerWiseTraining)][self.nDAggersValues.index(nDAggers)] = \
            self.trainer[self.layerWise.index(layerWiseTraining)][self.nDAggersValues.index(nDAggers)]\
                (self, data, nEpochs, batchSize, \
                 nDAggers, expertProb, aggregationSize, \
                     paramsLayerWiseTrain, layerWiseTraining, \
                         lossFunction, learningRate, beta1, beta2, useNonlinearity, **kwargs)        

        return self.trainer[self.layerWise.index(layerWiseTraining)][self.nDAggersValues.index(nDAggers)].train()

    def configure(self, data, nEpochs, batchSize, \
              nDAggers, expertProb, aggregationSize, \
                  paramsLayerWiseTrain, layerWiseTraining, \
                      lossFunction, learningRate, beta1, beta2, evalModel, useNonlinearity, **kwargs):
        
        self.evalModel = evalModel
        # self.archit.evalModel = evalModel        
        self.useNonlinearity = useNonlinearity
        # self.archit.useNonlinearity = useNonlinearity
        
        self.config[self.layerWise.index(layerWiseTraining)][self.nDAggersValues.index(nDAggers)] = \
            self.config[self.layerWise.index(layerWiseTraining)][self.nDAggersValues.index(nDAggers)]\
                (self, data, nEpochs, batchSize, \
                 nDAggers, expertProb, aggregationSize, \
                     paramsLayerWiseTrain, layerWiseTraining, \
                         lossFunction, learningRate, beta1, beta2, useNonlinearity, **kwargs)

        return self.config[self.layerWise.index(layerWiseTraining)][self.nDAggersValues.index(nDAggers)].configuration()    
    
    def evaluate(self, data, nDAggers, layerWiseTraining, **kwargs):        
        if (self.evalModel == False): 
            return self.evaluator(self, self.trainer[self.layerWise.index(layerWiseTraining)][self.nDAggersValues.index(nDAggers)], data, self.evalModel, self.useNonlinearity, **kwargs)
        else:
            return self.evaluator(self, self.config[self.layerWise.index(layerWiseTraining)][self.nDAggersValues.index(nDAggers)], data, self.evalModel, self.useNonlinearity, **kwargs)    
    
    def save(self, layerWiseTraining, nDAggers, l, iteration, epoch, batch, label = '', **kwargs):        
        
        if 'saveDir' in kwargs.keys():
            saveDir = kwargs['saveDir']
        else:
            saveDir = self.saveDir

        saveModelDir = os.path.join(saveDir,'savedModels')
        
        if layerWiseTraining == True:
            saveModelDir = os.path.join(saveModelDir,'layerWiseTraining')
        else:
            saveModelDir = os.path.join(saveModelDir,'endToEndTraining')

        if not os.path.exists(saveModelDir):
            os.makedirs(saveModelDir)

        if layerWiseTraining == True:
            saveFile = os.path.join(saveModelDir, self.name + '-LayerWise-' + str(l) + '-DAgger-' + str(iteration) + '-' + str(nDAggers) + '-Epoch-' + str(epoch) + '-Batch-' + str(batch))
        else:
            saveFile = os.path.join(saveModelDir, self.name + '-EndToEnd-' + str(l) + '-DAgger-' + str(iteration) + '-' + str(nDAggers) + '-Epoch-' + str(epoch) + '-Batch-' + str(batch))
                    
        torch.save(self.archit.state_dict(), saveFile+'-Archit-'+ label+'.ckpt')
        torch.save(self.optim.state_dict(), saveFile+'-Optim-'+label+'.ckpt')

    def load(self, layerWiseTraining, nDAggers, l, iteration, epoch, batch, label = '', **kwargs):
        
        if 'loadFiles' in kwargs.keys():
            (architLoadFile, optimLoadFile) = kwargs['loadFiles']
        else:
            saveModelDir = os.path.join(self.saveDir,'savedModels')
            
            if layerWiseTraining == True:
                saveModelDir = os.path.join(saveModelDir,'layerWiseTraining')
            else:
                saveModelDir = os.path.join(saveModelDir,'endToEndTraining')

            if layerWiseTraining == True:
                saveFile = os.path.join(saveModelDir, self.name + '-LayerWise-' + str(l) + '-DAgger-' + str(iteration) + '-' + str(nDAggers) + '-Epoch-' + str(epoch) + '-Batch-' + str(batch))
            else:
                saveFile = os.path.join(saveModelDir, self.name + '-EndToEnd-' + str(l) + '-DAgger-' + str(iteration) + '-' + str(nDAggers) + '-Epoch-' + str(epoch) + '-Batch-' + str(batch))
            
            architLoadFile = saveFile + '-Archit-' + label +'.ckpt'
            optimLoadFile = saveFile + '-Optim-' + label + '.ckpt'
        self.archit.load_state_dict(torch.load(architLoadFile))
        self.optim.load_state_dict(torch.load(optimLoadFile))
