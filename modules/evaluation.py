import numpy as np

def evaluate(model, data, **kwargs):
    initPosTest = data.getData('initOffset', 'test')
    initVelTest = data.getData('initSkew', 'test')
    graphTest = data.getData('commGraph','test')   
    clockNoiseTest = data.getData('clockNoise','test')   
    measurementNoiseTest = data.getData('packetExchangeDelay','test')   
    processingNoiseTest = data.getData('processingDelay','test')   
                    
    model.load(label='Best')

    print("\tComputing learned time synchronisation for best model...",
          end = ' ', flush = True)

    offsetTestBest, \
    skewTestBest, \
    adjTestBest, \
    stateTestBest, \
    commGraphTestBest = \
        data.computeTrajectory(initPosTest, initVelTest, \
                               measurementNoiseTest, processingNoiseTest, clockNoiseTest, 
                               graphTest, data.duration,
                               archit = model.archit)
                
    SavedPath ='./gnn_test.npz'
    np.savez(SavedPath, offsetTestBest=offsetTestBest, skewTestBest=skewTestBest, \
             adjTestBest=adjTestBest, stateTestBest=stateTestBest, \
                 commGraphTestBest=commGraphTestBest)    
    print("OK")

    model.load(label = 'Last')

    print("\tComputing learned time synchronisation for last model...",
          end = ' ', flush = True)

    offsetTestLast, \
    skewTestLast, \
    adjTestLast, \
    stateTestLast, \
    commGraphTestLast = \
        data.computeTrajectory(initPosTest, initVelTest,\
                               measurementNoiseTest, processingNoiseTest, clockNoiseTest, \
                               graphTest, data.duration,
                               archit = model.archit)

    print("OK")
