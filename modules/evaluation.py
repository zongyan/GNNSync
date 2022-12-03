import numpy as np

def evaluate(model, data, **kwargs):
    # Initial data
    initPosTest = data.getData('initOffset', 'test')
    initVelTest = data.getData('initSkew', 'test')
    graphTest = data.getData('commGraph','test')   
    clockNoiseTest = data.getData('clockNoise','test')   
    measurementNoiseTest = data.getData('packetExchangeDelay','test')   
    processingNoiseTest = data.getData('processingDelay','test')   
                    
    model.load(label = 'Best')

    print("\tComputing learned trajectory for best model...",
          end = ' ', flush = True)

    posTestBest, \
    velTestBest, \
    accelTestBest, \
    stateTestBest, \
    commGraphTestBest = \
        data.computeTrajectory(initPosTest, initVelTest, measurementNoiseTest, processingNoiseTest, clockNoiseTest, graphTest, data.duration,
                               archit = model.archit)
                
    SavedPath ='./gnn_test.npz'
    np.savez(SavedPath, posTestBest=posTestBest, velTestBest=velTestBest, \
             accelTestBest=accelTestBest, stateTestBest=stateTestBest, \
                 commGraphTestBest=commGraphTestBest)
    print("\tSaved the test data to the following path: ./gnn_test.npz...", end = ' ')
    print("OK", flush = True)        
    
    print("OK")

    model.load(label = 'Last')

    print("\tComputing learned trajectory for last model...",
          end = ' ', flush = True)

    posTestLast, \
    velTestLast, \
    accelTestLast, \
    stateTestLast, \
    commGraphTestLast = \
        data.computeTrajectory(initPosTest, initVelTest, measurementNoiseTest, processingNoiseTest, clockNoiseTest, graphTest, data.duration,
                               archit = model.archit)

    print("OK")
