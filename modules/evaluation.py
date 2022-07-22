import os
import torch
import pickle
import numpy as np

def evaluate(model, data, **kwargs):
                    
    ########
    # Data #
    ########

    # Initial data
    initOffsetTest = data.getData('initOffset', 'test')
    initSkewTest = data.getData('initSkew', 'test')
    netwkTopologyTest = data.getData('commNetwk','test')         

    ##############
    # Best Model #
    ##############

    model.load(label = 'Best')

    print("\tComputing learned synchronisation strategy for best model...", end = ' ', flush = True)

    offsetTestBest, skewTestBest, offsetCorrectionTestBest, skewCorrectionTestBest = \
        data.computeTimeSynchronisation(initOffsetTest, initSkewTest, netwkTopologyTest, 
                                   data.duration, model.archit, True)
           
    SavedPath ='./gnn_test.npz'
    np.savez(SavedPath, offsetTestBest=offsetTestBest, skewTestBest=skewTestBest, \
             offsetCorrectionTestBest=offsetCorrectionTestBest, skewCorrectionTestBest=skewCorrectionTestBest, \
                 networkTopologyTest=netwkTopologyTest)
    print("\tSaved the test data to the following path: ./gnn_test.npz...", end = ' ')
    print("OK", flush = True)

    ##############
    # Last Model #
    ##############

    model.load(label = 'Last')

    print("\tComputing learned synchronisation strategy for last model...",
          end = ' ', flush = True)

    offsetTestLast, skewTestLast, offsetCorrectionTestLast, skewCorrectionTestLast = \
        data.computeTimeSynchronisation(initOffsetTest, initSkewTest, netwkTopologyTest, 
                                   data.duration, model.archit, True)

    print("OK", flush = True)



