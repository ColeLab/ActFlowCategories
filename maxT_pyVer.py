# C. Cocuzza, 2021
# Python version: adapted from maxT.m (MATLAB function) by Michael Cole & Taku Ito

# Statistical significance for (1) nonparametric data, and (2) family-wise error correction of nonparametric data
# Max-T NPPT, Nichols TE, Holmes AP. (2002). Nonparametric permutation tests for functional neuroimaging: 
# A primer with Examples. Hum. Brain Mapp., 15: 1-25. doi:10.1002/hbm.1058

########################
# IMPORTS 
import numpy as np
from scipy import stats

########################
# FUNCTION 
def maxT(diffArr,nullMean,alphaVal=0.05,tailToUse=0,numPermsToRun=1000):
    '''
    INPUT: 
        diffArr: MxN matrix; set of M independent tests for condition 1 minus condition 2 across N participants;
                 NOTE: in python convention, if M=1 (1 test), then diffArr should be in shape: N,
                 diffArr can also be an array of multiple values (or tests) compared against the null mean
        nullMean: value to compare against (for ex: 0)
        alphaVal: OPTIONAL; significance level; defaults to 0.05
        tailsToUse: OPTIONAL; defaults to 0; 0=two-tailed testing; 1=upper tailed testing; -1=lower-tailed testing 
        numPermsToRun: OPTIONAL; defaults to 1000; number of permutations to build null distribution; check computational limits of your environment
        
    OUTPUT: 
        realT: the real t-stats (d.f. will = N-1); if M>1, then this will be a vector of size M
        maxTthresh: the critical threshold derived from the NPPT procedure; 1 scalar for all M tests
        maxTdistSorted: the NPPT derived null distribution, based on full family of tests (all M)
    '''
    
    # Set up data
    if diffArr.ndim == 2:
        numRows = diffArr.shape[0]
        nSubjsHere = diffArr.shape[1]
    elif diffArr.ndim == 1: 
        numRows = 0
        nSubjsHere = diffArr.shape[0]

    # Get real t stats
    if diffArr.ndim == 2: # if M>1
        realT = np.zeros((numRows))
        for rowNum in range(numRows):
            vecHere = diffArr[rowNum,:]
            tHere, pHere = stats.ttest_1samp(vecHere,nullMean)
            realT[rowNum] = tHere
    elif diffArr.ndim == 1: # if M=1 (ie shape of diffArr was N,)
        realT = np.zeros((numRows+1))
        vecHere = diffArr.copy()
        tHere, pHere = stats.ttest_1samp(vecHere,nullMean)
        realT = tHere

    # Permute to get null distrbution -- **NOTE** consider using a seeding approach for the np.randn function
    maxTdist = np.zeros((numPermsToRun))
    for permNum in range(numPermsToRun):
        
        if diffArr.ndim == 2: # if M>1
            shuffleMatGauss = np.random.randn(numRows,nSubjsHere)
            shuffleMat = np.zeros((numRows,nSubjsHere))
            posRows, posCols = np.where(shuffleMatGauss>0)
            negRows, negCols = np.where(shuffleMatGauss<0)
            shuffleMat[posRows,posCols] = 1
            shuffleMat[negRows,negCols] = -1;
            diffArrShuffled = diffArr * shuffleMat
            maxTtemp = np.zeros((numRows))
            
            for rowNum in range(numRows): 
                vecHere = diffArrShuffled[rowNum,:]
                tHere, pHere = stats.ttest_1samp(vecHere,nullMean)
                maxTtemp[rowNum] = tHere
            if tailToUse==1:
                maxTdist[permNum] = np.max(maxTtemp)
            elif tailToUse==-1:
                maxTdist[permNum] = np.min(maxTtemp)
            elif tailToUse==0: 
                maxTdist[permNum] = np.max(abs(maxTtemp))

        elif diffArr.ndim == 1: # if M=1 (ie shape of diffArr was N,)
            shuffleMatGauss = np.random.randn(nSubjsHere)
            shuffleMat = np.zeros((nSubjsHere))
            posCols = np.where(shuffleMatGauss>0)[0]
            negCols = np.where(shuffleMatGauss<0)[0]
            shuffleMat[posCols] = 1
            shuffleMat[negCols] = -1
            diffArrShuffled = diffArr * shuffleMat
            maxTtemp = np.zeros((numRows+1)) 
            vecHere = diffArrShuffled.copy()
            tHere, pHere = stats.ttest_1samp(vecHere,nullMean)
            maxTtemp = tHere;
            if tailToUse==1:
                maxTdist[permNum] = np.max(maxTtemp)
            elif tailToUse==-1:
                maxTdist[permNum] = np.min(maxTtemp)
            elif tailToUse==0: 
                maxTdist[permNum] = np.max(abs(maxTtemp))

    # Find alpha threshold from permuted null distribution 
    maxTdistSorted = np.sort(maxTdist)
    if tailToUse==1:
        topIx = round(numPermsToRun * (1 - alphaVal))
        #tailStr = 'upper tailed'; 
    elif tailToUse==-1:
        topIx = round(numPermsToRun * (alphaVal))
        #tailStr = 'lower tailed'; 
    elif tailToUse==0: 
        topIx = round(numPermsToRun * (1 - alphaVal))
        #tailStr = 'two tailed'; 

    maxTthresh = maxTdistSorted[topIx-1]

    return realT, maxTthresh, maxTdistSorted