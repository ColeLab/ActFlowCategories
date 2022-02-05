# Dominance analysis, following math here: https://github.com/dominance-analysis/dominance-analysis
# C. Cocuzza, 2021

# Multiple regression model to test how each network influences the explained variance (accuracy) of the act-flow prediction. 
# So, the IVs (x regressors) here are 12 network AF-predicted betas (pre-computed) across 24 conditions, 
# and the DV here (y response/outcome) is the actual beta across 24 conditions. Doing this across conditions (for a given fROI/complex of interest)
# is what we're calling the "response profile" level. 

# This is done per subject so that they can be parallelized on SLURM 
# See movieActFlow_Master_2020.ipynb for more details on calling this script 

# Note that here, n and r refer to classic nCr combination formula. 12 = # of predictors in multiple regression model, r = subset for each model. 
# (done pseudo-iteratively, so r varies from 1-11, and =12 for full model). 

# The total number of models = (2^n)-1. So 12 networks as IVs --> 4095 models total. See annotation throughout for how many models are in each subset. 
####################################################################################################################
# INPUTS 
# NOTE: make sure arrays are indexed to full subjects before input  

####################################################################################################################
# OUTPUTS (per subject; can populate results arrays with below variables with added dimension of size: number of subjects, ex: rsqFull_All = 1x176)
# rsqFull = 1 value for explained variance of full model, where n=12 and r=12 
# netVarExplDom = 12 values for partial explained variance per network (n=12, r is varied per subset of models)
# netVarImportance = 12 values for percent of relative importance per network (based on above values/full model rsq)

####################################################################################################################
# Imports 
import pkg_resources;
import numpy as np; 
import os;
import h5py;
import scipy.stats as stats;
import scipy;
import operator as op; 
from functools import reduce; 
from itertools import combinations;
import pandas as pd; 
import sklearn;
from sklearn.linear_model import LinearRegression;

####################################################################################################################
# Instantiate paths and core variables; change these for different projects / datasets
nNetsTotal = 12; # NOTE: total number of models with this method = (2^nNetsTotal)-1; which = iterative # of nCr models added up
useNetMeans = 0; # If not using net means (ie net mean flow products), use sum over net nodes (ie what adds up to AF prediction)
allStrs = ['VIS1','VIS2','SMN','CON','DAN','LAN','FPN','AUD','DMN','PMM','VMM','OAN']; 
numNets = 12;
nConditionsHere = 24;
#docDir = '/projects/f_mc1689_1/MovieActFlow/docs/scripts/HCP_3T_7Task/';

#actPredArray_NetMeans_All = np.zeros((numNets,len(parcelOfInterest),nConditionsHere,nSubjsHere));
#for subjNum in range(nSubjsHere):
#    for netNum in range(numNets):
#        startNode = int(boundariesCA[netNum,0]); endNode = int(boundariesCA[netNum,1]+1);
#        actPredArray_NetMeans_All[netNum,:,:,subjNum] = np.nanmean(flowArrAll[parcelOfInterest,:,:,subjNum][:,netorder,:][:,startNode:endNode,:],axis=1);


####################################################################################################################
# Function 

def dominance_analysis_slurm(predictedArray_NetRestricted,predictedArray_NetMeans,actualArray,parcelOfInterest):
    #netVarExplDom = np.zeros((nNetsTotal)); netVarImportance = np.zeros((nNetsTotal));

    def ncr(n, r):
        r = min(r, n-r); numer = reduce(op.mul, range(n, n-r, -1), 1); denom = reduce(op.mul, range(1, r+1), 1);
        return numer // denom
    
    parcelIx = []; #parcelOfInterestFile = docDir + parcelOfInterest;
    parcelOfInterestFile = parcelOfInterest;
    with open(parcelOfInterestFile, 'r') as fileHandle:
        for line in fileHandle:
            currentPlace = line[:-1]; parcelIx.append(int(currentPlace.strip()));
    parcelOfInterest = parcelIx;     

    if useNetMeans==0:
        #predDataHere = np.reshape(predictedArray_NetRestricted[:,:,:,subjNum],(numNets,nConditionsHere*len(parcelOfInterest))); 
        predDataHere = np.reshape(predictedArray_NetRestricted,(numNets,nConditionsHere*len(parcelOfInterest))); 
    if useNetMeans==1:
        #predDataHere = np.reshape(predictedArray_NetMeans[:,:,:,subjNum],(numNets,nConditionsHere*len(parcelOfInterest)));
        predDataHere = np.reshape(predictedArray_NetMeans,(numNets,nConditionsHere*len(parcelOfInterest)));
        
    #actualDataHere = np.reshape(actualArray[parcelOfInterest,:,subjNum],(nConditionsHere*len(parcelOfInterest)));
    actualDataHere = np.reshape(actualArray[parcelOfInterest,:],(nConditionsHere*len(parcelOfInterest)));
    
    #predFull = predictedArrayFull[parcelOfInterest,:,:][:,taskIxs,:];
    dataHere = {'VIS1':predDataHere[0,:],'VIS2':predDataHere[1,:],'SMN':predDataHere[2,:],'CON':predDataHere[3,:],'DAN':predDataHere[4,:],'LAN':predDataHere[5,:],
                'FPN':predDataHere[6,:],'AUD':predDataHere[7,:],'DMN':predDataHere[8,:],'PMM':predDataHere[9,:],'VMM':predDataHere[10,:],'OAN':predDataHere[11,:],
                'Actual':actualDataHere}; dfHere = pd.DataFrame(dataHere);

    xData = dfHere[['VIS1','VIS2','SMN','CON','DAN','LAN','FPN','AUD','DMN','PMM','VMM','OAN']]; yData = dfHere['Actual']; 

    # Full model with 12 predictors, k=12
    netsReduced = allStrs.copy(); #netsReduced = np.delete(netsReduced,netIxsRed).tolist(); 
    xData = dfHere[netsReduced];
    linReg = LinearRegression(); linReg.fit(xData,yData); predModel = linReg.predict(xData);
    #regScores = linReg.score(xData,yData); betaCoefs = linReg.coef_; modelR = linReg.score(xData,yData); 
    #sseFull = np.sum((yData - predModel)**2); sstFull = np.sum((yData - np.nanmean(yData)) ** 2); # NOTE: rsq should = 1 - (sseFull/sstFull)
    rsqFull = sklearn.metrics.explained_variance_score(yData,predModel); 
          
    # 12 models with 11 predictors each, k=11
    k = 11; numCombs = ncr(nNetsTotal,k); combosHere = list(combinations(np.arange(nNetsTotal),k)); rsqEleven = np.zeros((numCombs)); 
    addedVarMatEleven = np.zeros((numCombs,nNetsTotal)); addedVarMatEleven[:] = np.NaN;
    for modelNum in range(numCombs):
        thisModelSet = np.asarray(combosHere[modelNum]); netStrHere = netsReduced.copy(); netStrHere = list(np.asarray(netsReduced)[thisModelSet]); #print(netStrHere)
        xData = dfHere[netStrHere]; linReg = LinearRegression(); linReg.fit(xData,yData); predModel = linReg.predict(xData);
        rsqEleven[modelNum] = sklearn.metrics.explained_variance_score(yData,predModel);    
        refList = np.arange(nNetsTotal); missingPreds = np.asarray(list(set(refList).difference(thisModelSet)));
        nMissing = missingPreds.shape[0]; #netStrHereInit = netStrHere.copy();
        for heldOutNum in range(nMissing):
            thisIx = missingPreds[heldOutNum]; netsReducedCopy = netsReduced.copy(); addStr = list(np.asarray(netsReducedCopy)[[thisIx]]);
            netStrHereInit = netStrHere + addStr; #print(netStrHereInit);
            xData = dfHere[netStrHereInit]; linReg = LinearRegression(); linReg.fit(xData,yData); predModel = linReg.predict(xData);
            addedVarMatEleven[modelNum,thisIx] = sklearn.metrics.explained_variance_score(yData,predModel) - rsqEleven[modelNum]; 
            
    # 66 models with 10 predictors each, k=10
    k = 10; numCombs = ncr(nNetsTotal,k); combosHere = list(combinations(np.arange(nNetsTotal),k)); rsqTen = np.zeros((numCombs)); 
    addedVarMatTen = np.zeros((numCombs,nNetsTotal)); addedVarMatTen[:] = np.NaN;
    for modelNum in range(numCombs):
        thisModelSet = np.asarray(combosHere[modelNum]); netStrHere = netsReduced.copy(); netStrHere = list(np.asarray(netsReduced)[thisModelSet]); #print(netStrHere)
        xData = dfHere[netStrHere]; linReg = LinearRegression(); linReg.fit(xData,yData); predModel = linReg.predict(xData);
        rsqTen[modelNum] = sklearn.metrics.explained_variance_score(yData,predModel);    
        refList = np.arange(nNetsTotal); missingPreds = np.asarray(list(set(refList).difference(thisModelSet)));
        nMissing = missingPreds.shape[0]; #netStrHereInit = netStrHere.copy();
        for heldOutNum in range(nMissing):
            thisIx = missingPreds[heldOutNum]; netsReducedCopy = netsReduced.copy(); addStr = list(np.asarray(netsReducedCopy)[[thisIx]]);
            netStrHereInit = netStrHere + addStr; #print(netStrHereInit);
            xData = dfHere[netStrHereInit]; linReg = LinearRegression(); linReg.fit(xData,yData); predModel = linReg.predict(xData);
            addedVarMatTen[modelNum,thisIx] = sklearn.metrics.explained_variance_score(yData,predModel) - rsqTen[modelNum]; 
            
    # 220 models with 9 predictors each, k=9
    k = 9; numCombs = ncr(nNetsTotal,k); combosHere = list(combinations(np.arange(nNetsTotal),k)); rsqNine = np.zeros((numCombs)); 
    addedVarMatNine = np.zeros((numCombs,nNetsTotal)); addedVarMatNine[:] = np.NaN;
    for modelNum in range(numCombs):
        thisModelSet = np.asarray(combosHere[modelNum]); netStrHere = netsReduced.copy(); netStrHere = list(np.asarray(netsReduced)[thisModelSet]); #print(netStrHere)
        xData = dfHere[netStrHere]; linReg = LinearRegression(); linReg.fit(xData,yData); predModel = linReg.predict(xData);
        rsqNine[modelNum] = sklearn.metrics.explained_variance_score(yData,predModel);    
        refList = np.arange(nNetsTotal); missingPreds = np.asarray(list(set(refList).difference(thisModelSet)));
        nMissing = missingPreds.shape[0]; #netStrHereInit = netStrHere.copy();
        for heldOutNum in range(nMissing):
            thisIx = missingPreds[heldOutNum]; netsReducedCopy = netsReduced.copy(); addStr = list(np.asarray(netsReducedCopy)[[thisIx]]);
            netStrHereInit = netStrHere + addStr; #print(netStrHereInit);
            xData = dfHere[netStrHereInit]; linReg = LinearRegression(); linReg.fit(xData,yData); predModel = linReg.predict(xData);
            addedVarMatNine[modelNum,thisIx] = sklearn.metrics.explained_variance_score(yData,predModel) - rsqNine[modelNum]; 
            
    # 495 models with 8 predictors each, k=8
    k = 8; numCombs = ncr(nNetsTotal,k); combosHere = list(combinations(np.arange(nNetsTotal),k)); rsqEight = np.zeros((numCombs)); 
    addedVarMatEight = np.zeros((numCombs,nNetsTotal)); addedVarMatEight[:] = np.NaN;
    for modelNum in range(numCombs):
        thisModelSet = np.asarray(combosHere[modelNum]); netStrHere = netsReduced.copy(); netStrHere = list(np.asarray(netsReduced)[thisModelSet]); #print(netStrHere)
        xData = dfHere[netStrHere]; linReg = LinearRegression(); linReg.fit(xData,yData); predModel = linReg.predict(xData);
        rsqEight[modelNum] = sklearn.metrics.explained_variance_score(yData,predModel);    
        refList = np.arange(nNetsTotal); missingPreds = np.asarray(list(set(refList).difference(thisModelSet)));
        nMissing = missingPreds.shape[0]; #netStrHereInit = netStrHere.copy();
        for heldOutNum in range(nMissing):
            thisIx = missingPreds[heldOutNum]; netsReducedCopy = netsReduced.copy(); addStr = list(np.asarray(netsReducedCopy)[[thisIx]]);
            netStrHereInit = netStrHere + addStr; #print(netStrHereInit);
            xData = dfHere[netStrHereInit]; linReg = LinearRegression(); linReg.fit(xData,yData); predModel = linReg.predict(xData);
            addedVarMatEight[modelNum,thisIx] = sklearn.metrics.explained_variance_score(yData,predModel) - rsqEight[modelNum];
            
    # 792 models with 7 predictors each, k=7
    k = 7; numCombs = ncr(nNetsTotal,k); combosHere = list(combinations(np.arange(nNetsTotal),k)); rsqSeven = np.zeros((numCombs)); 
    addedVarMatSeven = np.zeros((numCombs,nNetsTotal)); addedVarMatSeven[:] = np.NaN;
    for modelNum in range(numCombs):
        thisModelSet = np.asarray(combosHere[modelNum]); netStrHere = netsReduced.copy(); netStrHere = list(np.asarray(netsReduced)[thisModelSet]); #print(netStrHere)
        xData = dfHere[netStrHere]; linReg = LinearRegression(); linReg.fit(xData,yData); predModel = linReg.predict(xData);
        rsqSeven[modelNum] = sklearn.metrics.explained_variance_score(yData,predModel);    
        refList = np.arange(nNetsTotal); missingPreds = np.asarray(list(set(refList).difference(thisModelSet)));
        nMissing = missingPreds.shape[0]; #netStrHereInit = netStrHere.copy();
        for heldOutNum in range(nMissing):
            thisIx = missingPreds[heldOutNum]; netsReducedCopy = netsReduced.copy(); addStr = list(np.asarray(netsReducedCopy)[[thisIx]]);
            netStrHereInit = netStrHere + addStr; #print(netStrHereInit);
            xData = dfHere[netStrHereInit]; linReg = LinearRegression(); linReg.fit(xData,yData); predModel = linReg.predict(xData);
            addedVarMatSeven[modelNum,thisIx] = sklearn.metrics.explained_variance_score(yData,predModel) - rsqSeven[modelNum];  
            
    # 924 models with 6 predictors each, k=6
    k = 6; numCombs = ncr(nNetsTotal,k); combosHere = list(combinations(np.arange(nNetsTotal),k)); rsqSix = np.zeros((numCombs)); 
    addedVarMatSix = np.zeros((numCombs,nNetsTotal)); addedVarMatSix[:] = np.NaN;
    for modelNum in range(numCombs):
        thisModelSet = np.asarray(combosHere[modelNum]); netStrHere = netsReduced.copy(); netStrHere = list(np.asarray(netsReduced)[thisModelSet]); #print(netStrHere)
        xData = dfHere[netStrHere]; linReg = LinearRegression(); linReg.fit(xData,yData); predModel = linReg.predict(xData);
        rsqSix[modelNum] = sklearn.metrics.explained_variance_score(yData,predModel);    
        refList = np.arange(nNetsTotal); missingPreds = np.asarray(list(set(refList).difference(thisModelSet)));
        nMissing = missingPreds.shape[0]; #netStrHereInit = netStrHere.copy();
        for heldOutNum in range(nMissing):
            thisIx = missingPreds[heldOutNum]; netsReducedCopy = netsReduced.copy(); addStr = list(np.asarray(netsReducedCopy)[[thisIx]]);
            netStrHereInit = netStrHere + addStr; #print(netStrHereInit);
            xData = dfHere[netStrHereInit]; linReg = LinearRegression(); linReg.fit(xData,yData); predModel = linReg.predict(xData);
            addedVarMatSix[modelNum,thisIx] = sklearn.metrics.explained_variance_score(yData,predModel) - rsqSix[modelNum];  
            
    # 792 models with 5 predictors each, k=5
    k = 5; numCombs = ncr(nNetsTotal,k); combosHere = list(combinations(np.arange(nNetsTotal),k)); rsqFive = np.zeros((numCombs)); 
    addedVarMatFive = np.zeros((numCombs,nNetsTotal)); addedVarMatFive[:] = np.NaN;
    for modelNum in range(numCombs):
        thisModelSet = np.asarray(combosHere[modelNum]); netStrHere = netsReduced.copy(); netStrHere = list(np.asarray(netsReduced)[thisModelSet]); #print(netStrHere)
        xData = dfHere[netStrHere]; linReg = LinearRegression(); linReg.fit(xData,yData); predModel = linReg.predict(xData);
        rsqFive[modelNum] = sklearn.metrics.explained_variance_score(yData,predModel);    
        refList = np.arange(nNetsTotal); missingPreds = np.asarray(list(set(refList).difference(thisModelSet)));
        nMissing = missingPreds.shape[0]; #netStrHereInit = netStrHere.copy();
        for heldOutNum in range(nMissing):
            thisIx = missingPreds[heldOutNum]; netsReducedCopy = netsReduced.copy(); addStr = list(np.asarray(netsReducedCopy)[[thisIx]]);
            netStrHereInit = netStrHere + addStr; #print(netStrHereInit);
            xData = dfHere[netStrHereInit]; linReg = LinearRegression(); linReg.fit(xData,yData); predModel = linReg.predict(xData);
            addedVarMatFive[modelNum,thisIx] = sklearn.metrics.explained_variance_score(yData,predModel) - rsqFive[modelNum];  
            
    # 495 models with 4 predictors each, k=4
    k = 4; numCombs = ncr(nNetsTotal,k); combosHere = list(combinations(np.arange(nNetsTotal),k)); rsqFour = np.zeros((numCombs)); 
    addedVarMatFour = np.zeros((numCombs,nNetsTotal)); addedVarMatFour[:] = np.NaN;
    for modelNum in range(numCombs):
        thisModelSet = np.asarray(combosHere[modelNum]); netStrHere = netsReduced.copy(); netStrHere = list(np.asarray(netsReduced)[thisModelSet]); #print(netStrHere)
        xData = dfHere[netStrHere]; linReg = LinearRegression(); linReg.fit(xData,yData); predModel = linReg.predict(xData);
        rsqFour[modelNum] = sklearn.metrics.explained_variance_score(yData,predModel);    
        refList = np.arange(nNetsTotal); missingPreds = np.asarray(list(set(refList).difference(thisModelSet)));
        nMissing = missingPreds.shape[0]; #netStrHereInit = netStrHere.copy();
        for heldOutNum in range(nMissing):
            thisIx = missingPreds[heldOutNum]; netsReducedCopy = netsReduced.copy(); addStr = list(np.asarray(netsReducedCopy)[[thisIx]]);
            netStrHereInit = netStrHere + addStr; #print(netStrHereInit);
            xData = dfHere[netStrHereInit]; linReg = LinearRegression(); linReg.fit(xData,yData); predModel = linReg.predict(xData);
            addedVarMatFour[modelNum,thisIx] = sklearn.metrics.explained_variance_score(yData,predModel) - rsqFour[modelNum];  
            
    # 220 models with 3 predictors each, k=3
    k = 3; numCombs = ncr(nNetsTotal,k); combosHere = list(combinations(np.arange(nNetsTotal),k)); rsqThree = np.zeros((numCombs)); 
    addedVarMatThree = np.zeros((numCombs,nNetsTotal)); addedVarMatThree[:] = np.NaN;
    for modelNum in range(numCombs):
        thisModelSet = np.asarray(combosHere[modelNum]); netStrHere = netsReduced.copy(); netStrHere = list(np.asarray(netsReduced)[thisModelSet]); #print(netStrHere)
        xData = dfHere[netStrHere]; linReg = LinearRegression(); linReg.fit(xData,yData); predModel = linReg.predict(xData);
        rsqThree[modelNum] = sklearn.metrics.explained_variance_score(yData,predModel);    
        refList = np.arange(nNetsTotal); missingPreds = np.asarray(list(set(refList).difference(thisModelSet)));
        nMissing = missingPreds.shape[0]; #netStrHereInit = netStrHere.copy();
        for heldOutNum in range(nMissing):
            thisIx = missingPreds[heldOutNum]; netsReducedCopy = netsReduced.copy(); addStr = list(np.asarray(netsReducedCopy)[[thisIx]]);
            netStrHereInit = netStrHere + addStr; #print(netStrHereInit);
            xData = dfHere[netStrHereInit]; linReg = LinearRegression(); linReg.fit(xData,yData); predModel = linReg.predict(xData);
            addedVarMatThree[modelNum,thisIx] = sklearn.metrics.explained_variance_score(yData,predModel) - rsqThree[modelNum];        

    # 66 Models with 2 predictors each, k=2
    k = 2; numCombs = ncr(nNetsTotal,k); combosHere = list(combinations(np.arange(nNetsTotal),k)); rsqTwo = np.zeros((numCombs)); 
    addedVarMatTwo = np.zeros((numCombs,nNetsTotal)); addedVarMatTwo[:] = np.NaN;
    for modelNum in range(numCombs):
        thisModelSet = np.asarray(combosHere[modelNum]); netStrHere = netsReduced.copy(); netStrHere = list(np.asarray(netsReduced)[thisModelSet]); #print(netStrHere)
        xData = dfHere[netStrHere]; linReg = LinearRegression(); linReg.fit(xData,yData); predModel = linReg.predict(xData);
        rsqTwo[modelNum] = sklearn.metrics.explained_variance_score(yData,predModel);   
        refList = np.arange(nNetsTotal); missingPreds = np.asarray(list(set(refList).difference(thisModelSet)));
        nMissing = missingPreds.shape[0]; netStrHereInit = netStrHere.copy(); 
        for heldOutNum in range(nMissing):
            thisIx = missingPreds[heldOutNum]; netsReducedCopy = netsReduced.copy(); addStr = list(np.asarray(netsReducedCopy)[[thisIx]]);
            netStrHereInit = netStrHere + addStr; #print(netStrHereInit);
            xData = dfHere[netStrHereInit]; linReg = LinearRegression(); linReg.fit(xData,yData); predModel = linReg.predict(xData);
            addedVarMatTwo[modelNum,thisIx] = sklearn.metrics.explained_variance_score(yData,predModel) - rsqTwo[modelNum];        

    # 12 models with 1 predictor each, k=1
    k = 1; numCombs = ncr(nNetsTotal,k); combosHere = list(combinations(np.arange(nNetsTotal),k)); rsqOne = np.zeros((numCombs)); 
    addedVarMatOne = np.zeros((numCombs,nNetsTotal)); addedVarMatOne[:] = np.NaN;
    for modelNum in range(numCombs):
        thisModelSet = np.asarray(combosHere[modelNum]); netStrHere = netsReduced.copy(); netStrHere = list(np.asarray(netsReduced)[thisModelSet]); #print(netStrHere)
        xData = dfHere[netStrHere]; linReg = LinearRegression(); linReg.fit(xData,yData); predModel = linReg.predict(xData);
        rsqOne[modelNum] = sklearn.metrics.explained_variance_score(yData,predModel);
        refList = np.arange(nNetsTotal); missingPreds = np.asarray(list(set(refList).difference(thisModelSet)));
        nMissing = missingPreds.shape[0]; netStrHereInit = netStrHere.copy();
        for heldOutNum in range(nMissing):
            thisIx = missingPreds[heldOutNum]; netsReducedCopy = netsReduced.copy(); addStr = list(np.asarray(netsReducedCopy)[[thisIx]]);
            netStrHereInit = netStrHere + addStr; #print(netStrHereInit);
            xData = dfHere[netStrHereInit]; linReg = LinearRegression(); linReg.fit(xData,yData); predModel = linReg.predict(xData);
            addedVarMatOne[modelNum,thisIx] = sklearn.metrics.explained_variance_score(yData,predModel) - rsqOne[modelNum];        

    # Combine means: note that the last row is the k=0 avg., which is equal to rsq's from each k=1 model
    modelMeans = np.reshape(np.concatenate((np.nanmean(addedVarMatOne,axis=0),np.nanmean(addedVarMatTwo,axis=0),np.nanmean(addedVarMatThree,axis=0),
                                            np.nanmean(addedVarMatFour,axis=0),np.nanmean(addedVarMatFive,axis=0),np.nanmean(addedVarMatSix,axis=0),
                                            np.nanmean(addedVarMatSeven,axis=0),np.nanmean(addedVarMatEight,axis=0),np.nanmean(addedVarMatNine,axis=0),
                                            np.nanmean(addedVarMatTen,axis=0),np.nanmean(addedVarMatEleven,axis=0),rsqOne)),(nNetsTotal,nNetsTotal));
    if not np.round(np.nansum(np.nanmean(modelMeans,axis=0)),10)==np.round(rsqFull,10): # should be true if conducted properly (precision diffs handled w/ round)
        print('Partial variances of model subsets do not add up to full r-sq for subject, please check.');

    # Results 
    netVarExplDom = np.nanmean(modelMeans,axis=0); # Use this for overall avg. partial r2 per network
    netVarImportance = (netVarExplDom / rsqFull) * 100; # Use this for percentage of relative importance of each network; should sum to 100%
    
    return rsqFull, netVarExplDom, netVarImportance