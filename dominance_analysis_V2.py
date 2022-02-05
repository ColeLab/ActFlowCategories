# Dominance analysis, following math here: https://github.com/dominance-analysis/dominance-analysis
# C. Cocuzza, 2021

# This version is flexible to inputting difference sources

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


def dominance_analysis_slurm(predictedData_ThisSubj,
                             actualData_ThisSubj,
                             kFull,
                             concatTargets=False,
                             printModels=False,
                             targetNode=None):
    '''
    INPUTS:
        predictedData_ThisSubj: target node (IV) x source node (DVs) x conditions/tasks; predicted from AF model
        actualData_ThisSubj: target node (IV) x conditions/task; actual data 
        kFull: scalar; number of source nodes / IVs / predictors (can be networks, source regions, etc.) 
        concatTargets: boolean; if true, targets treated as a complex/concatenated; if false, iterated over (takes longer)
        printModels: boolean; if true will print model numbers and how many subsets 
        targetNode: if using concatTargets, don't set this to anything; if not, then set it to the target node you wish to assess 
    
    OUTPUTS: 
        rsqFull: the full model explained variance. 
        netVarExplDom: unmixed partial explained variance per source 
        netVarImportance: relative importance of each source/DV to the full model.
        addedVarMat_Dict: dictionary saving out all variances (~ full table in github linked above)
        rsqDict: dictionary saving out all rsq components 
    '''
    
    # Extract modeling specs from input data 
    kVec = np.arange(0,kFull); kAllRegs = kVec[-1]; 

    # Where number of sources = number of regressors in full model (also=kFull)
    nTargets, nSources, nTasks = predictedData_ThisSubj.shape; 
    
    # Either concatenate target nodes or iterate
    if concatTargets:
        predDataHere = np.reshape(predictedData_ThisSubj,(nSources,(nTasks*nTargets))); 
        actDataHere = np.reshape(actualData_ThisSubj,(nTasks*nTargets));
    else: 
        predDataHere = predictedData_ThisSubj[targetNode,:,:]; 
        actDataHere = actualData_ThisSubj[targetNode,:];
    
    # Put data into a dataframe that linear regression can pull from 
    dataDict = {}; allStrs = [];
    for sourceNode in range(nSources): # sourceNodes can be nets or source regions, etc. 
        thisKey = 'source_' + str(sourceNode); 
        allStrs.append(thisKey);
        dataDict[thisKey] = predDataHere[sourceNode,:].copy(); 
    dataDict['target'] = actDataHere.copy(); 
    dataDF = pd.DataFrame(dataDict);

    # Set up DV (target) and IVs (sources)
    xData = dataDF[allStrs]; # IVs/regressors/predictors/sources
    yData = dataDF['target']; # DV/outcome/response/target

    # Log number of subset models that will be tested
    if printModels:
        modelLog = np.zeros((nSources)); modelLog[0] = 1;
        for k in range(1,nSources):
            numCombs = ncr(nSources,k); modelLog[k] = numCombs; 
        print('Across ' + str(nSources) + ' iterations, there will be ' + str(int(np.sum(modelLog))) + 
              ' subset models assessed.');
    
    # Full model: get full variance explained
    linReg = LinearRegression();
    linReg.fit(xData,yData);
    predModel = linReg.predict(xData);
    rsqFull = sklearn.metrics.explained_variance_score(yData,predModel);
    
    if printModels:
        print(str(nSources) + ' regressors explain ' + str(np.round(rsqFull*100,2)) + 
              '% of variance for the given response variable.\n');

    # Iterate over all other sub-models 
    addedVarMat_Dict = {}; rsqDict = {}; 
    for k in range(1,nSources):
    #for k in range(1,5):
        numCombs = ncr(kFull,k);
        if printModels:
            print('running subset k = ' + str(k) + ', number of models = ' + str(numCombs));
        combosHere = list(combinations(np.arange(nSources),k))
        rsqHere = np.zeros((numCombs));
        addedVarMatHere = np.zeros((numCombs,nSources));
        addedVarMatHere[:] = np.NaN;

        for modelNum in range(numCombs):
            thisModelSet = np.asarray(combosHere[modelNum]);
            netStrHere = allStrs.copy();
            netStrHere = list(np.asarray(allStrs)[thisModelSet]);
            xData = dataDF[netStrHere]; 
            linReg = LinearRegression();
            linReg.fit(xData,yData); 
            predModel = linReg.predict(xData);
            rsqHere[modelNum] = sklearn.metrics.explained_variance_score(yData,predModel);
            refList = np.arange(nSources);
            missingPreds = np.asarray(list(set(refList).difference(thisModelSet)));
            nMissing = missingPreds.shape[0];
            for heldOutNum in range(nMissing):
                thisIx = missingPreds[heldOutNum];
                netsReducedCopy = allStrs.copy();
                addStr = list(np.asarray(netsReducedCopy)[[thisIx]]);
                netStrHereInit = netStrHere + addStr;
                xData = dataDF[netStrHereInit];
                linReg = LinearRegression();
                linReg.fit(xData,yData);
                predModel = linReg.predict(xData);
                addedVarMatHere[modelNum,
                                thisIx] = sklearn.metrics.explained_variance_score(yData,predModel)-rsqHere[modelNum];
        addedVarMat_Dict[k] = addedVarMatHere; 
        rsqDict[k] = rsqHere;
    
    # Combine means: note that the last row is the k=0 avg., which is equal to rsq's from each k=1 model
    combinedMeans = np.zeros((nSources * nSources)); initIx = 0;
    for k in range(1,nSources):
        startIx = initIx; endIx = startIx + nSources;
        combinedMeans[startIx:endIx] = np.nanmean(addedVarMat_Dict[k],axis=0);

        initIx = endIx;

    # add in last row 
    combinedMeans[endIx:] = rsqDict[1];
    # reshape to square 
    modelMeans = np.reshape(combinedMeans,(nSources,nSources));

    # should be true if conducted properly (precision diffs handled w/ round)
    if not np.round(np.nansum(np.nanmean(modelMeans,axis=0)),10)==np.round(rsqFull,10): 
        print('Partial variances of model subsets do not add up to full r-sq for subject, please check.');

    # Final results
    # overall avg. partial r2 per source
    netVarExplDom = np.nanmean(modelMeans,axis=0);
    # Use this for % of relative importance of each source; should sum to 100%
    netVarImportance = (netVarExplDom / rsqFull) * 100;

    return rsqFull, netVarExplDom, netVarImportance, addedVarMat_Dict, rsqDict

########################################################################
# Combination function
def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom