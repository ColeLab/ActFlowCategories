# C. Cocuzza, 2021
# Function to compute category selectivity: quantifying the selective responsiveness of a 
# given set of nodes (vertices, functional ROIs, etc) to a semantic category. 
# EX: the regions comprising the FFA to images of faces. NOTE: this may also be applicable 
# to non-semantic-category conditions. 


# This version ("category_selectivity.py") of the function is used for when category selectivity is computed 
# for both predicted and actual data. The modification here is such that the min(x) and max(x) of the predicted data 
# is based on the actual data, so that the predicted data is not overly driven by distributed processing. 
# This is a specific consideration for activity flow predictions and the later computation (which uses category 
# selectivity scores) of % distributed processes; so that this estimate is not inflated. 
# In the other "category_selectivity_perSet" version, the data input is considered in isolation. This can be used for
# computing category selectivity for any individual vector.

# IMPORTS 
import numpy as np
np.seterr(divide='ignore', invalid='ignore') # legacy setting; from when using NaNs to replace potential infs
import sklearn.preprocessing as skp
import math

# FUNCTION 
def category_selectivity(data_actual,data_predicted,category_indices,noncategory_indices,norm_method='min-max',min_val=0,max_val=1,hcp_nbacks=True): 
    '''
    INPUT: 
        data_actual: neural data (e.g., actual activations) in shape of nodes x conditions or categories (nodes can be vertices, regions, etc.); can also have 3rd dim of subjects 
        data_predicted: neural data (e.g., activity-flow-predicted activations) in shape of nodes x conditions or categories (nodes can be vertices, regions, etc.); can also have 3rd dim of subjects. NOTE: predicted and actual data need to be the same shape.
        category_indices: index values for category of interest, in list or array (integers required); how to index 2nd dimension of data for the category you'd like to measure selectivity for (ex: indices for face categories)
        noncategory_indices: index values for non-category(ies) of interest, in list or array (integers required); how to index 2nd dimension of data for the reference category(ies) to compare against to obtain category selectivity (ex: indices for non-face categories)
        norm_method: either 'min-max' (i.e., min-max feature scaling / normalization / unity) or 'none' (no normalization performed in-function; may consider using if you have your own norming method)
        min_val: any non-zero (suggested to also use positive) number to set the lower bound of the rescaled data to. 
        max_val: any non-zero (suggested to also use positive; must use a number larger than min_val) number to set the upper bound of the rescaled data to.
        hcp_nbacks: whether or not to average across hcp n-back conditions, suggested to keep infs (which will be set to nan) from data (set to False if using a different dataset; TBA: modification that can handle flexible avg'ing dims)
    
    OUTPUT: 
        selectivity_score_actual: the selectivity estimate for the given data (actual)
        selectivity_score_predicted: the selectivity estimate for the given data (predicted)
        selectivity_array_actual: the expanded version of the above (actual); suggested to use for when you want to probe the scores for any reason
        selectivity_array_predicted: the expanded version of the above (predicted); suggested to use for when you want to probe the scores for any reason
    '''
    # Get dimensions (note that actual and predicted data need to be the same shape; TBA: an error to catch this)
    num_nodes = data_actual.shape[0]
    num_conditions = data_actual.shape[1]
    if np.ndim(data_actual)>2:
        num_subjs = data_actual.shape[2]
    rIX = 2 # rounding index for selectivity scores 
    
    # (1) normalize data: min-max feature scaling with specified range (so that downstream step doesn't return inf if denom is 0; any non-neg. min and max's should work)
    if norm_method=='min-max':
        if np.ndim(data_actual)>2: 
            data_norm_actual = np.zeros((num_nodes,num_conditions,num_subjs))
            data_norm_predicted = np.zeros((num_nodes,num_conditions,num_subjs))
            for subj_num in range(num_subjs):
                for nodeIx in range(num_nodes):
                    dataHere_actual = data_actual[nodeIx,:,subj_num]
                    dataHere_predicted = data_predicted[nodeIx,:,subj_num]
                    data_norm_actual_vec = (((dataHere_actual - np.min(dataHere_actual))) / (np.max(dataHere_actual)-np.min(dataHere_actual)))
                    data_norm_actual[nodeIx,:,subj_num] = skp.minmax_scale(data_norm_actual_vec,feature_range=(min_val,max_val))
                    data_norm_predicted_vec = (((dataHere_predicted - np.min(dataHere_actual))) / (np.max(dataHere_actual)-np.min(dataHere_actual)))
                    data_norm_predicted[nodeIx,:,subj_num] = skp.minmax_scale(data_norm_predicted_vec,feature_range=(min_val,max_val))  
                
        elif np.ndim(data_actual)==2:
            data_norm_actual = np.zeros((num_nodes,num_conditions))
            data_norm_predicted = np.zeros((num_nodes,num_conditions))
            for nodeIx in range(num_nodes):
                dataHere_actual = data_actual[nodeIx,:,:]
                dataHere_predicted = data_predicted[nodeIx,:,:]
                data_norm_actual_vec = (((dataHere_actual - np.min(dataHere_actual))) / (np.max(dataHere_actual)-np.min(dataHere_actual)))
                data_norm_actual[nodeIx,:] = skp.minmax_scale(data_norm_actual_vec,feature_range=(min_val,max_val))
                data_norm_predicted_vec = (((dataHere_predicted - np.min(dataHere_actual))) / (np.max(dataHere_actual)-np.min(dataHere_actual)))
                data_norm_predicted[nodeIx,:] = skp.minmax_scale(data_norm_predicted_vec,feature_range=(min_val,max_val))
                
    elif norm_method=='none':
        data_norm_actual = data_actual.copy()
        data_norm_predicted = data_predicted.copy()
        
 
    # (2) compute category selectivity 
    if hcp_nbacks: 
        # the paired HCP n-back conditions to non-categories (with noncategory_indices considered)
        non_ix_1 = [0,3]
        non_ix_2 = [1,4]
        non_ix_3 = [2,5]
        num_non_cats = 3
        if np.ndim(data_actual)>2:
            act_mat = np.zeros((num_nodes,num_non_cats,num_subjs))
            pred_mat = np.zeros((num_nodes,num_non_cats,num_subjs))
            
            act_mat[:,0,:] = (np.nanmean(np.round(data_norm_actual[:,category_indices,:],rIX),axis=1)/np.nanmean(np.round(data_norm_actual[:,noncategory_indices,:],rIX)[:,non_ix_1,:],axis=1))
            act_mat[:,1,:] = (np.nanmean(np.round(data_norm_actual[:,category_indices,:],rIX),axis=1)/np.nanmean(np.round(data_norm_actual[:,noncategory_indices,:],rIX)[:,non_ix_2,:],axis=1))
            act_mat[:,2,:] = (np.nanmean(np.round(data_norm_actual[:,category_indices,:],rIX),axis=1)/np.nanmean(np.round(data_norm_actual[:,noncategory_indices,:],rIX)[:,non_ix_3,:],axis=1))
            r,c,d1 = np.where(np.isinf(act_mat))
            act_mat[r,c,d1] = math.nan
            selectivity_score_actual = np.nanmean(np.nanmean(act_mat,axis=0),axis=0)
            selectivity_array_actual = act_mat.copy()

            pred_mat[:,0,:] = (np.nanmean(np.round(data_norm_predicted[:,category_indices,:],rIX),axis=1)/np.nanmean(np.round(data_norm_predicted[:,noncategory_indices,:],rIX)[:,non_ix_1,:],axis=1))
            pred_mat[:,1,:] = (np.nanmean(np.round(data_norm_predicted[:,category_indices,:],rIX),axis=1)/np.nanmean(np.round(data_norm_predicted[:,noncategory_indices,:],rIX)[:,non_ix_2,:],axis=1))
            pred_mat[:,2,:] = (np.nanmean(np.round(data_norm_predicted[:,category_indices,:],rIX),axis=1)/np.nanmean(np.round(data_norm_predicted[:,noncategory_indices,:],rIX)[:,non_ix_3,:],axis=1))
            r,c,d1 = np.where(np.isinf(pred_mat))
            pred_mat[r,c,d1] = math.nan
            selectivity_score_predicted = np.nanmean(np.nanmean(pred_mat,axis=0),axis=0)
            selectivity_array_predicted = pred_mat.copy()
            
        elif np.ndim(data_actual)==2:
            act_mat = np.zeros((num_nodes,num_non_cats))
            pred_mat = np.zeros((num_nodes,num_non_cats))
            
            act_mat[:,0] = (np.nanmean(np.round(data_norm_actual[:,category_indices],rIX),axis=1)/np.nanmean(np.round(data_norm_actual[:,noncategory_indices],rIX)[:,non_ix_1,:],axis=1))
            act_mat[:,1] = (np.nanmean(np.round(data_norm_actual[:,category_indices],rIX),axis=1)/np.nanmean(np.round(data_norm_actual[:,noncategory_indices],rIX)[:,non_ix_2,:],axis=1))
            act_mat[:,2] = (np.nanmean(np.round(data_norm_actual[:,category_indices],rIX),axis=1)/np.nanmean(np.round(data_norm_actual[:,noncategory_indices],rIX)[:,non_ix_3,:],axis=1))
            r,c = np.where(np.isinf(act_mat))
            act_mat[r,c] = math.nan
            selectivity_score_actual = np.nanmean(np.nanmean(act_mat,axis=0))
            selectivity_array_actual = act_mat.copy()

            pred_mat[:,0] = (np.nanmean(np.round(data_norm_predicted[:,category_indices],rIX),axis=1)/np.nanmean(np.round(data_norm_predicted[:,noncategory_indices],rIX)[:,non_ix_1,:],axis=1))
            pred_mat[:,1] = (np.nanmean(np.round(data_norm_predicted[:,category_indices],rIX),axis=1)/np.nanmean(np.round(data_norm_predicted[:,noncategory_indices],rIX)[:,non_ix_2,:],axis=1))
            pred_mat[:,2] = (np.nanmean(np.round(data_norm_predicted[:,category_indices],rIX),axis=1)/np.nanmean(np.round(data_norm_predicted[:,noncategory_indice:],rIX)[:,non_ix_3,:],axis=1))
            r,c = np.where(np.isinf(pred_mat))
            pred_mat[r,c] = math.nan
            selectivity_score_predicted = np.nanmean(np.nanmean(pred_mat,axis=0))
            selectivity_array_predicted = pred_mat.copy()

    elif not hcp_nbacks:
        tempArr_actual = np.zeros((num_nodes,len(category_indices),len(noncategory_indices)))
        tempArr_predicted = np.zeros((num_nodes,len(category_indices),len(noncategory_indices)))
        for nodeIx in range(num_nodes):
            for catIx in range(len(category_indices)):
                for nonCatIx in range(len(noncategory_indices)):
                    data_cat_actual = data_norm_actual[nodeIx,category_indices][catIx]
                    data_noncat_actual = data_norm_actual[nodeIx,noncategory_indices][nonCatIx]
                    tempArr_actual[nodeIx,catIx,nonCatIx] = (data_cat_actual / data_noncat_actual) 

                    data_cat_predicted = data_norm_predicted[nodeIx,category_indices][catIx]
                    data_noncat_predicted = data_norm_predicted[nodeIx,noncategory_indices][nonCatIx]
                    tempArr_predicted[nodeIx,catIx,nonCatIx] = (data_cat_predicted / data_noncat_predicted) 

        selectivity_array_actual = tempArr_actual.copy()
        selectivity_array_predicted = tempArr_predicted.copy()
        selectivity_score_actual = np.nanmean(tempArr_actual)
        selectivity_score_predicted = np.nanmean(tempArr_predicted)
    
    # note: scores will be scalars if done per subject, but will be a N (sample size) vector if entered 3 dimensional data (3rd dim = subjects)
    return selectivity_score_actual, selectivity_score_predicted, selectivity_array_actual, selectivity_array_predicted