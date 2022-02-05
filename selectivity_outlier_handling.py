# C. Cocuzza, 2021
# Function to identify outlier participants in category selectivity (MAF project).
# Using a very conservative number of deviations above or below the median: 5 
# See here for more: Leys, C., Ley, C., Klein, O., Bernard, P., & Licata, L. (2013). Detecting outliers: Do not use standard deviation around the mean, use absolute deviation around the median. Journal of Experimental Social Psychology, 49(4), 764–766. https://doi.org/10.1016/j.jesp.2013.03.013

# IMPORTS 
import numpy as np
import scipy.stats as stats 

# FUNCTION 
def selectivity_outlier_handling(data_actual,data_predicted,bidirectional=True,num_devs=5):
    '''
    INPUT: 
        data_actual: actual (empirical) category selectivity scores, vector of subject data (shape = N or sample size)
        data_predicted: act-flow predicted category selectivity scores, vector of subject data (shape = N or sample size)
        bidirectional: boolean, if True will detect and remove outliers above and below median; if false will only detect outliers above median 
        num_devs: number of deviations from the median to look for outliers; default is 5 (very conservative; suggested minimum number to use here is 3)
    
    OUTPUT: 
        outlier_subjs_act: indices of outlier subjects in actual data (uses python indexing starting at 0)
        outlier_subjs_predicted: indices of outlier subjects in predicted data (uses python indexing starting at 0)
        outlier_subjs_both: indices of unique (non-overlapping) outlier subjects across both actual and predicted data (uses python indexing starting at 0); sorted
    '''    
    # Compute median absolute deviation (MAD), see scipy.stats.median_abs_deviation for more on scale factor etc. 
    pred_MAD = stats.median_abs_deviation(data_predicted,nan_policy='omit',scale='normal')
    act_MAD = stats.median_abs_deviation(data_actual,nan_policy='omit',scale='normal')
    
    # Find positive and negative thresholds 
    pred_MAD_pos = np.nanmedian(data_predicted) + (pred_MAD * num_devs)
    pred_MAD_neg = np.nanmedian(data_predicted) - (pred_MAD * num_devs)
    act_MAD_pos = np.nanmedian(data_actual) + (act_MAD * num_devs)
    act_MAD_neg = np.nanmedian(data_actual) - (act_MAD * num_devs)
    
    # Find outlier (subjects) indices 
    if bidirectional:
        outlier_subjs_pred = np.where(np.logical_or(data_predicted>=pred_MAD_pos,data_predicted<=pred_MAD_neg))[0]
        outlier_subjs_act = np.where(np.logical_or(data_actual>=act_MAD_pos,data_actual<=act_MAD_neg))[0]
        
    elif not bidirectional:
        outlier_subjs_pred = np.where(data_predicted>=pred_MAD_pos)[0]
        outlier_subjs_act = np.where(data_actual>=act_MAD_pos)[0]
    
    # Combine outlier subjects across actual and predicted data-sets (needed downstream to compute % distributed; so there's the same # of subjects)
    outlier_subjs_both = np.sort(np.unique(np.concatenate((outlier_subjs_act,outlier_subjs_pred))));

    return outlier_subjs_act, outlier_subjs_pred, outlier_subjs_both, act_MAD_pos, act_MAD_neg, pred_MAD_pos, pred_MAD_neg 