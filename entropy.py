# required packages
import numpy as np
import pandas as pd

# general Shannon entropy functions for joint and marginal entropies
def entropy(x, bins = 10):
    
    counts = x.value_counts(bins = bins)
    probs = counts / np.sum(counts)
    
    return - np.sum(probs * np.log2(probs))

def joint_entropy(x, y, bins = 10):
    
    x.reset_index(drop = True, inplace = True)
    y.reset_index(drop = True, inplace = True)
    combined = pd.concat([x, y], axis=1).dropna() #drop all rows with NA in both variables
    j_probs = pd.crosstab(pd.cut(combined.iloc[:, 0], bins = bins), 
                          pd.cut(combined.iloc[:, 1], bins = bins)) / len(combined)
    
    return - np.sum(np.sum(j_probs * np.log2(j_probs)))

def entropy_3d(x, y, z, bins = 10): # This doesn't work but is required for Transfer Ent
    
    combined = pd.concat([x, y, z], axis = 1).dropna()
    j_probs = pd.crosstab(pd.cut(combined.iloc[:, 0], bins = bins),
                          pd.cut(combined.iloc[:, 1], bins = bins),
                          pd.cut(combined.iloc[:, 2], bins = bins))
    
    return - np.sum(np.sum(j_probs * np.log2(j_probs)))

# Mutual information calculator
def mutual_information(x, y, bins = 10, normalize = True):
    
    Hx = entropy(x, bins = bins)
    Hy = entropy(y, bins = bins)
    Hxy = joint_entropy(x, y, bins = bins)
    
    if normalize == True:
        MI = (Hx + Hy - Hxy) / Hy
    else:
        MI = Hx + Hy - Hxy
        
    return MI

# Calculate a mutual information timeseries in monthly chunks
# includes loop for calculating shuffled surrogates    
def MI_timeseries(x_var, y_var, bins = 10, normalize = True,
                  runs = 100, alpha = 0.05, MC_runs = True):
    
    #combine variables and create monthly stamp
    combined = pd.concat([x_var, y_var], axis=1).dropna()
    combined['Yr_mnth'] = combined.index.strftime('%Y%m')
    cols = combined.columns
    
    #subset data my month, calculate and save mutual information
    time_index = combined['Yr_mnth'].unique()
    MI_out = np.empty(len(time_index)) #empty MI array
    
    if MC_runs == True:
        MI_MC = np.empty([len(time_index), runs]) #raw MC runs matrix
    
    for i in range(len(time_index)):
        
        monthly = combined[combined.Yr_mnth == time_index[i]]
        MI_out[i] = mutual_information(monthly[cols[0]], monthly[cols[1]], 
                                       bins = bins, normalize=normalize)
        
        if MC_runs == True:
            
            for j in range(runs):
                x_shuffle = monthly[cols[0]].sample(len(monthly), replace = False) #shuffle x
                
                MI_MC[i, j] = mutual_information(x_shuffle, monthly[cols[1]], 
                                                 bins = bins, normalize = normalize)         
    
    if MC_runs == True:
        MI_MC = np.percentile(MI_MC, q = (1 - alpha) * 100, axis = 1)                
        return MI_out, MI_MC
    
    return MI_out
