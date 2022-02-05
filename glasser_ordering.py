# C. Cocuzza, 2021

# A function to take a vector of 360 (Glasser et al, 2016; MMP Atlas) node's data (e.g., activation estimates) 
# that have been sorted/ordered into CAB-NP assignments (Ji et al., 2019; https://github.com/ColeLab/ColeAnticevicNetPartition) and 
# un-order back into original MMP ordering

# Main use: to properly order nodes so they can be visualized on a connectome (HCP) workbench brain schematic; either 
# via wb_view gui (https://www.humanconnectome.org/software/connectome-workbench) or in-notebook via wbplot (https://pypi.org/project/wbplot/)

################################################
# MODULES 
import numpy as np
import nibabel as nib
import scipy.io as spio
import math as math

################################################
# HELPER VARIABLES  ***** TBA: make these available on MAF project GitHub & change call to Amarel directory (Cole lab HPC) *****

dirHere = '/projects/f_mc1689_1/MovieActFlow/docs/scripts/HCP_3T_7Task/'

# General ordering info relating MMP and CAB-NP
nodeIndices = spio.loadmat(dirHere + 'nodeIndices.mat')['nodeIndices'][:,0] # 360,, network assignment (CAB-NP) numbers 1-12
nodeOrder = (spio.loadmat(dirHere + 'nodeOrder.mat')['nodeOrder'] - 1)[:,0] # 360, node order into networks (CAB-NP) 

# Left nodes: 32492 vector, left hemisphere cortical vertex labels (label = Glasser parcel, or NaN)
ciftiFileLeft = dirHere + 'Q1-Q6_RelatedValidation210.L.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii'
leftNodes = nib.load(ciftiFileLeft).get_fdata()
leftNodes = np.squeeze(leftNodes)

# Right nodes: 32492 vector, right hemisphere cortical vertex labels (label = Glasser parcel, or NaN)
ciftiFileRight = dirHere + 'Q1-Q6_RelatedValidation210.R.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii'
rightNodes = nib.load(ciftiFileRight).get_fdata()
rightNodes = np.squeeze(rightNodes)

# Network references: 96854 vector, whole brain vertex labels (label = CABNP network 1-12, or NaN)
netRef = spio.loadmat(dirHere + 'netRefNew.mat')['netRefNew'] # 96854 x 1, whole brain vertex labels (label = CABNP network 1-12, or NaN); NOTE: hand fixed 

nParcels = 360
numNets = 12
numTotal = 96854

################################################
# FUNCTION
def glasser_ordering(dataVectorSorted):
    '''
    INPUT: 
        dataVectorSorted: a vector of data (e.g., activity, network metrics, contrast values, predicted activity, etc. etc.) for 360 Glasser (MMP) nodes sorted into CAB-NP networks, in size 360,
    
    OUTPUT: 
        dataVector: properly un-ordered version of dataVectorSorted, so that Glasser parcel numbers match what workbench expects (ie un-order to MMP space)
        dataVector_Vertices: same as above, at vertex level for use in workbench 
    '''
    ################################################
    # SET-UP
    orderedVec = nodeIndices[nodeOrder]
    newOrder = np.zeros((nParcels))
    for netNum in range(1,numNets+1):
        orderedFinder = np.where(orderedVec==netNum)[0]
        refFinder = np.where(nodeIndices==netNum)[0]
        newOrder[refFinder] = orderedFinder
    vecReOrdered = dataVectorSorted[newOrder.astype(int)]

    # Write over vertex labels with parcel-appropriate dataVectorSorted values, L-->R
    numBalsa = len(leftNodes)
    maxBalsa = numBalsa * 2
    netRefLeft = netRef[:numBalsa]
    netRefRight = netRef[numBalsa:maxBalsa]
    newRegionVec = np.zeros((numTotal))
    
    # Left hemisphere
    for vertNum in range(numBalsa):
        parcelIx = leftNodes[vertNum]
        testNaN = parcelIx==0
        if testNaN:
            newRegionVec[vertNum] = math.nan
            newRegionVec[vertNum] = 0
        else:
            adjIndexHemi = parcelIx - (nParcels/2)
            newRegionVec[vertNum] = vecReOrdered[adjIndexHemi.astype(int) - 1] # -1 for python

    # Right hemisphere
    for vertNum in range(numBalsa,maxBalsa):
        adjIndex = vertNum - numBalsa
        parcelIx = rightNodes[adjIndex]
        testNaN = parcelIx==0
        if testNaN:
            newRegionVec[vertNum] = math.nan
            newRegionVec[vertNum] = 0
        else:
            adjIndexHemi = parcelIx + (nParcels/2)
            newRegionVec[vertNum] = vecReOrdered[adjIndexHemi.astype(int) - 1] # -1 for python
    newRegionVec[maxBalsa:] = math.nan
    
    dataVector = vecReOrdered.copy()
    dataVector_Vertices = newRegionVec.copy()
    
    return dataVector, dataVector_Vertices