import numpy as np
from scipy.stats import beta
from scipy.stats import dirichlet
from scipy import linalg
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


########################################################################################
# This is one function to find the MIchaelis Tranzformation
#########################################################################################
def calc_V(VNEList):
    vRecipricalArray = []    
    
    Michaelis_constant = 0
    # This goes through and shows the elements of each simulation for specific rho, gamma, and beta combinations
    # and then it calculates the 'V' variable needed to send to the Lineweaver Burke file so we can plot it.
    maximum_element = np.amax(VNEList)
    half_element = maximum_element/2
 #   print('max then half\n', maximum_element, '\n', half_element)
    for t in range(len(VNEList)):
 #       print(VNEList[t])
        if VNEList[t] >= half_element and Michaelis_constant == 0:
            Michaelis_constant = t
            continue
    for s in range(len(VNEList)):
        numeratorValue = maximum_element*(s+1)
        denominatorValue = Michaelis_constant + s + 1
                                
        # We're combining this way so that we can appropriately graph it accurately since
        # the axis that we're graphing on is the reciprical
        vRecipricalArray.append(denominatorValue/numeratorValue)
        
    return(vRecipricalArray)
    
####################################################################################################
# This code brings in the numpy arrays for us to use
####################################################################################################
    
## This block of code loads a desired np.array so we don't have to run the simulations
realIxSVNE = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Real_data_work\09Data\VNEValues\IxSVNERealData09.npy')
realIxIVNE = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Real_data_work\09Data\VNEValues\IxIVNERealData09.npy')
realTripartiteVNE = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Real_data_work\09Data\VNEValues\tripartiteVNERealData09.npy')
realHomerangeVNE = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Real_data_work\09Data\VNEValues\homerangeVNERealData09.npy')
realSxSVNE = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Real_data_work\09Data\VNEValues\SxSVNERealData09.npy')


###########################################################################################
# This will calculate the MIchaelis constant so we can find an estimated line and 
# use the Lineweaver Burke Linearization
###########################################################################################
# These are the calculated Michaleis outputs for the data
vRecipricalIxS = calc_V(realIxSVNE)            
vRecipricalIxI = calc_V(realIxIVNE)            
vRecipricalTripartite = calc_V(realTripartiteVNE)
vRecipricalHomerange = calc_V(realHomerangeVNE)
vRecipricalSxS = calc_V(realSxSVNE)

# This is computing the Michaelis inputs for each output already calculated
michaelisInputList = 1/(np.arange(len(vRecipricalIxS)) + 1)

# These save the arrays so I can call it in another file.
np.save('realDataMichaelisInputs', michaelisInputList)
np.save('realDataIxSOutputs', vRecipricalIxS)
np.save('realDataIxIOutputs', vRecipricalIxI)
np.save('realDataTripartiteOutputs', vRecipricalTripartite)
np.save('realDataHomerangeOutputs', vRecipricalHomerange)
np.save('realDataSxSOutputs', vRecipricalSxS)
