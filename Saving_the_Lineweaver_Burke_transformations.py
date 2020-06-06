import numpy as np
import pandas as pd

########################################################################################
# This is one function to find the MIchaelis Tranzformation
#########################################################################################
def calc_V(collection, time_stamps, amount_of_simulations):
    
    # This goes through and shows the elements of each simulation for specific rho, gamma, and beta combinations
    # and then it calculates the 'v' variable needed to get a Lineweaver-Burk plot
    full_v_array = []
    # These 4 for loop are trying to get to the data.
    for j in range(amount_of_simulations):
        per_simulation_v_array = []
        for i in range(len(gamma_values)):
            per_gamma_v_array = []
            for k in range(len(beta_values)):
                per_beta_v_array = []
                for z in range(len(rho_values)):
                    Michaelis_constant = 0
                    # Grabbing VNE values for a specific simulation, gamma, beta, rho, combination
                    all_VNE_for_a_gamma_beta_rho_combination_per_simulation = collection[j][i][k][z]
                    maximum_element = np.amax(all_VNE_for_a_gamma_beta_rho_combination_per_simulation)
                    minimum_element = np.amin(all_VNE_for_a_gamma_beta_rho_combination_per_simulation)
                    
                    # There were some that had multiple zero values for VNE so this next condition
                    # statement helps sift through that
                    if minimum_element == 0 or minimum_element == -0:
                        copied_list = all_VNE_for_a_gamma_beta_rho_combination_per_simulation.copy()
                        sorted_list = sorted(copied_list)
                        for q in range(len(all_VNE_for_a_gamma_beta_rho_combination_per_simulation)):
                            minimum_element = sorted_list[q]
                            if minimum_element != 0 and minimum_element != -0:
                                break
                    # Creating Middle element used to compute the Michaelis Constant
                    middle_element = maximum_element/2
                    per_rho_v_array = []
                    # Finding the Michaelis constant
                    for t in range(time_stamps):
                        if collection[j][i][k][z][t] >= middle_element and Michaelis_constant == 0:
                            Michaelis_constant = collection[j][i][k][z][t]
                    for s in range(time_stamps):
                        numerator_per_gamma_beta_rho_combination = maximum_element*(s+1)
                        denominator_per_gamma_beta_rho_combination = Michaelis_constant + s + 1
                                                
                        # We're combining this way so that we can appropriately graph it accurately since
                        # the axis that we're graphing on is the reciprical
                        vOutputs = denominator_per_gamma_beta_rho_combination/numerator_per_gamma_beta_rho_combination
                                                
                        per_rho_v_array.append(vOutputs)

                    per_beta_v_array.append(per_rho_v_array)

                per_gamma_v_array.append(per_beta_v_array)

            per_simulation_v_array.append(per_gamma_v_array)

        full_v_array.append(per_simulation_v_array)

    return(full_v_array)
    
####################################################################################################
# This is the 'main' function
####################################################################################################
location_amount = 10
individual_amount = 10
time_stamps = 20
gamma_values = np.linspace(1, 10, 10, endpoint = True)
beta_values = np.linspace(1, 10, 10, endpoint = True)
rho_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 0.99, 0.999, 0.9999, 1.0]#np.linspace(0.1, 1.0, 10, endpoint=True)
simulation_amount = 10

full_IxS_collection = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\IxS10By10.npy')
full_IxI_collection = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\IxI10By10.npy')
full_tripartite_collection = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Tripartite10By10.npy')
full_homerange_IxI_collection = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Homerange10By10.npy')
full_SxS_collection = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\SxS10By10.npy')

###########################################################################################
# This will calculate the MIchaelis constant so we can find an estimated line and 
# use the Lineweaver Burke Linearization
###########################################################################################
# These are the calculated Michaleis outputs for the data
outputs_IxS = calc_V(full_IxS_collection, time_stamps, simulation_amount)            
outputs_IxI = calc_V(full_IxI_collection, time_stamps, simulation_amount)            
outputs_tripartite = calc_V(full_tripartite_collection, time_stamps, simulation_amount)
outputs_homerange_IxI = calc_V(full_homerange_IxI_collection, time_stamps, simulation_amount)
outputs_SxS = calc_V(full_SxS_collection, time_stamps, simulation_amount)

np.save('IxSVOutputs10By10', outputs_IxS)
np.save('IxIVOutputs10By10', outputs_IxI)
np.save('TripartiteVOutputs10By10', outputs_tripartite)
np.save('homerangeIxIVOutputs10By10', outputs_homerange_IxI)
np.save('SxSVOutputs10By10', outputs_SxS)
