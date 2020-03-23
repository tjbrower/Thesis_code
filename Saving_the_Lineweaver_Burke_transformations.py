import numpy as np
from scipy.stats import beta
from scipy.stats import dirichlet
from scipy import linalg
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


location_amount = 25
#location_amount = 6
individual_amount = 25
#individual_amount = 4
time_stamps = 20
#groups = 5
gamma_values = np.linspace(1, 10, 10, endpoint = True)
beta_values = np.linspace(1, 10, 10, endpoint = True)
rho_values = np.linspace(0.1, 1.0, 10, endpoint=True)
#simulation_amount = 10
simulation_amount = 10

########################################################################################
# This is one function to find the MIchaelis Tranzformation
#########################################################################################
def calc_V(collection, time_stamps, amount_of_simulations, variable):
    
    # This chunk calculates the 'midpoint' between the lowest and largest VNE
    if variable == 1:
        display = 'IxS!!!'
    elif variable == 2:
        display = 'IxI!!!'
    elif variable == 3:
        display = 'Tripartite!!!'
    elif variable == 4:
        display = 'HOMERANGE IxI!!!'
    elif variable == 5:
        display = 'SxS!!!!!!'
    
    Michaelis_constant = 0
    # This goes through and shows the elements of each simulation for specific rho, gamma, and beta combinations
    # and then it calculates the 'V' variable needed to send to the Lineweaver Burke file so we can plot it.
    full_v_array = []
    for j in range(amount_of_simulations):
        per_simulation_v_array = []
        for i in range(len(gamma_values)):
            per_gamma_v_array = []
            for k in range(len(beta_values)):
                per_beta_v_array = []
                for z in range(len(rho_values)):
                    all_VNE_for_a_gamma_beta_rho_combination_per_simulation = collection[j][i][k][z]
                    maximum_element = np.amax(all_VNE_for_a_gamma_beta_rho_combination_per_simulation)
                    minimum_element = np.amin(all_VNE_for_a_gamma_beta_rho_combination_per_simulation)
                    if minimum_element == 0 or minimum_element == -0:
                        copied_list = all_VNE_for_a_gamma_beta_rho_combination_per_simulation.copy()
                        sorted_list = sorted(copied_list)
 #                       print('this is the sorted list\n', sorted_list)
                        for q in range(len(all_VNE_for_a_gamma_beta_rho_combination_per_simulation)):
                        #print('This should be the list sorted\n', sorted_list)
                            minimum_element = sorted_list[q]
                            if minimum_element != 0 and minimum_element != -0:
 #                               print('I should be breaking')
                                break
#                            print('did I break?')
#                        print('still in the minimum element = 0 section')
                    distance = maximum_element - minimum_element
                    middle_element = distance/2 + minimum_element
                    per_rho_v_array = []
                    for t in range(time_stamps):
                        if collection[j][i][k][z][t] >= middle_element and Michaelis_constant == 0:
                            Michaelis_constant = collection[j][i][k][z][t]
                    for s in range(time_stamps):
                        numerator_per_gamma_beta_rho_combination = maximum_element*(s+1)
                        denominator_per_gamma_beta_rho_combination = Michaelis_constant + s + 1
                                                
                        # We're combining this way so that we can appropriately graph it accurately since
                        # the axis that we're graphing on is the reciprical
                        v = denominator_per_gamma_beta_rho_combination/numerator_per_gamma_beta_rho_combination
                        
                        if distance < 0.000001:
                            v = 1.0

#                        print('max element ', maximum_element, '\n min element', minimum_element,
#                              '\n distance', distance, '\n middle element', middle_element,
#                              '\n numerator', denominator_per_gamma_beta_rho_combination,
#                              '\n denominator', numerator_per_gamma_beta_rho_combination,
#                              '\n all VNE', all_VNE_for_a_gamma_beta_rho_combination_per_simulation,
#                              '\n and this is the value of v\n', v)
                        
                        # This will have one element in it
                        per_rho_v_array.append(v)
                    # This will have the amount of time stamps that we've got
                    per_beta_v_array.append(per_rho_v_array)
                # This has rho amount of vectors each with length of timestamps
                per_gamma_v_array.append(per_beta_v_array)
            # This is the amount of gamma of vectors each with rho amount of vectors each with length of timestamps
            per_simulation_v_array.append(per_gamma_v_array)
        # This is # of simulations of vectors where each vector has number of gamma of vectors
        # each with the length of rho         
        full_v_array.append(per_simulation_v_array)

    # I believe that I have computed the v array for each gamma, beta, rho combination
    # The resulting v array should have one less dimension of vectors since we're not considering
    # all of the time stamps.
#
#            if variable == 5:
#                print('this is the max element\n', max_element_per_beta_for_given_simulation)
#                print(display, collection[j][i])
#            v_array_per_beta = []
#            for t in range(time_stamps):
#                numerator_per_beta = max_element_per_beta_for_given_simulation*(t+1)
#                denominator_per_beta = Michaelis_constant + t + 1
#                v = denominator_per_beta/numerator_per_beta
#                v_array_per_beta.append(v)
#     #       print('this should be my v_array per beta\n', v_array_per_beta)
#            per_simulation_v_array.append(v_array_per_beta)
#            
#            #I have now calculated my v values for a given beta value. I now need to save it per simulation and per simulation as well
#        full_V_vector_reciprical.append(per_simulation_v_array)
#    full_V_vector_reciprical = np.array(full_V_vector_reciprical)
#    print('full reciprical list\n', full_V_vector_reciprical.shape)
    return(full_v_array)
    
####################################################################################################
# This code brings in the numpy arrays for us to use
####################################################################################################
    
## This block of code loads a desired np.array so we don't have to run the simulations
full_IxS_collection = np.load(r'C:\Users\tjbro\Desktop\Grid_search_projections\25_by_25_by_20_by_10\numpy_arrays_of_VNE\IxS_iterating_through_all_3_knobs_where_location_25_individual_25_time_20_simulations_10.npy')
full_IxI_collection = np.load(r'C:\Users\tjbro\Desktop\Grid_search_projections\25_by_25_by_20_by_10\numpy_arrays_of_VNE\IxI_iterating_through_all_3_knobs_where_location_25_individual_25_time_20_simulations_10.npy')
full_tripartite_collection = np.load(r'C:\Users\tjbro\Desktop\Grid_search_projections\25_by_25_by_20_by_10\numpy_arrays_of_VNE\full_tripartite_iterating_through_all_3_knobs_where_location_25_individual_25_time_20_simulations_10.npy')
full_homerange_IxI_collection = np.load(r'C:\Users\tjbro\Desktop\Grid_search_projections\25_by_25_by_20_by_10\numpy_arrays_of_VNE\homerange_IxI_iterating_through_all_3_knobs_where_location_25_individual_25_time_20_simulations_10.npy')
full_SxS_collection = np.load(r'C:\Users\tjbro\Desktop\Grid_search_projections\25_by_25_by_20_by_10\numpy_arrays_of_VNE\SxS_iterating_through_all_3_knobs_where_location_25_individual_25_time_20_simulations_10.npy')
##beta_binary_IxS_collection = np.load('IxS_binary_beta_collection_50_time_10_simulations.npy')
##beta_binary_IxI_collection = np.load('IxI_binary_beta_collection_50_time_10_simulations.npy')

full_input_list = np.load(r'C:\Users\tjbro\Desktop\Grid_search_projections\25_by_25_by_20_by_10\numpy_arrays_of_VNE\full_input_list_for_location_25_individuals_25_time_20_simulations_10.npy')

print('From left to right I have simulation\n gamma\n Beta\n and rho values\n and here is its shape\n', full_IxS_collection.shape)
print('this is the IxS collection\n', full_IxS_collection)

###########################################################################################
# This will calculate the MIchaelis constant so we can find an estimated line and 
# use the Lineweaver Burke Linearization
###########################################################################################
# These are the calculated Michaleis outputs for the data
outputs_IxS = calc_V(full_IxS_collection, time_stamps, simulation_amount, 1)            
outputs_IxI = calc_V(full_IxI_collection, time_stamps, simulation_amount, 2)            
outputs_tripartite = calc_V(full_tripartite_collection, time_stamps, simulation_amount, 3)
outputs_homerange_IxI = calc_V(full_homerange_IxI_collection, time_stamps, simulation_amount, 4)
outputs_SxS = calc_V(full_SxS_collection, time_stamps, simulation_amount, 5)

# This is computing the Michaelis inputs for each output already calculated
input_list = list(range(1, time_stamps + 1))
print('this is the input list\n', input_list)
one_vector = np.ones(time_stamps)
michaelis_inputs = np.divide(one_vector, input_list)

print('this the full output IxS\n', np.array(full_IxS_collection).shape)
print('and this is the outputs after the computing my v matrix\n', np.array(outputs_IxS).shape)

print('these are the michaelis inputs\n', michaelis_inputs)
# These save the arrays so I can call it in another file.
np.save('Michaelis_inputs_location_25_individual_25', michaelis_inputs)
np.save('IxS_outputs_location_25_individual_25', outputs_IxS)
np.save('IxI_outputs_location_25_individual_25', outputs_IxI)
np.save('Tripartite_outputs_location_25_individual_25', outputs_tripartite)
np.save('homerange_IxI_outputs_location_25_individual_25', outputs_homerange_IxI)
np.save('SxS_outputs_location_25_individual_25', outputs_SxS)
