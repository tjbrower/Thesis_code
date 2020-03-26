import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LinearRegression as LR
from numpy.polynomial.polynomial import polyfit as pfit
import pandas as pd
import seaborn as sns


location_amount = 10
#location_amount = 6
individual_amount = 10
#individual_amount = 4
time_stamps = 20
#groups = 5
gamma_values = np.linspace(1, 10, 10, endpoint = True)
beta_values = np.linspace(1, 10, 10, endpoint = True)
rho_values = np.linspace(0.1, 1.0, 10, endpoint=True)
#simulation_amount = 10
simulation_amount = 10

#print(pd.options.display.max_rows)
#print(pd.options.display.max_columns)

#michaelis_inputs = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Salvaging_Von_Neumann_Entropy\full_input_list_for_location_10_individuals_10_time_20_simulations_10.npy')
VNE_outputs_IxS = np.load(r'C:\Users\tjbro\Desktop\Grid_search_projections\10_by_10_by_20_by_10\numpy_arrays_of_VNE\IxS_iterating_through_all_3_knobs_where_location_10_individual_10_time_20_simulations_10.npy')
VNE_outputs_IxI = np.load(r'C:\Users\tjbro\Desktop\Grid_search_projections\10_by_10_by_20_by_10\numpy_arrays_of_VNE\IxI_iterating_through_all_3_knobs_where_location_10_individual_10_time_20_simulations_10.npy')
VNE_outputs_Tripartite = np.load(r'C:\Users\tjbro\Desktop\Grid_search_projections\10_by_10_by_20_by_10\numpy_arrays_of_VNE\full_tripartite_iterating_through_all_3_knobs_where_location_10_individual_10_time_20_simulations_10.npy')
VNE_outputs_Homerange_IxI = np.load(r'C:\Users\tjbro\Desktop\Grid_search_projections\10_by_10_by_20_by_10\numpy_arrays_of_VNE\homerange_IxI_iterating_through_all_3_knobs_where_location_10_individual_10_time_20_simulations_10.npy')
VNE_outputs_SxS = np.load(r'C:\Users\tjbro\Desktop\Grid_search_projections\10_by_10_by_20_by_10\numpy_arrays_of_VNE\SxS_iterating_through_all_3_knobs_where_location_10_individual_10_time_20_simulations_10.npy')

#inputs = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Salvaging_Von_Neumann_Entropy\Michaelis_inputs_location_10_individual_10.npy')
IxS_v_outputs = np.load(r'C:\Users\tjbro\Desktop\Grid_search_projections\10_by_10_by_20_by_10\arrays_needed_for_LWB\IxS_outputs_location_10_individual_10.npy')
IxI_v_outputs = np.load(r'C:\Users\tjbro\Desktop\Grid_search_projections\10_by_10_by_20_by_10\arrays_needed_for_LWB\IxI_outputs_location_10_individual_10.npy')
Tripartite_v_outputs = np.load(r'C:\Users\tjbro\Desktop\Grid_search_projections\10_by_10_by_20_by_10\arrays_needed_for_LWB\Tripartite_outputs_location_10_individual_10.npy')
Homerange_IxI_v_outputs = np.load(r'C:\Users\tjbro\Desktop\Grid_search_projections\10_by_10_by_20_by_10\arrays_needed_for_LWB\homerange_IxI_outputs_location_10_individual_10.npy')
SxS_v_outputs = np.load(r'C:\Users\tjbro\Desktop\Grid_search_projections\10_by_10_by_20_by_10\arrays_needed_for_LWB\SxS_outputs_location_10_individual_10.npy')

# This is computing the Michaelis inputs for each output already calculated
input_list = list(range(1, time_stamps + 1))
#print('this is the input list\n', input_list)
one_vector = np.ones(time_stamps)
michaelis_inputs = np.divide(one_vector, input_list)

print('these are the michaelis inputs\n', michaelis_inputs)


#print('this is the IxS outputs\n', outputs_IxS)
#beta_IxS_collection = np.load('IxS_beta_collection_50_time_10_simulations.npy')

#print('this is the check the shape\n', outputs_IxS.shape)
#print('this is also to check the shape\n', beta_IxS_collection.shape)
####################################################################################################
# Let's do our best to create a pandas dataframe and try to go from there
####################################################################################################
# The columns for my data fram are gamma, beta, rho, timesteps, and the v vector elements
# associated with each timestep and then the simulation number
# Let's first try getting everything in order of simulations
column_list_VNE = ['gamma', 'beta', 'rho', 'timeStep', 'outputValues']
column_list_v_outputs = ['gamma', 'beta', 'rho', 'michaelisInputs', 'outputValues']
# IxS_df = pd.DataFrame(columns = column_list)
# IxI_df = pd.DataFrame(columns = column_list)
# Homerange_IxI_df = pd.DataFrame(columns = column_list)
# SxS_df = pd.DataFrame(columns = column_list)
# Tripartite_df = pd.DataFrame(columns = column_list)
IxS_list = []
IxI_list = []
Homerange_IxI_list = []
SxS_list = []
Tripartite_list = []

IxS_v_values = []
IxI_v_values = []
Homerange_IxI_v_values = []
SxS_v_values = []
Tripartite_v_values = []

for i in range(simulation_amount):
    for j in range(len(gamma_values)):
        for k in range(len(beta_values)):
            for x in range(len(rho_values)):
                for y in range(time_stamps):
                    IxS_list.append([gamma_values[j], beta_values[k], rho_values[x], (y+1), VNE_outputs_IxS[i][j][k][x][y]])
                    IxI_list.append([gamma_values[j], beta_values[k], rho_values[x], (y+1), VNE_outputs_IxI[i][j][k][x][y]])
                    Homerange_IxI_list.append([gamma_values[j], beta_values[k], rho_values[x], (y+1), VNE_outputs_Homerange_IxI[i][j][k][x][y]])
                    SxS_list.append([gamma_values[j], beta_values[k], rho_values[x], (1+y), VNE_outputs_SxS[i][j][k][x][y]])
                    Tripartite_list.append([gamma_values[j], beta_values[k], rho_values[x], (y+1), VNE_outputs_Tripartite[i][j][k][x][y]])

                    IxS_v_values.append([gamma_values[j], beta_values[k], rho_values[x], 1/(y+1), IxS_v_outputs[i][j][k][x][y]])
                    IxI_v_values.append([gamma_values[j], beta_values[k], rho_values[x], 1/(y+1), IxI_v_outputs[i][j][k][x][y]])
                    Homerange_IxI_v_values.append([gamma_values[j], beta_values[k], rho_values[x], 1/(y+1), Homerange_IxI_v_outputs[i][j][k][x][y]])
                    SxS_v_values.append([gamma_values[j], beta_values[k], rho_values[x], 1/(y+1), SxS_v_outputs[i][j][k][x][y]])
                    Tripartite_v_values.append([gamma_values[j], beta_values[k], rho_values[x], 1/(y+1), Tripartite_v_outputs[i][j][k][x][y]])

IxS_VNE_df = pd.DataFrame(np.vstack(np.array(IxS_list)), columns = column_list_VNE)
IxI_VNE_df = pd.DataFrame(np.vstack(np.array(IxI_list)), columns = column_list_VNE)
Homerange_IxI_VNE_df = pd.DataFrame(np.vstack(np.array(Homerange_IxI_list)), columns = column_list_VNE)
SxS_VNE_df = pd.DataFrame(np.vstack(np.array(SxS_list)), columns = column_list_VNE)
Tripartite_VNE_df = pd.DataFrame(np.vstack(np.array(Tripartite_list)), columns = column_list_VNE)

IxS_v_outputs_df = pd.DataFrame(np.vstack(np.array(IxS_v_values)), columns = column_list_v_outputs)
IxI_v_outputs_df = pd.DataFrame(np.vstack(np.array(IxI_v_values)), columns = column_list_v_outputs)
Homerange_IxI_v_outputs_df = pd.DataFrame(np.vstack(np.array(Homerange_IxI_v_values)), columns = column_list_v_outputs)
SxS_v_outputs_df = pd.DataFrame(np.vstack(np.array(SxS_v_values)), columns = column_list_v_outputs)
Tripartite_v_outputs_df = pd.DataFrame(np.vstack(np.array(Tripartite_v_values)), columns = column_list_v_outputs)

#####################################################################################
#
#####################################################################################
IxS_list = []
IxI_list = []
Homerange_IxI_list = []
SxS_list = []
Tripartite_list = []

for i in gamma_values:
    gamma_IxS_df = IxS_v_outputs_df[IxS_v_outputs_df.gamma == i]
    gamma_IxI_df = IxI_v_outputs_df[IxI_v_outputs_df.gamma == i]
    gamma_Homerange_IxI_df = Homerange_IxI_v_outputs_df[Homerange_IxI_v_outputs_df.gamma == i]
    gamma_SxS_df = SxS_v_outputs_df[SxS_v_outputs_df.gamma == i]
    gamma_Tripartite_df = Tripartite_v_outputs_df[Tripartite_v_outputs_df.gamma == i]
    
    for j in beta_values:
        beta_IxS_df = gamma_IxS_df[gamma_IxS_df.beta == j]
        beta_IxI_df = gamma_IxI_df[gamma_IxI_df.beta == j]
        beta_Homerange_IxI_df = gamma_Homerange_IxI_df[gamma_Homerange_IxI_df.beta == j]
        beta_SxS_df = gamma_SxS_df[gamma_SxS_df.beta == j]
        beta_Tripartite_df = gamma_Tripartite_df[gamma_Tripartite_df.beta == j]
        
        for k in rho_values:
            rho_IxS_df = beta_IxS_df[beta_IxS_df.rho == k]
            rho_IxI_df = beta_IxI_df[beta_IxI_df.rho == k]
            rho_Homerange_IxI_df = beta_Homerange_IxI_df[beta_Homerange_IxI_df.rho == k]
            rho_SxS_df = beta_SxS_df[beta_SxS_df.rho == k]
            rho_Tripartite_df = beta_Tripartite_df[beta_Tripartite_df.rho == k]

            all_inputs = np.array(rho_IxS_df.michaelisInputs.values).reshape(-1,1)
            
            IxS_reg = LR().fit(all_inputs, np.array(rho_IxS_df.outputValues.values).reshape(-1,1))
            IxI_reg = LR().fit(all_inputs, np.array(rho_IxI_df.outputValues.values).reshape(-1,1))
            Homerange_IxI_reg = LR().fit(all_inputs, np.array(rho_Homerange_IxI_df.outputValues.values).reshape(-1,1))
            SxS_reg = LR().fit(all_inputs, np.array(rho_SxS_df.outputValues.values).reshape(-1,1))
            Tripartite_reg = LR().fit(all_inputs, np.array(rho_Tripartite_df.outputValues.values).reshape(-1,1))
            
            IxS_list.append([i, j, k, IxS_reg.intercept_[0]])
            IxI_list.append([i, j, k, IxI_reg.intercept_[0]])
            Homerange_IxI_list.append([i, j, k, Homerange_IxI_reg.intercept_[0]])
            SxS_list.append([i, j, k, SxS_reg.intercept_[0]])
            Tripartite_list.append([i, j, k, Tripartite_reg.intercept_[0]])

            
IxS_Max_VNE_df = pd.DataFrame(np.vstack(np.array(IxS_list)), columns = ['gamma', 'beta', 'rho', 'MaxVNE'])
IxI_Max_VNE_df = pd.DataFrame(np.vstack(np.array(IxI_list)), columns = ['gamma', 'beta', 'rho', 'MaxVNE'])
Homerange_IxI_Max_VNE_df = pd.DataFrame(np.vstack(np.array(Homerange_IxI_list)), columns = ['gamma', 'beta', 'rho', 'MaxVNE'])
Tripartite_Max_VNE_df = pd.DataFrame(np.vstack(np.array(Tripartite_list)), columns = ['gamma', 'beta', 'rho', 'MaxVNE'])
SxS_Max_VNE_df = pd.DataFrame(np.vstack(np.array(SxS_list)), columns = ['gamma', 'beta', 'rho', 'MaxVNE'])
            
#####################################################################################
#
#####################################################################################
SMALL_SIZE = 8
MEDIUM_SIZE = 16
BIGGER_SIZE = 24


plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tired_patchs
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure title

colors = ['r', 'g', 'k', 'b', 'y']

red_patch = mpatches.Patch(color='r', label='IxS')
green_patch = mpatches.Patch(color = 'g', label = 'IxI')
black_patch = mpatches.Patch(color='k', label='Tripartite')
blue_patch = mpatches.Patch(color='b', label='Homerange IxI')
yellow_patch = mpatches.Patch(color='y', label='SxS')

#####################################################################################
#
#####################################################################################
IxS_rho_graphing_df = IxS_Max_VNE_df[IxS_Max_VNE_df.gamma == 1]
IxS_rho_graphing_df = IxS_rho_graphing_df[IxS_rho_graphing_df.beta == 1]

IxI_rho_graphing_df = IxI_Max_VNE_df[IxI_Max_VNE_df.gamma == 1]
IxI_rho_graphing_df = IxI_rho_graphing_df[IxI_rho_graphing_df.beta == 1]

Homerange_IxI_rho_graphing_df = Homerange_IxI_Max_VNE_df[Homerange_IxI_Max_VNE_df.gamma == 1]
Homerange_IxI_rho_graphing_df = Homerange_IxI_rho_graphing_df[Homerange_IxI_rho_graphing_df.beta == 1]

Tripartite_rho_graphing_df = Tripartite_Max_VNE_df[Tripartite_Max_VNE_df.gamma == 1]
Tripartite_rho_graphing_df = Tripartite_rho_graphing_df[Tripartite_rho_graphing_df.beta == 1]

SxS_rho_graphing_df = SxS_Max_VNE_df[SxS_Max_VNE_df.gamma == 1]
SxS_rho_graphing_df = SxS_rho_graphing_df[SxS_rho_graphing_df.beta == 1]
#display(IxS_rho_graphing_df.shape)

#####################################################################################
#
#####################################################################################
IxS_beta_graphing_df = IxS_Max_VNE_df[IxS_Max_VNE_df.gamma == 1]
IxS_beta_graphing_df = IxS_beta_graphing_df[IxS_beta_graphing_df.rho == 0.1]

IxI_beta_graphing_df = IxI_Max_VNE_df[IxI_Max_VNE_df.gamma == 1]
IxI_beta_graphing_df = IxI_beta_graphing_df[IxI_beta_graphing_df.rho == 0.1]

Homerange_IxI_beta_graphing_df = Homerange_IxI_Max_VNE_df[Homerange_IxI_Max_VNE_df.gamma == 1]
Homerange_IxI_beta_graphing_df = Homerange_IxI_beta_graphing_df[Homerange_IxI_beta_graphing_df.rho == 0.1]

Tripartite_beta_graphing_df = Tripartite_Max_VNE_df[Tripartite_Max_VNE_df.gamma == 1]
Tripartite_beta_graphing_df = Tripartite_beta_graphing_df[Tripartite_beta_graphing_df.rho == 0.1]

SxS_beta_graphing_df = SxS_Max_VNE_df[SxS_Max_VNE_df.gamma == 1]
SxS_beta_graphing_df = SxS_beta_graphing_df[SxS_beta_graphing_df.rho == 0.1]

#display(IxS_beta_graphing_df.shape)

#####################################################################################
#
#####################################################################################
IxS_gamma_graphing_df = IxS_Max_VNE_df[IxS_Max_VNE_df.beta == 1]
IxS_gamma_graphing_df = IxS_gamma_graphing_df[IxS_gamma_graphing_df.rho == 0.1]

IxI_gamma_graphing_df = IxI_Max_VNE_df[IxI_Max_VNE_df.beta == 1]
IxI_gamma_graphing_df = IxI_gamma_graphing_df[IxI_gamma_graphing_df.rho == 0.1]

Homerange_IxI_gamma_graphing_df = Homerange_IxI_Max_VNE_df[Homerange_IxI_Max_VNE_df.beta == 1]
Homerange_IxI_gamma_graphing_df = Homerange_IxI_gamma_graphing_df[Homerange_IxI_gamma_graphing_df.rho == 0.1]

Tripartite_gamma_graphing_df = Tripartite_Max_VNE_df[Tripartite_Max_VNE_df.beta == 1]
Tripartite_gamma_graphing_df = Tripartite_gamma_graphing_df[Tripartite_gamma_graphing_df.rho == 0.1]

SxS_gamma_graphing_df = SxS_Max_VNE_df[SxS_Max_VNE_df.beta == 1]
SxS_gamma_graphing_df = SxS_gamma_graphing_df[SxS_gamma_graphing_df.rho == 0.1]

#display(IxS_gamma_graphing_df.shape)
#####################################################################################
#
#####################################################################################
plt.figure()



plt.show()



#####################################################################################
#
#####################################################################################
plt.figure()
#plt.subplot(221)
plt.grid(True)
plt.title('VNE Max for rho')
plt.xlabel('Rho Values')
plt.ylabel('Max VNE')
#plt.yticks(np.linspace(1.0, 1.5, 10, endpoint = True))
plt.legend(handles = [red_patch, green_patch, black_patch, blue_patch, yellow_patch], title='Projection')
plt.scatter(IxS_rho_graphing_df.rho.values, IxS_rho_graphing_df.MaxVNE.values, color = 'r')
plt.scatter(IxI_rho_graphing_df.rho.values, IxI_rho_graphing_df.MaxVNE.values, color = 'g')
plt.scatter(Homerange_IxI_rho_graphing_df.rho.values, Homerange_IxI_rho_graphing_df.MaxVNE.values, color = 'b')
plt.scatter(Tripartite_rho_graphing_df.rho.values, Tripartite_rho_graphing_df.MaxVNE.values, color = 'k')
plt.scatter(SxS_rho_graphing_df.rho.values, SxS_rho_graphing_df.MaxVNE.values, color = 'y')
plt.plot(IxS_rho_graphing_df.rho.values, IxS_rho_graphing_df.MaxVNE.values, color = 'r')
plt.plot(IxI_rho_graphing_df.rho.values, IxI_rho_graphing_df.MaxVNE.values, color = 'g')
plt.plot(Homerange_IxI_rho_graphing_df.rho.values, Homerange_IxI_rho_graphing_df.MaxVNE.values, color = 'b')
plt.plot(Tripartite_rho_graphing_df.rho.values, Tripartite_rho_graphing_df.MaxVNE.values, color = 'k')
plt.plot(SxS_rho_graphing_df.rho.values, SxS_rho_graphing_df.MaxVNE.values, color = 'y')
plt.show()

#plt.subplot(222)
plt.figure()
plt.grid(True)
plt.title('VNE Max for beta')
plt.xlabel('Beta Values')
plt.ylabel('Max VNE')
#plt.yticks(np.linspace(1.0, 1.5, 10, endpoint = True))
plt.legend(handles = [red_patch, green_patch, black_patch, blue_patch, yellow_patch], title='Projection')
plt.scatter(IxS_beta_graphing_df.beta.values, IxS_beta_graphing_df.MaxVNE.values, color = 'r')
plt.scatter(IxI_beta_graphing_df.beta.values, IxI_beta_graphing_df.MaxVNE.values, color = 'g')
plt.scatter(Homerange_IxI_beta_graphing_df.beta.values, Homerange_IxI_beta_graphing_df.MaxVNE.values, color = 'b')
plt.scatter(Tripartite_beta_graphing_df.beta.values, Tripartite_beta_graphing_df.MaxVNE.values, color = 'k')
plt.scatter(SxS_beta_graphing_df.beta.values, SxS_beta_graphing_df.MaxVNE.values, color = 'y')
plt.plot(IxS_beta_graphing_df.beta.values, IxS_beta_graphing_df.MaxVNE.values, color = 'r')
plt.plot(IxI_beta_graphing_df.beta.values, IxI_beta_graphing_df.MaxVNE.values, color = 'g')
plt.plot(Homerange_IxI_beta_graphing_df.beta.values, Homerange_IxI_beta_graphing_df.MaxVNE.values, color = 'b')
plt.plot(Tripartite_beta_graphing_df.beta.values, Tripartite_beta_graphing_df.MaxVNE.values, color = 'k')
plt.plot(SxS_beta_graphing_df.beta.values, SxS_beta_graphing_df.MaxVNE.values, color = 'y')
plt.show()

#plt.subplot(223)
plt.figure()
plt.grid(True)
plt.title('VNE Max for gamma')
plt.xlabel('Gamma Values')
plt.ylabel('Max VNE')
#plt.yticks(np.linspace(1.0, 1.5, 10, endpoint = True))
plt.legend(handles = [red_patch, green_patch, black_patch, blue_patch, yellow_patch], title='Projection')
plt.scatter(IxS_gamma_graphing_df.gamma.values, IxS_gamma_graphing_df.MaxVNE.values, color = 'r')
plt.scatter(IxI_gamma_graphing_df.gamma.values, IxI_gamma_graphing_df.MaxVNE.values, color = 'g')
plt.scatter(Homerange_IxI_gamma_graphing_df.gamma.values, Homerange_IxI_gamma_graphing_df.MaxVNE.values, color = 'b')
plt.scatter(Tripartite_gamma_graphing_df.gamma.values, Tripartite_gamma_graphing_df.MaxVNE.values, color = 'k')
plt.scatter(SxS_gamma_graphing_df.gamma.values, SxS_gamma_graphing_df.MaxVNE.values, color = 'y')
plt.plot(IxS_gamma_graphing_df.gamma.values, IxS_gamma_graphing_df.MaxVNE.values, color = 'r')
plt.plot(IxI_gamma_graphing_df.gamma.values, IxI_gamma_graphing_df.MaxVNE.values, color = 'g')
plt.plot(Homerange_IxI_gamma_graphing_df.gamma.values, Homerange_IxI_gamma_graphing_df.MaxVNE.values, color = 'b')
plt.plot(Tripartite_gamma_graphing_df.gamma.values, Tripartite_gamma_graphing_df.MaxVNE.values, color = 'k')
plt.plot(SxS_gamma_graphing_df.gamma.values, SxS_gamma_graphing_df.MaxVNE.values, color = 'y')
plt.show()

#print(SxS_gamma_graphing_df.gamma.values, SxS_gamma_graphing_df.MaxVNE.values)
#print(SxS_beta_graphing_df.beta.values, SxS_beta_graphing_df.MaxVNE.values)
#plt.subplot(224)
#plt.legend(handles = [red_patch, green_patch, black_patch, blue_patch, yellow_patch], title= 'Projection')
#plt.show()

#####################################################################################
#
#####################################################################################

#####################################################################################
#
#####################################################################################


