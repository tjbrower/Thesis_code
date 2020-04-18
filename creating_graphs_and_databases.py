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
rho_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.72, 0.74, 0.76, 0.78, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0]#np.linspace(0.1, 1.0, 10, endpoint=True)
#simulation_amount = 10
simulation_amount = 10

#print(pd.options.display.max_rows)
#print(pd.options.display.max_columns)

#michaelis_inputs = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Salvaging_Von_Neumann_Entropy\full_input_list_for_location_10_individuals_10_time_20_simulations_10.npy')
VNE_outputs_IxS = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\10_by_10_by_20_by_10\numpy_arrays_of_VNE\IxS_iterating_through_all_3_knobs_where_location_10_individual_10_time_20_simulations_10.npy')
VNE_outputs_IxI = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\10_by_10_by_20_by_10\numpy_arrays_of_VNE\IxI_iterating_through_all_3_knobs_where_location_10_individual_10_time_20_simulations_10.npy')
VNE_outputs_Tripartite = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\10_by_10_by_20_by_10\numpy_arrays_of_VNE\full_tripartite_iterating_through_all_3_knobs_where_location_10_individual_10_time_20_simulations_10.npy')
VNE_outputs_Homerange_IxI = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\10_by_10_by_20_by_10\numpy_arrays_of_VNE\homerange_IxI_iterating_through_all_3_knobs_where_location_10_individual_10_time_20_simulations_10.npy')
VNE_outputs_SxS = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\10_by_10_by_20_by_10\numpy_arrays_of_VNE\SxS_iterating_through_all_3_knobs_where_location_10_individual_10_time_20_simulations_10.npy')

#inputs = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Salvaging_Von_Neumann_Entropy\Michaelis_inputs_location_10_individual_10.npy')
IxS_v_outputs = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\10_by_10_by_20_by_10\arrays_needed_for_LWB\IxS_outputs_location_10_individual_10.npy')
IxI_v_outputs = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\10_by_10_by_20_by_10\arrays_needed_for_LWB\IxI_outputs_location_10_individual_10.npy')
Tripartite_v_outputs = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\10_by_10_by_20_by_10\arrays_needed_for_LWB\Tripartite_outputs_location_10_individual_10.npy')
Homerange_IxI_v_outputs = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\10_by_10_by_20_by_10\arrays_needed_for_LWB\homerange_IxI_outputs_location_10_individual_10.npy')
SxS_v_outputs = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\10_by_10_by_20_by_10\arrays_needed_for_LWB\SxS_outputs_location_10_individual_10.npy')

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
            
            IxS_list.append([i, j, k, 1/IxS_reg.intercept_[0]])
            IxI_list.append([i, j, k, 1/IxI_reg.intercept_[0]])
            Homerange_IxI_list.append([i, j, k, 1/Homerange_IxI_reg.intercept_[0]])
            SxS_list.append([i, j, k, 1/SxS_reg.intercept_[0]])
            Tripartite_list.append([i, j, k, 1/Tripartite_reg.intercept_[0]])


IxS_Max_VNE_df = pd.DataFrame(np.vstack(np.array(IxS_list)), columns = ['gamma', 'beta', 'rho', 'MaxVNE'])
IxI_Max_VNE_df = pd.DataFrame(np.vstack(np.array(IxI_list)), columns = ['gamma', 'beta', 'rho', 'MaxVNE'])
Homerange_IxI_Max_VNE_df = pd.DataFrame(np.vstack(np.array(Homerange_IxI_list)), columns = ['gamma', 'beta', 'rho', 'MaxVNE'])
Tripartite_Max_VNE_df = pd.DataFrame(np.vstack(np.array(Tripartite_list)), columns = ['gamma', 'beta', 'rho', 'MaxVNE'])
SxS_Max_VNE_df = pd.DataFrame(np.vstack(np.array(SxS_list)), columns = ['gamma', 'beta', 'rho', 'MaxVNE'])
            
#####################################################################################
# This is the size of the text for the graphs
#####################################################################################
SMALL_SIZE = 8
MEDIUM_SIZE = 16
BIGGER_SIZE = 24
BIGGEST_SIZE = 32


plt.rc('font', size=BIGGER_SIZE)         # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)   # fontsize of the tired_patchs
plt.rc('legend', fontsize=MEDIUM_SIZE)   # legend fontsize
plt.rc('figure', titlesize=SMALL_SIZE)   # fontsize of the figure title

colors = ['r', 'g', 'k', 'b', 'y']

red_patch = mpatches.Patch(color='r', label='IxS')
green_patch = mpatches.Patch(color = 'g', label = 'IxI')
black_patch = mpatches.Patch(color='k', label='Tripartite')
blue_patch = mpatches.Patch(color='b', label='Homerange')
yellow_patch = mpatches.Patch(color='y', label='SxS')

#####################################################################################
# This is getting the VNE for the base value parameters
#####################################################################################
IxS_VNE_graphing_df = IxS_VNE_df[IxS_VNE_df.gamma == 1]
IxS_VNE_graphing_df = IxS_VNE_graphing_df[IxS_VNE_graphing_df.beta == 1]
IxS_VNE_graphing_df = IxS_VNE_graphing_df[IxS_VNE_graphing_df.rho == 0.1]

IxI_VNE_graphing_df = IxI_VNE_df[IxI_VNE_df.gamma == 1]
IxI_VNE_graphing_df = IxI_VNE_graphing_df[IxI_VNE_graphing_df.beta == 1]
IxI_VNE_graphing_df = IxI_VNE_graphing_df[IxI_VNE_graphing_df.rho == 0.1]

Homerange_IxI_VNE_graphing_df = Homerange_IxI_VNE_df[Homerange_IxI_VNE_df.gamma == 1]
Homerange_IxI_VNE_graphing_df = Homerange_IxI_VNE_graphing_df[Homerange_IxI_VNE_graphing_df.beta == 1]
Homerange_IxI_VNE_graphing_df = Homerange_IxI_VNE_graphing_df[Homerange_IxI_VNE_graphing_df.rho == 0.1]

Tripartite_VNE_graphing_df = Tripartite_VNE_df[Tripartite_VNE_df.gamma == 1]
Tripartite_VNE_graphing_df = Tripartite_VNE_graphing_df[Tripartite_VNE_graphing_df.beta == 1]
Tripartite_VNE_graphing_df = Tripartite_VNE_graphing_df[Tripartite_VNE_graphing_df.rho == 0.1]

SxS_VNE_graphing_df = SxS_VNE_df[SxS_VNE_df.gamma == 1]
SxS_VNE_graphing_df = SxS_VNE_graphing_df[SxS_VNE_graphing_df.beta == 1]
SxS_VNE_graphing_df = SxS_VNE_graphing_df[SxS_VNE_graphing_df.rho == 0.1]

#####################################################################################
# Trying to get the avg value for each time value
#####################################################################################
IxS_avg = []
IxI_avg = []
Homerange_IxI_avg = []
Tripartite_avg = []
SxS_avg = []
for i in range(time_stamps):
    IxS_time_step_df = IxS_VNE_graphing_df[IxS_VNE_graphing_df.timeStep == i+1]
    IxI_time_step_df = IxI_VNE_graphing_df[IxI_VNE_graphing_df.timeStep == i+1]
    Homerange_IxI_time_step_df = Homerange_IxI_VNE_graphing_df[Homerange_IxI_VNE_graphing_df.timeStep == i+1]
    Tripartite_time_step_df = Tripartite_VNE_graphing_df[Tripartite_VNE_graphing_df.timeStep == i+1]
    SxS_time_step_df = SxS_VNE_graphing_df[SxS_VNE_graphing_df.timeStep == i+1]

    
    IxS_avg.append(np.sum(IxS_time_step_df.outputValues.values)/10)
    IxI_avg.append(np.sum(IxI_time_step_df.outputValues.values)/10)
    Homerange_IxI_avg.append(np.sum(Homerange_IxI_time_step_df.outputValues.values)/10)
    Tripartite_avg.append(np.sum(Tripartite_time_step_df.outputValues.values)/10)
    SxS_avg.append(np.sum(SxS_time_step_df.outputValues.values)/10)

print('this is the length of the avg array\n', np.array(IxS_avg).shape)
print(IxS_avg, '\n', IxI_avg, '\n', Homerange_IxI_avg, '\n', Tripartite_avg, '\n', 
      SxS_avg)
#####################################################################################
# The values for the Max VNE when looking at rho
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
# The values for the Max VNE when looking at beta
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
# The values for the Max VNE when looking at gamma
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
# This is getting the 
#####################################################################################
IxS_v_graphing_df = IxS_v_outputs_df[IxS_v_outputs_df.gamma == 1]
IxS_v_graphing_df = IxS_v_graphing_df[IxS_v_graphing_df.beta == 1]
IxS_v_graphing_df = IxS_v_graphing_df[IxS_v_graphing_df.rho == 0.1]

IxI_v_graphing_df = IxI_v_outputs_df[IxI_v_outputs_df.gamma == 1]
IxI_v_graphing_df = IxI_v_graphing_df[IxI_v_graphing_df.beta == 1]
IxI_v_graphing_df = IxI_v_graphing_df[IxI_v_graphing_df.rho == 0.1]

Homerange_IxI_v_graphing_df = Homerange_IxI_v_outputs_df[Homerange_IxI_v_outputs_df.gamma == 1]
Homerange_IxI_v_graphing_df = Homerange_IxI_v_graphing_df[Homerange_IxI_v_graphing_df.beta == 1]
Homerange_IxI_v_graphing_df = Homerange_IxI_v_graphing_df[Homerange_IxI_v_graphing_df.rho == 0.1]

Tripartite_v_graphing_df = Tripartite_v_outputs_df[Tripartite_v_outputs_df.gamma == 1]
Tripartite_v_graphing_df = Tripartite_v_graphing_df[Tripartite_v_graphing_df.beta == 1]
Tripartite_v_graphing_df = Tripartite_v_graphing_df[Tripartite_v_graphing_df.rho == 0.1]

SxS_v_graphing_df = SxS_v_outputs_df[SxS_v_outputs_df.gamma == 1]
SxS_v_graphing_df = SxS_v_graphing_df[SxS_v_graphing_df.beta == 1]
SxS_v_graphing_df = SxS_v_graphing_df[SxS_v_graphing_df.rho == 0.1]


#####################################################################################
# This is the graph of the VNE for each projection
####################################################################################
Big_font_size = 56


plt.figure()
plt.grid(True)
#plt.suptitle('IxS Lineweaver-Burk Plot with Regression Line', fontsize = Big_font_size)
plt.xlabel('Time Step', fontsize = Big_font_size)
plt.ylabel('VNE value', fontsize = Big_font_size)
plt.legend(handles = [red_patch, green_patch, black_patch, blue_patch, yellow_patch], title = 'Projection')
plt.xticks(np.linspace(1, 20, 20, endpoint = True), fontsize = 30)
plt.yticks(np.linspace(0, 1, 21, endpoint = True))
plt.scatter(IxS_VNE_graphing_df.timeStep.values, IxS_VNE_graphing_df.outputValues.values, color = 'r')
plt.scatter(IxI_VNE_graphing_df.timeStep.values, IxI_VNE_graphing_df.outputValues.values, color = 'g')
plt.scatter(Homerange_IxI_VNE_graphing_df.timeStep.values, Homerange_IxI_VNE_graphing_df.outputValues.values, color = 'b')
plt.scatter(Tripartite_VNE_graphing_df.timeStep.values, Tripartite_VNE_graphing_df.outputValues.values, color = 'k')
plt.scatter(SxS_VNE_graphing_df.timeStep.values, SxS_VNE_graphing_df.outputValues.values, color = 'y')

#plt.scatter([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], IxS_avg, color = 'r')
#plt.scatter([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], IxI_avg, color = 'g')
#plt.scatter([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], Homerange_IxI_avg, color = 'b')
#plt.scatter([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], Tripartite_avg, color = 'k')
#plt.scatter([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], SxS_avg, color = 'y')

plt.plot([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], IxS_avg, color = 'r')
plt.plot([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], IxI_avg, color = 'g')
plt.plot([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], Homerange_IxI_avg, color = 'b')
plt.plot([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], Tripartite_avg, color = 'k')
plt.plot([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], SxS_avg, color = 'y')

#plt.plot(IxS_VNE_graphing_df.timeStep.values, IxS_VNE_graphing_df.outputValues.values)
#plt.plot(IxI_VNE_graphing_df.timeStep.values, IxI_VNE_graphing_df.outputValues.values)
#plt.plot(Homerange_IxI_VNE_graphing_df.timeStep.values, Homerange_IxI_VNE_graphing_df.outputValues.values)
#plt.plot(Tripartite_VNE_graphing_df.timeStep.values, Tripartite_VNE_graphing_df.outputValues.values)
#plt.plot(SxS_VNE_graphing_df.timeStep.values, SxS_VNE_graphing_df.outputValues.values)
plt.show()

######################################################################################
# This is the visualization of the Lineweaver-Burk plot
######################################################################################
#IxS_reg = LR().fit(np.array(IxS_v_graphing_df.michaelisInputs.values).reshape(-1,1), np.array(IxS_v_graphing_df.outputValues.values).reshape(-1,1))
#intercept = IxS_reg.intercept_
#print('these are the coef', intercept)
#outputVector = []
##for i in IxS_v_graphing_df.michaelisInputs.values:
##    outputVector.append(coef[0]*i + coef[1])
#print(IxS_v_graphing_df.michaelisInputs.shape, IxS_v_graphing_df.outputValues.shape)
#plt.figure()
#plt.grid(True)
#plt.suptitle('IxS Lineweaver-Burk Plot with Regression Line', fontsize = Big_font_size)
#plt.xlabel('Michaelis Inputs', fontsize = Big_font_size)
#plt.ylabel('1/(Max VNE)', fontsize = Big_font_size)
##plt.scatter(IxS_v_graphing_df.michaelisInputs.values, IxS_v_graphing_df.outputValues.values, c = 'r')
#sns.regplot(x = 'michaelisInputs', y = 'outputValues', data = IxS_v_graphing_df, color = 'b')
#plt.show()
#####################################################################################
# This is thre distinct plots with the Lineweaver results
#####################################################################################
first_array = np.linspace(0.3, 0.9, 15, endpoint = False)
second_array = np.linspace(0.9, 1.0, 18, endpoint = True)
y_tick_array = np.hstack((first_array, second_array))


fig = plt.figure()
ax1 = plt.subplot2grid((3,3), (0,0), rowspan = 2)
ax2 = plt.subplot2grid((3,3), (0,1), rowspan = 2)
ax3 = plt.subplot2grid((3,3), (0,2), rowspan = 2)
ax4 = plt.subplot2grid((3,3), (2,0))
ax5 = plt.subplot2grid((3,3), (2,1))
ax6 = plt.subplot2grid((3,3), (2,2))

plt.legend(handles = [red_patch, green_patch, black_patch, blue_patch, yellow_patch], title='Projection', bbox_to_anchor=(1.05, 1.0), loc='upper left')

#fig.suptitle('Location = 10, Individuals = 10, Time Steps = 20, Simulations = 10', fontsize = Big_font_size)

ax1.grid(True)
#ax1.title('VNE Max for rho')
ax1.set_xlabel(r'$\rho$')
ax1.set_ylabel('Max VNE')
ax1.set_xticks(np.linspace(0.1, 1.0, 10, endpoint = True))
ax1.set_yticks(np.linspace(0.15, 1.0, 18, endpoint = True))
#ax1.legend(handles = [red_patch, green_patch, black_patch, blue_patch, yellow_patch], title='Projection')
ax1.scatter(IxS_rho_graphing_df.rho.values, IxS_rho_graphing_df.MaxVNE.values, color = 'r')
ax1.scatter(IxI_rho_graphing_df.rho.values, IxI_rho_graphing_df.MaxVNE.values, color = 'g')
ax1.scatter(Homerange_IxI_rho_graphing_df.rho.values, Homerange_IxI_rho_graphing_df.MaxVNE.values, color = 'b')
ax1.scatter(Tripartite_rho_graphing_df.rho.values, Tripartite_rho_graphing_df.MaxVNE.values, color = 'k')
ax1.scatter(SxS_rho_graphing_df.rho.values, SxS_rho_graphing_df.MaxVNE.values, color = 'y')
ax1.plot(IxS_rho_graphing_df.rho.values, IxS_rho_graphing_df.MaxVNE.values, color = 'r')
ax1.plot(IxI_rho_graphing_df.rho.values, IxI_rho_graphing_df.MaxVNE.values, color = 'g')
ax1.plot(Homerange_IxI_rho_graphing_df.rho.values, Homerange_IxI_rho_graphing_df.MaxVNE.values, color = 'b')
ax1.plot(Tripartite_rho_graphing_df.rho.values, Tripartite_rho_graphing_df.MaxVNE.values, color = 'k')
ax1.plot(SxS_rho_graphing_df.rho.values, SxS_rho_graphing_df.MaxVNE.values, color = 'y')
#plt.show()

ax2.grid(True)
#ax2.title('VNE Max for beta')
ax2.set_xlabel(r'$\beta$')
#ax2.set_ylabel('Max VNE')
ax2.set_xticks(np.linspace(1, 10, 10, endpoint = True))
ax2.set_yticks(np.linspace(0.15, 1.0, 18, endpoint = True))
#ax2.legend(handles = [red_patch, green_patch, black_patch, blue_patch, yellow_patch], title='Projection')
ax2.scatter(IxS_beta_graphing_df.beta.values, IxS_beta_graphing_df.MaxVNE.values, color = 'r')
ax2.scatter(IxI_beta_graphing_df.beta.values, IxI_beta_graphing_df.MaxVNE.values, color = 'g')
ax2.scatter(Homerange_IxI_beta_graphing_df.beta.values, Homerange_IxI_beta_graphing_df.MaxVNE.values, color = 'b')
ax2.scatter(Tripartite_beta_graphing_df.beta.values, Tripartite_beta_graphing_df.MaxVNE.values, color = 'k')
ax2.scatter(SxS_beta_graphing_df.beta.values, SxS_beta_graphing_df.MaxVNE.values, color = 'y')
ax2.plot(IxS_beta_graphing_df.beta.values, IxS_beta_graphing_df.MaxVNE.values, color = 'r')
ax2.plot(IxI_beta_graphing_df.beta.values, IxI_beta_graphing_df.MaxVNE.values, color = 'g')
ax2.plot(Homerange_IxI_beta_graphing_df.beta.values, Homerange_IxI_beta_graphing_df.MaxVNE.values, color = 'b')
ax2.plot(Tripartite_beta_graphing_df.beta.values, Tripartite_beta_graphing_df.MaxVNE.values, color = 'k')
ax2.plot(SxS_beta_graphing_df.beta.values, SxS_beta_graphing_df.MaxVNE.values, color = 'y')
#ax2.show()

ax3.grid(True)
#ax3.title('VNE Max for gamma')
ax3.set_xlabel(r'$\gamma$')
#ax3.set_ylabel('Max VNE')
ax3.set_xticks(np.linspace(1, 10, 10, endpoint = True))
ax3.set_yticks(np.linspace(0.15, 1.0, 18, endpoint = True))
#ax3.legend(handles = [red_patch, green_patch, black_patch, blue_patch, yellow_patch], title='Projection')
ax3.scatter(IxS_gamma_graphing_df.gamma.values, IxS_gamma_graphing_df.MaxVNE.values, color = 'r')
ax3.scatter(IxI_gamma_graphing_df.gamma.values, IxI_gamma_graphing_df.MaxVNE.values, color = 'g')
ax3.scatter(Homerange_IxI_gamma_graphing_df.gamma.values, Homerange_IxI_gamma_graphing_df.MaxVNE.values, color = 'b')
ax3.scatter(Tripartite_gamma_graphing_df.gamma.values, Tripartite_gamma_graphing_df.MaxVNE.values, color = 'k')
ax3.scatter(SxS_gamma_graphing_df.gamma.values, SxS_gamma_graphing_df.MaxVNE.values, color = 'y')
ax3.plot(IxS_gamma_graphing_df.gamma.values, IxS_gamma_graphing_df.MaxVNE.values, color = 'r')
ax3.plot(IxI_gamma_graphing_df.gamma.values, IxI_gamma_graphing_df.MaxVNE.values, color = 'g')
ax3.plot(Homerange_IxI_gamma_graphing_df.gamma.values, Homerange_IxI_gamma_graphing_df.MaxVNE.values, color = 'b')
ax3.plot(Tripartite_gamma_graphing_df.gamma.values, Tripartite_gamma_graphing_df.MaxVNE.values, color = 'k')
ax3.plot(SxS_gamma_graphing_df.gamma.values, SxS_gamma_graphing_df.MaxVNE.values, color = 'y')
#plt.show()


ax4.grid(True)
ax4.tick_params(labelsize = 'medium')
#ax3.title('VNE Max for gamma')
ax4.set_xlabel(r'$\rho$')
#ax3.set_ylabel('Max VNE')
ax4.set_xticks(np.linspace(0.1, 1.0, 10, endpoint = True))
ax4.set_yticks(second_array)
#ax3.legend(handles = [red_patch, green_patch, black_patch, blue_patch, yellow_patch], title='Projection')
ax4.scatter(IxI_rho_graphing_df.rho.values, IxI_rho_graphing_df.MaxVNE.values, color = 'g')
ax4.scatter(Homerange_IxI_rho_graphing_df.rho.values, Homerange_IxI_rho_graphing_df.MaxVNE.values, color = 'b')
ax4.plot(IxI_rho_graphing_df.rho.values, IxI_rho_graphing_df.MaxVNE.values, color = 'g')
ax4.plot(Homerange_IxI_rho_graphing_df.rho.values, Homerange_IxI_rho_graphing_df.MaxVNE.values, color = 'b')

ax5.tick_params(labelsize = 'medium')
ax5.grid(True)
#ax3.title('VNE Max for gamma')
ax5.set_xlabel(r'$\beta$')
#ax3.set_ylabel('Max VNE')
ax5.set_xticks(np.linspace(1, 10, 10, endpoint = True))
ax5.set_yticks(second_array)
#ax3.legend(handles = [red_patch, green_patch, black_patch, blue_patch, yellow_patch], title='Projection')
ax5.scatter(IxI_beta_graphing_df.beta.values, IxI_beta_graphing_df.MaxVNE.values, color = 'g')
ax5.scatter(Homerange_IxI_beta_graphing_df.beta.values, Homerange_IxI_beta_graphing_df.MaxVNE.values, color = 'b')
ax5.plot(IxI_beta_graphing_df.beta.values, IxI_beta_graphing_df.MaxVNE.values, color = 'g')
ax5.plot(Homerange_IxI_beta_graphing_df.beta.values, Homerange_IxI_beta_graphing_df.MaxVNE.values, color = 'b')

ax6.grid(True)
ax6.tick_params(labelsize = 'medium')
#ax3.title('VNE Max for gamma')
ax6.set_xlabel(r'$\gamma$')
#ax3.set_ylabel('Max VNE')
ax6.set_xticks(np.linspace(1, 10, 10, endpoint = True))
ax6.set_yticks(second_array)
#ax3.legend(handles = [red_patch, green_patch, black_patch, blue_patch, yellow_patch], title='Projection')
ax6.scatter(IxI_gamma_graphing_df.gamma.values, IxI_gamma_graphing_df.MaxVNE.values, color = 'g')
ax6.scatter(Homerange_IxI_gamma_graphing_df.gamma.values, Homerange_IxI_gamma_graphing_df.MaxVNE.values, color = 'b')
ax6.plot(IxI_gamma_graphing_df.gamma.values, IxI_gamma_graphing_df.MaxVNE.values, color = 'g')
ax6.plot(Homerange_IxI_gamma_graphing_df.gamma.values, Homerange_IxI_gamma_graphing_df.MaxVNE.values, color = 'b')

plt.tight_layout()
##################################################################################################
# This puts the Lineweaver results in the same subplot
###################################################################################################
#
#fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
#
#fig.suptitle(r'Max VNE for $\gamma, \beta, \gamma$', fontsize = Big_font_size)
#
##ax1.title(r'Iterating over $\rho$')
#ax1.set_xlabel(r'$\rho$')
#ax1.set_ylabel('Max VNE')
#ax1.grid(True)
#
#ax1.scatter(IxS_rho_graphing_df.rho.values, IxS_rho_graphing_df.MaxVNE.values, color = 'r')
#ax1.scatter(IxI_rho_graphing_df.rho.values, IxI_rho_graphing_df.MaxVNE.values, color = 'g')
#ax1.scatter(Homerange_IxI_rho_graphing_df.rho.values, Homerange_IxI_rho_graphing_df.MaxVNE.values, color = 'b')
#ax1.scatter(Tripartite_rho_graphing_df.rho.values, Tripartite_rho_graphing_df.MaxVNE.values, color = 'k')
#ax1.scatter(SxS_rho_graphing_df.rho.values, SxS_rho_graphing_df.MaxVNE.values, color = 'y')
#
#ax1.legend(handles = [red_patch, green_patch, black_patch, blue_patch, yellow_patch], title= 'Projection')
#
#ax1.plot(IxS_rho_graphing_df.rho.values, IxS_rho_graphing_df.MaxVNE.values, color = 'r')
#ax1.plot(IxI_rho_graphing_df.rho.values, IxI_rho_graphing_df.MaxVNE.values, color = 'g')
#ax1.plot(Homerange_IxI_rho_graphing_df.rho.values, Homerange_IxI_rho_graphing_df.MaxVNE.values, color = 'b')
#ax1.plot(Tripartite_rho_graphing_df.rho.values, Tripartite_rho_graphing_df.MaxVNE.values, color = 'k')
#ax1.plot(SxS_rho_graphing_df.rho.values, SxS_rho_graphing_df.MaxVNE.values, color = 'y')
#
#
#ax2.grid(True)
##ax2.title(r'Iterating over $\beta$')
#ax2.set_xlabel(r'$\beta$')
#ax2.set_ylabel('Max VNE')
#
#ax2.scatter(IxS_beta_graphing_df.beta.values, IxS_beta_graphing_df.MaxVNE.values, color = 'r')
#ax2.scatter(IxI_beta_graphing_df.beta.values, IxI_beta_graphing_df.MaxVNE.values, color = 'g')
#ax2.scatter(Homerange_IxI_beta_graphing_df.beta.values, Homerange_IxI_beta_graphing_df.MaxVNE.values, color = 'b')
#ax2.scatter(Tripartite_beta_graphing_df.beta.values, Tripartite_beta_graphing_df.MaxVNE.values, color = 'k')
#ax2.scatter(SxS_beta_graphing_df.beta.values, SxS_beta_graphing_df.MaxVNE.values, color = 'y')
#
##ax2.legend(handles = [red_patch, green_patch, black_patch, blue_patch, yellow_patch], title= 'Projection')
#
#ax2.plot(IxS_beta_graphing_df.beta.values, IxS_beta_graphing_df.MaxVNE.values, color = 'r')
#ax2.plot(IxI_beta_graphing_df.beta.values, IxI_beta_graphing_df.MaxVNE.values, color = 'g')
#ax2.plot(Homerange_IxI_beta_graphing_df.beta.values, Homerange_IxI_beta_graphing_df.MaxVNE.values, color = 'b')
#ax2.plot(Tripartite_beta_graphing_df.beta.values, Tripartite_beta_graphing_df.MaxVNE.values, color = 'k')
#ax2.plot(SxS_beta_graphing_df.beta.values, SxS_beta_graphing_df.MaxVNE.values, color = 'y')
#
#
#ax3.grid(True)
##ax3.title(r'Iterating over $\gamma$')
#ax3.set_xlabel(r'$\gamma$')
#ax3.set_ylabel('Max VNE')
#
#ax3.scatter(IxS_gamma_graphing_df.gamma.values, IxS_gamma_graphing_df.MaxVNE.values, color = 'r')
#ax3.scatter(IxI_gamma_graphing_df.gamma.values, IxI_gamma_graphing_df.MaxVNE.values, color = 'g')
#ax3.scatter(Homerange_IxI_gamma_graphing_df.gamma.values, Homerange_IxI_gamma_graphing_df.MaxVNE.values, color = 'b')
#ax3.scatter(Tripartite_gamma_graphing_df.gamma.values, Tripartite_gamma_graphing_df.MaxVNE.values, color = 'k')
#ax3.scatter(SxS_gamma_graphing_df.gamma.values, SxS_gamma_graphing_df.MaxVNE.values, color = 'y')
#
##ax3.legend(handles = [red_patch, green_patch, black_patch, blue_patch, yellow_patch], title= 'Projection')
#
#ax3.plot(IxS_gamma_graphing_df.gamma.values, IxS_gamma_graphing_df.MaxVNE.values, color = 'r')
#ax3.plot(IxI_gamma_graphing_df.gamma.values, IxI_gamma_graphing_df.MaxVNE.values, color = 'g')
#ax3.plot(Homerange_IxI_gamma_graphing_df.gamma.values, Homerange_IxI_gamma_graphing_df.MaxVNE.values, color = 'b')
#ax3.plot(Tripartite_gamma_graphing_df.gamma.values, Tripartite_gamma_graphing_df.MaxVNE.values, color = 'k')
#ax3.plot(SxS_gamma_graphing_df.gamma.values, SxS_gamma_graphing_df.MaxVNE.values, color = 'y')
#
##plt.subplot(224)
##plt.legend(handles = [red_patch, green_patch, black_patch, blue_patch, yellow_patch], title= 'Projection')
##plt.show()
#

