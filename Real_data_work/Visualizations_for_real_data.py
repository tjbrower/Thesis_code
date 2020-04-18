import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LinearRegression as LR
from numpy.polynomial.polynomial import polyfit as pfit
import pandas as pd
import seaborn as sns

#michaelis_inputs = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Salvaging_Von_Neumann_Entropy\full_input_list_for_location_10_individuals_10_time_20_simulations_10.npy')
VNE_outputs_IxS = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Real_data_work\09Data\VNEValues\IxSVNERealData09.npy')
VNE_outputs_IxI = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Real_data_work\09Data\VNEValues\IxIVNERealData09.npy')
VNE_outputs_Tripartite = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Real_data_work\09Data\VNEValues\tripartiteVNERealData09.npy')
VNE_outputs_Homerange_IxI = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Real_data_work\09Data\VNEValues\homerangeVNERealData09.npy')
VNE_outputs_SxS = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Real_data_work\09Data\VNEValues\SxSVNERealData09.npy')

#inputs = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Salvaging_Von_Neumann_Entropy\Michaelis_inputs_location_10_individual_10.npy')
IxS_v_outputs = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Real_data_work\09Data\LWBOutputs\realDataIxSOutputs.npy')
IxI_v_outputs = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Real_data_work\09Data\LWBOutputs\realDataIxIOutputs.npy')
Tripartite_v_outputs = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Real_data_work\09Data\LWBOutputs\realDataTripartiteOutputs.npy')
Homerange_IxI_v_outputs = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Real_data_work\09Data\LWBOutputs\realDataHomerangeOutputs.npy')
SxS_v_outputs = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Real_data_work\09Data\LWBOutputs\realDataSxSOutputs.npy')
michaelisInputs = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Real_data_work\09Data\LWBOutputs\realDataMichaelisInputs.npy')

####################################################################################################
# Let's do our best to create a pandas dataframe and try to go from there
####################################################################################################
# The columns for my data fram are gamma, beta, rho, timesteps, and the v vector elements
# associated with each timestep and then the simulation number
# Let's first try getting everything in order of simulations

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

red_patch = mpatches.Patch(color='r', label='IxS')
green_patch = mpatches.Patch(color = 'g', label = 'IxI')
black_patch = mpatches.Patch(color='k', label='Tripartite')
blue_patch = mpatches.Patch(color='b', label='Homerange')
yellow_patch = mpatches.Patch(color='y', label='SxS')

inputList = np.arange(len(VNE_outputs_IxS)) + 1
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
plt.xticks(fontsize = 30)
plt.yticks(np.linspace(0, 1, 21, endpoint = True))
plt.scatter(inputList, VNE_outputs_IxS, color = 'r')
plt.plot(inputList, VNE_outputs_IxS, color = 'r')
plt.scatter(inputList, VNE_outputs_IxI, color = 'g')
plt.plot(inputList, VNE_outputs_IxI, color = 'g')
plt.scatter(inputList, VNE_outputs_Tripartite, color = 'b')
plt.plot(inputList, VNE_outputs_Tripartite, color = 'b')
plt.scatter(inputList, VNE_outputs_Homerange_IxI, color = 'k')
plt.plot(inputList, VNE_outputs_Homerange_IxI, color = 'k')
plt.scatter(inputList, VNE_outputs_SxS, color = 'y')
plt.plot(inputList, VNE_outputs_SxS, color = 'y')

plt.show()

#####################################################################################
# This is thre distinct plots with the Lineweaver results
#####################################################################################
##first_array = np.linspace(0.3, 0.9, 15, endpoint = False)
##second_array = np.linspace(0.9, 1.0, 18, endpoint = True)
##y_tick_array = np.hstack((first_array, second_array))
##
#
#fig = plt.figure()
#ax1 = plt.subplot(111)
#plt.legend(handles = [red_patch, green_patch, black_patch, blue_patch, yellow_patch], title='Projection', bbox_to_anchor=(1.05, 1.0), loc='upper left')
#
##fig.suptitle('Location = 10, Individuals = 10, Time Steps = 20, Simulations = 10', fontsize = Big_font_size)
#
#ax1.grid(True)
##ax1.title('VNE Max for rho')
#ax1.set_xlabel(r'$\rho$')
#ax1.set_ylabel('Max VNE')
##ax1.set_xticks(np.linspace(0.1, 1.0, 10, endpoint = True))
##ax1.set_yticks(np.linspace(0.15, 1.0, 18, endpoint = True))
##ax1.legend(handles = [red_patch, green_patch, black_patch, blue_patch, yellow_patch], title='Projection')
#ax1.scatter(mic, IxS_rho_graphing_df.MaxVNE.values, color = 'r')
#ax1.scatter(IxI_rho_graphing_df.rho.values, IxI_rho_graphing_df.MaxVNE.values, color = 'g')
#ax1.scatter(Homerange_IxI_rho_graphing_df.rho.values, Homerange_IxI_rho_graphing_df.MaxVNE.values, color = 'b')
#ax1.scatter(Tripartite_rho_graphing_df.rho.values, Tripartite_rho_graphing_df.MaxVNE.values, color = 'k')
#ax1.scatter(SxS_rho_graphing_df.rho.values, SxS_rho_graphing_df.MaxVNE.values, color = 'y')
#ax1.plot(IxS_rho_graphing_df.rho.values, IxS_rho_graphing_df.MaxVNE.values, color = 'r')
#ax1.plot(IxI_rho_graphing_df.rho.values, IxI_rho_graphing_df.MaxVNE.values, color = 'g')
#ax1.plot(Homerange_IxI_rho_graphing_df.rho.values, Homerange_IxI_rho_graphing_df.MaxVNE.values, color = 'b')
#ax1.plot(Tripartite_rho_graphing_df.rho.values, Tripartite_rho_graphing_df.MaxVNE.values, color = 'k')
#ax1.plot(SxS_rho_graphing_df.rho.values, SxS_rho_graphing_df.MaxVNE.values, color = 'y')
#plt.show()