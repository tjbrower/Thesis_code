import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LinearRegression as LR
import pandas as pd
import seaborn as sns


location_amount = 10
individual_amount = 10
time_stamps = 20
gamma_values = np.linspace(1, 10, 10, endpoint = True)
beta_values = np.linspace(1, 10, 10, endpoint = True)
rho_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 0.99, 0.999, 0.9999, 1.0]#np.linspace(0.1, 1.0, 10, endpoint=True)
simulation_amount = 10

IxS_VNE_df = pd.read_csv(r'C:\Users\tjbro\Desktop\Correct_10_By_10_arrays\IxSVNE10By10.csv')
IxI_VNE_df = pd.read_csv(r'C:\Users\tjbro\Desktop\Correct_10_By_10_arrays\IxIVNE10By10.csv')
Tripartite_VNE_df = pd.read_csv(r'C:\Users\tjbro\Desktop\Correct_10_By_10_arrays\TripartiteVNE10By10.csv')
Homerange_IxI_VNE_df = pd.read_csv(r'C:\Users\tjbro\Desktop\Correct_10_By_10_arrays\HomerangeVNE10By10.csv')
SxS_VNE_df = pd.read_csv(r'C:\Users\tjbro\Desktop\Correct_10_By_10_arrays\SxSVNE10By10.csv')

IxS_Max_VNE_df = pd.read_csv(r'C:\Users\tjbro\Desktop\Correct_10_By_10_arrays\IxSMaxVNE10By10.csv')
IxI_Max_VNE_df = pd.read_csv(r'C:\Users\tjbro\Desktop\Correct_10_By_10_arrays\IxIMaxVNE10By10.csv')
Tripartite_Max_VNE_df = pd.read_csv(r'C:\Users\tjbro\Desktop\Correct_10_By_10_arrays\TripartiteMaxVNE10By10.csv')
Homerange_IxI_Max_VNE_df = pd.read_csv(r'C:\Users\tjbro\Desktop\Correct_10_By_10_arrays\HomerangeMaxVNE10By10.csv')
SxS_Max_VNE_df = pd.read_csv(r'C:\Users\tjbro\Desktop\Correct_10_By_10_arrays\SxSMaxVNE10By10.csv')

IxS_v_outputs_df = pd.read_csv(r'C:\Users\tjbro\Desktop\Correct_10_By_10_arrays\IxSLineweaver10By10.csv')
IxI_v_outputs_df = pd.read_csv(r'C:\Users\tjbro\Desktop\Correct_10_By_10_arrays\IxILineweaver10By10.csv')
Tripartite_v_outputs_df = pd.read_csv(r'C:\Users\tjbro\Desktop\Correct_10_By_10_arrays\TripartiteLineweaver10By10.csv')
Homerange_IxI_v_outputs_df = pd.read_csv(r'C:\Users\tjbro\Desktop\Correct_10_By_10_arrays\HomerangeLineweaver10By10.csv')
SxS_v_outputs_df = pd.read_csv(r'C:\Users\tjbro\Desktop\Correct_10_By_10_arrays\SxSLineweaver10By10.csv')


# This is computing the Michaelis inputs for each output already calculated
input_list = list(range(1, time_stamps + 1))
#print('this is the input list\n', input_list)
one_vector = np.ones(time_stamps)
michaelis_inputs = np.divide(one_vector, input_list)

#####################################################################################
# This is the size of the text for the graphs
#####################################################################################
SMALL_SIZE = 8
MEDIUM_SIZE = 16
BIGGER_SIZE = 24
BIGGEST_SIZE = 32
Big_font_size = 56


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
# The values for the Max VNE when looking at rho while gamma and beta are base values
#####################################################################################
# This is when gamma = 1
IxS_rho_graphing_df = IxS_Max_VNE_df[IxS_Max_VNE_df.gamma == 1]
IxS_rho_graphing_df_g01_b01 = IxS_rho_graphing_df[IxS_rho_graphing_df.beta == 1]
IxS_rho_graphing_df_g01_b05 = IxS_rho_graphing_df[IxS_rho_graphing_df.beta == 5]
IxS_rho_graphing_df_g01_b09 = IxS_rho_graphing_df[IxS_rho_graphing_df.beta == 9]

IxI_rho_graphing_df = IxI_Max_VNE_df[IxI_Max_VNE_df.gamma == 1]
IxI_rho_graphing_df_g01_b01 = IxI_rho_graphing_df[IxI_rho_graphing_df.beta == 1]
IxI_rho_graphing_df_g01_b05 = IxI_rho_graphing_df[IxI_rho_graphing_df.beta == 5]
IxI_rho_graphing_df_g01_b09 = IxI_rho_graphing_df[IxI_rho_graphing_df.beta == 9]

Homerange_IxI_rho_graphing_df = Homerange_IxI_Max_VNE_df[Homerange_IxI_Max_VNE_df.gamma == 1]
Homerange_IxI_rho_graphing_df_g01_b01 = Homerange_IxI_rho_graphing_df[Homerange_IxI_rho_graphing_df.beta == 1]
Homerange_IxI_rho_graphing_df_g01_b05 = Homerange_IxI_rho_graphing_df[Homerange_IxI_rho_graphing_df.beta == 5]
Homerange_IxI_rho_graphing_df_g01_b09 = Homerange_IxI_rho_graphing_df[Homerange_IxI_rho_graphing_df.beta == 9]

Tripartite_rho_graphing_df = Tripartite_Max_VNE_df[Tripartite_Max_VNE_df.gamma == 1]
Tripartite_rho_graphing_df_g01_b01 = Tripartite_rho_graphing_df[Tripartite_rho_graphing_df.beta == 1]
Tripartite_rho_graphing_df_g01_b05 = Tripartite_rho_graphing_df[Tripartite_rho_graphing_df.beta == 5]
Tripartite_rho_graphing_df_g01_b09 = Tripartite_rho_graphing_df[Tripartite_rho_graphing_df.beta == 9]

SxS_rho_graphing_df = SxS_Max_VNE_df[SxS_Max_VNE_df.gamma == 1]
SxS_rho_graphing_df_g01_b01 = SxS_rho_graphing_df[SxS_rho_graphing_df.beta == 1]
SxS_rho_graphing_df_g01_b05 = SxS_rho_graphing_df[SxS_rho_graphing_df.beta == 5]
SxS_rho_graphing_df_g01_b09 = SxS_rho_graphing_df[SxS_rho_graphing_df.beta == 9]

# This is when gamma = 5
IxS_rho_graphing_df = IxS_Max_VNE_df[IxS_Max_VNE_df.gamma == 5]
IxS_rho_graphing_df_g05_b01 = IxS_rho_graphing_df[IxS_rho_graphing_df.beta == 1]
IxS_rho_graphing_df_g05_b05 = IxS_rho_graphing_df[IxS_rho_graphing_df.beta == 5]
IxS_rho_graphing_df_g05_b09 = IxS_rho_graphing_df[IxS_rho_graphing_df.beta == 9]

IxI_rho_graphing_df = IxI_Max_VNE_df[IxI_Max_VNE_df.gamma == 5]
IxI_rho_graphing_df_g05_b01 = IxI_rho_graphing_df[IxI_rho_graphing_df.beta == 1]
IxI_rho_graphing_df_g05_b05 = IxI_rho_graphing_df[IxI_rho_graphing_df.beta == 5]
IxI_rho_graphing_df_g05_b09 = IxI_rho_graphing_df[IxI_rho_graphing_df.beta == 9]

Homerange_IxI_rho_graphing_df = Homerange_IxI_Max_VNE_df[Homerange_IxI_Max_VNE_df.gamma == 5]
Homerange_IxI_rho_graphing_df_g05_b01 = Homerange_IxI_rho_graphing_df[Homerange_IxI_rho_graphing_df.beta == 1]
Homerange_IxI_rho_graphing_df_g05_b05 = Homerange_IxI_rho_graphing_df[Homerange_IxI_rho_graphing_df.beta == 5]
Homerange_IxI_rho_graphing_df_g05_b09 = Homerange_IxI_rho_graphing_df[Homerange_IxI_rho_graphing_df.beta == 9]

Tripartite_rho_graphing_df = Tripartite_Max_VNE_df[Tripartite_Max_VNE_df.gamma == 5]
Tripartite_rho_graphing_df_g05_b01 = Tripartite_rho_graphing_df[Tripartite_rho_graphing_df.beta == 1]
Tripartite_rho_graphing_df_g05_b05 = Tripartite_rho_graphing_df[Tripartite_rho_graphing_df.beta == 5]
Tripartite_rho_graphing_df_g05_b09 = Tripartite_rho_graphing_df[Tripartite_rho_graphing_df.beta == 9]

SxS_rho_graphing_df = SxS_Max_VNE_df[SxS_Max_VNE_df.gamma == 5]
SxS_rho_graphing_df_g05_b01 = SxS_rho_graphing_df[SxS_rho_graphing_df.beta == 1]
SxS_rho_graphing_df_g05_b05 = SxS_rho_graphing_df[SxS_rho_graphing_df.beta == 5]
SxS_rho_graphing_df_g05_b09 = SxS_rho_graphing_df[SxS_rho_graphing_df.beta == 9]

# This is when gamma = 9
IxS_rho_graphing_df = IxS_Max_VNE_df[IxS_Max_VNE_df.gamma == 9]
IxS_rho_graphing_df_g09_b01 = IxS_rho_graphing_df[IxS_rho_graphing_df.beta == 1]
IxS_rho_graphing_df_g09_b05 = IxS_rho_graphing_df[IxS_rho_graphing_df.beta == 5]
IxS_rho_graphing_df_g09_b09 = IxS_rho_graphing_df[IxS_rho_graphing_df.beta == 9]

IxI_rho_graphing_df = IxI_Max_VNE_df[IxI_Max_VNE_df.gamma == 9]
IxI_rho_graphing_df_g09_b01 = IxI_rho_graphing_df[IxI_rho_graphing_df.beta == 1]
IxI_rho_graphing_df_g09_b05 = IxI_rho_graphing_df[IxI_rho_graphing_df.beta == 5]
IxI_rho_graphing_df_g09_b09 = IxI_rho_graphing_df[IxI_rho_graphing_df.beta == 9]

Homerange_IxI_rho_graphing_df = Homerange_IxI_Max_VNE_df[Homerange_IxI_Max_VNE_df.gamma == 9]
Homerange_IxI_rho_graphing_df_g09_b01 = Homerange_IxI_rho_graphing_df[Homerange_IxI_rho_graphing_df.beta == 1]
Homerange_IxI_rho_graphing_df_g09_b05 = Homerange_IxI_rho_graphing_df[Homerange_IxI_rho_graphing_df.beta == 5]
Homerange_IxI_rho_graphing_df_g09_b09 = Homerange_IxI_rho_graphing_df[Homerange_IxI_rho_graphing_df.beta == 9]

Tripartite_rho_graphing_df = Tripartite_Max_VNE_df[Tripartite_Max_VNE_df.gamma == 9]
Tripartite_rho_graphing_df_g09_b01 = Tripartite_rho_graphing_df[Tripartite_rho_graphing_df.beta == 1]
Tripartite_rho_graphing_df_g09_b05 = Tripartite_rho_graphing_df[Tripartite_rho_graphing_df.beta == 5]
Tripartite_rho_graphing_df_g09_b09 = Tripartite_rho_graphing_df[Tripartite_rho_graphing_df.beta == 9]

SxS_rho_graphing_df = SxS_Max_VNE_df[SxS_Max_VNE_df.gamma == 9]
SxS_rho_graphing_df_g09_b01 = SxS_rho_graphing_df[SxS_rho_graphing_df.beta == 1]
SxS_rho_graphing_df_g09_b05 = SxS_rho_graphing_df[SxS_rho_graphing_df.beta == 5]
SxS_rho_graphing_df_g09_b09 = SxS_rho_graphing_df[SxS_rho_graphing_df.beta == 9]

#display(IxS_rho_graphing_df.shape)

#####################################################################################
# The values for the Max VNE when looking at beta while rho and gamma are base values
#####################################################################################
# When gamma = 1
IxS_beta_graphing_df = IxS_Max_VNE_df[IxS_Max_VNE_df.gamma == 1]
IxS_beta_graphing_df_g01_r01 = IxS_beta_graphing_df[IxS_beta_graphing_df.rho == 0.1]
IxS_beta_graphing_df_g01_r05 = IxS_beta_graphing_df[IxS_beta_graphing_df.rho == 0.5]
IxS_beta_graphing_df_g01_r09 = IxS_beta_graphing_df[IxS_beta_graphing_df.rho == 0.9]

IxI_beta_graphing_df = IxI_Max_VNE_df[IxI_Max_VNE_df.gamma == 1]
IxI_beta_graphing_df_g01_r01 = IxI_beta_graphing_df[IxI_beta_graphing_df.rho == 0.1]
IxI_beta_graphing_df_g01_r05 = IxI_beta_graphing_df[IxI_beta_graphing_df.rho == 0.5]
IxI_beta_graphing_df_g01_r09 = IxI_beta_graphing_df[IxI_beta_graphing_df.rho == 0.9]

Homerange_IxI_beta_graphing_df = Homerange_IxI_Max_VNE_df[Homerange_IxI_Max_VNE_df.gamma == 1]
Homerange_IxI_beta_graphing_df_g01_r01 = Homerange_IxI_beta_graphing_df[Homerange_IxI_beta_graphing_df.rho == 0.1]
Homerange_IxI_beta_graphing_df_g01_r05 = Homerange_IxI_beta_graphing_df[Homerange_IxI_beta_graphing_df.rho == 0.5]
Homerange_IxI_beta_graphing_df_g01_r09 = Homerange_IxI_beta_graphing_df[Homerange_IxI_beta_graphing_df.rho == 0.9]

Tripartite_beta_graphing_df = Tripartite_Max_VNE_df[Tripartite_Max_VNE_df.gamma == 1]
Tripartite_beta_graphing_df_g01_r01 = Tripartite_beta_graphing_df[Tripartite_beta_graphing_df.rho == 0.1]
Tripartite_beta_graphing_df_g01_r05 = Tripartite_beta_graphing_df[Tripartite_beta_graphing_df.rho == 0.5]
Tripartite_beta_graphing_df_g01_r09 = Tripartite_beta_graphing_df[Tripartite_beta_graphing_df.rho == 0.9]

SxS_beta_graphing_df = SxS_Max_VNE_df[SxS_Max_VNE_df.gamma == 1]
SxS_beta_graphing_df_g01_r01 = SxS_beta_graphing_df[SxS_beta_graphing_df.rho == 0.1]
SxS_beta_graphing_df_g01_r05 = SxS_beta_graphing_df[SxS_beta_graphing_df.rho == 0.5]
SxS_beta_graphing_df_g01_r09 = SxS_beta_graphing_df[SxS_beta_graphing_df.rho == 0.9]

# When gamma = 5
IxS_beta_graphing_df = IxS_Max_VNE_df[IxS_Max_VNE_df.gamma == 5]
IxS_beta_graphing_df_g05_r01 = IxS_beta_graphing_df[IxS_beta_graphing_df.rho == 0.1]
IxS_beta_graphing_df_g05_r05 = IxS_beta_graphing_df[IxS_beta_graphing_df.rho == 0.5]
IxS_beta_graphing_df_g05_r09 = IxS_beta_graphing_df[IxS_beta_graphing_df.rho == 0.9]

IxI_beta_graphing_df = IxI_Max_VNE_df[IxI_Max_VNE_df.gamma == 5]
IxI_beta_graphing_df_g05_r01 = IxI_beta_graphing_df[IxI_beta_graphing_df.rho == 0.1]
IxI_beta_graphing_df_g05_r05 = IxI_beta_graphing_df[IxI_beta_graphing_df.rho == 0.5]
IxI_beta_graphing_df_g05_r09 = IxI_beta_graphing_df[IxI_beta_graphing_df.rho == 0.9]

Homerange_IxI_beta_graphing_df = Homerange_IxI_Max_VNE_df[Homerange_IxI_Max_VNE_df.gamma == 5]
Homerange_IxI_beta_graphing_df_g05_r01 = Homerange_IxI_beta_graphing_df[Homerange_IxI_beta_graphing_df.rho == 0.1]
Homerange_IxI_beta_graphing_df_g05_r05 = Homerange_IxI_beta_graphing_df[Homerange_IxI_beta_graphing_df.rho == 0.5]
Homerange_IxI_beta_graphing_df_g05_r09 = Homerange_IxI_beta_graphing_df[Homerange_IxI_beta_graphing_df.rho == 0.9]

Tripartite_beta_graphing_df = Tripartite_Max_VNE_df[Tripartite_Max_VNE_df.gamma == 5]
Tripartite_beta_graphing_df_g05_r01 = Tripartite_beta_graphing_df[Tripartite_beta_graphing_df.rho == 0.1]
Tripartite_beta_graphing_df_g05_r05 = Tripartite_beta_graphing_df[Tripartite_beta_graphing_df.rho == 0.5]
Tripartite_beta_graphing_df_g05_r09 = Tripartite_beta_graphing_df[Tripartite_beta_graphing_df.rho == 0.9]

SxS_beta_graphing_df = SxS_Max_VNE_df[SxS_Max_VNE_df.gamma == 5]
SxS_beta_graphing_df_g05_r01 = SxS_beta_graphing_df[SxS_beta_graphing_df.rho == 0.1]
SxS_beta_graphing_df_g05_r05 = SxS_beta_graphing_df[SxS_beta_graphing_df.rho == 0.5]
SxS_beta_graphing_df_g05_r09 = SxS_beta_graphing_df[SxS_beta_graphing_df.rho == 0.9]

# When gamma = 9
IxS_beta_graphing_df = IxS_Max_VNE_df[IxS_Max_VNE_df.gamma == 9]
IxS_beta_graphing_df_g09_r01 = IxS_beta_graphing_df[IxS_beta_graphing_df.rho == 0.1]
IxS_beta_graphing_df_g09_r05 = IxS_beta_graphing_df[IxS_beta_graphing_df.rho == 0.5]
IxS_beta_graphing_df_g09_r09 = IxS_beta_graphing_df[IxS_beta_graphing_df.rho == 0.9]

IxI_beta_graphing_df = IxI_Max_VNE_df[IxI_Max_VNE_df.gamma == 9]
IxI_beta_graphing_df_g09_r01 = IxI_beta_graphing_df[IxI_beta_graphing_df.rho == 0.1]
IxI_beta_graphing_df_g09_r05 = IxI_beta_graphing_df[IxI_beta_graphing_df.rho == 0.5]
IxI_beta_graphing_df_g09_r09 = IxI_beta_graphing_df[IxI_beta_graphing_df.rho == 0.9]

Homerange_IxI_beta_graphing_df = Homerange_IxI_Max_VNE_df[Homerange_IxI_Max_VNE_df.gamma == 9]
Homerange_IxI_beta_graphing_df_g09_r01 = Homerange_IxI_beta_graphing_df[Homerange_IxI_beta_graphing_df.rho == 0.1]
Homerange_IxI_beta_graphing_df_g09_r05 = Homerange_IxI_beta_graphing_df[Homerange_IxI_beta_graphing_df.rho == 0.5]
Homerange_IxI_beta_graphing_df_g09_r09 = Homerange_IxI_beta_graphing_df[Homerange_IxI_beta_graphing_df.rho == 0.9]

Tripartite_beta_graphing_df = Tripartite_Max_VNE_df[Tripartite_Max_VNE_df.gamma == 9]
Tripartite_beta_graphing_df_g09_r01 = Tripartite_beta_graphing_df[Tripartite_beta_graphing_df.rho == 0.1]
Tripartite_beta_graphing_df_g09_r05 = Tripartite_beta_graphing_df[Tripartite_beta_graphing_df.rho == 0.5]
Tripartite_beta_graphing_df_g09_r09 = Tripartite_beta_graphing_df[Tripartite_beta_graphing_df.rho == 0.9]

SxS_beta_graphing_df = SxS_Max_VNE_df[SxS_Max_VNE_df.gamma == 9]
SxS_beta_graphing_df_g09_r01 = SxS_beta_graphing_df[SxS_beta_graphing_df.rho == 0.1]
SxS_beta_graphing_df_g09_r05 = SxS_beta_graphing_df[SxS_beta_graphing_df.rho == 0.5]
SxS_beta_graphing_df_g09_r09 = SxS_beta_graphing_df[SxS_beta_graphing_df.rho == 0.9]

#display(IxS_beta_graphing_df.shape)

#####################################################################################
# The values for the Max VNE when looking at gamma while rho and beta are base values
#####################################################################################
# This is when beta = 1
IxS_gamma_graphing_df = IxS_Max_VNE_df[IxS_Max_VNE_df.beta == 1]
IxS_gamma_graphing_df_b01_r01 = IxS_gamma_graphing_df[IxS_gamma_graphing_df.rho == 0.1]
IxS_gamma_graphing_df_b01_r05 = IxS_gamma_graphing_df[IxS_gamma_graphing_df.rho == 0.5]
IxS_gamma_graphing_df_b01_r09 = IxS_gamma_graphing_df[IxS_gamma_graphing_df.rho == 0.9]

IxI_gamma_graphing_df = IxI_Max_VNE_df[IxI_Max_VNE_df.beta == 1]
IxI_gamma_graphing_df_b01_r01 = IxI_gamma_graphing_df[IxI_gamma_graphing_df.rho == 0.1]
IxI_gamma_graphing_df_b01_r05 = IxI_gamma_graphing_df[IxI_gamma_graphing_df.rho == 0.5]
IxI_gamma_graphing_df_b01_r09 = IxI_gamma_graphing_df[IxI_gamma_graphing_df.rho == 0.9]

Homerange_IxI_gamma_graphing_df = Homerange_IxI_Max_VNE_df[Homerange_IxI_Max_VNE_df.beta == 1]
Homerange_IxI_gamma_graphing_df_b01_r01 = Homerange_IxI_gamma_graphing_df[Homerange_IxI_gamma_graphing_df.rho == 0.1]
Homerange_IxI_gamma_graphing_df_b01_r05 = Homerange_IxI_gamma_graphing_df[Homerange_IxI_gamma_graphing_df.rho == 0.5]
Homerange_IxI_gamma_graphing_df_b01_r09 = Homerange_IxI_gamma_graphing_df[Homerange_IxI_gamma_graphing_df.rho == 0.9]

Tripartite_gamma_graphing_df = Tripartite_Max_VNE_df[Tripartite_Max_VNE_df.beta == 1]
Tripartite_gamma_graphing_df_b01_r01 = Tripartite_gamma_graphing_df[Tripartite_gamma_graphing_df.rho == 0.1]
Tripartite_gamma_graphing_df_b01_r05 = Tripartite_gamma_graphing_df[Tripartite_gamma_graphing_df.rho == 0.5]
Tripartite_gamma_graphing_df_b01_r09 = Tripartite_gamma_graphing_df[Tripartite_gamma_graphing_df.rho == 0.9]

SxS_gamma_graphing_df = SxS_Max_VNE_df[SxS_Max_VNE_df.beta == 1]
SxS_gamma_graphing_df_b01_r01 = SxS_gamma_graphing_df[SxS_gamma_graphing_df.rho == 0.1]
SxS_gamma_graphing_df_b01_r05 = SxS_gamma_graphing_df[SxS_gamma_graphing_df.rho == 0.5]
SxS_gamma_graphing_df_b01_r09 = SxS_gamma_graphing_df[SxS_gamma_graphing_df.rho == 0.9]

# This is when beta = 5
IxS_gamma_graphing_df = IxS_Max_VNE_df[IxS_Max_VNE_df.beta == 5]
IxS_gamma_graphing_df_b05_r01 = IxS_gamma_graphing_df[IxS_gamma_graphing_df.rho == 0.1]
IxS_gamma_graphing_df_b05_r05 = IxS_gamma_graphing_df[IxS_gamma_graphing_df.rho == 0.5]
IxS_gamma_graphing_df_b05_r09 = IxS_gamma_graphing_df[IxS_gamma_graphing_df.rho == 0.9]

IxI_gamma_graphing_df = IxI_Max_VNE_df[IxI_Max_VNE_df.beta == 5]
IxI_gamma_graphing_df_b05_r01 = IxI_gamma_graphing_df[IxI_gamma_graphing_df.rho == 0.1]
IxI_gamma_graphing_df_b05_r05 = IxI_gamma_graphing_df[IxI_gamma_graphing_df.rho == 0.5]
IxI_gamma_graphing_df_b05_r09 = IxI_gamma_graphing_df[IxI_gamma_graphing_df.rho == 0.9]

Homerange_IxI_gamma_graphing_df = Homerange_IxI_Max_VNE_df[Homerange_IxI_Max_VNE_df.beta == 5]
Homerange_IxI_gamma_graphing_df_b05_r01 = Homerange_IxI_gamma_graphing_df[Homerange_IxI_gamma_graphing_df.rho == 0.1]
Homerange_IxI_gamma_graphing_df_b05_r05 = Homerange_IxI_gamma_graphing_df[Homerange_IxI_gamma_graphing_df.rho == 0.5]
Homerange_IxI_gamma_graphing_df_b05_r09 = Homerange_IxI_gamma_graphing_df[Homerange_IxI_gamma_graphing_df.rho == 0.9]

Tripartite_gamma_graphing_df = Tripartite_Max_VNE_df[Tripartite_Max_VNE_df.beta == 5]
Tripartite_gamma_graphing_df_b05_r01 = Tripartite_gamma_graphing_df[Tripartite_gamma_graphing_df.rho == 0.1]
Tripartite_gamma_graphing_df_b05_r05 = Tripartite_gamma_graphing_df[Tripartite_gamma_graphing_df.rho == 0.5]
Tripartite_gamma_graphing_df_b05_r09 = Tripartite_gamma_graphing_df[Tripartite_gamma_graphing_df.rho == 0.9]

SxS_gamma_graphing_df = SxS_Max_VNE_df[SxS_Max_VNE_df.beta == 5]
SxS_gamma_graphing_df_b05_r01 = SxS_gamma_graphing_df[SxS_gamma_graphing_df.rho == 0.1]
SxS_gamma_graphing_df_b05_r05 = SxS_gamma_graphing_df[SxS_gamma_graphing_df.rho == 0.5]
SxS_gamma_graphing_df_b05_r09 = SxS_gamma_graphing_df[SxS_gamma_graphing_df.rho == 0.9]

# This is when beta = 9
IxS_gamma_graphing_df = IxS_Max_VNE_df[IxS_Max_VNE_df.beta == 9]
IxS_gamma_graphing_df_b09_r01 = IxS_gamma_graphing_df[IxS_gamma_graphing_df.rho == 0.1]
IxS_gamma_graphing_df_b09_r05 = IxS_gamma_graphing_df[IxS_gamma_graphing_df.rho == 0.5]
IxS_gamma_graphing_df_b09_r09 = IxS_gamma_graphing_df[IxS_gamma_graphing_df.rho == 0.9]

IxI_gamma_graphing_df = IxI_Max_VNE_df[IxI_Max_VNE_df.beta == 9]
IxI_gamma_graphing_df_b09_r01 = IxI_gamma_graphing_df[IxI_gamma_graphing_df.rho == 0.1]
IxI_gamma_graphing_df_b09_r05 = IxI_gamma_graphing_df[IxI_gamma_graphing_df.rho == 0.5]
IxI_gamma_graphing_df_b09_r09 = IxI_gamma_graphing_df[IxI_gamma_graphing_df.rho == 0.9]

Homerange_IxI_gamma_graphing_df = Homerange_IxI_Max_VNE_df[Homerange_IxI_Max_VNE_df.beta == 9]
Homerange_IxI_gamma_graphing_df_b09_r01 = Homerange_IxI_gamma_graphing_df[Homerange_IxI_gamma_graphing_df.rho == 0.1]
Homerange_IxI_gamma_graphing_df_b09_r05 = Homerange_IxI_gamma_graphing_df[Homerange_IxI_gamma_graphing_df.rho == 0.5]
Homerange_IxI_gamma_graphing_df_b09_r09 = Homerange_IxI_gamma_graphing_df[Homerange_IxI_gamma_graphing_df.rho == 0.9]

Tripartite_gamma_graphing_df = Tripartite_Max_VNE_df[Tripartite_Max_VNE_df.beta == 9]
Tripartite_gamma_graphing_df_b09_r01 = Tripartite_gamma_graphing_df[Tripartite_gamma_graphing_df.rho == 0.1]
Tripartite_gamma_graphing_df_b09_r05 = Tripartite_gamma_graphing_df[Tripartite_gamma_graphing_df.rho == 0.5]
Tripartite_gamma_graphing_df_b09_r09 = Tripartite_gamma_graphing_df[Tripartite_gamma_graphing_df.rho == 0.9]

SxS_gamma_graphing_df = SxS_Max_VNE_df[SxS_Max_VNE_df.beta == 9]
SxS_gamma_graphing_df_b09_r01 = SxS_gamma_graphing_df[SxS_gamma_graphing_df.rho == 0.1]
SxS_gamma_graphing_df_b09_r05 = SxS_gamma_graphing_df[SxS_gamma_graphing_df.rho == 0.5]
SxS_gamma_graphing_df_b09_r09 = SxS_gamma_graphing_df[SxS_gamma_graphing_df.rho == 0.9]

#display(IxS_gamma_graphing_df.shape)

#####################################################################################
# This is for the regression plot used when estimating the max VNE values
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

######################################################################################
## This is the graph of the VNE for each projection; FIRST GRAPH
#####################################################################################
plt.figure()
plt.grid(True)
plt.legend(handles = [red_patch, green_patch, black_patch, blue_patch, yellow_patch], title = 'Projection', bbox_to_anchor=(1.12, 1.0), loc='upper right')

#plt.suptitle('IxS Lineweaver-Burk Plot with Regression Line', fontsize = Big_font_size)
plt.xlabel('Time Step', fontsize = Big_font_size)
plt.ylabel('VNE value', fontsize = Big_font_size)
plt.xticks(np.linspace(1, 20, 20, endpoint = True), fontsize = 30)
plt.yticks(np.linspace(0, 1, 21, endpoint = True))
plt.scatter(IxS_VNE_graphing_df.timeStep.values, IxS_VNE_graphing_df.outputValues.values, color = 'r')
plt.scatter(IxI_VNE_graphing_df.timeStep.values, IxI_VNE_graphing_df.outputValues.values, color = 'g')
plt.scatter(Homerange_IxI_VNE_graphing_df.timeStep.values, Homerange_IxI_VNE_graphing_df.outputValues.values, color = 'b')
plt.scatter(Tripartite_VNE_graphing_df.timeStep.values, Tripartite_VNE_graphing_df.outputValues.values, color = 'k')
plt.scatter(SxS_VNE_graphing_df.timeStep.values, SxS_VNE_graphing_df.outputValues.values, color = 'y')

plt.plot([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], IxS_avg, color = 'r')
plt.plot([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], IxI_avg, color = 'g')
plt.plot([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], Homerange_IxI_avg, color = 'b')
plt.plot([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], Tripartite_avg, color = 'k')
plt.plot([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], SxS_avg, color = 'y')
plt.show()

# This next part may not be super necessary for a publication, but I included it just in case.
#######################################################################################
## This is the visualization of the Lineweaver-Burk plot; REGRESSION GRAPH FOR IXS
#######################################################################################
#IxS_reg = LR().fit(np.array(IxS_v_graphing_df.michaelisInputs.values).reshape(-1,1), np.array(IxS_v_graphing_df.outputValues.values).reshape(-1,1))
#intercept = IxS_reg.intercept_
#outputVector = []
##for i in IxS_v_graphing_df.michaelisInputs.values:
##    outputVector.append(coef[0]*i + coef[1])
#plt.figure()
#plt.grid(True)
##plt.suptitle('IxS Lineweaver-Burk Plot with Regression Line', fontsize = Big_font_size)
#sns.regplot(x = 'michaelisInputs', y = 'outputValues', data = IxS_v_graphing_df, color = 'b')
#plt.scatter(IxS_v_graphing_df.michaelisInputs.values, IxS_v_graphing_df.outputValues.values, c = 'r')
#plt.xlabel('Michaelis Inputs', fontsize = Big_font_size)
#plt.ylabel('1/v', fontsize = Big_font_size)
#plt.show()

# This next section is isolating just the beta graph instead of having it be on a 3 x 3 grid
######################################################################################
## This is the plot of interest
######################################################################################
#plt.figure()
#ax =plt.subplot(111)
#plt.grid(True)
##ax.set_xlabel(r'$\rho$')
##ax.set_ylabel('Max VNE')
#ax.set_xticks(np.linspace(1, 10, 10, endpoint = True))
#ax.set_yticks(np.linspace(0.0, 1.0, 21, endpoint = True))
#ax.legend(handles = [red_patch, green_patch, black_patch, blue_patch, yellow_patch], title='Projection')
#ax.scatter(IxS_beta_graphing_df_g01_r01.beta.values, IxS_beta_graphing_df_g01_r01.MaxVNE.values, color = 'r')
#ax.scatter(IxI_beta_graphing_df_g01_r01.beta.values, IxI_beta_graphing_df_g01_r01.MaxVNE.values, color = 'g')
#ax.scatter(Homerange_IxI_beta_graphing_df_g01_r01.beta.values, Homerange_IxI_beta_graphing_df_g01_r01.MaxVNE.values, color = 'b')
#ax.scatter(Tripartite_beta_graphing_df_g01_r01.beta.values, Tripartite_beta_graphing_df_g01_r01.MaxVNE.values, color = 'k')
#ax.scatter(SxS_beta_graphing_df_g01_r01.beta.values, SxS_beta_graphing_df_g01_r01.MaxVNE.values, color = 'y')
#ax.plot(IxS_beta_graphing_df_g01_r01.beta.values, IxS_beta_graphing_df_g01_r01.MaxVNE.values, color = 'r')
#ax.plot(IxI_beta_graphing_df_g01_r01.beta.values, IxI_beta_graphing_df_g01_r01.MaxVNE.values, color = 'g')
#ax.plot(Homerange_IxI_beta_graphing_df_g01_r01.beta.values, Homerange_IxI_beta_graphing_df_g01_r01.MaxVNE.values, color = 'b')
#ax.plot(Tripartite_beta_graphing_df_g01_r01.beta.values, Tripartite_beta_graphing_df_g01_r01.MaxVNE.values, color = 'k')
#ax.plot(SxS_beta_graphing_df_g01_r01.beta.values, SxS_beta_graphing_df_g01_r01.MaxVNE.values, color = 'y')
#plt.show()

#####################################################################################
# This is thre distinct plots showing the estimated Max VNE for gamma, beta, and rho; RESULTS COMBINED - ALL 3
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


ax1.grid(True)
ax1.set_xticks(np.linspace(0.1, 1.0, 10, endpoint = True))
ax1.set_yticks(np.linspace(0.0, 1.0, 21, endpoint = True))
ax1.scatter(IxS_rho_graphing_df_g01_b01.rho.values, IxS_rho_graphing_df_g01_b01.MaxVNE.values, color = 'r')
ax1.scatter(IxI_rho_graphing_df_g01_b01.rho.values, IxI_rho_graphing_df_g01_b01.MaxVNE.values, color = 'g')
ax1.scatter(Homerange_IxI_rho_graphing_df_g01_b01.rho.values, Homerange_IxI_rho_graphing_df_g01_b01.MaxVNE.values, color = 'b')
ax1.scatter(Tripartite_rho_graphing_df_g01_b01.rho.values, Tripartite_rho_graphing_df_g01_b01.MaxVNE.values, color = 'k')
ax1.scatter(SxS_rho_graphing_df_g01_b01.rho.values, SxS_rho_graphing_df_g01_b01.MaxVNE.values, color = 'y')
ax1.plot(IxS_rho_graphing_df_g01_b01.rho.values, IxS_rho_graphing_df_g01_b01.MaxVNE.values, color = 'r')
ax1.plot(IxI_rho_graphing_df_g01_b01.rho.values, IxI_rho_graphing_df_g01_b01.MaxVNE.values, color = 'g')
ax1.plot(Homerange_IxI_rho_graphing_df_g01_b01.rho.values, Homerange_IxI_rho_graphing_df_g01_b01.MaxVNE.values, color = 'b')
ax1.plot(Tripartite_rho_graphing_df_g01_b01.rho.values, Tripartite_rho_graphing_df_g01_b01.MaxVNE.values, color = 'k')
ax1.plot(SxS_rho_graphing_df_g01_b01.rho.values, SxS_rho_graphing_df_g01_b01.MaxVNE.values, color = 'y')

ax2.grid(True)
ax2.set_xticks(np.linspace(1, 10, 10, endpoint = True))
ax2.set_yticks(np.linspace(0.0, 1.0, 21, endpoint = True))
ax2.scatter(IxS_beta_graphing_df_g01_r01.beta.values, IxS_beta_graphing_df_g01_r01.MaxVNE.values, color = 'r')
ax2.scatter(IxI_beta_graphing_df_g01_r01.beta.values, IxI_beta_graphing_df_g01_r01.MaxVNE.values, color = 'g')
ax2.scatter(Homerange_IxI_beta_graphing_df_g01_r01.beta.values, Homerange_IxI_beta_graphing_df_g01_r01.MaxVNE.values, color = 'b')
ax2.scatter(Tripartite_beta_graphing_df_g01_r01.beta.values, Tripartite_beta_graphing_df_g01_r01.MaxVNE.values, color = 'k')
ax2.scatter(SxS_beta_graphing_df_g01_r01.beta.values, SxS_beta_graphing_df_g01_r01.MaxVNE.values, color = 'y')
ax2.plot(IxS_beta_graphing_df_g01_r01.beta.values, IxS_beta_graphing_df_g01_r01.MaxVNE.values, color = 'r')
ax2.plot(IxI_beta_graphing_df_g01_r01.beta.values, IxI_beta_graphing_df_g01_r01.MaxVNE.values, color = 'g')
ax2.plot(Homerange_IxI_beta_graphing_df_g01_r01.beta.values, Homerange_IxI_beta_graphing_df_g01_r01.MaxVNE.values, color = 'b')
ax2.plot(Tripartite_beta_graphing_df_g01_r01.beta.values, Tripartite_beta_graphing_df_g01_r01.MaxVNE.values, color = 'k')
ax2.plot(SxS_beta_graphing_df_g01_r01.beta.values, SxS_beta_graphing_df_g01_r01.MaxVNE.values, color = 'y')

ax3.grid(True)
ax3.set_xticks(np.linspace(1, 10, 10, endpoint = True))
ax3.set_yticks(np.linspace(0.0, 1.0, 21, endpoint = True))
ax3.scatter(IxS_gamma_graphing_df_b01_r01.gamma.values, IxS_gamma_graphing_df_b01_r01.MaxVNE.values, color = 'r')
ax3.scatter(IxI_gamma_graphing_df_b01_r01.gamma.values, IxI_gamma_graphing_df_b01_r01.MaxVNE.values, color = 'g')
ax3.scatter(Homerange_IxI_gamma_graphing_df_b01_r01.gamma.values, Homerange_IxI_gamma_graphing_df_b01_r01.MaxVNE.values, color = 'b')
ax3.scatter(Tripartite_gamma_graphing_df_b01_r01.gamma.values, Tripartite_gamma_graphing_df_b01_r01.MaxVNE.values, color = 'k')
ax3.scatter(SxS_gamma_graphing_df_b01_r01.gamma.values, SxS_gamma_graphing_df_b01_r01.MaxVNE.values, color = 'y')
ax3.plot(IxS_gamma_graphing_df_b01_r01.gamma.values, IxS_gamma_graphing_df_b01_r01.MaxVNE.values, color = 'r')
ax3.plot(IxI_gamma_graphing_df_b01_r01.gamma.values, IxI_gamma_graphing_df_b01_r01.MaxVNE.values, color = 'g')
ax3.plot(Homerange_IxI_gamma_graphing_df_b01_r01.gamma.values, Homerange_IxI_gamma_graphing_df_b01_r01.MaxVNE.values, color = 'b')
ax3.plot(Tripartite_gamma_graphing_df_b01_r01.gamma.values, Tripartite_gamma_graphing_df_b01_r01.MaxVNE.values, color = 'k')
ax3.plot(SxS_gamma_graphing_df_b01_r01.gamma.values, SxS_gamma_graphing_df_b01_r01.MaxVNE.values, color = 'y')

ax4.grid(True)
ax4.tick_params(labelsize = 'medium')
ax4.set_xticks(np.linspace(0.1, 1.0, 10, endpoint = True))
ax4.set_yticks(second_array)
ax4.scatter(IxI_rho_graphing_df_g01_b01.rho.values, IxI_rho_graphing_df_g01_b01.MaxVNE.values, color = 'g')
ax4.scatter(Homerange_IxI_rho_graphing_df_g01_b01.rho.values, Homerange_IxI_rho_graphing_df_g01_b01.MaxVNE.values, color = 'b')
ax4.plot(IxI_rho_graphing_df_g01_b01.rho.values, IxI_rho_graphing_df_g01_b01.MaxVNE.values, color = 'g')
ax4.plot(Homerange_IxI_rho_graphing_df_g01_b01.rho.values, Homerange_IxI_rho_graphing_df_g01_b01.MaxVNE.values, color = 'b')

ax5.tick_params(labelsize = 'medium')
ax5.grid(True)
ax5.set_xticks(np.linspace(1, 10, 10, endpoint = True))
ax5.set_yticks(second_array)
ax5.scatter(IxI_beta_graphing_df_g01_r01.beta.values, IxI_beta_graphing_df_g01_r01.MaxVNE.values, color = 'g')
ax5.scatter(Homerange_IxI_beta_graphing_df_g01_r01.beta.values, Homerange_IxI_beta_graphing_df_g01_r01.MaxVNE.values, color = 'b')
ax5.plot(IxI_beta_graphing_df_g01_r01.beta.values, IxI_beta_graphing_df_g01_r01.MaxVNE.values, color = 'g')
ax5.plot(Homerange_IxI_beta_graphing_df_g01_r01.beta.values, Homerange_IxI_beta_graphing_df_g01_r01.MaxVNE.values, color = 'b')

ax6.tick_params(labelsize = 'medium')
ax6.grid(True)
ax6.set_xticks(np.linspace(1, 10, 10, endpoint = True))
ax6.set_yticks(second_array)
ax6.scatter(IxI_gamma_graphing_df_b01_r01.gamma.values, IxI_gamma_graphing_df_b01_r01.MaxVNE.values, color = 'g')
ax6.scatter(Homerange_IxI_gamma_graphing_df_b01_r01.gamma.values, Homerange_IxI_gamma_graphing_df_b01_r01.MaxVNE.values, color = 'b')
ax6.plot(IxI_gamma_graphing_df_b01_r01.gamma.values, IxI_gamma_graphing_df_b01_r01.MaxVNE.values, color = 'g')
ax6.plot(Homerange_IxI_gamma_graphing_df_b01_r01.gamma.values, Homerange_IxI_gamma_graphing_df_b01_r01.MaxVNE.values, color = 'b')

plt.tight_layout()

###################################################################################################
# This is a 3 x 3 graph where either rho is varied when gamma, beta = 1, 5, and 9.
###################################################################################################
fig, ((ax11, ax12, ax13), (ax14, ax15, ax16), (ax17, ax18, ax19)) = plt.subplots(nrows = 3, ncols = 3)

ax11.grid(True)
ax11.set_xticks(np.linspace(0.1,1,10, endpoint = True))
ax11.set_yticks(np.linspace(0, 1, 21, endpoint = True))
ax11.scatter(IxS_rho_graphing_df_g01_b01.rho.values, IxS_rho_graphing_df_g01_b01.MaxVNE.values, color = 'r')
ax11.scatter(IxI_rho_graphing_df_g01_b01.rho.values, IxI_rho_graphing_df_g01_b01.MaxVNE.values, color = 'g')
ax11.scatter(Homerange_IxI_rho_graphing_df_g01_b01.rho.values, Homerange_IxI_rho_graphing_df_g01_b01.MaxVNE.values, color = 'b')
ax11.scatter(Tripartite_rho_graphing_df_g01_b01.rho.values, Tripartite_rho_graphing_df_g01_b01.MaxVNE.values, color = 'k')
ax11.scatter(SxS_rho_graphing_df_g01_b01.rho.values, SxS_rho_graphing_df_g01_b01.MaxVNE.values, color = 'y')
ax11.plot(IxS_rho_graphing_df_g01_b01.rho.values, IxS_rho_graphing_df_g01_b01.MaxVNE.values, color = 'r')
ax11.plot(IxI_rho_graphing_df_g01_b01.rho.values, IxI_rho_graphing_df_g01_b01.MaxVNE.values, color = 'g')
ax11.plot(Homerange_IxI_rho_graphing_df_g01_b01.rho.values, Homerange_IxI_rho_graphing_df_g01_b01.MaxVNE.values, color = 'b')
ax11.plot(Tripartite_rho_graphing_df_g01_b01.rho.values, Tripartite_rho_graphing_df_g01_b01.MaxVNE.values, color = 'k')
ax11.plot(SxS_rho_graphing_df_g01_b01.rho.values, SxS_rho_graphing_df_g01_b01.MaxVNE.values, color = 'y')

ax12.grid(True)
ax12.set_xticks(np.linspace(0.1,1,10, endpoint = True))
ax12.set_yticks(np.linspace(0, 1, 21, endpoint = True))
ax12.scatter(IxS_rho_graphing_df_g01_b05.rho.values, IxS_rho_graphing_df_g01_b05.MaxVNE.values, color = 'r')
ax12.scatter(IxI_rho_graphing_df_g01_b05.rho.values, IxI_rho_graphing_df_g01_b05.MaxVNE.values, color = 'g')
ax12.scatter(Homerange_IxI_rho_graphing_df_g01_b05.rho.values, Homerange_IxI_rho_graphing_df_g01_b05.MaxVNE.values, color = 'b')
ax12.scatter(Tripartite_rho_graphing_df_g01_b05.rho.values, Tripartite_rho_graphing_df_g01_b05.MaxVNE.values, color = 'k')
ax12.scatter(SxS_rho_graphing_df_g01_b05.rho.values, SxS_rho_graphing_df_g01_b05.MaxVNE.values, color = 'y')
ax12.plot(IxS_rho_graphing_df_g01_b05.rho.values, IxS_rho_graphing_df_g01_b05.MaxVNE.values, color = 'r')
ax12.plot(IxI_rho_graphing_df_g01_b05.rho.values, IxI_rho_graphing_df_g01_b05.MaxVNE.values, color = 'g')
ax12.plot(Homerange_IxI_rho_graphing_df_g01_b05.rho.values, Homerange_IxI_rho_graphing_df_g01_b05.MaxVNE.values, color = 'b')
ax12.plot(Tripartite_rho_graphing_df_g01_b05.rho.values, Tripartite_rho_graphing_df_g01_b05.MaxVNE.values, color = 'k')
ax12.plot(SxS_rho_graphing_df_g01_b05.rho.values, SxS_rho_graphing_df_g01_b05.MaxVNE.values, color = 'y')

ax13.grid(True)
ax13.set_xticks(np.linspace(0.1,1,10, endpoint = True))
ax13.set_yticks(np.linspace(0, 1, 21, endpoint = True))
ax13.scatter(IxS_rho_graphing_df_g01_b09.rho.values, IxS_rho_graphing_df_g01_b09.MaxVNE.values, color = 'r')
ax13.scatter(IxI_rho_graphing_df_g01_b09.rho.values, IxI_rho_graphing_df_g01_b09.MaxVNE.values, color = 'g')
ax13.scatter(Homerange_IxI_rho_graphing_df_g01_b09.rho.values, Homerange_IxI_rho_graphing_df_g01_b09.MaxVNE.values, color = 'b')
ax13.scatter(Tripartite_rho_graphing_df_g01_b09.rho.values, Tripartite_rho_graphing_df_g01_b09.MaxVNE.values, color = 'k')
ax13.scatter(SxS_rho_graphing_df_g01_b09.rho.values, SxS_rho_graphing_df_g01_b09.MaxVNE.values, color = 'y')
ax13.plot(IxS_rho_graphing_df_g01_b09.rho.values, IxS_rho_graphing_df_g01_b09.MaxVNE.values, color = 'r')
ax13.plot(IxI_rho_graphing_df_g01_b09.rho.values, IxI_rho_graphing_df_g01_b09.MaxVNE.values, color = 'g')
ax13.plot(Homerange_IxI_rho_graphing_df_g01_b09.rho.values, Homerange_IxI_rho_graphing_df_g01_b09.MaxVNE.values, color = 'b')
ax13.plot(Tripartite_rho_graphing_df_g01_b09.rho.values, Tripartite_rho_graphing_df_g01_b09.MaxVNE.values, color = 'k')
ax13.plot(SxS_rho_graphing_df_g01_b09.rho.values, SxS_rho_graphing_df_g01_b09.MaxVNE.values, color = 'y')

ax14.grid(True)
ax14.set_xticks(np.linspace(0.1,1,10, endpoint = True))
ax14.set_yticks(np.linspace(0, 1, 21, endpoint = True))
ax14.scatter(IxS_rho_graphing_df_g05_b01.rho.values, IxS_rho_graphing_df_g05_b01.MaxVNE.values, color = 'r')
ax14.scatter(IxI_rho_graphing_df_g05_b01.rho.values, IxI_rho_graphing_df_g05_b01.MaxVNE.values, color = 'g')
ax14.scatter(Homerange_IxI_rho_graphing_df_g05_b01.rho.values, Homerange_IxI_rho_graphing_df_g05_b01.MaxVNE.values, color = 'b')
ax14.scatter(Tripartite_rho_graphing_df_g05_b01.rho.values, Tripartite_rho_graphing_df_g05_b01.MaxVNE.values, color = 'k')
ax14.scatter(SxS_rho_graphing_df_g05_b01.rho.values, SxS_rho_graphing_df_g05_b01.MaxVNE.values, color = 'y')
ax14.plot(IxS_rho_graphing_df_g05_b01.rho.values, IxS_rho_graphing_df_g05_b01.MaxVNE.values, color = 'r')
ax14.plot(IxI_rho_graphing_df_g05_b01.rho.values, IxI_rho_graphing_df_g05_b01.MaxVNE.values, color = 'g')
ax14.plot(Homerange_IxI_rho_graphing_df_g05_b01.rho.values, Homerange_IxI_rho_graphing_df_g05_b01.MaxVNE.values, color = 'b')
ax14.plot(Tripartite_rho_graphing_df_g05_b01.rho.values, Tripartite_rho_graphing_df_g05_b01.MaxVNE.values, color = 'k')
ax14.plot(SxS_rho_graphing_df_g05_b01.rho.values, SxS_rho_graphing_df_g05_b01.MaxVNE.values, color = 'y')

ax15.grid(True)
ax15.set_xticks(np.linspace(0.1,1,10, endpoint = True))
ax15.set_yticks(np.linspace(0, 1, 21, endpoint = True))
ax15.scatter(IxS_rho_graphing_df_g05_b05.rho.values, IxS_rho_graphing_df_g05_b05.MaxVNE.values, color = 'r')
ax15.scatter(IxI_rho_graphing_df_g05_b05.rho.values, IxI_rho_graphing_df_g05_b05.MaxVNE.values, color = 'g')
ax15.scatter(Homerange_IxI_rho_graphing_df_g05_b05.rho.values, Homerange_IxI_rho_graphing_df_g05_b05.MaxVNE.values, color = 'b')
ax15.scatter(Tripartite_rho_graphing_df_g05_b05.rho.values, Tripartite_rho_graphing_df_g05_b05.MaxVNE.values, color = 'k')
ax15.scatter(SxS_rho_graphing_df_g05_b05.rho.values, SxS_rho_graphing_df_g05_b05.MaxVNE.values, color = 'y')
ax15.plot(IxS_rho_graphing_df_g05_b05.rho.values, IxS_rho_graphing_df_g05_b05.MaxVNE.values, color = 'r')
ax15.plot(IxI_rho_graphing_df_g05_b05.rho.values, IxI_rho_graphing_df_g05_b05.MaxVNE.values, color = 'g')
ax15.plot(Homerange_IxI_rho_graphing_df_g05_b05.rho.values, Homerange_IxI_rho_graphing_df_g05_b05.MaxVNE.values, color = 'b')
ax15.plot(Tripartite_rho_graphing_df_g05_b05.rho.values, Tripartite_rho_graphing_df_g05_b05.MaxVNE.values, color = 'k')
ax15.plot(SxS_rho_graphing_df_g05_b05.rho.values, SxS_rho_graphing_df_g05_b05.MaxVNE.values, color = 'y')

ax16.grid(True)
ax16.set_xticks(np.linspace(0.1,1,10, endpoint = True))
ax16.set_yticks(np.linspace(0, 1, 21, endpoint = True))
ax16.scatter(IxS_rho_graphing_df_g05_b09.rho.values, IxS_rho_graphing_df_g05_b09.MaxVNE.values, color = 'r')
ax16.scatter(IxI_rho_graphing_df_g05_b09.rho.values, IxI_rho_graphing_df_g05_b09.MaxVNE.values, color = 'g')
ax16.scatter(Homerange_IxI_rho_graphing_df_g05_b09.rho.values, Homerange_IxI_rho_graphing_df_g05_b09.MaxVNE.values, color = 'b')
ax16.scatter(Tripartite_rho_graphing_df_g05_b09.rho.values, Tripartite_rho_graphing_df_g05_b09.MaxVNE.values, color = 'k')
ax16.scatter(SxS_rho_graphing_df_g05_b09.rho.values, SxS_rho_graphing_df_g05_b09.MaxVNE.values, color = 'y')
ax16.plot(IxS_rho_graphing_df_g05_b09.rho.values, IxS_rho_graphing_df_g05_b09.MaxVNE.values, color = 'r')
ax16.plot(IxI_rho_graphing_df_g05_b09.rho.values, IxI_rho_graphing_df_g05_b09.MaxVNE.values, color = 'g')
ax16.plot(Homerange_IxI_rho_graphing_df_g05_b09.rho.values, Homerange_IxI_rho_graphing_df_g05_b09.MaxVNE.values, color = 'b')
ax16.plot(Tripartite_rho_graphing_df_g05_b09.rho.values, Tripartite_rho_graphing_df_g05_b09.MaxVNE.values, color = 'k')
ax16.plot(SxS_rho_graphing_df_g05_b09.rho.values, SxS_rho_graphing_df_g05_b09.MaxVNE.values, color = 'y')

ax17.grid(True)
ax17.set_xticks(np.linspace(0.1,1,10, endpoint = True))
ax17.set_yticks(np.linspace(0, 1, 21, endpoint = True))
ax17.scatter(IxS_rho_graphing_df_g09_b01.rho.values, IxS_rho_graphing_df_g09_b01.MaxVNE.values, color = 'r')
ax17.scatter(IxI_rho_graphing_df_g09_b01.rho.values, IxI_rho_graphing_df_g09_b01.MaxVNE.values, color = 'g')
ax17.scatter(Homerange_IxI_rho_graphing_df_g09_b01.rho.values, Homerange_IxI_rho_graphing_df_g09_b01.MaxVNE.values, color = 'b')
ax17.scatter(Tripartite_rho_graphing_df_g09_b01.rho.values, Tripartite_rho_graphing_df_g09_b01.MaxVNE.values, color = 'k')
ax17.scatter(SxS_rho_graphing_df_g09_b01.rho.values, SxS_rho_graphing_df_g09_b01.MaxVNE.values, color = 'y')
ax17.plot(IxS_rho_graphing_df_g09_b01.rho.values, IxS_rho_graphing_df_g09_b01.MaxVNE.values, color = 'r')
ax17.plot(IxI_rho_graphing_df_g09_b01.rho.values, IxI_rho_graphing_df_g09_b01.MaxVNE.values, color = 'g')
ax17.plot(Homerange_IxI_rho_graphing_df_g09_b01.rho.values, Homerange_IxI_rho_graphing_df_g09_b01.MaxVNE.values, color = 'b')
ax17.plot(Tripartite_rho_graphing_df_g09_b01.rho.values, Tripartite_rho_graphing_df_g09_b01.MaxVNE.values, color = 'k')
ax17.plot(SxS_rho_graphing_df_g09_b01.rho.values, SxS_rho_graphing_df_g09_b01.MaxVNE.values, color = 'y')

ax18.grid(True)
ax18.set_xticks(np.linspace(0.1,1,10, endpoint = True))
ax18.set_yticks(np.linspace(0, 1, 21, endpoint = True))
ax18.scatter(IxS_rho_graphing_df_g09_b05.rho.values, IxS_rho_graphing_df_g09_b05.MaxVNE.values, color = 'r')
ax18.scatter(IxI_rho_graphing_df_g09_b05.rho.values, IxI_rho_graphing_df_g09_b05.MaxVNE.values, color = 'g')
ax18.scatter(Homerange_IxI_rho_graphing_df_g09_b05.rho.values, Homerange_IxI_rho_graphing_df_g09_b05.MaxVNE.values, color = 'b')
ax18.scatter(Tripartite_rho_graphing_df_g09_b05.rho.values, Tripartite_rho_graphing_df_g09_b05.MaxVNE.values, color = 'k')
ax18.scatter(SxS_rho_graphing_df_g09_b05.rho.values, SxS_rho_graphing_df_g09_b05.MaxVNE.values, color = 'y')
ax18.plot(IxS_rho_graphing_df_g09_b05.rho.values, IxS_rho_graphing_df_g09_b05.MaxVNE.values, color = 'r')
ax18.plot(IxI_rho_graphing_df_g09_b05.rho.values, IxI_rho_graphing_df_g09_b05.MaxVNE.values, color = 'g')
ax18.plot(Homerange_IxI_rho_graphing_df_g09_b05.rho.values, Homerange_IxI_rho_graphing_df_g09_b05.MaxVNE.values, color = 'b')
ax18.plot(Tripartite_rho_graphing_df_g09_b05.rho.values, Tripartite_rho_graphing_df_g09_b05.MaxVNE.values, color = 'k')
ax18.plot(SxS_rho_graphing_df_g09_b05.rho.values, SxS_rho_graphing_df_g09_b05.MaxVNE.values, color = 'y')

ax19.grid(True)
ax19.set_xticks(np.linspace(0.1,1,10, endpoint = True))
ax19.set_yticks(np.linspace(0, 1, 21, endpoint = True))
ax19.scatter(IxS_rho_graphing_df_g09_b09.rho.values, IxS_rho_graphing_df_g09_b09.MaxVNE.values, color = 'r')
ax19.scatter(IxI_rho_graphing_df_g09_b09.rho.values, IxI_rho_graphing_df_g09_b09.MaxVNE.values, color = 'g')
ax19.scatter(Homerange_IxI_rho_graphing_df_g09_b09.rho.values, Homerange_IxI_rho_graphing_df_g09_b09.MaxVNE.values, color = 'b')
ax19.scatter(Tripartite_rho_graphing_df_g09_b09.rho.values, Tripartite_rho_graphing_df_g09_b09.MaxVNE.values, color = 'k')
ax19.scatter(SxS_rho_graphing_df_g09_b09.rho.values, SxS_rho_graphing_df_g09_b09.MaxVNE.values, color = 'y')
ax19.plot(IxS_rho_graphing_df_g09_b09.rho.values, IxS_rho_graphing_df_g09_b09.MaxVNE.values, color = 'r')
ax19.plot(IxI_rho_graphing_df_g09_b09.rho.values, IxI_rho_graphing_df_g09_b09.MaxVNE.values, color = 'g')
ax19.plot(Homerange_IxI_rho_graphing_df_g09_b09.rho.values, Homerange_IxI_rho_graphing_df_g09_b09.MaxVNE.values, color = 'b')
ax19.plot(Tripartite_rho_graphing_df_g09_b09.rho.values, Tripartite_rho_graphing_df_g09_b09.MaxVNE.values, color = 'k')
ax19.plot(SxS_rho_graphing_df_g09_b09.rho.values, SxS_rho_graphing_df_g09_b09.MaxVNE.values, color = 'y')

plt.show()

##################################################################################################
# This is a 3 x 3 graph where either gamma is varied when rho, beta = 1, 5, and 9.
##################################################################################################
fig, ((ax21, ax22, ax23), (ax24, ax25, ax26), (ax27, ax28, ax29)) = plt.subplots(nrows = 3, ncols = 3)

ax21.grid(True)
ax21.set_xticks(np.linspace(1,10,10, endpoint = True))
ax21.set_yticks(np.linspace(0, 1, 21, endpoint = True))
ax21.scatter(IxS_gamma_graphing_df_b01_r01.gamma.values, IxS_gamma_graphing_df_b01_r01.MaxVNE.values, color = 'r')
ax21.scatter(IxI_gamma_graphing_df_b01_r01.gamma.values, IxI_gamma_graphing_df_b01_r01.MaxVNE.values, color = 'g')
ax21.scatter(Homerange_IxI_gamma_graphing_df_b01_r01.gamma.values, Homerange_IxI_gamma_graphing_df_b01_r01.MaxVNE.values, color = 'b')
ax21.scatter(Tripartite_gamma_graphing_df_b01_r01.gamma.values, Tripartite_gamma_graphing_df_b01_r01.MaxVNE.values, color = 'k')
ax21.scatter(SxS_gamma_graphing_df_b01_r01.gamma.values, SxS_gamma_graphing_df_b01_r01.MaxVNE.values, color = 'y')
ax21.plot(IxS_gamma_graphing_df_b01_r01.gamma.values, IxS_gamma_graphing_df_b01_r01.MaxVNE.values, color = 'r')
ax21.plot(IxI_gamma_graphing_df_b01_r01.gamma.values, IxI_gamma_graphing_df_b01_r01.MaxVNE.values, color = 'g')
ax21.plot(Homerange_IxI_gamma_graphing_df_b01_r01.gamma.values, Homerange_IxI_gamma_graphing_df_b01_r01.MaxVNE.values, color = 'b')
ax21.plot(Tripartite_gamma_graphing_df_b01_r01.gamma.values, Tripartite_gamma_graphing_df_b01_r01.MaxVNE.values, color = 'k')
ax21.plot(SxS_gamma_graphing_df_b01_r01.gamma.values, SxS_gamma_graphing_df_b01_r01.MaxVNE.values, color = 'y')

ax22.grid(True)
ax22.set_xticks(np.linspace(1,10,10, endpoint = True))
ax22.set_yticks(np.linspace(0, 1, 21, endpoint = True))
ax22.scatter(IxS_gamma_graphing_df_b01_r05.gamma.values, IxS_gamma_graphing_df_b01_r05.MaxVNE.values, color = 'r')
ax22.scatter(IxI_gamma_graphing_df_b01_r05.gamma.values, IxI_gamma_graphing_df_b01_r05.MaxVNE.values, color = 'g')
ax22.scatter(Homerange_IxI_gamma_graphing_df_b01_r05.gamma.values, Homerange_IxI_gamma_graphing_df_b01_r05.MaxVNE.values, color = 'b')
ax22.scatter(Tripartite_gamma_graphing_df_b01_r05.gamma.values, Tripartite_gamma_graphing_df_b01_r05.MaxVNE.values, color = 'k')
ax22.scatter(SxS_gamma_graphing_df_b01_r05.gamma.values, SxS_gamma_graphing_df_b01_r05.MaxVNE.values, color = 'y')
ax22.plot(IxS_gamma_graphing_df_b01_r05.gamma.values, IxS_gamma_graphing_df_b01_r05.MaxVNE.values, color = 'r')
ax22.plot(IxI_gamma_graphing_df_b01_r05.gamma.values, IxI_gamma_graphing_df_b01_r05.MaxVNE.values, color = 'g')
ax22.plot(Homerange_IxI_gamma_graphing_df_b01_r05.gamma.values, Homerange_IxI_gamma_graphing_df_b01_r05.MaxVNE.values, color = 'b')
ax22.plot(Tripartite_gamma_graphing_df_b01_r05.gamma.values, Tripartite_gamma_graphing_df_b01_r05.MaxVNE.values, color = 'k')
ax22.plot(SxS_gamma_graphing_df_b01_r05.gamma.values, SxS_gamma_graphing_df_b01_r05.MaxVNE.values, color = 'y')

ax23.grid(True)
ax23.set_xticks(np.linspace(1,10,10, endpoint = True))
ax23.set_yticks(np.linspace(0, 1, 21, endpoint = True))
ax23.scatter(IxS_gamma_graphing_df_b01_r09.gamma.values, IxS_gamma_graphing_df_b01_r09.MaxVNE.values, color = 'r')
ax23.scatter(IxI_gamma_graphing_df_b01_r09.gamma.values, IxI_gamma_graphing_df_b01_r09.MaxVNE.values, color = 'g')
ax23.scatter(Homerange_IxI_gamma_graphing_df_b01_r09.gamma.values, Homerange_IxI_gamma_graphing_df_b01_r09.MaxVNE.values, color = 'b')
ax23.scatter(Tripartite_gamma_graphing_df_b01_r09.gamma.values, Tripartite_gamma_graphing_df_b01_r09.MaxVNE.values, color = 'k')
ax23.scatter(SxS_gamma_graphing_df_b01_r09.gamma.values, SxS_gamma_graphing_df_b01_r09.MaxVNE.values, color = 'y')
ax23.plot(IxS_gamma_graphing_df_b01_r09.gamma.values, IxS_gamma_graphing_df_b01_r09.MaxVNE.values, color = 'r')
ax23.plot(IxI_gamma_graphing_df_b01_r09.gamma.values, IxI_gamma_graphing_df_b01_r09.MaxVNE.values, color = 'g')
ax23.plot(Homerange_IxI_gamma_graphing_df_b01_r09.gamma.values, Homerange_IxI_gamma_graphing_df_b01_r09.MaxVNE.values, color = 'b')
ax23.plot(Tripartite_gamma_graphing_df_b01_r09.gamma.values, Tripartite_gamma_graphing_df_b01_r09.MaxVNE.values, color = 'k')
ax23.plot(SxS_gamma_graphing_df_b01_r09.gamma.values, SxS_gamma_graphing_df_b01_r09.MaxVNE.values, color = 'y')

ax24.grid(True)
ax24.set_xticks(np.linspace(1,10,10, endpoint = True))
ax24.set_yticks(np.linspace(0, 1, 21, endpoint = True))
ax24.scatter(IxS_gamma_graphing_df_b05_r01.gamma.values, IxS_gamma_graphing_df_b05_r01.MaxVNE.values, color = 'r')
ax24.scatter(IxI_gamma_graphing_df_b05_r01.gamma.values, IxI_gamma_graphing_df_b05_r01.MaxVNE.values, color = 'g')
ax24.scatter(Homerange_IxI_gamma_graphing_df_b05_r01.gamma.values, Homerange_IxI_gamma_graphing_df_b05_r01.MaxVNE.values, color = 'b')
ax24.scatter(Tripartite_gamma_graphing_df_b05_r01.gamma.values, Tripartite_gamma_graphing_df_b05_r01.MaxVNE.values, color = 'k')
ax24.scatter(SxS_gamma_graphing_df_b05_r01.gamma.values, SxS_gamma_graphing_df_b05_r01.MaxVNE.values, color = 'y')
ax24.plot(IxS_gamma_graphing_df_b05_r01.gamma.values, IxS_gamma_graphing_df_b05_r01.MaxVNE.values, color = 'r')
ax24.plot(IxI_gamma_graphing_df_b05_r01.gamma.values, IxI_gamma_graphing_df_b05_r01.MaxVNE.values, color = 'g')
ax24.plot(Homerange_IxI_gamma_graphing_df_b05_r01.gamma.values, Homerange_IxI_gamma_graphing_df_b05_r01.MaxVNE.values, color = 'b')
ax24.plot(Tripartite_gamma_graphing_df_b05_r01.gamma.values, Tripartite_gamma_graphing_df_b05_r01.MaxVNE.values, color = 'k')
ax24.plot(SxS_gamma_graphing_df_b05_r01.gamma.values, SxS_gamma_graphing_df_b05_r01.MaxVNE.values, color = 'y')

ax25.grid(True)
ax25.set_xticks(np.linspace(1,10,10, endpoint = True))
ax25.set_yticks(np.linspace(0, 1, 21, endpoint = True))
ax25.scatter(IxS_gamma_graphing_df_b05_r05.gamma.values, IxS_gamma_graphing_df_b05_r05.MaxVNE.values, color = 'r')
ax25.scatter(IxI_gamma_graphing_df_b05_r05.gamma.values, IxI_gamma_graphing_df_b05_r05.MaxVNE.values, color = 'g')
ax25.scatter(Homerange_IxI_gamma_graphing_df_b05_r05.gamma.values, Homerange_IxI_gamma_graphing_df_b05_r05.MaxVNE.values, color = 'b')
ax25.scatter(Tripartite_gamma_graphing_df_b05_r05.gamma.values, Tripartite_gamma_graphing_df_b05_r05.MaxVNE.values, color = 'k')
ax25.scatter(SxS_gamma_graphing_df_b05_r05.gamma.values, SxS_gamma_graphing_df_b05_r05.MaxVNE.values, color = 'y')
ax25.plot(IxS_gamma_graphing_df_b05_r05.gamma.values, IxS_gamma_graphing_df_b05_r05.MaxVNE.values, color = 'r')
ax25.plot(IxI_gamma_graphing_df_b05_r05.gamma.values, IxI_gamma_graphing_df_b05_r05.MaxVNE.values, color = 'g')
ax25.plot(Homerange_IxI_gamma_graphing_df_b05_r05.gamma.values, Homerange_IxI_gamma_graphing_df_b05_r05.MaxVNE.values, color = 'b')
ax25.plot(Tripartite_gamma_graphing_df_b05_r05.gamma.values, Tripartite_gamma_graphing_df_b05_r05.MaxVNE.values, color = 'k')
ax25.plot(SxS_gamma_graphing_df_b05_r05.gamma.values, SxS_gamma_graphing_df_b05_r05.MaxVNE.values, color = 'y')

ax26.grid(True)
ax26.set_xticks(np.linspace(1,10,10, endpoint = True))
ax26.set_yticks(np.linspace(0, 1, 21, endpoint = True))
ax26.scatter(IxS_gamma_graphing_df_b05_r09.gamma.values, IxS_gamma_graphing_df_b05_r09.MaxVNE.values, color = 'r')
ax26.scatter(IxI_gamma_graphing_df_b05_r09.gamma.values, IxI_gamma_graphing_df_b05_r09.MaxVNE.values, color = 'g')
ax26.scatter(Homerange_IxI_gamma_graphing_df_b05_r09.gamma.values, Homerange_IxI_gamma_graphing_df_b05_r09.MaxVNE.values, color = 'b')
ax26.scatter(Tripartite_gamma_graphing_df_b05_r09.gamma.values, Tripartite_gamma_graphing_df_b05_r09.MaxVNE.values, color = 'k')
ax26.scatter(SxS_gamma_graphing_df_b05_r09.gamma.values, SxS_gamma_graphing_df_b05_r09.MaxVNE.values, color = 'y')
ax26.plot(IxS_gamma_graphing_df_b05_r09.gamma.values, IxS_gamma_graphing_df_b05_r09.MaxVNE.values, color = 'r')
ax26.plot(IxI_gamma_graphing_df_b05_r09.gamma.values, IxI_gamma_graphing_df_b05_r09.MaxVNE.values, color = 'g')
ax26.plot(Homerange_IxI_gamma_graphing_df_b05_r09.gamma.values, Homerange_IxI_gamma_graphing_df_b05_r09.MaxVNE.values, color = 'b')
ax26.plot(Tripartite_gamma_graphing_df_b05_r09.gamma.values, Tripartite_gamma_graphing_df_b05_r09.MaxVNE.values, color = 'k')
ax26.plot(SxS_gamma_graphing_df_b05_r09.gamma.values, SxS_gamma_graphing_df_b05_r09.MaxVNE.values, color = 'y')

ax27.grid(True)
ax27.set_xticks(np.linspace(1,10,10, endpoint = True))
ax27.set_yticks(np.linspace(0, 1, 21, endpoint = True))
ax27.scatter(IxS_gamma_graphing_df_b09_r01.gamma.values, IxS_gamma_graphing_df_b09_r01.MaxVNE.values, color = 'r')
ax27.scatter(IxI_gamma_graphing_df_b09_r01.gamma.values, IxI_gamma_graphing_df_b09_r01.MaxVNE.values, color = 'g')
ax27.scatter(Homerange_IxI_gamma_graphing_df_b09_r01.gamma.values, Homerange_IxI_gamma_graphing_df_b09_r01.MaxVNE.values, color = 'b')
ax27.scatter(Tripartite_gamma_graphing_df_b09_r01.gamma.values, Tripartite_gamma_graphing_df_b09_r01.MaxVNE.values, color = 'k')
ax27.scatter(SxS_gamma_graphing_df_b09_r01.gamma.values, SxS_gamma_graphing_df_b09_r01.MaxVNE.values, color = 'y')
ax27.plot(IxS_gamma_graphing_df_b09_r01.gamma.values, IxS_gamma_graphing_df_b09_r01.MaxVNE.values, color = 'r')
ax27.plot(IxI_gamma_graphing_df_b09_r01.gamma.values, IxI_gamma_graphing_df_b09_r01.MaxVNE.values, color = 'g')
ax27.plot(Homerange_IxI_gamma_graphing_df_b09_r01.gamma.values, Homerange_IxI_gamma_graphing_df_b09_r01.MaxVNE.values, color = 'b')
ax27.plot(Tripartite_gamma_graphing_df_b09_r01.gamma.values, Tripartite_gamma_graphing_df_b09_r01.MaxVNE.values, color = 'k')
ax27.plot(SxS_gamma_graphing_df_b09_r01.gamma.values, SxS_gamma_graphing_df_b09_r01.MaxVNE.values, color = 'y')

ax28.grid(True)
ax28.set_xticks(np.linspace(1,10,10, endpoint = True))
ax28.set_yticks(np.linspace(0, 1, 21, endpoint = True))
ax28.scatter(IxS_gamma_graphing_df_b09_r05.gamma.values, IxS_gamma_graphing_df_b09_r05.MaxVNE.values, color = 'r')
ax28.scatter(IxI_gamma_graphing_df_b09_r05.gamma.values, IxI_gamma_graphing_df_b09_r05.MaxVNE.values, color = 'g')
ax28.scatter(Homerange_IxI_gamma_graphing_df_b09_r05.gamma.values, Homerange_IxI_gamma_graphing_df_b09_r05.MaxVNE.values, color = 'b')
ax28.scatter(Tripartite_gamma_graphing_df_b09_r05.gamma.values, Tripartite_gamma_graphing_df_b09_r05.MaxVNE.values, color = 'k')
ax28.scatter(SxS_gamma_graphing_df_b09_r05.gamma.values, SxS_gamma_graphing_df_b09_r05.MaxVNE.values, color = 'y')
ax28.plot(IxS_gamma_graphing_df_b09_r05.gamma.values, IxS_gamma_graphing_df_b09_r05.MaxVNE.values, color = 'r')
ax28.plot(IxI_gamma_graphing_df_b09_r05.gamma.values, IxI_gamma_graphing_df_b09_r05.MaxVNE.values, color = 'g')
ax28.plot(Homerange_IxI_gamma_graphing_df_b09_r05.gamma.values, Homerange_IxI_gamma_graphing_df_b09_r05.MaxVNE.values, color = 'b')
ax28.plot(Tripartite_gamma_graphing_df_b09_r05.gamma.values, Tripartite_gamma_graphing_df_b09_r05.MaxVNE.values, color = 'k')
ax28.plot(SxS_gamma_graphing_df_b09_r05.gamma.values, SxS_gamma_graphing_df_b09_r05.MaxVNE.values, color = 'y')

ax29.grid(True)
ax29.set_xticks(np.linspace(1,10,10, endpoint = True))
ax29.set_yticks(np.linspace(0, 1, 21, endpoint = True))
ax29.scatter(IxS_gamma_graphing_df_b09_r09.gamma.values, IxS_gamma_graphing_df_b09_r09.MaxVNE.values, color = 'r')
ax29.scatter(IxI_gamma_graphing_df_b09_r09.gamma.values, IxI_gamma_graphing_df_b09_r09.MaxVNE.values, color = 'g')
ax29.scatter(Homerange_IxI_gamma_graphing_df_b09_r09.gamma.values, Homerange_IxI_gamma_graphing_df_b09_r09.MaxVNE.values, color = 'b')
ax29.scatter(Tripartite_gamma_graphing_df_b09_r09.gamma.values, Tripartite_gamma_graphing_df_b09_r09.MaxVNE.values, color = 'k')
ax29.scatter(SxS_gamma_graphing_df_b09_r09.gamma.values, SxS_gamma_graphing_df_b09_r09.MaxVNE.values, color = 'y')
ax29.plot(IxS_gamma_graphing_df_b09_r09.gamma.values, IxS_gamma_graphing_df_b09_r09.MaxVNE.values, color = 'r')
ax29.plot(IxI_gamma_graphing_df_b09_r09.gamma.values, IxI_gamma_graphing_df_b09_r09.MaxVNE.values, color = 'g')
ax29.plot(Homerange_IxI_gamma_graphing_df_b09_r09.gamma.values, Homerange_IxI_gamma_graphing_df_b09_r09.MaxVNE.values, color = 'b')
ax29.plot(Tripartite_gamma_graphing_df_b09_r09.gamma.values, Tripartite_gamma_graphing_df_b09_r09.MaxVNE.values, color = 'k')
ax29.plot(SxS_gamma_graphing_df_b09_r09.gamma.values, SxS_gamma_graphing_df_b09_r09.MaxVNE.values, color = 'y')

plt.show()

##################################################################################################
# This is a 3 x 3 graph where beta is varied when gamma, rho = 1, 5, and 9.
##################################################################################################
fig, ((ax31, ax32, ax33), (ax34, ax35, ax36), (ax37, ax38, ax39)) = plt.subplots(nrows = 3, ncols = 3)

ax31.grid(True)
ax31.set_xticks(np.linspace(1,10,10, endpoint = True))
ax31.set_yticks(np.linspace(0, 1, 21, endpoint = True))
ax31.scatter(IxS_beta_graphing_df_g01_r01.beta.values, IxS_beta_graphing_df_g01_r01.MaxVNE.values, color = 'r')
ax31.scatter(IxI_beta_graphing_df_g01_r01.beta.values, IxI_beta_graphing_df_g01_r01.MaxVNE.values, color = 'g')
ax31.scatter(Homerange_IxI_beta_graphing_df_g01_r01.beta.values, Homerange_IxI_beta_graphing_df_g01_r01.MaxVNE.values, color = 'b')
ax31.scatter(Tripartite_beta_graphing_df_g01_r01.beta.values, Tripartite_beta_graphing_df_g01_r01.MaxVNE.values, color = 'k')
ax31.scatter(SxS_beta_graphing_df_g01_r01.beta.values, SxS_beta_graphing_df_g01_r01.MaxVNE.values, color = 'y')
ax31.plot(IxS_beta_graphing_df_g01_r01.beta.values, IxS_beta_graphing_df_g01_r01.MaxVNE.values, color = 'r')
ax31.plot(IxI_beta_graphing_df_g01_r01.beta.values, IxI_beta_graphing_df_g01_r01.MaxVNE.values, color = 'g')
ax31.plot(Homerange_IxI_beta_graphing_df_g01_r01.beta.values, Homerange_IxI_beta_graphing_df_g01_r01.MaxVNE.values, color = 'b')
ax31.plot(Tripartite_beta_graphing_df_g01_r01.beta.values, Tripartite_beta_graphing_df_g01_r01.MaxVNE.values, color = 'k')
ax31.plot(SxS_beta_graphing_df_g01_r01.beta.values, SxS_beta_graphing_df_g01_r01.MaxVNE.values, color = 'y')

ax32.grid(True)
ax32.set_xticks(np.linspace(1,10,10, endpoint = True))
ax32.set_yticks(np.linspace(0, 1, 21, endpoint = True))
ax32.scatter(IxS_beta_graphing_df_g01_r05.beta.values, IxS_beta_graphing_df_g01_r05.MaxVNE.values, color = 'r')
ax32.scatter(IxI_beta_graphing_df_g01_r05.beta.values, IxI_beta_graphing_df_g01_r05.MaxVNE.values, color = 'g')
ax32.scatter(Homerange_IxI_beta_graphing_df_g01_r05.beta.values, Homerange_IxI_beta_graphing_df_g01_r05.MaxVNE.values, color = 'b')
ax32.scatter(Tripartite_beta_graphing_df_g01_r05.beta.values, Tripartite_beta_graphing_df_g01_r05.MaxVNE.values, color = 'k')
ax32.scatter(SxS_beta_graphing_df_g01_r05.beta.values, SxS_beta_graphing_df_g01_r05.MaxVNE.values, color = 'y')
ax32.plot(IxS_beta_graphing_df_g01_r05.beta.values, IxS_beta_graphing_df_g01_r05.MaxVNE.values, color = 'r')
ax32.plot(IxI_beta_graphing_df_g01_r05.beta.values, IxI_beta_graphing_df_g01_r05.MaxVNE.values, color = 'g')
ax32.plot(Homerange_IxI_beta_graphing_df_g01_r05.beta.values, Homerange_IxI_beta_graphing_df_g01_r05.MaxVNE.values, color = 'b')
ax32.plot(Tripartite_beta_graphing_df_g01_r05.beta.values, Tripartite_beta_graphing_df_g01_r05.MaxVNE.values, color = 'k')
ax32.plot(SxS_beta_graphing_df_g01_r05.beta.values, SxS_beta_graphing_df_g01_r05.MaxVNE.values, color = 'y')

ax33.grid(True)
ax33.set_xticks(np.linspace(1,10,10, endpoint = True))
ax33.set_yticks(np.linspace(0, 1, 21, endpoint = True))
ax33.scatter(IxS_beta_graphing_df_g01_r09.beta.values, IxS_beta_graphing_df_g01_r09.MaxVNE.values, color = 'r')
ax33.scatter(IxI_beta_graphing_df_g01_r09.beta.values, IxI_beta_graphing_df_g01_r09.MaxVNE.values, color = 'g')
ax33.scatter(Homerange_IxI_beta_graphing_df_g01_r09.beta.values, Homerange_IxI_beta_graphing_df_g01_r09.MaxVNE.values, color = 'b')
ax33.scatter(Tripartite_beta_graphing_df_g01_r09.beta.values, Tripartite_beta_graphing_df_g01_r09.MaxVNE.values, color = 'k')
ax33.scatter(SxS_beta_graphing_df_g01_r09.beta.values, SxS_beta_graphing_df_g01_r09.MaxVNE.values, color = 'y')
ax33.plot(IxS_beta_graphing_df_g01_r09.beta.values, IxS_beta_graphing_df_g01_r09.MaxVNE.values, color = 'r')
ax33.plot(IxI_beta_graphing_df_g01_r09.beta.values, IxI_beta_graphing_df_g01_r09.MaxVNE.values, color = 'g')
ax33.plot(Homerange_IxI_beta_graphing_df_g01_r09.beta.values, Homerange_IxI_beta_graphing_df_g01_r09.MaxVNE.values, color = 'b')
ax33.plot(Tripartite_beta_graphing_df_g01_r09.beta.values, Tripartite_beta_graphing_df_g01_r09.MaxVNE.values, color = 'k')
ax33.plot(SxS_beta_graphing_df_g01_r09.beta.values, SxS_beta_graphing_df_g01_r09.MaxVNE.values, color = 'y')

ax34.grid(True)
ax34.set_xticks(np.linspace(1,10,10, endpoint = True))
ax34.set_yticks(np.linspace(0, 1, 21, endpoint = True))
ax34.scatter(IxS_beta_graphing_df_g05_r01.beta.values, IxS_beta_graphing_df_g05_r01.MaxVNE.values, color = 'r')
ax34.scatter(IxI_beta_graphing_df_g05_r01.beta.values, IxI_beta_graphing_df_g05_r01.MaxVNE.values, color = 'g')
ax34.scatter(Homerange_IxI_beta_graphing_df_g05_r01.beta.values, Homerange_IxI_beta_graphing_df_g05_r01.MaxVNE.values, color = 'b')
ax34.scatter(Tripartite_beta_graphing_df_g05_r01.beta.values, Tripartite_beta_graphing_df_g05_r01.MaxVNE.values, color = 'k')
ax34.scatter(SxS_beta_graphing_df_g05_r01.beta.values, SxS_beta_graphing_df_g05_r01.MaxVNE.values, color = 'y')
ax34.plot(IxS_beta_graphing_df_g05_r01.beta.values, IxS_beta_graphing_df_g05_r01.MaxVNE.values, color = 'r')
ax34.plot(IxI_beta_graphing_df_g05_r01.beta.values, IxI_beta_graphing_df_g05_r01.MaxVNE.values, color = 'g')
ax34.plot(Homerange_IxI_beta_graphing_df_g05_r01.beta.values, Homerange_IxI_beta_graphing_df_g05_r01.MaxVNE.values, color = 'b')
ax34.plot(Tripartite_beta_graphing_df_g05_r01.beta.values, Tripartite_beta_graphing_df_g05_r01.MaxVNE.values, color = 'k')
ax34.plot(SxS_beta_graphing_df_g05_r01.beta.values, SxS_beta_graphing_df_g05_r01.MaxVNE.values, color = 'y')

ax35.grid(True)
ax35.set_xticks(np.linspace(1,10,10, endpoint = True))
ax35.set_yticks(np.linspace(0, 1, 21, endpoint = True))
ax35.scatter(IxS_beta_graphing_df_g05_r05.beta.values, IxS_beta_graphing_df_g05_r05.MaxVNE.values, color = 'r')
ax35.scatter(IxI_beta_graphing_df_g05_r05.beta.values, IxI_beta_graphing_df_g05_r05.MaxVNE.values, color = 'g')
ax35.scatter(Homerange_IxI_beta_graphing_df_g05_r05.beta.values, Homerange_IxI_beta_graphing_df_g05_r05.MaxVNE.values, color = 'b')
ax35.scatter(Tripartite_beta_graphing_df_g05_r05.beta.values, Tripartite_beta_graphing_df_g05_r05.MaxVNE.values, color = 'k')
ax35.scatter(SxS_beta_graphing_df_g05_r05.beta.values, SxS_beta_graphing_df_g05_r05.MaxVNE.values, color = 'y')
ax35.plot(IxS_beta_graphing_df_g05_r05.beta.values, IxS_beta_graphing_df_g05_r05.MaxVNE.values, color = 'r')
ax35.plot(IxI_beta_graphing_df_g05_r05.beta.values, IxI_beta_graphing_df_g05_r05.MaxVNE.values, color = 'g')
ax35.plot(Homerange_IxI_beta_graphing_df_g05_r05.beta.values, Homerange_IxI_beta_graphing_df_g05_r05.MaxVNE.values, color = 'b')
ax35.plot(Tripartite_beta_graphing_df_g05_r05.beta.values, Tripartite_beta_graphing_df_g05_r05.MaxVNE.values, color = 'k')
ax35.plot(SxS_beta_graphing_df_g05_r05.beta.values, SxS_beta_graphing_df_g05_r05.MaxVNE.values, color = 'y')

ax36.grid(True)
ax36.set_xticks(np.linspace(1,10,10, endpoint = True))
ax36.set_yticks(np.linspace(0, 1, 21, endpoint = True))
ax36.scatter(IxS_beta_graphing_df_g05_r09.beta.values, IxS_beta_graphing_df_g05_r09.MaxVNE.values, color = 'r')
ax36.scatter(IxI_beta_graphing_df_g05_r09.beta.values, IxI_beta_graphing_df_g05_r09.MaxVNE.values, color = 'g')
ax36.scatter(Homerange_IxI_beta_graphing_df_g05_r09.beta.values, Homerange_IxI_beta_graphing_df_g05_r09.MaxVNE.values, color = 'b')
ax36.scatter(Tripartite_beta_graphing_df_g05_r09.beta.values, Tripartite_beta_graphing_df_g05_r09.MaxVNE.values, color = 'k')
ax36.scatter(SxS_beta_graphing_df_g05_r09.beta.values, SxS_beta_graphing_df_g05_r09.MaxVNE.values, color = 'y')
ax36.plot(IxS_beta_graphing_df_g05_r09.beta.values, IxS_beta_graphing_df_g05_r09.MaxVNE.values, color = 'r')
ax36.plot(IxI_beta_graphing_df_g05_r09.beta.values, IxI_beta_graphing_df_g05_r09.MaxVNE.values, color = 'g')
ax36.plot(Homerange_IxI_beta_graphing_df_g05_r09.beta.values, Homerange_IxI_beta_graphing_df_g05_r09.MaxVNE.values, color = 'b')
ax33.plot(Tripartite_beta_graphing_df_g05_r09.beta.values, Tripartite_beta_graphing_df_g05_r09.MaxVNE.values, color = 'k')
ax36.plot(SxS_beta_graphing_df_g05_r09.beta.values, SxS_beta_graphing_df_g05_r09.MaxVNE.values, color = 'y')

ax37.grid(True)
ax37.set_xticks(np.linspace(1,10,10, endpoint = True))
ax37.set_yticks(np.linspace(0, 1, 21, endpoint = True))
ax37.scatter(IxS_beta_graphing_df_g09_r01.beta.values, IxS_beta_graphing_df_g09_r01.MaxVNE.values, color = 'r')
ax37.scatter(IxI_beta_graphing_df_g09_r01.beta.values, IxI_beta_graphing_df_g09_r01.MaxVNE.values, color = 'g')
ax37.scatter(Homerange_IxI_beta_graphing_df_g09_r01.beta.values, Homerange_IxI_beta_graphing_df_g09_r01.MaxVNE.values, color = 'b')
ax37.scatter(Tripartite_beta_graphing_df_g09_r01.beta.values, Tripartite_beta_graphing_df_g09_r01.MaxVNE.values, color = 'k')
ax37.scatter(SxS_beta_graphing_df_g09_r01.beta.values, SxS_beta_graphing_df_g09_r01.MaxVNE.values, color = 'y')
ax37.plot(IxS_beta_graphing_df_g09_r01.beta.values, IxS_beta_graphing_df_g09_r01.MaxVNE.values, color = 'r')
ax37.plot(IxI_beta_graphing_df_g09_r01.beta.values, IxI_beta_graphing_df_g09_r01.MaxVNE.values, color = 'g')
ax37.plot(Homerange_IxI_beta_graphing_df_g09_r01.beta.values, Homerange_IxI_beta_graphing_df_g09_r01.MaxVNE.values, color = 'b')
ax37.plot(Tripartite_beta_graphing_df_g09_r01.beta.values, Tripartite_beta_graphing_df_g09_r01.MaxVNE.values, color = 'k')
ax37.plot(SxS_beta_graphing_df_g09_r01.beta.values, SxS_beta_graphing_df_g09_r01.MaxVNE.values, color = 'y')

ax38.grid(True)
ax38.set_xticks(np.linspace(1,10,10, endpoint = True))
ax38.set_yticks(np.linspace(0, 1, 21, endpoint = True))
ax38.scatter(IxS_beta_graphing_df_g09_r05.beta.values, IxS_beta_graphing_df_g09_r05.MaxVNE.values, color = 'r')
ax38.scatter(IxI_beta_graphing_df_g09_r05.beta.values, IxI_beta_graphing_df_g09_r05.MaxVNE.values, color = 'g')
ax38.scatter(Homerange_IxI_beta_graphing_df_g09_r05.beta.values, Homerange_IxI_beta_graphing_df_g09_r05.MaxVNE.values, color = 'b')
ax38.scatter(Tripartite_beta_graphing_df_g09_r05.beta.values, Tripartite_beta_graphing_df_g09_r05.MaxVNE.values, color = 'k')
ax38.scatter(SxS_beta_graphing_df_g09_r05.beta.values, SxS_beta_graphing_df_g09_r05.MaxVNE.values, color = 'y')
ax38.plot(IxS_beta_graphing_df_g09_r05.beta.values, IxS_beta_graphing_df_g09_r05.MaxVNE.values, color = 'r')
ax38.plot(IxI_beta_graphing_df_g09_r05.beta.values, IxI_beta_graphing_df_g09_r05.MaxVNE.values, color = 'g')
ax38.plot(Homerange_IxI_beta_graphing_df_g09_r05.beta.values, Homerange_IxI_beta_graphing_df_g09_r05.MaxVNE.values, color = 'b')
ax38.plot(Tripartite_beta_graphing_df_g09_r05.beta.values, Tripartite_beta_graphing_df_g09_r05.MaxVNE.values, color = 'k')
ax38.plot(SxS_beta_graphing_df_g09_r05.beta.values, SxS_beta_graphing_df_g09_r05.MaxVNE.values, color = 'y')

ax39.grid(True)
ax39.set_xticks(np.linspace(1,10,10, endpoint = True))
ax39.set_yticks(np.linspace(0, 1, 21, endpoint = True))
ax39.scatter(IxS_beta_graphing_df_g09_r09.beta.values, IxS_beta_graphing_df_g09_r09.MaxVNE.values, color = 'r')
ax39.scatter(IxI_beta_graphing_df_g09_r09.beta.values, IxI_beta_graphing_df_g09_r09.MaxVNE.values, color = 'g')
ax39.scatter(Homerange_IxI_beta_graphing_df_g09_r09.beta.values, Homerange_IxI_beta_graphing_df_g09_r09.MaxVNE.values, color = 'b')
ax39.scatter(Tripartite_beta_graphing_df_g09_r09.beta.values, Tripartite_beta_graphing_df_g09_r09.MaxVNE.values, color = 'k')
ax39.scatter(SxS_beta_graphing_df_g09_r09.beta.values, SxS_beta_graphing_df_g09_r09.MaxVNE.values, color = 'y')
ax39.plot(IxS_beta_graphing_df_g09_r09.beta.values, IxS_beta_graphing_df_g09_r09.MaxVNE.values, color = 'r')
ax39.plot(IxI_beta_graphing_df_g09_r09.beta.values, IxI_beta_graphing_df_g09_r09.MaxVNE.values, color = 'g')
ax39.plot(Homerange_IxI_beta_graphing_df_g09_r09.beta.values, Homerange_IxI_beta_graphing_df_g09_r09.MaxVNE.values, color = 'b')
ax39.plot(Tripartite_beta_graphing_df_g09_r09.beta.values, Tripartite_beta_graphing_df_g09_r09.MaxVNE.values, color = 'k')
ax39.plot(SxS_beta_graphing_df_g09_r09.beta.values, SxS_beta_graphing_df_g09_r09.MaxVNE.values, color = 'y')

plt.show()
