import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression as LR
from numpy.polynomial.polynomial import polyfit as pfit

location_amount = 10
individual_amount = 10
time_stamps = 20
gamma_values = np.linspace(1, 10, 10, endpoint = True)
beta_values = np.linspace(1, 10, 10, endpoint = True)
rho_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 0.99, 0.999, 0.9999, 1.0]#np.linspace(0.1, 1.0, 10, endpoint=True)
simulation_amount = 10

# Loading both the 
VNE_outputs_IxS = np.load(r'C:\Users\tjbro\Desktop\Correct_10_By_10_arrays\VNE\IxS10By10VNE.npy')
VNE_outputs_IxI = np.load(r'C:\Users\tjbro\Desktop\Correct_10_By_10_arrays\VNE\IxI10By10VNE.npy')
VNE_outputs_Tripartite = np.load(r'C:\Users\tjbro\Desktop\Correct_10_By_10_arrays\VNE\Tripartite10By10VNE.npy')
VNE_outputs_Homerange_IxI = np.load(r'C:\Users\tjbro\Desktop\Correct_10_By_10_arrays\VNE\Homerange10By10VNE.npy')
VNE_outputs_SxS = np.load(r'C:\Users\tjbro\Desktop\Correct_10_By_10_arrays\VNE\SxS10By10VNE.npy')

IxS_v_outputs = np.load(r'C:\Users\tjbro\Desktop\Correct_10_By_10_arrays\Lineweaver-Burk\IxSLineweaver10By10.npy')
IxI_v_outputs = np.load(r'C:\Users\tjbro\Desktop\Correct_10_By_10_arrays\Lineweaver-Burk\IxILineweaver10By10.npy')
Tripartite_v_outputs = np.load(r'C:\Users\tjbro\Desktop\Correct_10_By_10_arrays\Lineweaver-Burk\TripartiteLineweaver10By10.npy')
Homerange_IxI_v_outputs = np.load(r'C:\Users\tjbro\Desktop\Correct_10_By_10_arrays\Lineweaver-Burk\HomerangeIXILineweaver10By10.npy')
SxS_v_outputs = np.load(r'C:\Users\tjbro\Desktop\Correct_10_By_10_arrays\Lineweaver-Burk\SxSLineweaver10By10.npy')

####################################################################################################
# Let's do our best to create a pandas dataframe and try to go from there
####################################################################################################
# The columns for my data fram are gamma, beta, rho, timesteps, and the v vector elements
# associated with each timestep and then the simulation number
# Let's first try getting everything in order of simulations
column_list_VNE = ['gamma', 'beta', 'rho', 'timeStep', 'outputValues']
column_list_v_outputs = ['gamma', 'beta', 'rho', 'michaelisInputs', 'outputValues']

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
            #print('gamma = ', i, ' beta = ', j , ' rho = ', k, '\n', rho_SxS_df.outputValues.values)
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
            
# Saving the files
IxS_VNE_df.to_csv('IxSVNE10By50.csv')
IxI_VNE_df.to_csv('IxIVNE10By50.csv')
Homerange_IxI_VNE_df.to_csv('HomerangeVNE10By50.csv')
SxS_VNE_df.to_csv('SxSVNE10By50.csv')
Tripartite_VNE_df.to_csv('TripartiteVNE10By50.csv')

IxS_v_outputs_df.to_csv('IxSLineweaver10By50.csv')
IxI_v_outputs_df.to_csv('IxILineweaver10By50.csv')
Homerange_IxI_v_outputs_df.to_csv('HomerangeLineweaver10By50.csv')
SxS_v_outputs_df.to_csv('SxSLineweaver10By50.csv')
Tripartite_v_outputs_df.to_csv('TripartiteLineweaver10By50.csv')

IxS_Max_VNE_df.to_csv('IxSMaxVNE10By50.csv')
IxI_Max_VNE_df.to_csv('IxIMaxVNE10By50.csv')
Homerange_IxI_Max_VNE_df.to_csv('HomerangeMaxVNE10By50.csv')
Tripartite_Max_VNE_df.to_csv('TripartiteMaxVNE10By50.csv')
SxS_Max_VNE_df.to_csv('SxSMaxVNE10By50.csv')


