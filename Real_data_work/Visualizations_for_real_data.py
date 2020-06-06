import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LinearRegression as LR
from numpy.polynomial.polynomial import polyfit as pfit
import pandas as pd
import seaborn as sns

#################################################################################################################################
# Call the VNE and transformed VNE files for spatial density 03 here
#################################################################################################################################
VNE_outputs_IxS3 = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Real_data_work\03Data\VNEValues\IxSVNERealData03.npy')
VNE_outputs_IxI3 = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Real_data_work\03Data\VNEValues\IxIVNERealData03.npy')
VNE_outputs_Tripartite3 = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Real_data_work\03Data\VNEValues\tripartiteVNERealData03.npy')
VNE_outputs_Homerange_IxI3 = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Real_data_work\03Data\VNEValues\homerangeVNERealData03.npy')
VNE_outputs_SxS3 = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Real_data_work\03Data\VNEValues\SxSVNERealData03.npy')

IxS_v_outputs3 = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Real_data_work\03Data\LWBOutputs\realDataIxSOutputs.npy')
IxI_v_outputs3 = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Real_data_work\03Data\LWBOutputs\realDataIxIOutputs.npy')
Tripartite_v_outputs3 = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Real_data_work\03Data\LWBOutputs\realDataTripartiteOutputs.npy')
Homerange_IxI_v_outputs3 = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Real_data_work\03Data\LWBOutputs\realDataHomerangeOutputs.npy')
SxS_v_outputs3 = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Real_data_work\03Data\LWBOutputs\realDataSxSOutputs.npy')
michaelisInputs3 = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Real_data_work\03Data\LWBOutputs\realDataMichaelisInputs.npy')

#################################################################################################################################
# Call the VNE and transformed VNE files for spatial density 05 here
#################################################################################################################################
VNE_outputs_IxS5 = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Real_data_work\05Data\VNEValues\IxSVNERealData05.npy')
VNE_outputs_IxI5 = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Real_data_work\05Data\VNEValues\IxIVNERealData05.npy')
VNE_outputs_Tripartite5 = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Real_data_work\05Data\VNEValues\tripartiteVNERealData05.npy')
VNE_outputs_Homerange_IxI5 = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Real_data_work\05Data\VNEValues\homerangeVNERealData05.npy')
VNE_outputs_SxS5 = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Real_data_work\05Data\VNEValues\SxSVNERealData05.npy')

IxS_v_outputs5 = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Real_data_work\05Data\LWBOutputs\realDataIxSOutputs.npy')
IxI_v_outputs5 = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Real_data_work\05Data\LWBOutputs\realDataIxIOutputs.npy')
Tripartite_v_outputs5 = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Real_data_work\05Data\LWBOutputs\realDataTripartiteOutputs.npy')
Homerange_IxI_v_outputs5 = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Real_data_work\05Data\LWBOutputs\realDataHomerangeOutputs.npy')
SxS_v_outputs5 = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Real_data_work\05Data\LWBOutputs\realDataSxSOutputs.npy')
michaelisInputs5 = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Real_data_work\05Data\LWBOutputs\realDataMichaelisInputs.npy')

#################################################################################################################################
# Call the VNE and transformed VNE files for spatial density 07 here
#################################################################################################################################
VNE_outputs_IxS7 = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Real_data_work\07Data\VNEValues\IxSVNERealData07.npy')
VNE_outputs_IxI7 = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Real_data_work\07Data\VNEValues\IxIVNERealData07.npy')
VNE_outputs_Tripartite7 = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Real_data_work\07Data\VNEValues\tripartiteVNERealData07.npy')
VNE_outputs_Homerange_IxI7 = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Real_data_work\07Data\VNEValues\homerangeVNERealData07.npy')
VNE_outputs_SxS7 = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Real_data_work\07Data\VNEValues\SxSVNERealData07.npy')

IxS_v_outputs7 = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Real_data_work\07Data\LWBOutputs\realDataIxSOutputs.npy')
IxI_v_outputs7 = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Real_data_work\07Data\LWBOutputs\realDataIxIOutputs.npy')
Tripartite_v_outputs7 = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Real_data_work\07Data\LWBOutputs\realDataTripartiteOutputs.npy')
Homerange_IxI_v_outputs7 = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Real_data_work\07Data\LWBOutputs\realDataHomerangeOutputs.npy')
SxS_v_outputs7 = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Real_data_work\07Data\LWBOutputs\realDataSxSOutputs.npy')
michaelisInputs7 = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Real_data_work\07Data\LWBOutputs\realDataMichaelisInputs.npy')

#################################################################################################################################
# Call the VNE and transformed VNE files for spatial density 09 here
#################################################################################################################################
VNE_outputs_IxS9 = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Real_data_work\09Data\VNEValues\IxSVNERealData09.npy')
VNE_outputs_IxI9 = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Real_data_work\09Data\VNEValues\IxIVNERealData09.npy')
VNE_outputs_Tripartite9 = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Real_data_work\09Data\VNEValues\tripartiteVNERealData09.npy')
VNE_outputs_Homerange_IxI9 = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Real_data_work\09Data\VNEValues\homerangeVNERealData09.npy')
VNE_outputs_SxS9 = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Real_data_work\09Data\VNEValues\SxSVNERealData09.npy')

IxS_v_outputs9 = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Real_data_work\09Data\LWBOutputs\realDataIxSOutputs.npy')
IxI_v_outputs9 = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Real_data_work\09Data\LWBOutputs\realDataIxIOutputs.npy')
Tripartite_v_outputs9 = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Real_data_work\09Data\LWBOutputs\realDataTripartiteOutputs.npy')
Homerange_IxI_v_outputs9 = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Real_data_work\09Data\LWBOutputs\realDataHomerangeOutputs.npy')
SxS_v_outputs9 = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Real_data_work\09Data\LWBOutputs\realDataSxSOutputs.npy')
michaelisInputs9 = np.load(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Real_data_work\09Data\LWBOutputs\realDataMichaelisInputs.npy')

#####################################################################################
# This is the size of the text for the graphs
#####################################################################################
SMALL_SIZE = 8
MEDIUM_SIZE = 16
BIGGER_SIZE = 24
BIGGEST_SIZE = 32
Big_font_size = 42

plt.rc('font', size=BIGGER_SIZE)         # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)   # fontsize of the tired_patchs
plt.rc('legend', fontsize=BIGGER_SIZE)   # legend fontsize
plt.rc('figure', titlesize=SMALL_SIZE)   # fontsize of the figure title

#####################################################################################
# Defining inputs and color scheme for visualizations
#####################################################################################
red_patch = mpatches.Patch(color='r', label='3 data; small spatial variance')
green_patch = mpatches.Patch(color = 'g', label = '5 data; medium spatial variance')
black_patch = mpatches.Patch(color='k', label='7 data; large spatial variance')
blue_patch = mpatches.Patch(color='b', label='9 data; huge spatial variance')

inputList3 = np.arange(len(VNE_outputs_IxS3)) + 1
inputList5 = np.arange(len(VNE_outputs_IxS5)) + 1
inputList7 = np.arange(len(VNE_outputs_IxS7)) + 1
inputList9 = np.arange(len(VNE_outputs_IxS9)) + 1

######################################################################################
## This is the graph of the VNE for each projection
#####################################################################################
plt.figure()
plt.grid(True)
plt.xlabel('Time Step', fontsize = Big_font_size)
plt.ylabel('VNE value', fontsize = Big_font_size)
plt.legend(handles = [red_patch, green_patch, black_patch, blue_patch], title = 'Data set')
plt.xticks(fontsize = 30)
plt.yticks(np.linspace(0, 1, 21, endpoint = True))
plt.scatter(inputList3, VNE_outputs_IxS3, color = 'r')
plt.plot(inputList3, VNE_outputs_IxS3, color = 'r')
plt.scatter(inputList5, VNE_outputs_IxS5, color = 'g')
plt.plot(inputList5, VNE_outputs_IxS5, color = 'g')
plt.scatter(inputList7, VNE_outputs_IxS7, color = 'k')
plt.plot(inputList7, VNE_outputs_IxS7, color = 'k')
plt.scatter(inputList9, VNE_outputs_IxS9, color = 'b')
plt.plot(inputList9, VNE_outputs_IxS9, color = 'b')

plt.show()

plt.figure()
plt.grid(True)
plt.xlabel('Time Step', fontsize = Big_font_size)
plt.ylabel('VNE value', fontsize = Big_font_size)
plt.legend(handles = [red_patch, green_patch, black_patch, blue_patch], title = 'Data set')
plt.xticks(fontsize = 30)
plt.yticks(np.linspace(0, 1, 21, endpoint = True))
plt.scatter(inputList3, VNE_outputs_IxI3, color = 'r')
plt.plot(inputList3, VNE_outputs_IxI3, color = 'r')
plt.scatter(inputList5, VNE_outputs_IxI5, color = 'g')
plt.plot(inputList5, VNE_outputs_IxI5, color = 'g')
plt.scatter(inputList7, VNE_outputs_IxI7, color = 'k')
plt.plot(inputList7, VNE_outputs_IxI7, color = 'k')
plt.scatter(inputList9, VNE_outputs_IxI9, color = 'b')
plt.plot(inputList9, VNE_outputs_IxI9, color = 'b')
plt.show()

plt.figure()
plt.grid(True)
plt.xlabel('Time Step', fontsize = Big_font_size)
plt.ylabel('VNE value', fontsize = Big_font_size)
plt.legend(handles = [red_patch, green_patch, black_patch, blue_patch], title = 'Data set')
plt.xticks(fontsize = 30)
plt.yticks(np.linspace(0, 1, 21, endpoint = True))
plt.scatter(inputList3, VNE_outputs_Tripartite3, color = 'r')
plt.plot(inputList3, VNE_outputs_Tripartite3, color = 'r')
plt.scatter(inputList5, VNE_outputs_Tripartite5, color = 'g')
plt.plot(inputList5, VNE_outputs_Tripartite5, color = 'g')
plt.scatter(inputList7, VNE_outputs_Tripartite7, color = 'k')
plt.plot(inputList7, VNE_outputs_Tripartite7, color = 'k')
plt.scatter(inputList9, VNE_outputs_Tripartite9, color = 'b')
plt.plot(inputList9, VNE_outputs_Tripartite9, color = 'b')
plt.show()

plt.figure()
plt.grid(True)
plt.xlabel('Time Step', fontsize = Big_font_size)
plt.ylabel('VNE value', fontsize = Big_font_size)
plt.legend(handles = [red_patch, green_patch, black_patch, blue_patch], title = 'Data set')
plt.xticks(fontsize = 30)
plt.yticks(np.linspace(0, 1, 21, endpoint = True))
plt.scatter(inputList3, VNE_outputs_Homerange_IxI3, color = 'r')
plt.plot(inputList3, VNE_outputs_Homerange_IxI3, color = 'r')
plt.scatter(inputList5, VNE_outputs_Homerange_IxI5, color = 'g')
plt.plot(inputList5, VNE_outputs_Homerange_IxI5, color = 'g')
plt.scatter(inputList7, VNE_outputs_Homerange_IxI7, color = 'k')
plt.plot(inputList7, VNE_outputs_Homerange_IxI7, color = 'k')
plt.scatter(inputList9, VNE_outputs_Homerange_IxI9, color = 'b')
plt.plot(inputList9, VNE_outputs_Homerange_IxI9, color = 'b')
plt.show()

plt.figure()
plt.grid(True)
plt.xlabel('Time Step', fontsize = Big_font_size)
plt.ylabel('VNE value', fontsize = Big_font_size)
plt.legend(handles = [red_patch, green_patch, black_patch, blue_patch], title = 'Data set')
plt.xticks(fontsize = 30)
plt.yticks(np.linspace(0, 1, 21, endpoint = True))
plt.scatter(inputList3, VNE_outputs_SxS3, color = 'r')
plt.plot(inputList3, VNE_outputs_SxS3, color = 'r')
plt.scatter(inputList5, VNE_outputs_SxS5, color = 'g')
plt.plot(inputList5, VNE_outputs_SxS5, color = 'g')
plt.scatter(inputList7, VNE_outputs_SxS7, color = 'k')
plt.plot(inputList7, VNE_outputs_SxS7, color = 'k')
plt.scatter(inputList9, VNE_outputs_SxS9, color = 'b')
plt.plot(inputList9, VNE_outputs_SxS9, color = 'b')

plt.show()
