import numpy as np
from scipy import linalg
import pandas as pd
import glob

#############################################################################################
# The tripartite matrix assembled as one matrix
#############################################################################################
def assembling_tripartite(list_of_tripartites, time_stamp, individual_amount, location_amount):
    if time_stamp != 1:
#        print('this is the time stamp values\n', time_stamp)
        collection_of_rows_of_tripartite = []
#        row_of_matrix = np.zeros((time_stamp, individual_amount, location_amount))
        for i in range(time_stamp):
            row_of_matrix = np.zeros((time_stamp, len(individual_amount), len(location_amount)))
            row_of_matrix[i] = list_of_tripartites[i]
            collection_of_rows_of_tripartite.append(np.hstack(row_of_matrix))
        full_tripartite_matrix = np.vstack(collection_of_rows_of_tripartite)
#        print('Testint that I get a full tripartite\n', full_tripartite_matrix,
#              '\n and this is the shape\n', full_tripartite_matrix.shape)

        return(full_tripartite_matrix)
    else:
 #       print('THIS IS THE TRIPARTITE!\n', list_of_tripartites[0],
 #             '\n NAD THIS IS ITS SHAPE\n', np.array(list_of_tripartites).shape)
        return(list_of_tripartites[0])
##############################################################################################
# This creates our IxS matrix as desired
##############################################################################################
def IxS(list_of_tripartites, time_stamps, individual_amount, location_amount):
#    print(individual_amount)
    full_matrix = np.zeros((len(individual_amount), len(location_amount)))
    #print(len(list_of_tripartites))
    for i in range(time_stamps):
        full_matrix += list_of_tripartites[i]
     #   print(full_matrix, '\n')
#    print("This is the full IxS\n", full_matrix)
    return(full_matrix)

##############################################################################################
# This creates our individual by individual square matrix as desired.    
##############################################################################################
def unipartite_square_matrix(collection_of_matrices, time_stamps,
                             individuals, locations):
    
    # This is the emptry I by I matrix
    ind_matrix = np.zeros((len(individuals), len(individuals)))
    # We need to iterate through our collection of matrices; hence we start a for loop
    # going through the number of timestamps
    for i in range(time_stamps):
        # Only consider the ith matrix in our list of matrices
        matrix = collection_of_matrices[i]
        # we now need to go through each row; meaning we need to consider each individuals
        # relationship with everyone else at their location
        for j in range(len(individuals)):
            # we need to look if there are other individuals at that location
            for k in range(len(locations)):
                # if that individual is at that location we're gonna look to see if there are other people there
                if matrix[j][k] == 1:
                    # transpose the matrix for coding simplicity
                    tran_matrix = np.transpose(matrix)
                    # go through that specific location
                    for m in range(len(individuals)):
                        # if there is someone else there...
                        if tran_matrix[k][m] == 1:
                            # ...note it on the ind by ind matrix in the associate spot
                            ind_matrix[m][j] += 1
                        if m == j:
                            ind_matrix[m][j] = 0

#                else:
#                    print("There was a zero at this location\n", j, "row and the ", k, "column")
#    print('this is the association matrix\n', ind_matrix)
    return ind_matrix

##################################################################################
# SxS Unipartite Matrix
# Got the code right. Only increasing a node value if more individuals
# share the same location. Max value of a node will be equal to the number 
# of individuals in the simulation.
##################################################################################
def SxS_Matrix(IxS, individuals, locations):
    
#    print('This is the IxS matrix\n', IxS)
    
    #This creates an exmpty SxS matrix for me to fill
    SxS = np.zeros((len(locations), len(locations)))
    
    # I need to iterate through the I x S matrix so I start with the locations
    for i in range(len(locations)):
        
        # Grabbing the first column of my IxS matrix my 'starting' column
        first_column = IxS[:,i]    
#        print('This is the first column we are working with\n', first_column)
        # Creating an empty array to note the amount of overlap there is in locations
            
        # Iterating through all the individuals now
        for j in range(len(locations)):

            # We dont want to compare the same location with itself
            if i == j:
                continue
            else:
                # This is the 'next' column we are working with
                next_column = IxS[:,j]
#                print('this is hte next column we are working with\n', next_column)

            # this is to sum the amount of connections between locations
            node_of_interest = 0
                
            # Now going through all the locations from the columns that I've marked
            for k in range(len(individuals)):
                
                # If the certain 'location' is being inhabited in both arrays then investigate
                if first_column[k] > 0 and next_column[k] > 0:
#                    print('We have reached the first logical step!')
                    node_of_interest += 1                    
                
            # This is me filling in the SxS matrix with the summed up connections between locations
            SxS[i][j] = node_of_interest
            SxS[j][i] = node_of_interest

    return(SxS)
##################################################################################
# Homerange IxI matrix
# For this we are comparing individuals separately, meaning
# we will not have a symmetric matrix. Also we will be normalizing by the
# amount of inhabited locations by the individuals
##################################################################################
def Homerange_IxI(IxS_matrix, time_stamps, 
                  individuals, locations):
    
    ind_matrix = np.zeros((len(individuals), len(individuals)))
    
    for i in range(len(individuals)):
        firstIndividual = IxS_matrix[i]        
        for k in range(len(individuals)):
            if i == k:
                continue
            else:
                secondIndividual = IxS_matrix[k]
            nodeOfInterest = 0
            
            array = np.nonzero(secondIndividual)
            dividingValue = len(array[0])

            for j in range(len(locations)):
                if firstIndividual[j] == 0 or secondIndividual[j] == 0:
                    continue
                if firstIndividual[j] > 0 and secondIndividual[j] > 0:
                    nodeOfInterest += 1
                
                nodeValue = nodeOfInterest/dividingValue
                ind_matrix[i][k] = nodeValue

    return(ind_matrix)
##################################################################################
# This helps compute the VNE
##################################################################################
# This function calculates mu given our square matrix.
def cal_aida(desired_projection_laplacian):
    
 
    # This outputs the eigenvalues for our square matrix. We are only concerned
    # with the 'eigenvalues' and the dump is a place holder that we can ignore
    eigenvalues = linalg.eigvals(desired_projection_laplacian)
 
    # We sum up all the eigenvalues for our square matrix
    eigenvalue_sum = np.sum(eigenvalues)
 
    # This creates a zero matrix that is as long as the eigenvalue vector
    aida = np.zeros(len(eigenvalues))
    
    # This will compute the aida formula for all our eigenvalues
    for i in range(len(eigenvalues)):
        aida[i] = (eigenvalues[i])/(eigenvalue_sum)

    return aida

###################################################################################
# Functions computing the VNE and normalizing it
###################################################################################
# This calculates the von neumann entropy given our aida vector
def cal_von_entropy(aida):
    
    # This is an integer that we can add to based on the aida formula
    p = 0
    
    # This iterates through our aida vector so we can complete our calculations
    for i in range(len(aida)):
    
        # This is considering the case if aida is less than or equal to zero
        # That way we don't end up with 'nan' outputs
        imaginary = np.isnan(aida[i])
        #print("This is our aida[i] output\n", aida[i])
        #print("This should be the truth statement of nan", imaginary)
        if aida[i] <= 0.0 or imaginary == True:
            p += 0
    
        # This is the formula being calculated.
        else:
            p += aida[i]*np.log(aida[i])
    
    # This is the final portion of our calculation
    von_entropy_summed = -p
    
    return von_entropy_summed

###########################################################################################################
# This normalizes the von neumann entropy 
###########################################################################################################
def normalize_von_entropy(von_neumann_entropy_vector, dimension):

    von_neumann_entropy_value = von_neumann_entropy_vector/np.log(dimension[0])

    return von_neumann_entropy_value

#############################################################################################################
# Laplacian functions for unipartite and tripartite matrices
#############################################################################################################
def unipartite_laplacian(matrix):
    diag_vals = []
    shape_of_matrix = matrix.shape
    
    # I want to check with them that to compute the Laplacian of the square matrix we are just summing all 
    # the entries per row and that becomes the diagonal entries?
    for i in range(shape_of_matrix[0]):
        diag_vals.append(np.sum(matrix[i]))
    
        # This creates the adjacency matrix at the same time
        matrix[i][i] = 0
    diagonal_matrix = np.diag(diag_vals)
    
    # We now have the diagonal matrix which is the summation of each row and the adjacency matrix which
    # is represented by the matrix.
    laplacian = diagonal_matrix - matrix

    return(laplacian)
    
###########################################################################################################
# This computes the laplacian matrix for the IxS matrix
###########################################################################################################
def IxS_laplacian_matrix(matrix, time_stamps, individual_amount):
    
    # This takes the shape of our adjacency matrix and creates the transpose of it
    transposed_matrix = np.transpose(matrix)
    
    # This line gives us a diagonal matrix. For the IxS case the diagonals will be the same
    # value as the amount of timestamps because we are summing the amount of 'instances'
    # in each row. Each row only represents the amount of time stamps.
    diag_vect = [time_stamps] * len(individual_amount)
    diagonal_one = np.diag(diag_vect)
    
    # This takes the sum of each column which we use to create the diagonal two matrix
    diagonal_two_vals = matrix.sum(axis=0)
    diagonal_two = np.diag(diagonal_two_vals)
        
    # Negate adjacency and the transposed matrix
    neg_adjacency = -1*matrix
    neg_transposed = -1*transposed_matrix
    
    # This is the top row of the laplacian followed by the bottom
    top_row = np.hstack((diagonal_one, neg_adjacency))
    bottom_row = np.hstack((neg_transposed, diagonal_two))
    
    # Putting it all together
    laplacian = np.vstack((top_row, bottom_row))

    return laplacian

###########################################################################################################
# This computes the laplacian matrix for the tripartite matrix
###########################################################################################################
def Tripartite_laplacian_matrix(matrix, time_stamps, individual_amount):
    
    # This takes the shape of our adjacency matrix and creates the transpose of it
    transposed_matrix = np.transpose(matrix)
    shape_of_tripartite = np.array(matrix).shape
 
    # This will give us an identity matrix the same shape as the tripartite. The reason it
    # is an identity matrix is because it will be summing up each row and with how the
    # tripartite is structured it will only be one per row.
    diagonal_one = np.diag(np.ones(shape_of_tripartite[0]))
    
    # This takes the sum of each column which we use to create the diagonal two matrix
    diagonal_two_vals = matrix.sum(axis=0)
    diagonal_two = np.diag(diagonal_two_vals)
        
    # Negate adjacency and the transposed matrix
    neg_adjacency = -1*matrix
    neg_transposed = -1*transposed_matrix
    
    # This is the top row of the laplacian followed by the bottom
    top_row = np.hstack((diagonal_one, neg_adjacency))
    bottom_row = np.hstack((neg_transposed, diagonal_two))
    
    # Putting it all together
    tripartite_laplacian = np.vstack((top_row, bottom_row))

    return tripartite_laplacian

###########################################################################################################
# This gets the amount of individuals and locations being considered for 
# our real data
###########################################################################################################
def getMaxColumnsAndRows():
    columns = []
    indexes = []
    for file in glob.glob(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Real_data_work\05Data\tab_list_.05\*'):
        df = pd.read_csv(file)
        df = df.set_index('Unnamed: 0')
        del(df.index.name)
        columns.append(list(df.columns))
        indexes.append(list(df.index))
    indexes = np.hstack(indexes)
    maxRows = list(np.unique(np.sort(indexes)))
    columns = np.hstack(columns)
    maxColumns = list(np.unique(np.sort(columns)))        
    return(maxColumns, maxRows)
    
###########################################################################################################
# This adds individuals to time steps that are missing samples for
# indidividuals. Their placement is randomly chosen uniformly.
###########################################################################################################
def addRows(df, newDf, maxRows):
    if len(maxRows) == len(list(df.index)):
        return(df)
#    print('all rows', len(maxRows), 'current rows', len(list(df.index)))
    differentElements = set(maxRows).difference(list(df.index))
#    print(differentElements, len(differentElements))
    for i in differentElements:
        inhabitedLocations = newDf.loc[i].nonzero()[0]
#        print('inhabited locations\n', inhabitedLocations, 'length of them\n', len(inhabitedLocations))
        indexOfLocation = np.random.choice(len(inhabitedLocations), 1)
#        print('index of the location', indexOfLocation)
        newRow = np.zeros(len(list(df.columns)))
        newRow[inhabitedLocations[indexOfLocation]] = 1
        df.loc[i] = newRow 
    return(df)
    
###########################################################################################################
# This creates a new df that shows all the locations we have recorded so we can use
# those locations when using the probability method from np.numpy.choice()
###########################################################################################################    
def getNewDf(maxRows, maxColumns):
    data = np.zeros((len(maxRows), len(maxColumns)))
    newDf = pd.DataFrame(data, maxRows, maxColumns)
    for file in glob.glob(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Real_data_work\05Data\tab_list_.05\*'):
        df = pd.read_csv(file)
        df = df.set_index('Unnamed: 0')
        del(df.index.name)
        df = addColumns(df, maxColumns)
        for i in df.index:
            newDf.loc[i] += df.loc[i]
    return(newDf)

###########################################################################################################
# This adds empty locations to make sure the dimensions are all the same for each time step
###########################################################################################################    
def addColumns(df, maxColumns):
    df = df.loc[:, maxColumns].fillna(0)
    return(df)

###########################################################################################################
# This chooses the location the individual is located by looking at the area where they resided
# the most during the day and changes that value to a 1 and every other cell value to a zero.
###########################################################################################################
def binaryValues(df):
    for i in df.index:
        seriesRow = df.loc[i]
        nonzeroEntries = seriesRow.nonzero()[0]
        actualEntryValues = []
        for j in nonzeroEntries:
            actualEntryValues.append(df.loc[i][j])
            df.loc[i][j] = 0
        #display(nonzeroEntries, array)
        df.loc[i][nonzeroEntries[np.argmax(actualEntryValues)]] = 1
    return(df)
###########################################################################################################
# This calls the functions which alter the size and structure of each sample
# and returns the modified samples to compute VNE
###########################################################################################################
def createTimeStepMatrix(file, newDf, maxRows, maxColumns):
    df = pd.read_csv(file)
    df = df.set_index('Unnamed: 0')
    del(df.index.name)
    df = addColumns(df, maxColumns)
    df = addRows(df, newDf, maxRows)
    df = binaryValues(df)
    return(df, maxRows, maxColumns)
    
###########################################################################################################
# This constructs the networks using the real data
###########################################################################################################
def constructingNetworks(timeStepMatrices, maxRows, maxColumns):
    IxSMatrix = IxS(timeStepMatrices, np.array(timeStepMatrices).shape[0], maxRows, maxColumns)
    IxIMatrix = unipartite_square_matrix(timeStepMatrices, np.array(timeStepMatrices).shape[0], maxRows, maxColumns)
    tripartiteMatrix = assembling_tripartite(timeStepMatrices, np.array(timeStepMatrices).shape[0], maxRows, maxColumns)
    homerangeMatrix = Homerange_IxI(IxSMatrix, np.array(timeStepMatrices).shape[0], maxRows, maxColumns)
    SxSMatrix = SxS_Matrix(IxSMatrix, maxRows, maxColumns)
    
    return(IxSMatrix, IxIMatrix, tripartiteMatrix, 
           homerangeMatrix, SxSMatrix)

###########################################################################################################
# The laplacian matrix for each network so VNE can be computed
###########################################################################################################
def laplacianConstruction(IxSMatrix, IxIMatrix, tripartiteMatrix,
                          homerangeMatrix, SxSMatrix, timeStepMatrices,
                          maxRows):
    
    IxSLaplacian = IxS_laplacian_matrix(IxSMatrix, np.array(timeStepMatrices).shape[0], maxRows)
    IxILaplacian = unipartite_laplacian(IxIMatrix)
    tripartiteLaplacian = Tripartite_laplacian_matrix(tripartiteMatrix, np.array(timeStepMatrices).shape[0], maxRows)
    homerangeLaplacian = unipartite_laplacian(homerangeMatrix)
    SxSLaplacian = unipartite_laplacian(SxSMatrix)
    
    return(IxSLaplacian, IxILaplacian, tripartiteLaplacian,
           homerangeLaplacian, SxSLaplacian)    

###########################################################################################################
# Aida values for each to compute VNE
###########################################################################################################
def aidaValues(IxSLaplacian, IxILaplacian, tripartiteLaplacian,
               homerangeLaplacian, SxSLaplacian):
    
    IxSAida = cal_aida(IxSLaplacian)
    IxIAida = cal_aida(IxILaplacian)
    tripartiteAida = cal_aida(tripartiteLaplacian)
    homerangeAida = cal_aida(homerangeLaplacian)
    SxSAida = cal_aida(SxSLaplacian)
    
    return(IxSAida, IxIAida, tripartiteAida,
           homerangeAida, SxSAida)

###########################################################################################################
# This actually computes the VNE
###########################################################################################################
def VNECalculation(IxSAida, IxIAida, tripartiteAida,
                   homerangeAida, SxSAida):
    IxSVNE = cal_von_entropy(IxSAida)
    IxIVNE = cal_von_entropy(IxIAida)
    tripartiteVNE = cal_von_entropy(tripartiteAida)
    homerangeVNE = cal_von_entropy(homerangeAida)
    SxSVNE = cal_von_entropy(SxSAida)
    
    return(IxSVNE, IxIVNE, tripartiteVNE,
           homerangeVNE, SxSVNE)

###########################################################################################################
# This is the function which calls every other function to compute everything
###########################################################################################################
def fullFunction():
    timeStepMatrices = []
    fullIxSVNE = []
    fullIxIVNE = []
    fullTripartiteVNE = []
    fullHomerangeVNE = []
    fullSxSVNE = []
    i = 0
    maxColumns, maxRows = getMaxColumnsAndRows()
    newDf = getNewDf(maxRows, maxColumns)

    print('max rows then columns\n', len(maxRows), '\n', len(maxColumns))
    for file in glob.glob(r'C:\Users\tjbro\Desktop\Thesis_Project\Thesis_code\Real_data_work\05Data\tab_list_.05\*'):
        df, maxRows, maxColumns = createTimeStepMatrix(file, newDf, maxRows, maxColumns)
        timeStepMatrices.append(df.to_numpy())
        IxSMatrix, IxIMatrix, tripartiteMatrix, homerangeMatrix, SxSMatrix = constructingNetworks(timeStepMatrices, maxRows, maxColumns)
        IxSLaplacian, IxILaplacian, tripartiteLaplacian, homerangeLaplacian, SxSLaplacian = laplacianConstruction(IxSMatrix,
                                                                                                                  IxIMatrix,
                                                                                                                  tripartiteMatrix, 
                                                                                                                  homerangeMatrix,
                                                                                                                  SxSMatrix, 
                                                                                                                  timeStepMatrices, 
                                                                                                                  maxRows)
        
        IxSAida, IxIAida, tripartiteAida, homerangeAida, SxSAida = aidaValues(IxSLaplacian, IxILaplacian, tripartiteLaplacian,
                                                                              homerangeLaplacian, SxSLaplacian)

        IxSVNE, IxIVNE, tripartiteVNE, homerangeVNE, SxSVNE = VNECalculation(IxSAida, IxIAida, tripartiteAida, homerangeAida,
                                                                             SxSAida)
        
        fullIxSVNE.append(normalize_von_entropy(IxSVNE, IxSLaplacian.shape))
        fullIxIVNE.append(normalize_von_entropy(IxIVNE, IxILaplacian.shape))
        fullTripartiteVNE.append(normalize_von_entropy(tripartiteVNE, tripartiteLaplacian.shape))
        fullHomerangeVNE.append(normalize_von_entropy(homerangeVNE, homerangeLaplacian.shape))
        fullSxSVNE.append(normalize_von_entropy(SxSVNE, SxSLaplacian.shape))
        i += 1
        print('data file done!', i)
        
    return(fullIxSVNE, fullIxIVNE, fullTripartiteVNE, fullHomerangeVNE,
           fullSxSVNE)        

###########################################################################################################
# Calling the whole function together
###########################################################################################################
fullIxSVNE, fullIxIVNE, fullTripartiteVNE, fullHomerangeVNE, fullSxSVNE = fullFunction()


#print(np.array(fullIxSVNE).shape, np.array(fullIxIVNE).shape, np.array(fullTripartiteVNE).shape,
#      np.array(fullHomerangeVNE).shape, np.array(fullSxSVNE).shape)

#np.save('IxSVNERealData07', fullIxSVNE)
#np.save('IxIVNERealData07', fullIxIVNE)
#np.save('tripartiteVNERealData07', fullTripartiteVNE)
#np.save('homerangeVNERealData07', fullHomerangeVNE)
#np.save('SxSVNERealData07', fullSxSVNE)
