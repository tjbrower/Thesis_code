import numpy as np
from scipy.stats import beta
from scipy.stats import dirichlet
from scipy import linalg
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

###############################################################################################
# This is creating the our first tripartite matrix. We're passing the 'vector' out so we can 
# create more tripartite networks based on the amount of timestamps. 
###############################################################################################
#def create_initial_matrix_with_beta_distribution(number_of_locations, number_of_individuals, beta_value):#, time_stamps):
#    a, b = 1.0, beta_value
#    x = np.linspace(0.1, 1.0, number_of_locations)
#    alpha_for_dirichlet = beta.pdf(x,a,b)
#    for i in range(number_of_locations):
#        if alpha_for_dirichlet[i] == 0.0:
#            alpha_for_dirichlet[i] = 0.000000001
#    print('this is the alpha for dirichlet\n', alpha_for_dirichlet)
#    vector = dirichlet.rvs(alpha_for_dirichlet, size=1)
#    print('This is the vector\n', vector)
#        # all of the above is completely necessary to give the 'weights' required to 
#        # each location 
#
#    full_matrix = np.zeros((number_of_individuals, number_of_locations))
#    #print('full matrix shape\n', full_matrix.shape)
#    list_of_tripartite = []
#    #print("alpha vector\n", vector[0])
#    #for p in range(time_stamps):
#    #print('Number of locations\n', number_of_locations, '\n alpha vector\n', vector[0],
#    #      '\nhere are their shapes\n', len(vector[0]))
#    #print('number of ind\n', number_of_individuals)
#    for q in range(number_of_individuals):
#        j = np.random.choice(number_of_locations, 1, p = vector[0])
#        #j = individual_location
#        full_matrix[q][j] = 1.0
#    list_of_tripartite.append(full_matrix)
#    #print("This should be our new row combined matrix\n", full_matrix)
#        
#    return (full_matrix, list_of_tripartite, vector[0])
#
##############################################################################################
# This is creating the spatial preference vector used when describing the stickiness
# of each group as well as the initial placement of the groups.
##############################################################################################
def creating_spatial_preference_vector(number_of_locations, number_of_individuals, beta_value):#, time_stamps):
    a, b = 1.0, beta_value
    x = np.linspace(0.1, 1.0, number_of_locations)
    alpha_for_dirichlet = beta.pdf(x,a,b)
    for i in range(number_of_locations):
        if alpha_for_dirichlet[i] == 0.0:
            alpha_for_dirichlet[i] = 0.000000001
    #print('this is the alpha for dirichlet\n', alpha_for_dirichlet)
    vector = dirichlet.rvs(alpha_for_dirichlet, size=1)
#    print('This is the vector\n', vector)
        # all of the above is completely necessary to give the 'weights' required to 
        # each location 

    return(vector)

##############################################################################################
# This creates the group preference vector used to assign how many individuals will be
# in each group.
##############################################################################################
def creating_group_preference_vector(number_of_locations, number_of_individuals, gamma_value):#, time_stamps)
    
    # Establishing the parameters used to create a dirichlet distribution
    a, b = 1.0, gamma_value
    
    # This creates the amount of 'groups' which are the same as the number of locations
    x = np.linspace(0.1, 1.0, number_of_locations)

    # This creates the dirichlet distribution
    alpha_for_dirichlet = beta.pdf(x,a,b)
    
    # This guarantees that we are not dividing by zero
    for i in range(number_of_locations):
        if alpha_for_dirichlet[i] == 0.0:
            alpha_for_dirichlet[i] = 0.000000001
#    print('this is the alpha for dirichlet\n', alpha_for_dirichlet)

    # This is creating the actual group preference vector
    group_preference_vector = dirichlet.rvs(alpha_for_dirichlet, size=1)
#    print('This is the vector\n', vector)
        # all of the above is completely necessary to give the 'weights' required to 
        # each location 

    # This should be assigning how many individuals are in each group
    individuals_in_each_group = np.zeros(number_of_locations)
    for i in range(number_of_individuals):
        j = np.random.choice(number_of_locations, 1, p = group_preference_vector[0])
        individuals_in_each_group[j] += 1
#    print('this should be the vector that has the individuals assigned to each group\n', 
#          individuals_in_each_group)
    
    return(group_preference_vector, individuals_in_each_group)

##############################################################################################
# This is creating the initial matrix for any simulation
##############################################################################################
def each_timestamp_matrix(number_of_locations, number_of_individuals, spatial_preference_vector,
                          group_preference_vector, individuals_in_each_group, rho):

    # I need to assign each group to a location; this code will give me the indices
    # that are non-zero. So individuals_in_each_group with the indices tells me what
    # groups to use.
    indices_of_interest = np.nonzero(individuals_in_each_group)
    indices_of_interest = indices_of_interest[0]
#    print('these are the groups we are working with\n', indices_of_interest)
    
    # This will be what we're adding to until we're done and then we'll stack
    # them on top of each other to get our beginning matrix.
    matrices_for_each_group = []
#    print('spatial vector\n', spatial_preference_vector)
#    amount_of_zero_entries = len(spatial_preference_vector) - np.nonzero(spatial_preference_vector)
#    print('amoun of zero\n', amount_of_zero_entries, '\n len of indices\n', len(indices_of_interest))
    
    amount_of_zeros = 0
    for i in range(number_of_locations):
        if spatial_preference_vector[0][i] == 0:
#            print('this is the entry in the spatial preference vector\n',
#                  spatial_preference_vector[0][i])
            amount_of_zeros += 1
            
    non_zero_entries = number_of_locations - amount_of_zeros
#    print('This is the vector length\n', spatial_preference_vector[0], 
#          '\nthis is the amount of locations that have a probability being here\n', non_zero_entries, 
#          '\nlength of vector\n', len(indices_of_interest))
#     I now need to assign each group to a location WITHOUT replacement
    all_locations_for_the_groups_without_replacement = np.random.choice(number_of_locations, len(indices_of_interest), replace =True, p = spatial_preference_vector[0])


    if non_zero_entries < len(indices_of_interest):
        all_locations_for_the_groups_without_replacement = np.random.choice(number_of_locations, len(indices_of_interest), replace =True, p = spatial_preference_vector[0])
    else:
        all_locations_for_the_groups_without_replacement = np.random.choice(number_of_locations, len(indices_of_interest), replace =False, p = spatial_preference_vector[0])
    
#    print('This is sampling without replacement for the locations for all GROUPS!\n', 
#          all_locations_for_the_groups_without_replacement)
    
    # This is the same length as the number of groups that are 'filled'
    # a.k.a this is the amount of groups we are considering
    for i in range(len(all_locations_for_the_groups_without_replacement)):
        current_group_index = indices_of_interest[i]
        amount_of_individuals_in_group = individuals_in_each_group[current_group_index]
#        print('here is the current group index\n', current_group_index,
#              '\n this is the individuals in each group vector\n', individuals_in_each_group,
#              '\n and this is the amount of individuals in this group\n', amount_of_individuals_in_group)
        groups_location = all_locations_for_the_groups_without_replacement[i]
#        print('This is where the current group is going to go\n', groups_location)
        groups_matrix = np.zeros((int(amount_of_individuals_in_group), number_of_locations))
        
        # This next bit of code will determine the 'rho' vector dependent on the groups
        # current location
        rho_vector = np.zeros(number_of_locations)
        rho_vector[groups_location] = rho
        
        # This for loop is making every other location less than ideal
        for j in range(number_of_locations):
            if rho_vector[j] == rho:
                continue
            else:
                rho_vector[j] = (1-rho)
        
#        print('this should be our rho vector\n', rho_vector, '\n and the sum of it\n',
#              sum(rho_vector))

        # Just checking that the rho vector equals 1
#        if sum(rho_vector) >= 0.98 and sum(rho_vector) <= 1.01:
#            print('YEAH!!!')
#        else:
#            print('NOOOO!!!!!\n', rho_vector, '\n resulting sum\n', sum(rho_vector))
            
        # This is creating our stickiness vector which will take into consideration
        # whether indiviuals want to stick with their group and if the don't, that
        # they still go to an ideal location.
 #       print('rho vector\n', rho_vector, '\n spatial preference vector\n', spatial_preference_vector[0])
        complete_stickiness_vector = rho_vector * spatial_preference_vector[0]
        
#        print('this is the stickiness vector\n', complete_stickiness_vector,
#              '\n and this is the sum of the vector\n', sum(complete_stickiness_vector))
        
        # Now I normalize?
        normalized_stickiness_vector = complete_stickiness_vector/sum(complete_stickiness_vector)
#        print('This is the normalized stickiness vector\n', normalized_stickiness_vector,
#              '\n and this is the sum\n', sum(normalized_stickiness_vector))
        
        # Now I'm going to create a vector where each individual is assigned to a location
        # with the stickiness vector.
        location_of_individuals = np.random.choice(number_of_locations, int(amount_of_individuals_in_group), p = normalized_stickiness_vector)
        
        # I now need to take this vector right above and create a matrix with it
        for k in range(int(amount_of_individuals_in_group)):
            actual_location = location_of_individuals[k]
            groups_matrix[k][actual_location] = 1
#        print('this is the matrix\n', groups_matrix)    
        matrices_for_each_group.append(groups_matrix)
    
 #   print('these are the matrices that we are dealing with\n', matrices_for_each_group)
    # We finish up by stacking all the matrices on top of each other.
    full_timestamp_matrix = np.vstack(matrices_for_each_group)
        
    return(full_timestamp_matrix)
    
##############################################################################################
# This function is computing the rho value. I need to decide how many 'groups' I'm going to 
# choose for each simulation.
# I NEED TO MAKE SURE ALL OF THIS IS WORKING GOOD!!!! THIS IS JUST FOR THE INITIAL MATRIX.
# HOW WOULD I CREATE A FUNCTION THAT WOULD ALLOW ME TO DO THIS AS WELL?.
###############################################################################################
#def create_initial_matrix_using_rho(locations, individuals, rho, time_stamp, groups):    
#
#    full_matrix = np.zeros((individuals, locations))
#    #print('full matrix shape\n', full_matrix.shape)
#    list_of_tripartite = []
#    
#    q = 0
#    
#    for i in range(groups):
#        # this determines where the 'group' has set down
#        number = np.random.randint(0, locations - 1)
#        
#        # This is creating the weight vector
#        weight_vector = np.zeros(locations)
#        
#        # This for loop establishes the weight vector based off of rho
#        for j in range(locations):
#            if j == number:
#                weight_vector[j] = rho
#            else:
#                weight_vector[j] = (1-rho)/(locations-1)
##        print('this is the weight vector now\n', weight_vector)
#
#        for k in range(int(individuals/groups)):
#            j = np.random.choice(locations, 1, p = weight_vector)
#            #j = individual_location
#            full_matrix[q][j] = 1.0
#            q += 1
#
#    list_of_tripartite.append(full_matrix)
##    print("This should be our new row combined matrix\n", full_matrix)
#        
#    return (full_matrix, list_of_tripartite)
#        
##############################################################################################
# This function adds to the tripartite list
##############################################################################################
#def adding_to_the_matrix_using_beta(alpha_vector, number_of_locations, number_of_individuals,
#                                    time_stamps, current_matrix, list_of_tripartite):
#    
##    empty_list = np.zeros((time_stamps, time_stamps, number_of_individuals, number_of_locations))
##    final_product = np.zeros((time_stamps, number_of_individuals, number_of_locations*time_stamps))
##    added_matrix = np.zeros((number_of_individuals, number_of_locations))
##    for p in range(time_stamps):
##        if p == (time_stamps - 1):
##            for q in range(number_of_individuals):
##                j = np.random.choice(number_of_locations, 1, p = alpha_vector)
##    #            j = individual_location
###                empty_list[p][p][q][j] = 1
##                added_matrix[q][j] = 1.0
##            empty_list[p][p] = added_matrix
##            list_of_tripartite.append(added_matrix)
##        else:
##            empty_list[p][p] = list_of_tripartite[p]
##        final_product[p] = np.hstack(empty_list[p])
##
##    full_matrix = np.vstack(final_product)
##    return(full_matrix, list_of_tripartite)
#
#
#    next_matrix = np.zeros((number_of_individuals, number_of_locations))
#    for i in range(number_of_individuals):
#        j = np.random.choice(number_of_locations, 1, p = alpha_vector)
#        next_matrix[i][j] = 1.0
#    #print(next_matrix, '\n', next_matrix.shape)
#    return(next_matrix)
#        
#############################################################################################
# Time stamps being added using the rho method
#############################################################################################
#def adding_to_the_matrix_using_rho(locations, individuals, time_stamps,
#                                   current_matrix, list_of_tripartite, rho, groups):
#
#    # This creates the matrix we're going to fill for our given time stamp    
#    next_matrix = np.zeros((individuals, locations))
#    
#    q = 0
#    
#    for i in range(groups):
#        # this determines where the 'group' has set down
#        number = np.random.randint(0, locations - 1)
#        
#        # This is creating the weight vector; empty at first
#        weight_vector = np.zeros(locations)
#        
#        # This for loop establishes the weight vector based off of rho
#        for j in range(locations):
#            if j == number:
#                weight_vector[j] = rho
#            else:
#                weight_vector[j] = (1-rho)/(locations-1)
##        print('this is the weight vector now\n', weight_vector)
#
#        for k in range(int(individuals/groups)):
#            j = np.random.choice(locations, 1, p = weight_vector)
#            #j = individual_location
#            next_matrix[q][j] = 1.0
#            q += 1
#
#    
#    return(next_matrix)

#############################################################################################
# The tripartite matrix assembled as one matrix
#############################################################################################
def assembling_tripartite(list_of_tripartites, individual_amount, location_amount, time_stamp):
    if time_stamp != 1:
#        print('this is the time stamp values\n', time_stamp)
        collection_of_rows_of_tripartite = []
#        row_of_matrix = np.zeros((time_stamp, individual_amount, location_amount))
        for i in range(time_stamp):
            row_of_matrix = np.zeros((time_stamp, individual_amount, location_amount))
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
    full_matrix = np.zeros((individual_amount, location_amount))
    #print(len(list_of_tripartites))
    for i in range(time_stamps):
        full_matrix += list_of_tripartites[i]
     #   print(full_matrix, '\n')
 #   print("this should be the summed up full_matrix\n", full_matrix)
    return(full_matrix)

##############################################################################################
# This creates our individual by individual square matrix as desired.    
##############################################################################################
def unipartite_square_matrix(collection_of_matrices, individuals, 
                             locations, time_stamps):
    
    # This is the emptry I by I matrix
    ind_matrix = np.zeros((individuals, individuals))
    # We need to iterate through our collection of matrices; hence we start a for loop
    # going through the number of timestamps
    for i in range(time_stamps):
        # Only consider the ith matrix in our list of matrices
        matrix = collection_of_matrices[i]
        # we now need to go through each row; meaning we need to consider each individuals
        # relationship with everyone else at their location
        for j in range(individuals):
            # we need to look if there are other individuals at that location
            for k in range(locations):
                # if that individual is at that location we're gonna look to see if there are other people there
                if matrix[j][k] == 1:
                    # transpose the matrix for coding simplicity
                    tran_matrix = np.transpose(matrix)
                    # go through that specific location
                    for m in range(individuals):
                        # if there is someone else there...
                        if tran_matrix[k][m] == 1:
                            # ...note it on the ind by ind matrix in the associate spot
                            ind_matrix[m][j] += 1
                        if m == j:
                            ind_matrix[m][j] = 0

#                else:
#                    print("There was a zero at this location\n", j, "row and the ", k, "column")
    print('this is the association matrix\n', ind_matrix)
    return ind_matrix

##################################################################################
# SxS Unipartite Matrix; I am having issues with this code. I need to spend a few hours
# on it where I am actually capturing the correct rows and columns. I think I might just
# need to change a few things around with the i and j variables but other than that I
# think it will work fine.
##################################################################################
def SxS_Matrix(IxS, locations, individuals):
    
#    print('This is the IxS matrix\n', IxS)
    
    #This creates an exmpty SxS matrix for me to fill
    SxS = np.zeros((locations, locations))
    
    # I need to iterate through the I x S matrix so I start with the locations
    for i in range(locations):
        
        # Grabbing the first column of my IxS matrix my 'starting' column
        first_column = IxS[:,i]    
#        print('This is the first column we are working with\n', first_column)
        # Creating an empty array to note the amount of overlap there is in locations
            
        # Iterating through all the individuals now
        for j in range(locations):

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
            for k in range(individuals):
                
                # If the certain 'location' is being inhabited in both arrays then investigate
                if first_column[k] > 0 and next_column[k] > 0:
#                    print('We have reached the first logical step!')
                    
                    # if they're the same value, then mark that number in the array above
                    if first_column[k] == next_column[k]:
                        node_of_interest += first_column[k]
                    
                    # Otherwise, choose the smalle of the two numbers and mark that instead
                    elif first_column[k] < next_column[k]:
                        node_of_interest += first_column[k]
                    else:
                        node_of_interest += next_column[k]
        
            # This is me filling in the SxS matrix with the summed up connections between locations
            SxS[i][j] = node_of_interest
            SxS[j][i] = node_of_interest
#            print('this is the SxS matrix so far\n', SxS)
            
    print('this is SxS\n', SxS)
    return(SxS)
##################################################################################
# Homerange IxI matrix
##################################################################################
def Homerange_IxI(IxS_matrix, individuals,
                  locations, time_stamps):
    
    ind_matrix = np.zeros((individuals, individuals))
    tran_matrix = np.transpose(IxS_matrix)
    # Going through the IxS matrix so that we can create an IxI matrix 
    for i in range(individuals):
        for j in range(locations):
            if tran_matrix[j][i] > 0:
                for k in range(individuals):
                    if tran_matrix[j][k] > 0:
                        if tran_matrix[j][i] < tran_matrix[j][k]:
                            ind_matrix[k][i] = tran_matrix[j][i]
                        else:
                            ind_matrix[k][i] = tran_matrix[j][k]
                        if k == i:
                            ind_matrix[k][i] = 0
                            
    print('This is the IxS matrix\n', IxS_matrix, '\nThis is the Homerange IxI\n', ind_matrix)
    return(ind_matrix)
##################################################################################
# This helps compute the VNE
##################################################################################
# This function calculates mu given our square matrix.
def cal_aida(desired_projection_laplacian):
    
 #   print("This is the desired projections")
 
    # This outputs the eigenvalues for our square matrix. We are only concerned
    # with the 'eigenvalues' and the dump is a place holder that we can ignore
    eigenvalues = linalg.eigvals(desired_projection_laplacian)
    #print("Eigenvalues\n", eigenvalues)
 #   print("After the eigenvalues")
 
    # We sum up all the eigenvalues for our square matrix
    eigenvalue_sum = np.sum(eigenvalues)
 #   print("This is after the sum of eigenvalues")
 
    # This creates a zero matrix that is as long as the eigenvalue vector
    aida = np.zeros(len(eigenvalues))
    
    # This will compute the aida formula for all our eigenvalues
    for i in range(len(eigenvalues)):
        aida[i] = (eigenvalues[i])/(eigenvalue_sum)
   # print("this is the aida vector\n", aida)
    return aida, eigenvalues

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
    
    #print("This is the von entropy\n", von_entropy_summed)
    return von_entropy_summed

###########################################################################################################
# This normalizes the von neumann entropy 
###########################################################################################################
def normalize_von_entropy(von_neumann_entropy_vector, dimension):
#    print("This is the von entropy vector before\n", von_neumann_entropy_vector)
#    print("This is the dimension for our given case", dimension)
    von_neumann_entropy_value = von_neumann_entropy_vector/np.log(dimension[0])
#    print("And this is it after\n", von_neumann_entropy_vector)
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
   # print("This should be the laplcian\n", laplacian)
#    print("This should be my diagonal values\n", diag_vals)
    return(laplacian)
    
###########################################################################################################
# This computes the laplacian matrix for the IxS matrix
###########################################################################################################
def IxS_laplacian_matrix(matrix, time_stamps, individual_amount):
    
    # This takes the shape of our adjacency matrix and creates the transpose of it
    transposed_matrix = np.transpose(matrix)
   # print("Still the matrix\n", matrix)
    
    # This line gives us a diagonal matrix. For the IxS case the diagonals will be the same
    # value as the amount of timestamps because we are summing the amount of 'instances'
    # in each row. Each row only represents the amount of time stamps.
    diag_vect = [time_stamps] * individual_amount
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

   # print("tripartite laplacian\n", laplacian)
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

   # print("tripartite laplacian\n", laplacian)
    return tripartite_laplacian


############################################################################################################
## This function computes the binary matrix; not being used currently.
############################################################################################################
#def binary_matrix(matrix):
#    shape = matrix.shape
#    for i in range(shape[0]):
#        for j in range(shape[1]):
#            if matrix[i][j] > 1:
#                matrix[i][j] = 1
#    #print('binary matrix\n', matrix)
#    return(matrix)

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
    
#    Michaelis_constant = 0
    # This goes through and shows the elements of each simulation for specific rho, gamma, and beta combinations
    # and then it calculates the 'V' variable needed to send to the Lineweaver Burke file so we can plot it.
    full_v_array = []
    for j in range(amount_of_simulations):
        per_simulation_v_array = []
#        print('simulation ', j)
        for i in range(len(gamma_values)):
            per_gamma_v_array = []
#            print('gamma value ', i)
            for k in range(len(beta_values)):
                per_beta_v_array = []
#                print('beta value ', k)
                for z in range(len(rho_values)):
 #                   print('rho value ', z)
                    all_VNE_for_a_gamma_beta_rho_combination_per_simulation = collection[j][i][k][z]
#                    print('this should only have a shape of 20\n', np.array(all_VNE_for_a_gamma_beta_rho_combination_per_simulation).shape)
#                    print('this should have a shape of rho times 20\n', np.array(collection[j][i][k]).shape)
                    maximum_element = np.amax(all_VNE_for_a_gamma_beta_rho_combination_per_simulation)
                    minimum_element = np.amin(all_VNE_for_a_gamma_beta_rho_combination_per_simulation)
                    print('this is the list of VNE first\n', all_VNE_for_a_gamma_beta_rho_combination_per_simulation)
                    if minimum_element == 0 or minimum_element == -0:
                        copied_list = all_VNE_for_a_gamma_beta_rho_combination_per_simulation.copy()
                        sorted_list = sorted(copied_list)
                        print('this is the sorted list\n', sorted_list)
                        for q in range(len(all_VNE_for_a_gamma_beta_rho_combination_per_simulation)):
                        #print('This should be the list sorted\n', sorted_list)
                            minimum_element = sorted_list[q]
                            if minimum_element != 0 and minimum_element != -0:
                                print('I should be breaking')
                                break
                            print('did I break?')
                        print('still in the minimum element = 0 section')
                    distance = maximum_element - minimum_element
                    middle_element = distance/2 + minimum_element
                    per_rho_v_array = []
                    Michaelis_constant = 0
                    for t in range(time_stamps):
                        if collection[j][i][k][z][t] >= middle_element and Michaelis_constant == 0:
                            Michaelis_constant = collection[j][i][k][z][t]
                            break
#                    print('I am still working even after using break!!!!!!!!!!!!!!!')
                    for s in range(time_stamps):
#                        print('time step', t)
                        numerator_per_gamma_beta_rho_combination = maximum_element*(s+1)
                        denominator_per_gamma_beta_rho_combination = Michaelis_constant + s + 1
                        
                        # We're combining this way so that we can appropriately graph it accurately since
                        # the axis that we're graphing on is the reciprical
                        v = denominator_per_gamma_beta_rho_combination/numerator_per_gamma_beta_rho_combination
                        #print('simulation', j, '\ntime step', t, '\nvalue of v\n', v)
                        # This will have one element in it
                        print('max element ', maximum_element, '\n min element', minimum_element,
                              '\n distance', distance, '\n middle element', middle_element,
                              '\n numerator', denominator_per_gamma_beta_rho_combination,
                              '\n denominator', numerator_per_gamma_beta_rho_combination,
                              '\n all VNE', all_VNE_for_a_gamma_beta_rho_combination_per_simulation,
                              '\n and this is the value of v\n', v)
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
#####################################################################################
    
# next computation is to run it through with location = 100 and individual = 50
location_amount = 5
#location_amount = 6
individual_amount = 6
#individual_amount = 4
time_stamps = 5
#groups = 5
#gamma_values = [1, 3, 5, 7, 9]
#beta_values = np.linspace(1, 10, 19, endpoint = True)
#rho_values = np.linspace(0.1, 1.0, 10, endpoint=True)
#simulation_amount = 10
simulation_amount = 1

gamma_values = [2]
beta_values = [10]
rho_values = [1]

full_IxS_collection = []
full_IxI_collection = []
full_tripartite_collection = []
full_homerange_IxI_collection = []
full_SxS_collection = []
#full_binary_collection_IxS = []
#full_binary_collection_IxI = []

per_rho_inputs = list(range(1, time_stamps+1))
per_beta_inputs = [per_rho_inputs] * len(beta_values)
per_gamma_inputs = [per_beta_inputs] * len(gamma_values)
full_input_list = [per_gamma_inputs] * simulation_amount

# THIS WHOLE CHUNK IS TO CREATE THE PROJECTIONS!!!!
for j in range(simulation_amount):
    print(j, ' simulation')
    gamma_collection_IxS = []
    gamma_collection_IxI = []
    gamma_collection_tripartite = []
    gamma_collection_homerange_IxI = []
    gamma_collection_SxS = []


    for z in range(len(gamma_values)):
        print(z, ' gamma value which is ', gamma_values[z])
        # We only need to create the group preference vector once per gamma value so here it is
        group_preference_vector, individuals_in_each_group = creating_group_preference_vector(location_amount, individual_amount, gamma_values[z])

        beta_collection_IxS = []
        beta_collection_IxI = []
        beta_collection_tripartite = []
        beta_collection_homerange_IxI = []
        beta_collection_SxS = []

        for k in range(len(beta_values)):
            print(j+1, 'simulation value', z+1, 'gamma spot', k+1, ' beta value')
            # We only need to create the spatial preference vector once per beta values
            spatial_preference_vector = creating_spatial_preference_vector(location_amount, individual_amount, beta_values[k])

            rho_tripartite_collection = []
            rho_IxS_collection = []
            rho_IxI_collection = []
            rho_homerange_IxI_collection = []
            rho_SxS_collection = []
    #        IxS_binary_collection = []
    #        IxI_binary_collection = []

            
            # Both spatial and group locations are unaffected by the value of rho so
            # that is why it is the last for loop here.
            for y in range(len(rho_values)):
                #print(y, 'rho values which is', rho_values[y])
                # This is creating the initial placement of all individuals in the landscape in the
                # first time stamp based on the beta distribution. 
                # This is establishing location preference and NOT group
#                original_large_matrix, large_tripartite_list, large_alpha_vector = create_initial_matrix_with_beta_distribution(location_amount, individual_amount, beta_values[k], time_stamps)       
        #        original_large_matrix, large_tripartite_list = create_initial_matrix_using_rho(location_amount, individual_amount, rho_values[k], time_stamps, groups)
                # I think these next two lines are redundant
                #original_matrix, tripartite_list, alpha_vector = create_initial_matrix(location_amount, large_individual_amount, beta_values[k], time_stamps)    
                #print(original_large_matrix, large_tripartite_list)#, original_matrix)

                # This is the new code that will incorporate all changes and all 3 varying parameters.
                # Namely, beta, rho, and gamma; the spatial preference, group stickiness, and group preference.
                beginning_matrix = each_timestamp_matrix(location_amount, individual_amount, spatial_preference_vector, 
                                                         group_preference_vector, individuals_in_each_group, rho_values[y])

                # This is putting all of the IxS networks at each timestamp into one matrix so we can 
                # create the full, diagonal tripartite network as well
                large_tripartite_list = []                
                
                current_rho_IxS_collection = []
                current_rho_IxI_collection = []
                current_rho_tripartite_collection = []
                current_rho_homerange_IxI_collection = []
                current_rho_SxS_collection = []
                
                # This goes through and adds different 'snapshots' to our network depending on 
                # how many timestamps we are considering.
                for i in range(time_stamps):
#                    if i == 0:
#                        continue
#                    else:
#                        list_for_the_full_tripartite.append(#adding_to_the_matrix_using_beta(large_alpha_vector, location_amount, individual_amount,
                                                            #                                i+1, original_large_matrix, large_tripartite_list))
        #                large_tripartite_list.append(adding_to_the_matrix_using_rho(location_amount, individual_amount,
        #                                                                            i+1, original_large_matrix, large_tripartite_list, rho_values[k], groups))
 
                    # This is adding each timestamp to a list of all the IxS projections at each
                    # timestamp. We are calling the function 'each_timestamp_matrix' to give us
                    # the desired IxS implementing Gamma, Beta, and Rho.
                    large_tripartite_list.append(each_timestamp_matrix(location_amount, individual_amount, spatial_preference_vector,
                                                                       group_preference_vector, individuals_in_each_group, rho_values[y]))
                        
                    # This is creating the different projections based on the large tripartite list
                    # we got from above per timestamp.
                    IxS_matrix = IxS(large_tripartite_list, i+1, individual_amount, location_amount)
                    IxI_matrix = unipartite_square_matrix(large_tripartite_list, individual_amount, location_amount, i+1)
                    tripartite_matrix = assembling_tripartite(large_tripartite_list, individual_amount, location_amount, i+1)
                    homerange_matrix = Homerange_IxI(IxS_matrix, individual_amount, location_amount, i+1)
                    SxS_matrix = SxS_Matrix(IxS_matrix, location_amount, individual_amount)
                    
                    # This code was when we were considering the binary cases, but we didn't really
                    # notice that much of a difference so we are ignoring them currently.
        #            IxS_binary = binary_matrix(IxS_matrix)
        #            IxI_binary = binary_matrix(IxI_matrix)
                    
                    # This is computing the laplacian matrix for each of our above projections
                    IxS_laplacian = IxS_laplacian_matrix(IxS_matrix, i+1, individual_amount)
                    IxI_laplacian = unipartite_laplacian(IxI_matrix)
                    tripartite_laplacian = Tripartite_laplacian_matrix(tripartite_matrix, i+1, individual_amount)
                    homerange_IxI_laplacian = unipartite_laplacian(homerange_matrix)
                    SxS_laplacian = unipartite_laplacian(SxS_matrix)
                    
                    # Again, this code is for the binary case which we are not currently considering.
        #            IxS_binary_laplacian = tripartite_laplacian_matrix(IxS_binary, i+1, large_individual_amount)
        #            IxI_binary_laplacian = unipartite_laplacian(IxI_binary)
        
                    # This next step is calculating the aida value for each projection which we need
                    # to find the VNE of each projection            
                    IxS_aida, IxS_eigenvalues = cal_aida(IxS_laplacian)
                    IxI_aida, IxI_eigenvalues = cal_aida(IxI_laplacian)
                    tripartite_aida, tripartite_eigenvalues = cal_aida(tripartite_laplacian)
                    homerange_IxI_aida, homerange_eigenvalues = cal_aida(homerange_IxI_laplacian)
                    SxS_aida, SxS_eigenvalues = cal_aida(SxS_laplacian)
                    
                    # Again, binary case is not being considered currently.
        #            IxS_binary_aida, IxS_eigenvalues = cal_aida(IxS_binary_laplacian)
        #            IxI_binary_aida, IxI_eigenvalues = cal_aida(IxI_binary_laplacian)
                    
                    # This is calculating the VNE for each projection now
                    IxS_VNE = cal_von_entropy(IxS_aida)
                    IxI_VNE = cal_von_entropy(IxI_aida)
                    tripartite_VNE = cal_von_entropy(tripartite_aida)
                    homerange_IxI_VNE = cal_von_entropy(homerange_IxI_aida)
                    SxS_VNE = cal_von_entropy(SxS_aida)
        
                    # Binary case is not being considered.
        #            IxS_binary_VNE = cal_von_entropy(IxS_binary_aida)
        #            IxI_binary_VNE = cal_von_entropy(IxI_binary_aida)
                    
                    # This code is normalizing the VNE which I should have done in the VNE
                    # calculation anyway, but didnt.
                    IxS_normalized = normalize_von_entropy(IxS_VNE, IxS_laplacian.shape)
                    IxI_normalized = normalize_von_entropy(IxI_VNE, IxI_laplacian.shape)
                    tripartite_normalized = normalize_von_entropy(tripartite_VNE, tripartite_laplacian.shape)
                    homerange_IxI_normalized = normalize_von_entropy(homerange_IxI_VNE, homerange_IxI_laplacian.shape)
                    SxS_normalized = normalize_von_entropy(SxS_VNE, SxS_laplacian.shape)
        
                    # Binary not being considered.
        #            IxS_binary_normalized = normalize_von_entropy(IxS_binary_VNE, IxS_laplacian.shape)
        #            IxI_binary_normalized = normalize_von_entropy(IxI_binary_VNE, IxI_laplacian.shape)
        
                    # This is collecting the VNE for each simulation so we can graph them
                    current_rho_IxS_collection.append(IxS_normalized)
                    current_rho_IxI_collection.append(IxI_normalized)
                    current_rho_tripartite_collection.append(tripartite_normalized)
                    current_rho_homerange_IxI_collection.append(homerange_IxI_normalized)
                    current_rho_SxS_collection.append(SxS_normalized)
                    
#                print('all the VNE for a given gamma, beta, rho combination.\n This means they will have length equal to timestamps whch is 3', np.array(current_rho_IxS_collection).shape)
#                      '\n these are the 3 VNE values for our timestamps\n', current_rho_IxS_collection)
                # This next section collects all the VNE for a given rho value. If my rho vector
                # has a length of 5 then this collection should have 5 vectors inside of it
                # with length equal to the number of timestamps
                rho_IxS_collection.append(current_rho_IxS_collection)
                rho_IxI_collection.append(current_rho_IxI_collection)
                rho_tripartite_collection.append(current_rho_tripartite_collection)
                rho_homerange_IxI_collection.append(current_rho_homerange_IxI_collection)
                rho_SxS_collection.append(current_rho_SxS_collection)
        #        Tripartite_collection.append(Tripartite_normalized)
    #            IxS_binary_collection.append(IxS_binary_normalized)
    #            IxI_binary_collection.append(IxI_binary_normalized)
                
#            print('VNE for a given gamma and beta for ALL rho meaning this should have 10 vectors each with length equal to timestamps\n',
#                  np.array(rho_IxS_collection).shape)
            # This next section collects all the rho collections for each beta.
            # If the length of my beta vector is 20 and length of my rho vector is 5 then
            # these collections should have 20 vectors each with 5 vectors with length
            # equal to the number of timestamps.
            beta_collection_IxS.append(rho_IxS_collection)
            beta_collection_IxI.append(rho_IxI_collection)
            beta_collection_tripartite.append(rho_tripartite_collection)
            beta_collection_homerange_IxI.append(rho_homerange_IxI_collection)
            beta_collection_SxS.append(rho_SxS_collection)
    
            # Binary is not considered
    #        Full_binary_collection_IxS.append(IxS_binary_collection)
    #        Full_binary_collection_IxI.append(IxI_binary_collection)

            # I need to figure out what my new input vector will need to look like.
#            full_input_list.append(list(range(1, time_stamps + 1)))
            
#        print('VNE for a given gamma for ALL beta and ALL rho.\n This means we will have 20 vectors each with 10 vectors all with length equal to the amount of timestamps\n',
#              np.array(beta_collection_IxS).shape)
        
        # If my gamma vecor has length of 3 then each of these will have 3 vectors 
        # of 20 vectors, where each of those 20 vectors have 5 vectors where each of those
        # have length = timestamps
        gamma_collection_IxS.append(beta_collection_IxS)
        gamma_collection_IxI.append(beta_collection_IxI)
        gamma_collection_tripartite.append(beta_collection_tripartite)
        gamma_collection_homerange_IxI.append(beta_collection_homerange_IxI)
        gamma_collection_SxS.append(beta_collection_SxS)
        
        # Binary is not considered
    #    beta_binary_IxS_collection.append(Full_binary_collection_IxS)
    #    beta_binary_IxI_collection.append(Full_binary_collection_IxI)
        
    # If I'm running 50 simulations then I'll have 50 vectors that satisfy all the other
    # things in the previous collection.
#    print('VNE for ALL gamma, rho, and beta.\n This means that we will have 5 vectors, each with 20 vectors, each with 10 vectors, all with length equal to timestamps\n',
#          np.array(gamma_collection_IxS).shape)
    
    full_IxS_collection.append(gamma_collection_IxS)
    full_IxI_collection.append(gamma_collection_IxI)
    full_tripartite_collection.append(gamma_collection_tripartite)
    full_homerange_IxI_collection.append(gamma_collection_homerange_IxI)
    full_SxS_collection.append(gamma_collection_SxS)
    

### This block of code saves the simulation's array so we don't have to run the simulation every time
###np.save('IxS_beta_collection_50_time_10_simulations', beta_IxS_collection)
###np.save('IxI_beta_collection_50_time_10_simulations', beta_IxI_collection)
###np.save('full_tripartite_collection_100_time_10_simulations', beta_tripartite_collection)
###np.save('homerange_IxI_collection_100_time_10_simulations', beta_homerange_IxI_collection)
#np.save('IxS_iterating_through_all_3_knobs_where_location_25_individual_25_time_20_simulations_10', full_IxS_collection)
#np.save('IxI_iterating_through_all_3_knobs_where_location_25_individual_25_time_20_simulations_10', full_IxI_collection)
#np.save('full_tripartite_iterating_through_all_3_knobs_where_location_25_individual_25_time_20_simulations_10', full_tripartite_collection)
#np.save('homerange_IxI_iterating_through_all_3_knobs_where_location_25_individual_25_time_20_simulations_10', full_homerange_IxI_collection)
#np.save('SxS_iterating_through_all_3_knobs_where_location_25_individual_25_time_20_simulations_10', full_SxS_collection)
##print('beta_ IxS collection\n', beta_IxS_collection)
##np.save('IxS_binary_beta_collection_50_time_10_simulations', beta_binary_IxS_collection)
##np.save('IxI_binary_beta_collection_50_time_10_simulations', beta_binary_IxI_collection)
#
#np.save('full_input_list_for_location_25_individuals_25_time_20_simulations_10', full_input_list)


### This block of code loads a desired np.array so we don't have to run the simulations
#full_IxS_collection = np.load('IxS_iterating_through_all_3_knobs_where_location_100_individual_50_time_20_simulations_10.npy')
#full_IxI_collection = np.load('IxI_iterating_through_all_3_knobs_where_location_100_individual_50_time_20_simulations_10.npy')
#full_tripartite_collection = np.load('full_tripartite_iterating_through_all_3_knobs_where_location_100_individual_50_time_20_simulations_10.npy')
#full_homerange_IxI_collection = np.load('homerange_IxI_iterating_through_all_3_knobs_where_location_100_individual_50_time_20_simulations_10.npy')
#full_SxS_collection = np.load('SxS_iterating_through_all_3_knobs_where_location_100_individual_50_time_20_simulations_10.npy')
###beta_binary_IxS_collection = np.load('IxS_binary_beta_collection_50_time_10_simulations.npy')
###beta_binary_IxI_collection = np.load('IxI_binary_beta_collection_50_time_10_simulations.npy')
#
#full_input_list = np.load('full_input_list_for_100_location_50_individuals_20_time_10_simulations.npy')

#print('From left to right I have simulation\n gamma\n Beta\n and rho values\n and here is its shape\n', full_IxS_collection.shape)
#print('this is the IxS collection\n', full_IxS_collection)
#########################################################################
# This is my rough computations to find the information needed to get the
# Michaelis Constant
 
# Michaelis Constant for 10 locations, 10 individuals, 10 timestamps, and 10 simulations
# IxS = 2
# IxI = 2
# Tripartite = 2
# Homerange IxI = 3
# SxS = 2

# Michaelis Constant for 50 locations, 10 individuals, 10 timestamps, and 10 simulations
# IxS = 2
# IxI = 2
# Tripartite = 3
# Homerange IxI = 2
# SxS = 

# Michaelis Constant for 100 locations, 10 individuals, 10 timestamps, and 10 simulations
# IxS = 3
# IxI = 2
# Tripartite = 3
# Homerange IxI = 2
# SxS = 

# Michaelis Constant for 10 locations, 50 individuals, 10 timestamps, and 10 simulations
# IxS = 2
# IxI = 2
# Tripartite = 2
# Homerange IxI = 2
# SxS = 

# Michaelis Constant for 50 locations, 50 individuals, 10 timestamps, and 10 simulations
# IxS = 2
# IxI = 2
# Tripartite = 2
# Homerange IxI = 2
# SxS = 

# Michaelis Constant for 100 locations, 50 individuals, 10 timestamps, and 10 simulations
# IxS = 2; check this array
# IxI = 2
# Tripartite = 3
# Homerange IxI = 2; Homerange IxI array needs to be looked at
# SxS = 

# Michaelis Constant for 10 locations, 100 individuals, 10 timestamps, and 10 simulations
# IxS = 2; maybe look at this one....it might not be needed though
# IxI = 2
# Tripartite = 2
# Homerange IxI = 2; This needs to be looked at closely
# SxS = 

# Michaelis Constant for 50 locations, 100 individuals, 10 timestamps, and 10 simulations
# IxS = 2; maybe needs to be checked?
# IxI = 2
# Tripartite = 3
# Homerange IxI = 2; a couple cases where it decreases
# SxS = 

# Michaelis Constant for 100 locations, 100 individuals, 10 timestamps, and 10 simulations
# IxS = 2; one case where the thousandth integer was decreasing after while
# IxI = 2
# Tripartite = 3
# Homerange IxI = 2
# SxS = 

###########################################################################################
# This is the Rho the transformations with the rho values. 
###########################################################################################
# Michaelis Constant for 10 locations, 10 individuals, 10 timestamps, and 10 simulations
# IxS = 2
# IxI = 2
# Tripartite = 2
# Homerange IxI = 2
# SxS = 2

# Michaelis Constant for 50 locations, 10 individuals, 10 timestamps, and 10 simulations
# IxS = 3
# IxI = 2
# Tripartite = 3
# Homerange IxI = 3
# SxS = 3

# Michaelis Constant for 100 locations, 10 individuals, 10 timestamps, and 10 simulations
# IxS = 3
# IxI = 2
# Tripartite = 3
# Homerange IxI = 2
# SxS = 3

# Michaelis Constant for 10 locations, 50 individuals, 10 timestamps, and 10 simulations
# IxS = 2
# IxI = 2
# Tripartite = 3
# Homerange IxI = 2
# SxS = 2

# Michaelis Constant for 50 locations, 50 individuals, 10 timestamps, and 10 simulations
# IxS = 3
# IxI = 3
# Tripartite = 3
# Homerange IxI = 2
# SxS = 2

# Michaelis Constant for 100 locations, 50 individuals, 10 timestamps, and 10 simulations
# IxS = 3
# IxI = 2
# Tripartite = 3
# Homerange IxI = 3
# SxS = 3

# Michaelis Constant for 10 locations, 100 individuals, 10 timestamps, and 10 simulations
# IxS = 2
# IxI = 2
# Tripartite = 3
# Homerange IxI = 2
# SxS = 2

# Michaelis Constant for 50 locations, 100 individuals, 10 timestamps, and 10 simulations
# IxS = 3
# IxI = 2
# Tripartite = 3
# Homerange IxI = 2
# SxS = 2

# Michaelis Constant for 100 locations, 100 individuals, 10 timestamps, and 10 simulations
# IxS = 3
# IxI = 2
# Tripartite = 3
# Homerange IxI = 2
# SxS = 3

###########################################################################################
# This will calculate the MIchaelis constant so we can find an estimated line and 
# use the Lineweaver Burke Linearization
###########################################################################################
## These are the calculated Michaleis outputs for the data
#outputs_IxS = calc_V(full_IxS_collection, time_stamps, simulation_amount, 1)            
#outputs_IxI = calc_V(full_IxI_collection, time_stamps, simulation_amount, 2)            
#outputs_tripartite = calc_V(full_tripartite_collection, time_stamps, simulation_amount, 3)
#outputs_homerange_IxI = calc_V(full_homerange_IxI_collection, time_stamps, simulation_amount, 4)
outputs_SxS = calc_V(full_SxS_collection, time_stamps, simulation_amount, 5)
#
## This is computing the Michaelis inputs for each output already calculated
#input_list = list(range(1, time_stamps + 1))
##print('this is the input list\n', input_list)
#one_vector = np.ones(time_stamps)
#michaelis_inputs = np.divide(one_vector, input_list)
#
##print('this the full output IxS\n', np.array(full_IxS_collection).shape)
##print('and this is the outputs after the computing my v matrix\n', np.array(outputs_IxS).shape)
#
## These save the arrays so I can call it in another file.
#np.save('Michaelis_inputs_location_10_individual_10', michaelis_inputs)
#np.save('IxS_outputs_location_10_individual_10', outputs_IxS)
#np.save('IxI_outputs_location_10_individual_10', outputs_IxI)
#np.save('Tripartite_outputs_location_10_individual_10', outputs_tripartite)
#np.save('homerange_IxI_outputs_location_10_individual_10', outputs_homerange_IxI)
#np.save('SxS_outputs_location_10_individual_10', outputs_SxS)
##print('Michaelis inputs\n', Michaelis_inputs)

############################################################################################
# This is me normalizing the data before graphing it
############################################################################################
#log_beta_IxS_collection = np.log(beta_IxS_collection)
#log_beta_IxI_collection = np.log(beta_IxI_collection)
#log_beta_Tripartite_collection = np.log(beta_tripartite_collection)
#log_beta_homerange_IxI_collection = np.log(beta_homerange_IxI_collection)
#log_beta_binary_IxS_collection = np.log(beta_binary_IxS_collection)
#log_beta_binary_IxI_collection = np.log(beta_binary_IxI_collection)

#################################################################################
## This section is me setting up the paramters for graphing
#################################################################################
#SMALL_SIZE = 8
#MEDIUM_SIZE = 16
#BIGGER_SIZE = 24
#
#color = ['r', 'g', 'b']
#true_color = color*simulation_amount*time_stamps
# 
# # This is us creating our labels for our legend so we can understand the data we have
#red_patch = mpatches.Patch(color='r', label='1')
#black_patch = mpatches.Patch(color='k', label='9')
#green_patch = mpatches.Patch(color = 'g', label = '5')
#
#plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
#plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
#plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
#plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
#plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
#plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
#plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure title
#

#############################################################################
## This chunk of code is giving me the MIchaelis transformation plotted out
## They are plotted out based on the beta value variations
#############################################################################
#colors = ['r', 'g', 'k']
#for j in range(3):
#    plt.figure()
#    for i in range(simulation_amount):
#        plt.scatter(michaelis_inputs, outputs_IxS[i][j], c = colors[j])
#    plt.title('Lineweaver-Burk Linearization with I x S\n time stamps being 50 and running 10 simulations per rho')
#    plt.yticks(np.linspace(1.0, 6.0, 11, endpoint=True))
#    plt.xlabel('1/S which is 1/Time Stamps in our case')
#    plt.ylabel('1/v')
#    plt.legend(handles = [red_patch, green_patch, black_patch], loc = 'lower right', title = 'Beta Values')
#    plt.show()
#
#plt.figure()
#for i in range(simulation_amount):
#    plt.scatter(michaelis_inputs, outputs_IxI[i][0], c = 'r')
#    plt.plot(michaelis_inputs, outputs_IxI[i][1], c = 'g')
#    plt.plot(michaelis_inputs, outputs_IxI[i][2], c = 'k')
#
#plt.title('Lineweaver-Burk Linearization with I x I\n time stamps being 50 and running 10 simulations per rho')
#plt.yticks(np.linspace(1.0, 6.0, 11, endpoint=True))
#plt.xlabel('Time Stamps')
#plt.ylabel('VNE')
#plt.legend(handles = [red_patch, green_patch, black_patch], loc = 'lower right', title = 'Beta Values')
#
#plt.show()
###########################################################################################
# This chunk of code is just graphing the raw VNE data; before the MIchaleis transformation
###########################################################################################
#colors = ['r', 'g', 'k', 'b', 'y']
#
#red_patch = mpatches.Patch(color='r', label='IxS')
#green_patch = mpatches.Patch(color = 'g', label = 'IxI')
#black_patch = mpatches.Patch(color='k', label='Tripartite')
#blue_patch = mpatches.Patch(color='b', label='Homerange IxI')
#yellow_patch = mpatches.Patch(color='y', label='SxS')
#
#fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8), (ax9, ax10, ax11, ax12)) = plt.subplots(3,4)
#axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12]
#for i in range(simulation_amount):
#    # This is for the IxS VNE
#    axes[i].scatter(full_input_list[0], beta_IxS_collection[i][0], c = colors[0])
#    axes[i].scatter(full_input_list[0], beta_IxI_collection[i][0], c = colors[1])
#    axes[i].scatter(full_input_list[0], beta_tripartite_collection[i][0], c = colors[2])
#    axes[i].scatter(full_input_list[0], beta_homerange_IxI_collection[i][0], c = colors[3])
#    axes[i].scatter(full_input_list[0], beta_SxS_collection[i][0], c = colors[4])
#    axes[i].set_title('Beta = 1 for all projections')
#    #plt.yticks(np.linspace(1.0, 6.0, 11, endpoint=True))
#    axes[i].set_xlabel('Time Stamps')
#    axes[i].set_ylabel('VNE')
#    axes[i].legend(handles = [red_patch, green_patch, black_patch, blue_patch, yellow_patch], loc = 'lower right', title = 'Projections')
#    axes[i].grid(True)
#    plt.show()

#for i in range(simulation_amount):    
#    # This is for the IxI VNE
#    axes[i].scatter(full_input_list[0], beta_IxS_collection[i][1], c = colors[0])
#    axes[i].scatter(full_input_list[0], beta_IxI_collection[i][1], c = colors[1])
#    axes[i].scatter(full_input_list[0], beta_tripartite_collection[i][1], c = colors[2])
#    axes[i].scatter(full_input_list[0], beta_homerange_IxI_collection[i][1], c = colors[3])
#    axes[i].scatter(full_input_list[0], beta_SxS_collection[i][1], c = colors[4])
#    axes[i].set_title('Beta = 3 for all projections')
#    #plt.yticks(np.linspace(1.0, 6.0, 11, endpoint=True))
#    axes[i].set_xlabel('Time Stamps')
#    axes[i].set_ylabel('VNE')
#    axes[i].legend(handles = [red_patch, green_patch, black_patch, blue_patch, yellow_patch], loc = 'lower right', title = 'Projections')
#    axes[i].grid(True)
#    plt.show()
    
#for i in range(simulation_amount):
#    # This is the for the tripartite VNE
#    axes[i].scatter(full_input_list[0], beta_IxS_collection[i][2], c = colors[0])
#    axes[i].scatter(full_input_list[0], beta_IxI_collection[i][2], c = colors[1])
#    axes[i].scatter(full_input_list[0], beta_tripartite_collection[i][2], c = colors[2])
#    axes[i].scatter(full_input_list[0], beta_homerange_IxI_collection[i][2], c = colors[3])
#    axes[i].scatter(full_input_list[0], beta_SxS_collection[i][2], c = colors[4])
#    axes[i].set_title('Beta = 5 for all projections')
#    #plt.yticks(np.linspace(1.0, 6.0, 11, endpoint=True))
#    axes[i].set_xlabel('Time Stamps')
#    axes[i].set_ylabel('VNE')
#    axes[i].legend(handles = [red_patch, green_patch, black_patch, blue_patch, yellow_patch], loc = 'lower right', title = 'Projections')
#    axes[i].grid(True)
#    plt.show()
    
#for i in range(simulation_amount):
#    # This is for the homerange_IxI VNE
#    axes[i].scatter(full_input_list[0], beta_IxS_collection[i][3], c = colors[0])
#    axes[i].scatter(full_input_list[0], beta_IxI_collection[i][3], c = colors[1])
#    axes[i].scatter(full_input_list[0], beta_tripartite_collection[i][3], c = colors[2])
#    axes[i].scatter(full_input_list[0], beta_homerange_IxI_collection[i][3], c = colors[3])
#    axes[i].scatter(full_input_list[0], beta_SxS_collection[i][3], c = colors[4])
#    axes[i].set_title('Beta = 7 for all projections')
#    #plt.yticks(np.linspace(1.0, 6.0, 11, endpoint=True))
#    axes[i].set_xlabel('Time Stamps')
#    axes[i].set_ylabel('VNE')
#    axes[i].legend(handles = [red_patch, green_patch, black_patch, blue_patch, yellow_patch], loc = 'lower right', title = 'Projections')
#    axes[i].grid(True)
#    plt.show()
#    
#for i in range(simulation_amount):
#    axes[i].scatter(full_input_list[0], beta_IxS_collection[i][4], c = colors[0])
#    axes[i].scatter(full_input_list[0], beta_IxI_collection[i][4], c = colors[1])
#    axes[i].scatter(full_input_list[0], beta_tripartite_collection[i][4], c = colors[2])
#    axes[i].scatter(full_input_list[0], beta_homerange_IxI_collection[i][4], c = colors[3])
#    axes[i].scatter(full_input_list[0], beta_SxS_collection[i][4], c = colors[4])
#    axes[i].set_title('Beta = 9 for all projections')
#    #plt.yticks(np.linspace(1.0, 6.0, 11, endpoint=True))
#    axes[i].set_xlabel('Time Stamps')
#    axes[i].set_ylabel('VNE')
#    axes[i].legend(handles = [red_patch, green_patch, black_patch, blue_patch, yellow_patch], loc = 'lower right', title = 'Projections')
#    axes[i].grid(True)
#    plt.show()

###############################################################################
## This chunk of code is just graphing the log data; before the MIchaleis transformation
#############################################################################

#log_inputs = np.log(input_list)
#plt.figure()
#for i in range(simulation_amount):
#    plt.plot(log_inputs, beta_IxS_collection[i][0], c = 'r')
#    plt.plot(log_inputs, beta_IxS_collection[i][1], c = 'g')
#    plt.plot(log_inputs, beta_IxS_collection[i][2], c = 'k')
#
#plt.title('I x S with just inputs logged\n time stamps being 50 and running 10 simulations per rho')
##plt.yticks(np.linspace(1.0, 6.0, 11, endpoint=True))
#plt.xlabel('Time Stamps')
#plt.ylabel('VNE')
#plt.legend(handles = [red_patch, green_patch, black_patch], loc = 'lower right', title = 'Beta Values')
#
#plt.show()
#
#plt.figure()
#for i in range(simulation_amount):
#    plt.plot(log_inputs, beta_IxI_collection[i][0], c = 'r')
#    plt.plot(log_inputs, beta_IxI_collection[i][1], c = 'g')
#    plt.plot(log_inputs, beta_IxI_collection[i][2], c = 'k')
#
#plt.title('I x I with just inputs logged\n time stamps being 50 and running 10 simulations per rho')
##plt.yticks(np.linspace(1.0, 6.0, 11, endpoint=True))
#plt.xlabel('Time Stamps')
#plt.ylabel('VNE')
#plt.legend(handles = [red_patch, green_patch, black_patch], loc = 'lower right', title = 'Beta Values')
#
#plt.show()
###############################################################################
#
#############################################################################
## This chunk of code is just graphing the data without normalizing it; before the MIchaleis transformation
#plt.figure()
#for i in range(simulation_amount):
#    plt.plot(input_list, beta_IxS_collection[i][0], c = 'r')
#    plt.plot(input_list, beta_IxS_collection[i][1], c = 'g')
#    plt.plot(input_list, beta_IxS_collection[i][2], c = 'k')
#
#plt.title(' I x S data without normalizing\n time stamps being 50 and running 10 simulations per rho')
##plt.yticks(np.linspace(1.0, 6.0, 11, endpoint=True))
#plt.xlabel('Time Stamps')
#plt.ylabel('VNE')
#plt.legend(handles = [red_patch, green_patch, black_patch], loc = 'lower right', title = 'Beta Values')
#
#plt.show()
#
#plt.figure()
#for i in range(simulation_amount):
#    plt.plot(input_list, beta_IxI_collection[i][0], c = 'r')
#    plt.plot(input_list, beta_IxI_collection[i][1], c = 'g')
#    plt.plot(input_list, beta_IxI_collection[i][2], c = 'k')
#
#plt.title('I x S data without normalizing\n time stamps being 50 and running 10 simulations per rho')
##plt.yticks(np.linspace(1.0, 6.0, 11, endpoint=True))
#plt.xlabel('Time Stamps')
#plt.ylabel('VNE')
#plt.legend(handles = [red_patch, green_patch, black_patch], loc = 'lower right', title = 'Beta Values')
#
#plt.show()
##############################################################################



#plt.subplot(221)
#for i in range(simulation_amount):
#    plt.title('Both full and binary IxS VNE;\n 1 - 100 Time Stamps; beta = 5\n Locations = Individuals')
#    #plt.yticks(np.linspace(0.7, 0.95, 6, endpoint=True))
#    plt.xlabel('Time Stamps')
#    plt.ylabel('VNE')
#    plt.legend(handles = [red_patch, black_patch], loc = 'lower right')
#
#
#plt.subplot(222)
#for i in range(simulation_amount):
#    plt.title('Both full and binary IxS VNE;\n 1 - 100 Time Stamps; beta = 9\n Locations = Individuals')
#    #plt.yticks(np.linspace(0.7, .95, 6, endpoint=True))
#    plt.xlabel('Time Stamps')
#    plt.ylabel('VNE')
#    plt.legend(handles = [red_patch, black_patch], loc = 'lower right')

#
#for i in range(simulation_amount):
#    plt.scatter(input_list, log_beta_binary_IxS_collection[i][0], c = 'k')
#
#for i in range(simulation_amount):
#    plt.scatter(input_list, log_beta_binary_IxS_collection[i][1], c = 'k')
#
#for i in range(simulation_amount):
#    plt.scatter(input_list, log_beta_binary_IxS_collection[i][2], c = 'k')
#
############################################################################################################
#plt.figure()
#plt.subplot(221)
#for i in range(simulation_amount):
#for i in range(simulation_amount):
#    plt.scatter(input_list, log_beta_binary_IxI_collection[i][0], c = 'k')
#plt.title('Both full and binary IxI VNE;\n 1 - 100 Time Stamps; beta = 1\n Locations = Individuals')
##plt.yticks(np.linspace(0.75, 1.05, 6, endpoint=True))
#plt.xlabel('Time Stamps')
#plt.ylabel('VNE')
#plt.legend(handles = [red_patch, black_patch], loc = 'lower right')
#
#plt.subplot(222)
#for i in range(simulation_amount):
#    plt.scatter(input_list, log_beta_IxI_collection[i][1], c = 'r')
#for i in range(simulation_amount):
#    plt.scatter(input_list, log_beta_binary_IxI_collection[i][1], c = 'k')
#plt.title('Both full and binary IxI VNE;\n 1 - 100 Time Stamps; beta = 5\n Locations = Individuals')
##plt.yticks(np.linspace(0.75, 1.05, 6, endpoint=True))
#plt.xlabel('Time Stamps')
#plt.ylabel('VNE')
#plt.legend(handles = [red_patch, black_patch], loc = 'lower right')
#
#
#plt.subplot(223)
#for i in range(simulation_amount):
#    plt.scatter(input_list, log_beta_IxI_collection[i][2], c = 'r')
#for i in range(simulation_amount):
#    plt.scatter(input_list, log_beta_binary_IxI_collection[i][2], c = 'k')
#plt.title('Both full and binary IxI VNE;\n 1 - 100 Time Stamps; beta = 9\n Locations = Individuals')
##plt.yticks(np.linspace(0.75, 1.05, 6, endpoint=True))
#plt.xlabel('Time Stamps')
#plt.ylabel('VNE')
#plt.legend()
#plt.legend(handles = [red_patch, black_patch], loc = 'lower right')
#
#plt.show()
#
#############################################################################################################
#
#
#plt.subplot(222)
#plt.scatter(full_input_list[:,1], beta_IxS_collection[:,1], c = 'g')
#plt.title('IxI VNE from 2 - 101 Time Stamps varying beta 1, 5, and 9')
#plt.yticks([0.7, 1.0, 1.25, 1.5, 1.75])
#plt.xlabel('Time Stamps')
#plt.ylabel('VNE')
#plt.legend(handles = green_patch, title = 'beta values',
#           loc = 'lower right')
#
#plt.subplot(223)
#plt.scatter(full_input_list[:,2], beta_IxS_collection[:,2], c = 'b')
#plt.title('IxS beta = 9 VNE from 2 - 101 Time Stamps')
#plt.yticks([0.7, 1.0, 1.25, 1.5, 1.75])
#plt.xlabel('Time Stamps')
#plt.ylabel('VNE')
#plt.legend(handles = blue_patch, title = 'beta values',
#           loc = 'lower right')
#
#
#
#plt.subplot(224)
#plt.scatter(full_input_list, beta_binary_IxI_collection, c = true_color)
#plt.title('IxI binary case VNE from 2 - 101 Time Stamps varying beta 1, 5, and 9')
##plt.yticks([0.75, 1.0, 1.25, 1.5, 1.75])
#plt.xlabel('Time Stamps')
#plt.ylabel('VNE')
#plt.legend(handles = [red_patch, green_patch, blue_patch], title = 'beta values',
#           loc = 'lower right')
#
