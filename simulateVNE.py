import numpy as np
from scipy.stats import beta
from scipy.stats import dirichlet
from scipy import linalg

##############################################################################################
# This is creating the spatial preference vector used to determine group and individual
# placement in the simulations.
##############################################################################################
def creating_spatial_preference_vector(number_of_locations, number_of_individuals, beta_value):
    # This chunk defines the beta distribution
    a, b = 1.0, beta_value
    x = np.linspace(0.0, 1.0, number_of_locations)
    alpha_for_dirichlet = beta.pdf(x,a,b)
        
    # This uses the defined beta distribution to pull from a dirichlet distribution
    for i in range(number_of_locations):
        if alpha_for_dirichlet[i] == 0.0:
            alpha_for_dirichlet[i] = 0.000000001
    vector = dirichlet.rvs(alpha_for_dirichlet, size=1)
    
    # This is our spatial preference vector
    return(vector)

##############################################################################################
# This is the group preference vector used to assign individuals to groups
##############################################################################################
def creating_group_preference_vector(number_of_locations, number_of_individuals, gamma_value):
    
    # Establishing the parameters used to create a dirichlet distribution
    a, b = 1.0, gamma_value
    
    # This creates the amount of 'groups' which are the same as the number of locations
    x = np.linspace(0.0, 1.0, number_of_locations)

    # This creates the dirichlet distribution
    alpha_for_dirichlet = beta.pdf(x,a,b)

    # This guarantees that we are not dividing by zero
    for i in range(number_of_locations):
        if alpha_for_dirichlet[i] == 0.0:
            alpha_for_dirichlet[i] = 0.000000001

    # This is creating the actual group preference vector
    group_preference_vector = dirichlet.rvs(alpha_for_dirichlet, size=1)

    # This assigns an amount of indidividuals to each of our groups based on the group preference vector
    individuals_in_each_group = np.zeros(number_of_locations)
    for i in range(number_of_individuals):
        j = np.random.choice(number_of_locations, 1, p = group_preference_vector[0])
        individuals_in_each_group[j] += 1
    
    return(group_preference_vector, individuals_in_each_group)

##############################################################################################
# This is the meat. This calculates the biadjacency matrices and constructs
# each network for a particular time step
##############################################################################################
def each_timestamp_matrix(number_of_locations, number_of_individuals, spatial_preference_vector,
                          group_preference_vector, individuals_in_each_group, rho):

    # I need to assign each group to a location; this code will give me the indices
    # that are non-zero. So individuals_in_each_group with the indices tells me what
    # groups to use.
    indices_of_interest = np.nonzero(individuals_in_each_group)
    indices_of_interest = indices_of_interest[0]
    
    # This will be what we're adding to until we're done and then we'll stack each matrix
    # on top of each other to get our biadjacency matrix.
    matrices_for_each_group = []
    
    # This calculates the amount of spaces that are non-zero
    amount_of_zeros = 0
    for i in range(number_of_locations):
        if spatial_preference_vector[0][i] == 0:
            amount_of_zeros += 1
    non_zero_entries = number_of_locations - amount_of_zeros

    # This determines is we can use sampling without replacement or not
    if non_zero_entries < len(indices_of_interest):
        all_locations_for_the_groups_without_replacement = np.random.choice(number_of_locations, len(indices_of_interest), replace =True, p = spatial_preference_vector[0])
    else:
        all_locations_for_the_groups_without_replacement = np.random.choice(number_of_locations, len(indices_of_interest), replace =False, p = spatial_preference_vector[0])
        
    # This is the same length as the number of groups that are 'filled'
    # a.k.a this is the amount of groups we are considering
    for i in range(len(all_locations_for_the_groups_without_replacement)):
        current_group_index = indices_of_interest[i]
        amount_of_individuals_in_group = individuals_in_each_group[current_group_index]
        groups_location = all_locations_for_the_groups_without_replacement[i]
        groups_matrix = np.zeros((int(amount_of_individuals_in_group), number_of_locations))
        
        
        # This next little bit determines our group affinity vector.
        rho_vector = np.zeros(number_of_locations)
        rho_vector[groups_location] = rho
        
        # This assigns each location a probability value
        for j in range(number_of_locations):
            if rho_vector[j] == rho:
                continue
            else:
                rho_vector[j] = (1-rho)
                    
        complete_stickiness_vector = rho_vector * spatial_preference_vector[0]
        normalized_stickiness_vector = complete_stickiness_vector/sum(complete_stickiness_vector)
        
        # Having constructed the group affinity vector, we're going to assign each individual
        # to a location now.
        location_of_individuals = np.random.choice(number_of_locations, int(amount_of_individuals_in_group), p = normalized_stickiness_vector)
        
        # Create a vector based on the individuals placement.
        for k in range(int(amount_of_individuals_in_group)):
            actual_location = location_of_individuals[k]
            groups_matrix[k][actual_location] = 1
        matrices_for_each_group.append(groups_matrix)
    
    # We finish up by stacking all the matrices on top of each other.
    full_timestamp_matrix = np.vstack(matrices_for_each_group)
        
    return(full_timestamp_matrix)
#############################################################################################
# This constructs the tripartite network using the bi adjacency matrices
#############################################################################################
def assembling_tripartite(list_of_tripartites, individual_amount, location_amount, time_stamp):
    if time_stamp != 1:
        collection_of_rows_of_tripartite = []
        for i in range(time_stamp):
            row_of_matrix = np.zeros((time_stamp, individual_amount, location_amount))
            row_of_matrix[i] = list_of_tripartites[i]
            collection_of_rows_of_tripartite.append(np.hstack(row_of_matrix))
        full_tripartite_matrix = np.vstack(collection_of_rows_of_tripartite)
        return(full_tripartite_matrix)
    else:
        return(list_of_tripartites[0])
##############################################################################################
# This creates our IxS network using the bi adjacency matrices
##############################################################################################
def IxS(list_of_tripartites, time_stamps, individual_amount, location_amount):
    full_matrix = np.zeros((individual_amount, location_amount))
    # Entry wise matrix summation using the list of biadjacency matrices
    for i in range(time_stamps):
        full_matrix += list_of_tripartites[i]
    return(full_matrix)

##############################################################################################
# This creates our IxI association network using the bi adjacency matrices    
##############################################################################################
def unipartite_square_matrix(collection_of_matrices, individuals, 
                             locations, time_stamps):
    
    # This is the empty IxI matrix
    ind_matrix = np.zeros((individuals, individuals))
    # We need to iterate through our collection of matrices; hence we start a for loop
    # going through the number of timestamps
    for i in range(time_stamps):
        # Only consider the ith matrix in our list of matrices
        matrix = collection_of_matrices[i]
        # We now compare pairings of individuals to see if there are relationships
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
                            continue
    return ind_matrix

##################################################################################
# This calculates the SxS network using the IxS network we already computed
##################################################################################
def SxS_Matrix(IxS, locations, individuals):
    
    #This creates an exmpty SxS matrix for me to fill
    SxS = np.zeros((locations, locations))
    
    # I need to iterate through the I x S matrix so I start with the locations
    for i in range(locations):
        
        # Grabbing the first column of my IxS matrix my 'starting' column
        first_column = IxS[:,i]    
        # Creating an empty array to note the amount of overlap there is in locations
            
        # Iterating through all the individuals now
        for j in range(locations):

            # We dont want to compare the same location with itself
            if i == j:
                continue
            else:
                # This is the 'next' column we are working with
                next_column = IxS[:,j]

            # this is to sum the amount of connections between locations
            node_of_interest = 0
                
            # Now going through all the locations from the columns that I've marked
            for k in range(individuals):
                
                # If the certain 'location' is being inhabited in both arrays then investigate
                if first_column[k] > 0 and next_column[k] > 0:
                    node_of_interest += 1                    
                
            # This is me filling in the SxS matrix with the summed up connections between locations
            SxS[i][j] = node_of_interest
            SxS[j][i] = node_of_interest
            
    return(SxS)
##################################################################################
# This constructs the Homerange network using the IxS network already constructed
# Note, each entry will be a ratio unlike all the other networks.
##################################################################################
def Homerange_IxI(IxS_matrix, individuals,
                  locations, time_stamps):
    # creates an empty IxI Matrix
    ind_matrix = np.zeros((individuals, individuals))
    
    # Going through each row of our IxS Matrix
    for i in range(individuals):
        firstIndividual = IxS_matrix[i]        
        for k in range(individuals):
            if i == k:
                continue
            else:
                secondIndividual = IxS_matrix[k]
            nodeOfInterest = 0
            
            # This creates the value which we will be dividing by giving us a ratio
            array = np.nonzero(secondIndividual)
            dividingValue = len(array[0])

            # Makes sure that there both indidivuals are present to reflect a relationship
            for j in range(locations):
                if firstIndividual[j] == 0 or secondIndividual[j] == 0:
                    continue
                if firstIndividual[j] > 0 and secondIndividual[j] > 0:
                    nodeOfInterest += 1
                
                # Seeing how many locations were inhabited by the pairing of individuals
                # then dividing by the amount of inhabited locations by individual 1
                # creating our ratio
                nodeValue = nodeOfInterest/dividingValue
                ind_matrix[i][k] = nodeValue

    return(ind_matrix)
##################################################################################
# Used to compute the VNE for each network
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

    return aida, eigenvalues

###################################################################################
# Functions computing the VNE
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
        if aida[i] <= 0.0 or imaginary == True:
            p += 0
    
        # This is the formula being calculated.
        else:
            p += aida[i]*np.log(aida[i])
    
    # This is the final portion of our calculation
    von_entropy_summed = -p
    
    return von_entropy_summed

###########################################################################################################
# This normalizes the von neumann entropy allowing for netowkrs to be compared
# with one another of differing dimnesions
###########################################################################################################
def normalize_von_entropy(von_neumann_entropy_vector, dimension):
    von_neumann_entropy_value = von_neumann_entropy_vector/np.log(dimension[0])
    return von_neumann_entropy_value

#############################################################################################################
# Laplacian construction for unipartite networks
#############################################################################################################
def unipartite_laplacian(matrix):
    diag_vals = []
    shape_of_matrix = matrix.shape
    
    
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

#####################################################################################
# This method is the for loop that calculates the VNE for all the time steps
#####################################################################################
def maxValue(IxS, IxI, Tripartite, Homerange, SxS):
    fullList = [IxS, IxI, Tripartite, Homerange, SxS]
    for i in fullList:
        maximum = np.amax(i)
        if maximum < 0.0002 or maximum > 100:
            return False
            break
    return True

#####################################################################################
# This method is the for loop that calculates the VNE for all the time steps
#####################################################################################
def timeSteps(locationAmount, individualAmount, spatialPreferenceVector, groupPreferenceVector,
              indEachGroup, rhoValue):
    largeTripartiteList = []
    truth = False
    if truth == False:
        IxS = []
        IxI = []
        Tripartite = []
        Homerange = []
        SxS = []
        for i in range(time_stamps):
            largeTripartiteList.append(each_timestamp_matrix(locationAmount, individualAmount, spatialPreferenceVector,
                                                               groupPreferenceVector, indEachGroup, rhoValue))
                
            # This is creating the different projections based on the large tripartite list
            # we got from above per timestamp.
            IxSMatrix = IxS(largeTripartiteList, i+1, individualAmount, locationAmount)
            IxIMatrix = unipartite_square_matrix(largeTripartiteList, individualAmount, locationAmount, i+1)
            tripartiteMatrix = assembling_tripartite(largeTripartiteList, individualAmount, locationAmount, i+1)
            homerangeMatrix = Homerange_IxI(IxSMatrix, individualAmount, locationAmount, i+1)
            SxSMatrix = SxS_Matrix(IxSMatrix, locationAmount, individualAmount)
            
            # This code was when we were considering the binary cases, but we didn't really
            # notice that much of a difference so we are ignoring them currently.
            
            # This is computing the laplacian matrix for each of our above projections
            IxS_laplacian = IxS_laplacian_matrix(IxSMatrix, i+1, individualAmount)
            IxI_laplacian = unipartite_laplacian(IxIMatrix)
            tripartite_laplacian = Tripartite_laplacian_matrix(tripartiteMatrix, i+1, individualAmount)
            homerange_IxI_laplacian = unipartite_laplacian(homerangeMatrix)
            SxS_laplacian = unipartite_laplacian(SxSMatrix)
                    
            # This next step is calculating the aida value for each projection which we need
            # to find the VNE of each projection            
            IxS_aida, IxS_eigenvalues = cal_aida(IxS_laplacian)
            IxI_aida, IxI_eigenvalues = cal_aida(IxI_laplacian)
            tripartite_aida, tripartite_eigenvalues = cal_aida(tripartite_laplacian)
            homerange_IxI_aida, homerange_eigenvalues = cal_aida(homerange_IxI_laplacian)
            SxS_aida, SxS_eigenvalues = cal_aida(SxS_laplacian)
            
            # This is calculating the VNE for each projection now
            IxS_VNE = cal_von_entropy(IxS_aida)
            IxI_VNE = cal_von_entropy(IxI_aida)
            tripartite_VNE = cal_von_entropy(tripartite_aida)
            homerange_IxI_VNE = cal_von_entropy(homerange_IxI_aida)
            SxS_VNE = cal_von_entropy(SxS_aida)
            
            # This code is normalizing the VNE which I should have done in the VNE
            # calculation anyway, but didnt.
            IxS_normalized = normalize_von_entropy(IxS_VNE, IxS_laplacian.shape)
            IxI_normalized = normalize_von_entropy(IxI_VNE, IxI_laplacian.shape)
            tripartite_normalized = normalize_von_entropy(tripartite_VNE, tripartite_laplacian.shape)
            homerange_IxI_normalized = normalize_von_entropy(homerange_IxI_VNE, homerange_IxI_laplacian.shape)
            SxS_normalized = normalize_von_entropy(SxS_VNE, SxS_laplacian.shape)
            
            IxS.append(IxS_normalized)
            IxI.append(IxI_normalized)
            Tripartite.append(tripartite_normalized)
            Homerange.append(homerange_IxI_normalized)
            SxS.append(SxS_normalized)
            
        truth = maxValue(IxS, IxI, Tripartite, Homerange, SxS)  
        
    return(IxS, IxI, Tripartite, Homerange, SxS)

#####################################################################################
# This is the 'main' function
#####################################################################################

location_amount = 10
individual_amount = 10
time_stamps = 20
simulation_amount = 10
gamma_values = np.linspace(1, 10, 10, endpoint =True)
beta_values = np.linspace(1, 10, 10, endpoint = True)
rho_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 0.99, 0.999, 0.9999, 1.0]#np.linspace(0.1, 1.0, 10, endpoint=True)

full_VNE_IxS_collection = []
full_VNE_IxI_collection = []
full_VNE_tripartite_collection = []
full_VNE_homerange_IxI_collection = []
full_VNE_SxS_collection = []

# This begins iterating for the simualtion amount as well as all gamma, beta, and rho values
for j in range(simulation_amount):
    gamma_collection_VNE_IxS = []
    gamma_collection_VNE_IxI = []
    gamma_collection_VNE_tripartite = []
    gamma_collection_VNE_homerange_IxI = []
    gamma_collection_VNE_SxS = []

    for z in range(len(gamma_values)):
        # We only need to create the group preference vector once per gamma value so here it is
        group_preference_vector, individuals_in_each_group = creating_group_preference_vector(location_amount, individual_amount, gamma_values[z])

        beta_collection_VNE_IxS = []
        beta_collection_VNE_IxI = []
        beta_collection_VNE_tripartite = []
        beta_collection_VNE_homerange_IxI = []
        beta_collection_VNE_SxS = []

        for k in range(len(beta_values)):
            # We only need to create the spatial preference vector once per beta values
            spatial_preference_vector = creating_spatial_preference_vector(location_amount, individual_amount, beta_values[k])

            rho_VNE_tripartite_collection = []
            rho_VNE_IxS_collection = []
            rho_VNE_IxI_collection = []
            rho_VNE_homerange_IxI_collection = []
            rho_VNE_SxS_collection = []
            
            # Both spatial and group locations are unaffected by the value of rho so
            # that is why it is the last for loop here.
            for y in range(len(rho_values)):


                rhoVNEIxS, rhoVNEIxI, rhoVNETripartite, rhoVNEHomerange, rhoVNESxS = timeSteps(location_amount, individual_amount, spatial_preference_vector,
                                                                                               group_preference_vector, individuals_in_each_group, rho_values[y])
                   
                # This next section collects all the VNE for a given rho value. If my rho vector
                # has a length of 5 then this collection should have 5 vectors inside of it
                # with length equal to the number of timestamps
                    
                rho_VNE_IxS_collection.append(rhoVNEIxS)
                rho_VNE_IxI_collection.append(rhoVNEIxI)
                rho_VNE_tripartite_collection.append(rhoVNETripartite)
                rho_VNE_homerange_IxI_collection.append(rhoVNEHomerange)
                rho_VNE_SxS_collection.append(rhoVNESxS)

            # This next section collects all the rho collections for each beta.
            # If the length of my beta vector is 20 and length of my rho vector is 5 then
            # these collections should have 20 vectors each with 5 vectors with length
            # equal to the number of timestamps.
            beta_collection_VNE_IxS.append(rho_VNE_IxS_collection)
            beta_collection_VNE_IxI.append(rho_VNE_IxI_collection)
            beta_collection_VNE_tripartite.append(rho_VNE_tripartite_collection)
            beta_collection_VNE_homerange_IxI.append(rho_VNE_homerange_IxI_collection)
            beta_collection_VNE_SxS.append(rho_VNE_SxS_collection)
    
           
        # If my gamma vecor has length of 3 then each of these will have 3 vectors 
        # of 20 vectors, where each of those 20 vectors have 5 vectors where each of those
        # have length = timestamps
        gamma_collection_VNE_IxS.append(beta_collection_VNE_IxS)
        gamma_collection_VNE_IxI.append(beta_collection_VNE_IxI)
        gamma_collection_VNE_tripartite.append(beta_collection_VNE_tripartite)
        gamma_collection_VNE_homerange_IxI.append(beta_collection_VNE_homerange_IxI)
        gamma_collection_VNE_SxS.append(beta_collection_VNE_SxS)
                
    # If I'm running 50 simulations then I'll have 50 vectors that satisfy all the other
    # things in the previous collection.
    
    full_VNE_IxS_collection.append(gamma_collection_VNE_IxS)
    full_VNE_IxI_collection.append(gamma_collection_VNE_IxI)
    full_VNE_tripartite_collection.append(gamma_collection_VNE_tripartite)
    full_VNE_homerange_IxI_collection.append(gamma_collection_VNE_homerange_IxI)
    full_VNE_SxS_collection.append(gamma_collection_VNE_SxS)
        
### This block of code saves the simulation's array so we don't have to run the simulation every time
np.save('IxSVNE10By10', full_VNE_IxS_collection)
np.save('IxIVNE10By10', full_VNE_IxI_collection)
np.save('TripartiteVNE10By10', full_VNE_tripartite_collection)
np.save('HomerangeVNE10By10', full_VNE_homerange_IxI_collection)
np.save('SxSVNE10By10', full_VNE_SxS_collection)
