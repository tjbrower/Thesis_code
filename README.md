# Thesis_code
This is a little late but here is my code for my thesis


The order in which I used these files is as follows:

1) simulateVNE.py or simulateVNEUsingParallelization.py - Creates the actual VNE values from our simulations. 
    Saves the VNE values in 5 different arrays
    
2) Saving_the_Lineweaver_Burk_transformations.py - Pulls the 5 arrays with the VNE values and computes the 'v' value for each.
    Also saves these values in 5 different arrays
    
3) creatingDatabases.py - This takes our VNE values and their transformed 'v' values and estimates the max VNE values.
    This also creates 15 different databses and saves them as csv's.
    
4) allVisualizationsFile.py - This takes the 15 csv files and creates a bunch of visualizations for them using the seaborn and matplotlib
    libraries.
   
When using empirical data the I used these files under the file RealDataWork as follows:

1) VNE_calculation_for_real_data.py - This cleans up the data and adds in individuals that don't have readings a particular day
      and adds missing locations as well. There are 5 arrays that are saved with VNE values for each of the 5 networks
      
2) Visualizations_for_real_data.py - This takes the saved arrays for the 5 networks and creates visualizations over the 4 spatial variances.
      Each plot is of each network over the 4 variances.

