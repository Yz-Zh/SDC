# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import matplotlib.pyplot as plt 
import numpy as np              
from sklearn.cluster import KMeans  
import sklearn.metrics as sklm  

#select variables to plot
x_select=0
y_select=1

data_filename = 'C:\\Users\\Yz Zh\\Desktop\\sdc-clustering-data0.csv'
num_clusters = 3
figure_width, figure_height = 7,7

data = np.genfromtxt(data_filename,delimiter = ',')


fig_title = 'Outlier removed plot'
x_label   = 'traffic folws'
y_label   = 'carbon emissions'
title_fontsize = 20
label_fontsize = 18
#x_min, x_max = 0.8*np.min(data[:,x_select]), 1.05*np.max(data[:,x_select])
#y_min, y_max = 0.4*np.min(data[:,y_select]), 1.2*np.max(data[:,y_select])

#x_min, x_max = -0.15, 1.15
#y_min, y_max = -0.15, 1.15

x_min, x_max = -0.15, 1.05*np.max(data[:,x_select])
y_min, y_max = -0.15, 1.2*np.max(data[:,y_select])

# PERFORM CLUSTERING




# This line performs the k-means clustering:
kmeans_output = KMeans(n_clusters=num_clusters, n_init=1).fit(data)

# This line creates a list giving the final cluster number of each point:
clustering_ids_kmeans = kmeans_output.labels_


# DATA PROCESSING

# These lines add the cluster IDs to the original data and save the data with these added cluster IDs.
complete_data_with_clusters = np.hstack((data,np.array([clustering_ids_kmeans]).T))


# The loop below creates a separate data array for each cluster, and puts these arrays together in a list:
data_by_cluster = []

for i in range(num_clusters):
    
    this_data = []
    
    for row in complete_data_with_clusters:
        
        if row[-1] == i:
            this_data.append(row)
    
    this_data = np.array(this_data)
    
    data_by_cluster.append(this_data)
 

# CREATE FIGURES

# This is a function that sets up each figure's x-limits and y-limits and axis labels.

def setup_figure():
    
    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])
    plt.xlabel(x_label,fontsize=label_fontsize)
    plt.ylabel(y_label,fontsize=label_fontsize)


# FIGURE 0 : UNCLUSTERED DATA

# These lines extract the y-values and the x-values from the data:
x_values = data[:,x_select]
y_values = data[:,y_select]

# The next lines create and save the plot:
plt.figure(0,figsize=(figure_width,figure_height))
setup_figure()
plt.title(fig_title,fontsize=title_fontsize)
plt.plot(x_values,y_values,'k.')



# FIGURES 1-N : SEPARATE CLUSTER PLOTS

# This is a list of colours to differentiate each cluster.
color_list = ['b','r','g','m']



# FIGURE N + 1 : COMBINED CLUSTER PLOT

# These lines create a plot with all the data points, coloured by cluster.
plt.figure(num_clusters + 1,figsize=(figure_width,figure_height))
setup_figure()
plt.title(fig_title ,fontsize=title_fontsize)

for i in range(num_clusters):
    
    x_values = data_by_cluster[i][:,x_select]
    y_values = data_by_cluster[i][:,y_select]
    
    plt.plot(x_values,y_values,color_list[i % num_clusters] + '.')
      



# SILHOUETTE SCORE

# These lines calculate the silhouette score...
silhouette_kmeans = sklm.silhouette_score(data,clustering_ids_kmeans)

# ... and print it:
print("Silhouette Score:", silhouette_kmeans)


SSE = []
for k in range(1, 9):
    model = KMeans(n_clusters=k).fit(data)
    SSE.append(model.inertia_)
X = range(1, 9)
fig_title = 'SSE-K plot'
plt.title(fig_title)
plt.xlabel('k')
plt.ylabel('SSE')
plt.plot(X, SSE, 'o-')

clustering_ids_kmeans = kmeans_output.labels_
print(clustering_ids_kmeans)




