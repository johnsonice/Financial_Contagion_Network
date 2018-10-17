import sys 
import os
lib_path = './lib'
sys.path.insert(0, lib_path)     ## add lib path to sys path 
import igraph
import random
import copy
import scipy.stats
import pandas as pd
import numpy as np
from matplotlib import pylab as plt
from scipy.sparse import linalg as LA
import MultiContagion as mc


year = 2013                                                #### specify which year you want the structural statistics for
import_path = "./data/adjacency_matrix/"                   #### set up data path 
export_path_csv = "./results/csv_structuralmeasurements/"  #### set up export data path 

#import the aggregate adjacency matrix, and make graph with names
aggregate_am = np.genfromtxt(import_path + 'AM4_all_nodes_aggregate' + \
                            str(year) + '.csv', delimiter=",")
df_names = pd.read_csv(import_path + 'all_country_name4.csv', header=None)
names = list(df_names[0])
aggregate_g = igraph.Graph.Weighted_Adjacency(list(aggregate_am))
aggregate_g.vs["name"] = copy.deepcopy(names)

sum(aggregate_g.es['weight'])

#Import the adjacency matrix and make graphs of all the layers.
cdis_equity_am = np.genfromtxt(import_path + 'AM4_all_nodesCDIS-equity' + \
                              str(year) + '.csv', delimiter=",")
cdis_equity_g = igraph.Graph.Weighted_Adjacency(list(cdis_equity_am))
cdis_equity_g.vs["name"] = copy.deepcopy(names)

cdis_debt_am = np.genfromtxt(import_path + 'AM4_all_nodesCDIS-debt'+ \
                            str(year) + '.csv', delimiter=",")
cdis_debt_g = igraph.Graph.Weighted_Adjacency(list(cdis_debt_am))
cdis_debt_g.vs["name"] = copy.deepcopy(names)

cpis_equity_am = np.genfromtxt(import_path + 'AM4_all_nodesCPIS-equity' + \
                             str(year) + '.csv', delimiter=",")
cpis_equity_g = igraph.Graph.Weighted_Adjacency(list(cpis_equity_am))
cpis_equity_g.vs["name"] = copy.deepcopy(names)

cpis_debt_am = np.genfromtxt(import_path + 'AM4_all_nodesCPIS-debt' + \
                            str(year)+'.csv', delimiter=",")
cpis_debt_g = igraph.Graph.Weighted_Adjacency(list(cpis_debt_am))
cpis_debt_g.vs["name"] = copy.deepcopy(names)

bis_am = np.genfromtxt(import_path + 'AM4_all_nodesBIS' + str(year) + '.csv', \
                        delimiter=",")
bis_g = igraph.Graph.Weighted_Adjacency(list(bis_am))
bis_g.vs["name"] = copy.deepcopy(names)

G_list = [cdis_equity_g, cdis_debt_g, cpis_equity_g, cpis_debt_g, bis_g, \
        aggregate_g]
G_names = ["CDIS_equity", "CDIS_debt", "CPIS_equity", "CPIS_debt", "BIS", \
            "Aggregated"]

def save_basic_measurements(y, list_of_graphs, names_of_graphs, exp_path=export_path_csv):
    '''Saves into a csv file the basic structural measurements of list of graphs
    Args:
        y(int): integer used to name the file
        list_of_graphs(list): list contaning graph type elements
        names_of_graphs(list): list containing the names of the graphs
    '''
    f = open(exp_path + "monoplex_basic_struc_measures"+ str(y) +".csv"\
            , "w")
    f.write("network,diameter (unweighted-directed),density(unweighted-directed),\
            mean geodesic distance(unweighted-directed), \
            reciprocity(unweighted-directed),\
            global clustering(unweighted-undirected),\
            average clustering (unweighted-undirected),\
            sum of weights\n")
    if len(list_of_graphs) != len(names_of_graphs):
        print("warning dimention of lists do not match")
    n = len(list_of_graphs)
    for i in range(n):
        f.write(names_of_graphs[i] + ",")
        g = list_of_graphs[i]
        weights = np.array(copy.deepcopy(g.es["weight"]))
        f.write(str(round(g.diameter(), 2) ) + ",")
        f.write(str(round(g.density(), 2) ) + ",")
        f.write(str(round(g.average_path_length(), 2) ) + ",")
        f.write(str(round(g.reciprocity(), 2) ) + ",")
        f.write(str(round(g.transitivity_undirected(), 2) ) + ",")
        f.write(str(round(np.mean(g.transitivity_local_undirected(mode="zero"))\
                , 2) ) + ",")
        f.write(str(round(sum(weights)) ) + "\n")
    f.close()

save_basic_measurements(year, G_list, G_names)

#print  " average path length =  ", G_agg_2015.average_path_length()
#print  " global clustering =  ", G_agg_2015.transitivity_undirected()
#print " average clustering (unweighted) =  ", np.mean(G_agg_2015.transitivity_local_undirected( mode = "zero"))
#print  " average clustering =  ", np.mean(G_agg_2015.transitivity_local_undirected(weights = G_agg_2015.es["weight"],  mode = "zero" ))
