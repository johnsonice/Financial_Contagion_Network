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

year = 2016
import_path = "./data/adjacency_matrix/"
export_path_csv = "./results/csv_structuralmeasurements/"
#import the aggregate adjacency matrix, and make graph with names
aggregate_am = np.genfromtxt(import_path + 'AM4_all_nodes_aggregate' + \
                            str(year) + '.csv', delimiter=",")
df_names = pd.read_csv(import_path + 'all_country_name4.csv', header=None)
names = list(df_names[0])
aggregate_g = igraph.Graph.Weighted_Adjacency(list(aggregate_am))
aggregate_g.vs["name"] = copy.deepcopy(names)


def order_centrality(centrality_vector, names=names):
    '''Takes eigenvector of centrality and returns list of ranked
    nodes and another of their score.
    Args
    centrality_vector(numpy array): the centrality measure for each node-layer
    names(list of strings): name of nodes (countries)
    Return:
    sort_names(list of strings): names of countries ordered by centrality
    sort_centrality(list of flots): sorted score of nodes
    '''
    node_names = np.array(copy.deepcopy(names))
    inds = centrality_vector.argsort()[::-1][:]
    sort_names= node_names[inds]
    sort_centrality = np.array(centrality_vector)[inds]
    return sort_names, sort_centrality

def save_centrality(y, namefile, namecent, country, cent, path=export_path_csv):
    f = open(path+namefile+str(y) +".csv", "w")
    f.write("Country,"+namecent+"\n")
    n = len(country)
    for i in range(n):
        f.write(str(country[i]) + "," + str(cent[i])+ "\n")
    f.close()

hubs_names, hub_score = order_centrality(np.array(aggregate_g.hub_score(weights=aggregate_g.es["weight"])))
auth_names, auth_score = order_centrality(np.array(aggregate_g.authority_score(weights=aggregate_g.es["weight"])))
pr_names, pr_score = order_centrality(np.array(\
                    aggregate_g.personalized_pagerank(weights=aggregate_g.es["weight"])))

save_centrality(year, "Aggregate_hub", "Hub score" , hubs_names, hub_score)
save_centrality(year, "Aggregate_auth", "Authority score" , auth_names, auth_score)
save_centrality(year, "Aggregate_PageRank", "PageRank score" , pr_names, pr_score)

aggregate_gT = igraph.Graph.Weighted_Adjacency(list(np.transpose(aggregate_am)))
aggregate_gT.vs["name"] = copy.deepcopy(names)

hubs_namesT, hub_scoreT = order_centrality(np.array(aggregate_gT.hub_score(weights=aggregate_gT.es["weight"])))
auth_namesT, auth_scoreT = order_centrality(np.array(aggregate_gT.authority_score(weights=aggregate_gT.es["weight"])))
pr_namesT, pr_scoreT = order_centrality(np.array(\
                    aggregate_gT.personalized_pagerank(weights=aggregate_gT.es["weight"])))

save_centrality(year, "AggregateT_hub", "Hub score" , hubs_namesT, hub_scoreT)
save_centrality(year, "AggregateT_auth", "Authority score" , auth_namesT, auth_scoreT)
save_centrality(year, "AggregateT_PageRank", "PageRank score" , pr_namesT, pr_scoreT)
