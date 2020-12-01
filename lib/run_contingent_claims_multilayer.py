# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 11:17:17 2019

@author: RdelRioChanona
"""
import sys 
import networkx as nx
import numpy as np
import pandas as pd
lib_path = './lib'
sys.path.insert(0, lib_path) 
import sprmulticontagion as mc
import scipy
from matplotlib import pylab as plt
from scipy.sparse import linalg as LA
from scipy import linalg as la
import copy
import matplotlib.gridspec as gridspec

data_path = './data/'
file_multilayer = "data_gff_02092018.csv"
df = pd.read_csv(data_path + file_multilayer)
data_path = './data/'
export_path_csv = './results/csv_internship/'
export_path_fig = './results/fig_internship/'
year = 2016
# getting the layers
layer_out = set(df["country_sector"].value_counts().index.tolist())
layer_in = set(df["counterpart_sector"].value_counts().index.tolist())
layer_set = layer_out.intersection(layer_in)
layers = list(layer_set)
# getting the countries
countries_out = set(df["country"].value_counts().index.tolist())
countries_in = set(df["counterpart"].value_counts().index.tolist())
countries_set = countries_out.intersection(countries_in)
countries = list(countries_set)
# getting node-layers

df['node_layer_out'] = df.country + "-" + df.country_sector
df['node_layer_in'] = df.counterpart + "-" + df.counterpart_sector
df_multilayer_edgelist = df[["node_layer_out", "node_layer_in", str(year) ]]
df_multilayer_edgelist.rename(columns={str(year):'weight'}, inplace=True)

G = nx.from_pandas_edgelist(df_multilayer_edgelist, source="node_layer_out",\
                            target="node_layer_in", create_using=nx.DiGraph(),
                            edge_attr="weight")
        
A = np.array(nx.to_numpy_matrix(G))
plt.imshow(A)
plt.show()
# make dictionary of countries and layers
nodedict = {}
layerdict = {}
for i, c in enumerate(countries):
    nodedict[i] = c
for i, l in enumerate(layers):
    layerdict[i] = l

mx_net = mc.MultiplexNet(supra_adj=A, nodedict=nodedict, layerdict=layerdict)

car = 0.01*np.array([16, 16, 16, 14.33, 14.33, 14.33, 19.56, 19.56, 19.56])
ir = 0.01*np.array([-0.02625, -0.22792, -0.1, 0.395, 0.325833, 0.3958, 0.229, 0.325833, 0.3958])
sigma = 0.01*np.array([0.149219, 0.20958, 0.1830, 2.066, 1.91866, 2.0580, 2.241244, 2.1797, 2.196109])


#car = 0.1*np.array([16, 16, 16, 14.33, 14.33, 14.33, 19.56, 19.56, 19.56])
#ir = 0.1*np.array([-0.02625, -0.22792, -0.1, 0.395, 0.325833, 0.3958, 0.229, 0.325833, 0.3958])
#sigma = 0.1*np.array([0.149219, 0.20958, 0.1830, 2.066, 1.91866, 2.0580, 2.241244, 2.1797, 2.196109])

loss_2 = mx_net.contigent_claims_spread(car, defaulted_country="Japan",\
                                           defaulted_sector="Financial Corporations", T=5, r=ir,\
                                           sigma=sigma, shock_size=0.0)

country_list = list(mx_net.countries.keys())
layer_list = list(mx_net.sectors.keys())
country_names = ["Japan Banks", "Japan Sovereign", "Japan Firms", "USA Banks",\
                 "USA Sovereign", "USA Firms", "UK Banks", "UK Sovereign",\
                 "UK Firms", "No shock"]
colors = ["#63ACBE", "#63ACBE", "#63ACBE", "#601A4A", "#601A4A","#601A4A",\
          "#EE442F","#EE442F","#EE442F", "k"]
marker = marker_style = ["-", ":", "--"]

fontsize_ticks = 30
fontsize_axis = 30
fontsize_title = 40
fontsize_legend = 30
f = plt.figure(figsize=(16,10))
f.subplots_adjust(hspace=0.2)
gs = gridspec.GridSpec(1, 1)
ax = plt.subplot(gs[0])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(labelsize=fontsize_ticks)

i = 0
shock=0.5
for country in country_list:
    for layer in layer_list:
        loss = mx_net.contigent_claims_spread(car, defaulted_country=country,\
                                           defaulted_sector=layer, T=100, r=ir,\
                                           sigma=sigma, shock_size=shock)
        score = loss/mx_net.supra_adj_matrix.sum() 
        ax.plot(100*score[:], color=colors[i], linestyle=marker[i%3],linewidth=5, \
                label=country_names[i], alpha=0.8)
        i += 1
# add no shock as baseline
loss_noshock = mx_net.contigent_claims_spread(car, defaulted_country="Japan",
                                defaulted_sector="Financial Corporations",\
                                T=100, r=ir, sigma=sigma, shock_size=0.0)
score = loss_noshock/mx_net.supra_adj_matrix.sum() 
ax.plot(100*score[:], color="k", linestyle="-",linewidth=5, \
            label="No Shock", alpha=0.8)

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=fontsize_legend)   
plt.ylabel("Lost assets (%)", fontsize=fontsize_axis)
plt.xlabel("Time step", fontsize=fontsize_axis)
plt.title("Contigent Claims Multilayer,\nassuming a "+ str(shock*100)[:2] +"% shock", fontsize=fontsize_title)
plt.savefig(export_path_fig + "claims_testsig_multilayer"+str(shock*100)[:2]+".png", bbox_inches="tight")
plt.show()
#
#
#
#nnodes = 3
#nlayers= 3
#countries = {}
#for i in range(nnodes*nlayers):
#    countries[list(G.nodes)[i]] = i
#
#
##### defining contigent claims outside library
#def compute_d2(assets, DB, r, sigma, tau):
#    """Computes d2, from the Black Scholes Eqs. Later used to compute
#    probability of default
#    A(array) (nl,) array
#    """
#    # we use mask array to avoid error when taking log of assets that are 0
#    log_a_db = np.ma.log(assets/DB)
#    d2 = (log_a_db  + (r - 0.5*sigma**2)*tau)/(sigma * \
#               np.sqrt(tau))
#    # if there was a problem with the log, substitue with negative large #
#    d2.filled(-1e10)
##        print("a/db = ", (assets[self.countries["France"]]/DB[self.countries["France"]]))
##        print("d2 = ", d2[self.countries["France"]] )
#    return d2
#
#
#def p_default(d2, r, sigma):
#    """ give probability of default for a country-sector given d2
#    """
#    #TODO check
##        return 1 - scipy.stats.norm.cdf(d2, loc=r, scale=sigma)
#    return 1 - scipy.stats.norm.cdf(d2, loc=0, scale=1)
#    
#def update_contingent_claims_matrix(A, p, exch_rate=1.0):
#    """
#    WARNING be careful you don't update the actual supra adjacency matrix
#    """
##        print("p = ", p[:10] )
#    return A * (1-p) * exch_rate
#
#def contigent_claims_spread(supra_adj_matrix, car, r=1, sigma=0.5, tau=10, T=10, \
#                          exch_rate=1.0, defaulted_country=None, \
#                          shock_size=1):
#    """
#    if T = None, returns rank until tolerance is met
#    """
#    
#    # record the initial value of assets
#    initial_assets = supra_adj_matrix.sum(axis=1)
#    
#    # adapt car to multilayer if needed
#    if car.shape[0] == nnodes:
#        ext_car = np.concatenate((car, car), axis=None)
#        for i in range(1, nlayers-1):
#            ext_car = np.concatenate((ext_car, car), axis=None)
#    else:
#        ext_car = car
#    # use car to compute distress barrier
#    DB = (1 - ext_car) * initial_assets
#    
#    # make a compy of supraadj matrix which will decay in value
#    A = copy.deepcopy(supra_adj_matrix)
#    # record lost assets
#    assets_lost = np.zeros(T + 1)
#    # wipe out the assets of the defaulted country
#    if type(defaulted_country) != type(None):
#        
#        assets_lost[0] = (shock_size) * A[:, countries[defaulted_country]].sum()
#        A[:, countries[defaulted_country]] = (1 - shock_size) * \
#            A[:, countries[defaulted_country]]
#    else:
#        assets_lost[0] = 0
#    # assets is sum over columns
#    assets = A.sum(axis=1)
#    
#
#    for t in range(1,T+1):
##            print("assets = ", (assets)[:10])
##            print("All assets = ", assets[self.countries["France"]].sum())
#        # compute parameters needed to update claims
#        d2 = compute_d2(assets, DB, r, sigma, tau)
#        p = p_default(d2, r, sigma)
#        A = update_contingent_claims_matrix(A, p, exch_rate)
#        assets = A.sum(axis=1)
#        # compute the loss of the whole network
#        loss = (initial_assets - assets).sum()
#        assets_lost[t] = loss
#        
#    return assets_lost


