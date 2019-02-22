import sys
import os
import igraph
import random
import scipy.stats
import copy
import pandas as pd
import numpy as np
lib_path = './lib'
sys.path.insert(0, lib_path)     ## add lib path to sys path
import MultiContagion as mc
import helpfunctions as  hf

from matplotlib import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
#%matplotlib inline

year = 2016
# Paths to data
import_path = "./data/adjacency_matrix/"
export_path = "./results/csv_contagion2019/"
export_fig_path = "./results/fig_contagion2019/"

# Import aggregate adjacency matrix and get name of countries
aggregate_am = np.genfromtxt (import_path +'AM4_all_nodes_aggregate'+str(year)+'.csv', delimiter=",")
df_names = pd.read_csv(import_path +'all_country_name4.csv', header=None)
names = list(df_names[0])

#Import the adjacency matrix and make graphs of all the layers.
cdis_equity_am = np.genfromtxt (import_path +'AM4_all_nodesCDIS-equity'+str(year)+'.csv', delimiter=",")
cdis_debt_am = np.genfromtxt (import_path +'AM4_all_nodesCDIS-debt'+str(year)+'.csv', delimiter=",")
cpis_equity_am = np.genfromtxt (import_path +'AM4_all_nodesCPIS-equity'+str(year)+'.csv', delimiter=",")
cpis_debt_am = np.genfromtxt (import_path +'AM4_all_nodesCPIS-debt'+str(year)+'.csv', delimiter=",")
bis_am = np.genfromtxt (import_path +'AM4_all_nodesBIS'+str(year)+'.csv', delimiter=",")
#list of adjacency matrices
A_list = [cdis_equity_am, cdis_debt_am, cpis_equity_am, cpis_debt_am, bis_am]
# get mean degree
g = igraph.Graph.Weighted_Adjacency(list(bis_am)) # mean deg ~ 77
g = igraph.Graph.Weighted_Adjacency(list(cpis_debt_am))# mean deg ~ 50
g = igraph.Graph.Weighted_Adjacency(list(cdis_debt_am))# mean deg ~ 50



# dimensions of multiples
n_countries = A_list[0].shape[0]
n_layers = len(A_list)
# the coupling is done via the indegree for the Threshold model, however we also
# show the outdegree since it can be used for the CreditFlow coupling.
C_indegree = np.zeros([n_countries, n_layers])
C_outdegree = np.zeros([n_countries, n_layers])
for l in range(n_layers):
    C_indegree[:, l] = np.sum(A_list[l], axis=0)
    C_outdegree[:, l] = np.sum(A_list[l], axis=1)

# set a simple one for the aggregate network run
C_agg = np.zeros([n_countries, 1])
C_agg[:,0] = np.sum(aggregate_am, axis=1)

#seeds lists
Large_economies = ["United States", "United Kingdom", "Netherlands", \
"Luxembourg", "France", "Germany", "China  P.R.: Hong Kong",\
"China  P.R.: Mainland"]


states, states_agg = hf.make_initial_states(seed_names=["United Kingdom"], \
                    names=names)#, seed_layers=[1,2])


#####
# Example with moving horizontal threshold. Multiplex vs Aggregate
# No vertical threshold
#####
tau_ver_none = 0.000001#0.1#1.5#0
ti = 0.1
tf = 0.95
steps = 18
tau_hor_list = np.linspace(ti, tf, steps)


for country in Large_economies:
    #seeting initial states for seeds
    seed_name = [country]
    states, states_agg = hf.make_initial_states(seed_name, names)

    # list in which results will be added
    list_defaulted = []#np.zeros(len(tau_hor_list))
    list_defaulted_agg = []

    # running simulartion for different horizontal thresholds
    for tau_hor in tau_hor_list[:]:
        # run for multiplex
        state_history, hor_contagion, ver_contagion, mix_contagion, t_end = \
        hf.run_contagion(A_list, C_indegree, states,\
        hf.default_horizontal_homogenous, hf.default_vertical_homogenous, \
        tau_hor, tau_ver_none, save_csv=False, \
        save_name="results/csv_contagion2019/multilayer_uk",country_names=names)
        # append to list the results (for plotting)
        list_defaulted.append(hf.countries_defaulted(state_history))

        # run for aggregate
        state_history_agg, hor_contagion_agg, ver_contagion_agg, \
        mix_contagion_agg, t_end_agg= hf.run_contagion([aggregate_am], \
        C_agg, states_agg,hf.default_horizontal_homogenous, \
        hf.default_vertical_homogenous, tau_hor, tau_ver_none, save_csv=False, \
        save_name="results/csv_contagion2019/aggregate_uk", country_names=names)
        # append to list the results (for plotting)
        list_defaulted_agg.append(hf.countries_defaulted(state_history_agg))

    # plotting the results
    plt.title("Contagion caused by " + seed_name[0])
    plt.xlabel("intralayer fragility")
    plt.ylabel("Number of affected countries")
    plt.plot(tau_hor_list[::-1], list_defaulted[::-1], "o-", label="multiplex")
    plt.plot(tau_hor_list[::-1], list_defaulted_agg[::-1], "o-", label="aggregate")
    plt.gca().invert_xaxis()
    plt.legend()
    #plt.savefig(export_fig_path + "MultiVsAgg_" + seed_name[0] + str(year) +".png")
    plt.show()

# print("total horizontal contagion = ", sum(hor_contagion))
# print("total vertical contagion = ", sum(ver_contagion))
# print("total mixed contagion = ", sum(mix_contagion))
# print("total countries affected = ", hf.countries_defaulted(state_history))
# print("total defaulted = ", sum(hor_contagion) + sum(ver_contagion) - sum(mix_contagion))
# print("ver" ,ver_contagion)
# print("hor" ,hor_contagion)
# print("mix" ,mix_contagion)



#####
# Example with moving horizontal threshold. Two types of multiplex,
# playing with vertical threshold
#####
#
# tau_ver_none = 0.001#0.1#1.5#0
# tau_ver = 1#5.0
# ti = 0.1
# tf = 0.95
# steps = 18
# tau_hor_list = np.linspace(ti, tf, steps)
# list_defaulted = []#np.zeros(len(tau_hor_list))
# list_defaulted2 = []
# list_defaulted_agg = []
# list_vert = []
# list_vert2 = []
# for tau_hor in tau_hor_list[:]:
#     state_history, hor_contagion, ver_contagion, mix_contagion, t_end = hf.run_contagion(A_list, C_indegree, states,\
#     hf.default_horizontal_homogenous, hf.default_vertical_homogenous, tau_hor, tau_ver_none, save_csv=False, save_name="results/csv_contagion2019/multilayer_uk", country_names=names)
#     list_defaulted.append(hf.countries_defaulted(state_history))
#     list_vert.append(sum(ver_contagion))
#     state_history2, hor_contagion2, ver_contagion2, mix_contagion2, t_end2 = hf.run_contagion(A_list, C_indegree, states,\
#     hf.default_horizontal_homogenous, hf.default_vertical_homogenous, tau_hor, tau_ver, save_csv=False, save_name="results/csv_contagion2019/multilayer_uk", country_names=names)
#     list_defaulted2.append(hf.countries_defaulted(state_history2))
#     list_vert2.append(sum(ver_contagion2))
#     state_history_agg, hor_contagion_agg, ver_contagion_agg, mix_contagion_agg, t_end_agg= hf.run_contagion([aggregate_am], C_agg, states_agg,\
#     hf.default_horizontal_homogenous, hf.default_vertical_homogenous, tau_hor, tau_ver, save_csv=False, save_name="results/csv_contagion2019/aggregate_uk", country_names=names)
#     list_defaulted_agg.append(hf.countries_defaulted(state_history_agg))
#
# plt.title("Contagion caused by " + seed_name)
# plt.xlabel("intralayer fragility")
# plt.ylabel("Number of affected countries")
# plt.plot(tau_hor_list[::-1], list_defaulted[::-1], "o-", label="multiplex1")
# plt.plot(tau_hor_list[::-1], list_defaulted2[::-1], "o-", label="multiplex2")
# plt.plot(tau_hor_list[::-1], list_defaulted_agg[::-1], "o-", label="aggregate")
# plt.plot(tau_hor_list[::-1], list_vert[::-1], "o-", label="multiplex1")
# plt.plot(tau_hor_list[::-1], list_vert2[::-1], "o-", label="multiplex2")
# plt.gca().invert_xaxis()
# plt.legend()
# plt.show()
#
# print("total horizontal contagion = ", sum(hor_contagion))
# print("total vertical contagion = ", sum(ver_contagion))
# print("total mixed contagion = ", sum(mix_contagion))
# print("total countries affected = ", hf.countries_defaulted(state_history))
# print("total defaulted = ", sum(hor_contagion) + sum(ver_contagion) - sum(mix_contagion))
# print("ver" ,ver_contagion)
# print("hor" ,hor_contagion)
# print("mix" ,mix_contagion)
#
# print("total horizontal contagion = ", sum(hor_contagion2))
# print("total vertical contagion = ", sum(ver_contagion2))
# print("total mixed contagion = ", sum(mix_contagion2))
# print("total countries affected = ", hf.countries_defaulted(state_history2))
# print("total defaulted = ", sum(hor_contagion2) + sum(ver_contagion2) - sum(mix_contagion2))
# print("ver" ,ver_contagion2)
# print("hor" ,hor_contagion2)
# print("mix" ,mix_contagion2)
#

#
# EU_names = [ "Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Estonia", "Finland", "France" , "Germany", "Greece", "Hungary", "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Netherlands", "Poland", "Portugal", "Romania", "Slovak Republic", "Slovenia", "Spain", "Sweden", "United Kingdom"]
# EU_names_brexit = [ "Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Estonia", "Finland", "France" , "Germany", "Greece", "Hungary", "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Netherlands", "Poland", "Portugal", "Romania", "Slovak Republic", "Slovenia", "Spain", "Sweden"]
# Large_economies = ["United States", "United Kingdom", "Netherlands", "Luxembourg", "France", "Germany", "China  P.R.: Hong Kong","China  P.R.: Mainland"]
# China_and_HK = ["China  P.R.: Hong Kong","China  P.R.: Mainland"]
#
# China_and_HK_ind = [names.index(c) for c in China_and_HK]
# China_and_HK_ind
#
#
# states = np.zeros([n_countries, n_layers])
# # default USA in all layers
# for l in range(n_layers):
#     for c in China_and_HK_ind:
#         states[c,l] = 1
#
#
#
# tau_ver = 0.0001#0.1#1.5#0
# ti = 0.1
# tf = 0.95
# steps = 18
# tau_hor_list = np.linspace(ti, tf, steps)
# list_defaulted = []#np.zeros(len(tau_hor_list))
# list_defaulted2 = []
# list_defaulted_agg = []
# for tau_hor in tau_hor_list[:]:
#     state_history, hor_contagion, ver_contagion, mix_contagion, t_end = hf.run_contagion(A_list, C_indegree, states,\
#     hf.default_horizontal_homogenous, hf.default_vertical_homogenous, tau_hor, tau_ver, save_csv=False, save_name="results/csv_contagion2019/multilayer_uk", country_names=names)
#     list_defaulted.append(hf.countries_defaulted(state_history))
#     state_history2, hor_contagion2, ver_contagion2, mix_contagion2, t_end2 = hf.run_contagion(A_list, C_indegree, states,\
#     hf.default_horizontal_homogenous, hf.default_vertical_homogenous, tau_hor, 1.5, save_csv=False, save_name="results/csv_contagion2019/multilayer_uk", country_names=names)
#     list_defaulted2.append(hf.countries_defaulted(state_history2))
#     state_history_agg, hor_contagion_agg, ver_contagion_agg, mix_contagion_agg, t_end_agg= hf.run_contagion([aggregate_am], C_agg, states_agg,\
#     hf.default_horizontal_homogenous, hf.default_vertical_homogenous, tau_hor, tau_ver, save_csv=False, save_name="results/csv_contagion2019/aggregate_uk", country_names=names)
#     list_defaulted_agg.append(hf.countries_defaulted(state_history_agg))
#
# plt.title("total number of defaulted countries")
# plt.xlabel("Horizontal threshold")
# plt.ylabel("defaulted countries")
# plt.plot(tau_hor_list[::-1], list_defaulted[::-1], "o-")
# plt.plot(tau_hor_list[::-1], list_defaulted2[::-1], "o-")
# plt.plot(tau_hor_list[::-1], list_defaulted_agg[::-1], "o-")
# plt.gca().invert_xaxis()
# plt.show()
#
# print("total horizontal contagion = ", sum(hor_contagion))
# print("total vertical contagion = ", sum(ver_contagion))
# print("total mixed contagion = ", sum(mix_contagion))
# print("total countries affected = ", hf.countries_defaulted(state_history))
# print("total defaulted = ", sum(hor_contagion) + sum(ver_contagion) - sum(mix_contagion))
# print("ver" ,ver_contagion)
# print("hor" ,hor_contagion)
# print("mix" ,mix_contagion)
