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
import_thresh_path = "./data/Buffers for JA/"
export_path = "./results/csv_contagion2019/"
export_fig_path = "./results/fig_contagion2019/"

# countries names ordered as in adjacency matrix
df_names = pd.read_csv(import_path +'all_country_name4.csv', header=None)
names = list(df_names[0])
n_countries = len(names)
n_layers = 5
# df of iip_gdp data from which thresholds can be determined
df_threshold = pd.read_stata(import_thresh_path + "iip_gdp.dta")
# crosswalk for some countries names between adj mat and threshold
df_crosswalk = pd.read_csv(import_thresh_path + "crosswalk.csv")
dict_crosswalk = dict(zip(df_crosswalk.network,df_crosswalk.threshold))
# get the mean iipgdp value
mean_iipgdp = np.mean(df_threshold.loc[(df_threshold['year'] == year) & \
            np.isfinite(df_threshold["iip_gdp"]) ]["iip_gdp"])
# another suggestion would be to use 0
input_value = 0#mean_iipgdp

df_threshold.loc[(df_threshold['year'] == year) & (df_threshold["iip_gdp"] < -2) ]#["country"]

# getting the scores for countries
iip_gdp = np.zeros(n_countries)
for (i, name) in enumerate(names):
    # if the name is in the threhold list get the threshold
    if name in list(df_threshold["country"]):
        value = float(df_threshold.loc[(df_threshold['year']==year) & \
        (df_threshold["country"]==name) ]["iip_gdp"])
        if np.isfinite(value):
            iip_gdp[i] = value
        else:
            iip_gdp[i] = input_value
    # else try the crosswalk
    elif name in dict_crosswalk.keys():
        value = float(df_threshold.loc[(df_threshold['year']==year) & \
        (df_threshold["country"]==dict_crosswalk[name]) ]["iip_gdp"])
        iip_gdp[i] = value
        if np.isfinite(value):
            iip_gdp[i] = value
        else:
            iip_gdp[i] = input_value
    # if neither works then just input the average
    else:
        iip_gdp[i] = input_value

# defining the thresholds.
# max and min value seem to be around -4 and 4 thus the spread
norm_factor = 8
# window to the right and left of the centre. Total window is 2*window
window = 0.2
# fragility, parameter which moves
fragility = 0.3
# must check again this assertion, remmeber max vlaue is actuall 6
assert(0 < fragility - window < fragility + window < 1)
# actual list of thresholds


def make_hetero_thresholds(country_score, norm_factor, window, fragility):
    return fragility + window * country_score / norm_factor

thresholds = make_hetero_thresholds(iip_gdp, norm_factor, window, fragility)

# Import aggregate adjacency matrix and get name of countries
aggregate_am = np.genfromtxt (import_path +'AM4_all_nodes_aggregate'+str(year)+'.csv', delimiter=",")

#Import the adjacency matrix and make graphs of all the layers.
cdis_equity_am = np.genfromtxt (import_path +'AM4_all_nodesCDIS-equity'+str(year)+'.csv', delimiter=",")
cdis_debt_am = np.genfromtxt (import_path +'AM4_all_nodesCDIS-debt'+str(year)+'.csv', delimiter=",")
cpis_equity_am = np.genfromtxt (import_path +'AM4_all_nodesCPIS-equity'+str(year)+'.csv', delimiter=",")
cpis_debt_am = np.genfromtxt (import_path +'AM4_all_nodesCPIS-debt'+str(year)+'.csv', delimiter=",")
bis_am = np.genfromtxt (import_path +'AM4_all_nodesBIS'+str(year)+'.csv', delimiter=",")
#list of adjacency matrices
A_list = [cdis_equity_am, cdis_debt_am, cpis_equity_am, cpis_debt_am, bis_am]
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
#####

tau_ver_none = 0.000001#0.1#1.5#0
tau_ver_none_list = [0.000001 for i in range(n_layers)]#0.1#1.5#0
window = 0.2
ti = 0.2
tf = 0.8
steps = 10
tau_hor_list = np.linspace(ti, tf, steps)


tau_ver_none = 0.000001#0.1#1.5#0
tau_ver_none_list = [0.000001 for i in range(n_layers)]#0.1#1.5#0
window = 0.2
ti = 0.2
tf = 0.8
steps = 10
tau_hor_list = np.linspace(ti, tf, steps)

for country in Large_economies[:]:
    #seeting initial states for seeds
    seed_name = [country]
    states, states_agg = hf.make_initial_states(seed_name, names)

    # list in which results will be added
    list_defaulted = []#np.zeros(len(tau_hor_list))
    list_defaulted_agg = []
    list_defaulted_homo = []#np.zeros(len(tau_hor_list))
    list_defaulted_agg_homo = []

    # running simulartion for different horizontal thresholds
    for fragility in tau_hor_list[:]:
        tau_hor = make_hetero_thresholds(iip_gdp, norm_factor, window, fragility)

        state_history, hor_contagion, ver_contagion, mix_contagion, t_end = hf.run_contagion(A_list, C_indegree, states,\
        hf.default_horizontal, hf.default_vertical, tau_hor, tau_ver_none_list, save_csv=False, save_name="results/csv_contagion2019/multilayer" + country + "homogenous_frag_" + str(fragility)[2:4] , country_names=names)
        list_defaulted.append(hf.countries_defaulted(state_history))

        state_history, hor_contagion, ver_contagion, mix_contagion, t_end = hf.run_contagion(A_list, C_indegree, states,\
        hf.default_horizontal_homogenous, hf.default_vertical_homogenous, fragility, tau_ver_none, save_csv=False, save_name="results/csv_contagion2019/multilayer" + country + "hetero_frag_" + str(fragility)[2:4], country_names=names)
        list_defaulted_homo.append(hf.countries_defaulted(state_history))


        state_history_agg, hor_contagion_agg, ver_contagion_agg, mix_contagion_agg, t_end_agg= hf.run_contagion([aggregate_am], C_agg, states_agg,\
        hf.default_horizontal, hf.default_vertical, tau_hor, tau_ver_none_list, save_csv=True, save_name="results/csv_contagion2019/aggregate" + country + "homogenous_frag_" + str(fragility)[2:4], country_names=names)
        list_defaulted_agg.append(hf.countries_defaulted(state_history_agg))

        state_history_agg, hor_contagion_agg, ver_contagion_agg, mix_contagion_agg, t_end_agg= hf.run_contagion([aggregate_am], C_agg, states_agg,\
        hf.default_horizontal_homogenous, hf.default_vertical_homogenous, fragility, tau_ver_none, save_csv=True, save_name="results/csv_contagion2019/aggregate" + country + "hetero_frag_" + str(fragility)[2:4], country_names=names)
        list_defaulted_agg_homo.append(hf.countries_defaulted(state_history_agg))

    # plotting the results
    plt.title("Contagion caused by " + seed_name[0] + "\n using heterogenous horizontal thresholds")
    plt.xlabel("intralayer fragility")
    plt.ylabel("Number of affected countries")
    plt.plot(tau_hor_list[::-1], list_defaulted_homo[::-1], "o-", label="multiplex")
    plt.plot(tau_hor_list[::-1], list_defaulted[::-1], "o-", label="multiplex hetero")
    plt.plot(tau_hor_list[::-1], list_defaulted_agg_homo[::-1], "o-", label="aggregate")
    plt.plot(tau_hor_list[::-1], list_defaulted_agg[::-1], "o-", label="aggregate hetero")
    plt.xlim([0, 1])
    plt.gca().invert_xaxis()
    plt.legend()
    #plt.savefig(export_fig_path + "MultiVsAgg_hetero" + seed_name[0] + str(year) +".png")
    plt.show()



state_history_agg, hor_contagion_agg, ver_contagion_agg, mix_contagion_agg, t_end_agg= hf.run_contagion([aggregate_am], C_agg, states_agg,\
hf.default_horizontal_homogenous, hf.default_vertical_homogenous, 0.333333333, tau_ver_none, save_csv=False, save_name="results/csv_contagion2019/aggregate_uk", country_names=names)
list_defaulted_agg_homo.append(hf.countries_defaulted(state_history_agg))

hf.countries_defaulted(state_history_agg)

state_history_agg, hor_contagion_agg, ver_contagion_agg, mix_contagion_agg, t_end_agg= hf.run_contagion([aggregate_am], C_agg, states_agg,\
hf.default_horizontal_homogenous, hf.default_vertical_homogenous, 0.34, tau_ver_none, save_csv=False, save_name="results/csv_contagion2019/aggregate_uk", country_names=names)
list_defaulted_agg_homo.append(hf.countries_defaulted(state_history_agg))

hf.countries_defaulted(state_history_agg)


state_history_agg, hor_contagion_agg, ver_contagion_agg, mix_contagion_agg, t_end_agg= hf.run_contagion([aggregate_am], C_agg, states_agg,\
hf.default_horizontal_homogenous, hf.default_vertical_homogenous, 0.4, tau_ver_none, save_csv=False, save_name="results/csv_contagion2019/aggregate_uk", country_names=names)
list_defaulted_agg_homo.append(hf.countries_defaulted(state_history_agg))

hf.countries_defaulted(state_history_agg)
