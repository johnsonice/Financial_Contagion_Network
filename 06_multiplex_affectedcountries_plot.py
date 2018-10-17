import sys 
import os
lib_path = './lib'
sys.path.insert(0, lib_path)     ## add lib path to sys path 
import pandas as pd
import MultiContagion as mc
import igraph
import random
import numpy as np
from matplotlib import pylab as plt
import scipy.stats
import copy
from mpl_toolkits.mplot3d import Axes3D

import_path = "./results/csv_contagionJuly/"
export_fig_path = "./results/fig_contagionJuly/"
year0 = 2009
year1=2016
year2=2015
countries_name_starting = ['EU', 'EU-uk', 'China', "United States", "United Kingdom", "China  P.R.: Hong Kong", "China  P.R.: Mainland", "Netherlands", "Luxembourg", "France", "Germany" ]
countries_abre_name = ['EU', 'EU-UK', 'China & HK', 'USA', 'UK', 'Hong Kong', 'China', 'Netherlands', 'Luxembourg', 'France', 'Germany']
for i in range(len(countries_name_starting)):
    df09 = pd.read_csv(import_path + 'MultAllLayNumAffected_seed' + countries_name_starting[i].replace(':','') + str(year0) + '.csv')
    df15 = pd.read_csv(import_path + 'MultAllLayNumAffected_seed' + countries_name_starting[i].replace(':','') + str(year1) + '.csv')
    df11 = pd.read_csv(import_path + 'MultAllLayNumAffected_seed' + countries_name_starting[i].replace(':','') + str(year2) + '.csv')
    plt.plot(df09['Threshold'], df09['Number of Affected Countries '], 'r^-', label = str(year0))
    plt.plot(df15['Threshold'], df15['Number of Affected Countries '], 'c*-', label = str(year1))
    if i < 2:
        plt.plot(df11['Threshold'], df11['Number of Affected Countries '], 'y+-', label = str(year2))
    plt.title(str(year0) + " and " + str(year1) +"multiplex contagion comparison " + "\n seed = "+ countries_abre_name[i] )
    plt.xlabel("Threshold")
    plt.ylabel("Number of affected countries")
    #box = ax.get_position()
    plt.legend(loc='upper left')
    plt.gca().invert_xaxis()
    if i < 2:
        plt.plot(df11['Threshold'], df11['Number of Affected Countries '], 'y+-', label = str(year2))
        plt.savefig(export_fig_path+"yearComparison091115"+"seed"+ countries_name_starting[i].replace(':','') + ".png", bbox_inches='tight')
    else:
        plt.savefig(export_fig_path+"yearComparison0915"+"seed"+ countries_name_starting[i].replace(':','') + ".png", bbox_inches='tight')
    plt.show()

#%%

thresh35_index = 4
np.around(df09['Threshold'][thresh35_index])
affected_at_3509 = []
affected_at_3515 = []
for i in range(len(countries_name_starting)):
    df09 = pd.read_csv(import_path + 'MultAllLayNumAffected_seed' + countries_name_starting[i].replace(':','') + str(year0) + '.csv')
    df15 = pd.read_csv(import_path + 'MultAllLayNumAffected_seed' + countries_name_starting[i].replace(':','') + str(year1) + '.csv')
    df11 = pd.read_csv(import_path + 'MultAllLayNumAffected_seed' + countries_name_starting[i].replace(':','') + str(year2) + '.csv')
    affected_at_3509.append(df09['Number of Affected Countries '][thresh35_index])
    affected_at_3515.append(df15['Number of Affected Countries '][thresh35_index])

y_pos = np.arange(len(countries_name_starting))
y_pos2 = y_pos + 0.3
plt.bar(y_pos, affected_at_3509, 0.3,align='center', alpha=0.8, color='r', label=str(year0))
plt.bar(y_pos2, affected_at_3515, 0.3,align='center', alpha=0.8, color='c',label=str(year1))
plt.xticks(y_pos, countries_abre_name, rotation=60)
plt.xlim([-0.5, len(countries_abre_name) + 1.5])
plt.ylabel('Number of affected countries')
plt.title('Countries Affected \n (Threshold {})'.format(np.around(df09['Threshold'][thresh35_index],2)))
plt.legend()
plt.savefig(export_fig_path+"Summary" + str(year0)+str(year1)+'TH' + str(np.around(df09['Threshold'][thresh35_index],2)) + ".png", bbox_inches='tight')
plt.show()
