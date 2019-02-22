'''Helper functions
'''
import numpy as np
import copy

def make_initial_states(seed_names=None, names=None, seed_nodes=None, \
                        seed_layers=None,n_countries=213, n_layers=5):
    """ function that received a list of nodes/countries which are
    seeds for contagion and returns the array(n x l) of initial states
    for both the aggregate and the multiplexexi
    seed_names(list): if not none should be a list of names of seed nodes
    names(list): if not none, list of countries names
    seed_nodes(list): if not none, list of index of seed nodes
    seed_layers(list): if not none should be a list of layers of seed node
    """
    assert (seed_names is None or seed_nodes is None)
    # making the seed node list if not given
    if seed_nodes == None:
        seed_nodes = [names.index(i) for i in seed_names]
    if seed_layers==None:
        seed_layers = [l for l in range(n_layers)]

    # making the states array and adding the seeds
    states = np.zeros([n_countries, n_layers])
    states_agg = np.zeros([n_countries, 1])

    for c in seed_nodes:
        states_agg[c, :] = 1
        for l in seed_layers:
            states[c,l] = 1

    return states, states_agg

def default_horizontal(c, l, defaulted_assets, instrength_layer_list, tau_hor):
    ''' old since threhsolds are int
    Rule for contagion considers all infected layers
    Args:   c(int): country index
            l(int): layer index
            defaulted_assets(1d-array):(horizontal) defaulted assests
            instrength_layer_list(2d-array): array of instr of node,layers
            tau_hor((nx1) np array): array with threshold of country
    Returns: (Bool): True if default conditon is met, False otherwise'''
    if instrength_layer_list[c, l] == 0:
        # print("no indegree")
        return False

    elif defaulted_assets[c] / instrength_layer_list[c, l] > tau_hor[c]:
        return True
    else:
        return False


def default_vertical(c, l, defaulted_assets, instrength_layer_list, tau_ver, type_ver="layer"):
    '''
    Rule for contagion considers all infected layers
    Args:   c(int): country index
            l(int): layer index
            defaulted_assets(1d-array): (vertical) defaulted assests
            instrength_layer_list(2d-array): array of instr of node,layers
            tau_ver((lx1)nparray): threshold for each layer
            type(str): layer if instregth of layer is taken into account
                        or global is it is instrength over all layers
    Returns: (Bool): True if default conditon is met, False otherwise'''

    if type_ver == "layer":
        if instrength_layer_list[c, l] == 0:
            # print("no indegree")
            return False
        elif defaulted_assets[c] / instrength_layer_list[c, l] > tau_ver[l]:
            return True
        else:
            return False
    if type_ver == "global":
        if instrength_layer_list[c, l] == 0:
            # print("no indegree")
            return False
        elif defaulted_assets[c] / sum(instrength_layer_list[c, :]) > tau_ver[l]:
            return True
        else:
            return False

def default_horizontal_homogenous(c, l, defaulted_assets, instrength_layer_list, tau_hor):
    ''' homogenous since threhsolds are the same for all countries
    Rule for contagion considers all infected layers
    Args:   c(int): country index
            l(int): layer index
            defaulted_assets(1d-array):(horizontal) defaulted assests
            instrength_layer_list(2d-array): array of instr of node,layers
    Returns: (Bool): True if default conditon is met, False otherwise'''
    if instrength_layer_list[c, l] == 0:
        # print("no indegree")
        return False

    elif defaulted_assets[c] / instrength_layer_list[c, l] > tau_hor:

        return True
    else:

        return False


def default_vertical_homogenous(c, l, defaulted_assets, instrength_layer_list, tau_ver, type_ver="layer"):
    '''homogenous since threhsolds are the same for all countries
    Rule for contagion considers all infected layers
    Args:   c(int): country index
            l(int): layer index
            defaulted_assets(1d-array): (vertical) defaulted assests
            instrength_layer_list(2d-array): array of instr of node,layers
            type(str): layer if instregth of layer is taken into account
                        or global is it is instrength over all layers
    Returns: (Bool): True if default conditon is met, False otherwise'''

    if type_ver == "layer":
        if instrength_layer_list[c, l] == 0:
            # print("no indegree")
            if defaulted_assets[c] > 0:
                return True
            else:
                return False
        elif defaulted_assets[c] / instrength_layer_list[c, l] > tau_ver:
            return True
        else:
            return False
    if type_ver == "global":
        if instrength_layer_list[c, l] == 0:
            # print("no indegree")
            if defaulted_assets[c] > 0:
                return True
            else:
                return False
        elif defaulted_assets[c] / sum(instrength_layer_list[c, :]) > tau_ver:
            return True
        else:
            return False

def countries_defaulted(state_history):
    """Args(array): state array of countries, layer
    Return(Int): number of countries infected in at least one layer
    """
    # vector with number of layers in which the coutnry has defaulted
    country_default = np.sum(state_history[-1, :, :], axis=1)
    # return number of times country defaulted in at least one layer
    return np.count_nonzero(country_default)

def add_to_csv(file, t, solvent, l, direction, country_names=0):
    """
    Args:   file(file): csv file in which to write
            t(int): time step
            solvent(int): solvent node index
            l(int): layer index of solvent node
            direction(str): horizontal or vertical
            country_names(list or int): if list, names of countries
    """
    if country_names != 0:
        country = country_names[solvent]
    else:
    # otherwise save the index of country
        country = str(solvent)
    file.write(str(t) + "," + str(l) + "," + country + \
            "," + direction + "\n")

def one_step_contagion(n_layers, states, t, defaulted_assets_hor, defaulted_assets_ver,\
                instregth, tau_hor, tau_ver, hor_contagion, ver_contagion,\
                mix_contagion, default_hor, default_ver, add_to_csv, f, \
                country_names=0, save_csv=False,type_ver="layer",):
    """
    Args:   n_layers(int): number of layers
            states((n,l) nparray): states(0 or 1) of node-layers
            t(int): time step of the simulation
            defaulted_assets_hor((n,l)np array): defaulted assets of node layer
            due to defaults in their same layer
            defaulted_assets_hor((n)np array): defaultes assets of nodes due to
            defaults in other layer (same node)
            instregth(np array): total assets of a country
            tau_hor(np array): array with horizontal threholds per country
            tau_ver(np array): array with horizontal threholds per layer
            for now all are the same but left as array for flexibility

            hor_contagion((t_sim,)np array): array with number of horizontal
            defaults at that time

            default_hor(function):

            f(file): csv file where to save data
    Returns:states((n,l) nparray): updates states of node-layers
            hor_contagion: updated counter of horizontal contagion
            ver_contagion: updated counter of vertical contagion
            mix_contagion: updated counter of mix contagion
    """

    for l in range(n_layers):
        solvent_nodes = np.where(states[:, l] == 0)[0]
        for solvent in solvent_nodes:
            # variable that checks if contagion is both hor and ver
            mix = 0
            # check condition for horizontal contagion

            if default_hor(solvent, l, defaulted_assets_hor[:, l], instregth, tau_hor):
                # if default, update status and add to hor and mix count
                # print("horizontal contagion ", solvent, "lay ", l)
                states[solvent, l] = 1
                hor_contagion[t] += 1
                mix += 0.5
                if save_csv == True:
                    add_to_csv(f, t, solvent, l, "horizontal", \
                                country_names=country_names)


            if default_ver(solvent, l, defaulted_assets_ver, instregth, tau_ver, type_ver):
                # if default, update status and add to ver count
                # print("vertical contagion ", solvent, "lay ", l)
                # print("solvent node ", solvent, "lay ", l)
                # print("def assets ", defaulted_assets_ver[solvent])
                # print("in str ", instregth[solvent, l])
                states[solvent, l] = 1
                ver_contagion[t] += 1
                mix += 0.5

                if save_csv == True:
                    add_to_csv(f, t, solvent, l, "vertical", \
                                country_names=country_names)
            if mix == 1: #if it was both hor and vert then it is mixed
                mix_contagion[t] += 1

    return states, hor_contagion, ver_contagion, mix_contagion



def run_contagion(A_list, C, states_original, default_hor, default_ver, tau_hor,\
    tau_ver, type_ver="layer", save_csv=False, \
    save_name="results/csv_contagion2019/test", country_names=0, add_to_csv=add_to_csv):
    """
    Args:   A_list(list of np array): list with adjacency matrices of layers
            C(array): Coupling array, should be n_countries x n_layers
            states_original(array): initial states of counties 0 for liquid
                                    1 for defaulted
            default_hor(function): condition for horizontal contagion
            default_ver(function): condition for horizontal contagion
            tau_hor(float): horizontal threshold for contagion
            tau_ver(float): vertical threshold for contagion
            type(str): layer if instregth of layer is taken into account
                        or global is it is instrength over all layers

            add_to_csv(functions): modifies csv file
    Returns:
    """

    assert(A_list[0].shape[0] == C.shape[0])
    assert(len(A_list) == C.shape[1])

    # getting the number of countries and layers
    n_layers = len(A_list)
    n_countries = A_list[0].shape[0]
    # max number of interations is n x l since it is deterministic percolation
    n_sim = n_layers * n_countries

    #setting initial conditions and vectors of stored information
    seed_country = np.where(states_original[:, :] == 1)[0]
    seed_layer = np.where(states_original[:, :] == 1)[1]
    state_history = np.zeros([n_sim, n_countries, n_layers])
    hor_contagion = np.zeros([n_sim])
    ver_contagion = np.zeros([n_sim])
    mix_contagion = np.zeros([n_sim])
    state_history[0, :, :] = copy.deepcopy(states_original)

    # setting vectors of partially stored information
    states = copy.deepcopy(states_original)
    defaulted_assets_hor = np.zeros([n_countries, n_layers])
    # nodes of the same country have == vertical defaulted assets
    defaulted_assets_ver = np.zeros([n_countries])

    if save_csv == True:
        f = open(save_name + ".csv", "w")
        f.write("time step,layer,country,contagion type\n")
    else:
        f = None

    # (in)strength vector with which lost assests will be compared
    instregth = np.zeros([n_countries, n_layers])
    for l in range(n_layers):
        # for outstrength axis=1
        instregth[:, l] = np.sum(A_list[l], axis=0)

    # simulation starts
    for t in range(1, n_sim):

        # get intralayer(horizontal) defaulted assets
        for l in range(n_layers):
            # indegree contagion thus, sum_j s_jA_ji; s vec and Adjmat of layer l
            defaulted_assets_hor[:, l] = states[:, l].dot(A_list[l])


        # get interlayer(vertical) defaulted assets
        for c in range(n_countries):
            defaulted_assets_ver[c] = sum(np.multiply(states[c, :], C[c, :]))
        # NOTE "self-loop" is considered but since if s=1, it is already
        # defaulted we don't care. also same as C_array.dot(states[0,:])

        # run contagion for one time step
        states, hor_contagion, ver_contagion, mix_contagion = \
        one_step_contagion(n_layers, states, t, defaulted_assets_hor,\
        defaulted_assets_ver, instregth, tau_hor, tau_ver, hor_contagion,\
        ver_contagion, mix_contagion, default_hor, default_ver,add_to_csv, f,
        country_names=country_names, save_csv=save_csv, type_ver=type_ver)

        # record the state history
        state_history[t, :, :] = copy.deepcopy(states)

        # if nothing happend in two time steps, then break
        if ver_contagion[t - 2] == ver_contagion[t] == 0 and \
            hor_contagion[t - 2]  == hor_contagion[t] == 0:
            t_end = t
            # print("break happened ")
            # print(hor_contagion[t - 2], " ",  hor_contagion[t])
            break

    if save_csv == True:
        f.close()

    return state_history[:t_end, :, :], hor_contagion[:t_end],\
            ver_contagion[:t_end], mix_contagion[:t_end], t_end


# def run_contagion(A_list, C, states_original, default_hor, default_ver, tau_hor,\
#     tau_ver, type_ver="layer", save_csv=False, \
#     save_name="results/csv_contagion2019/test", country_names=0):
#     """
#     Args:   A_list(list of np array): list with adjacency matrices of layers
#             C(array): Coupling array, should be n_countries x n_layers
#             states_original(array): initial states of counties 0 for liquid
#                                     1 for defaulted
#             default_hor(function): condition for horizontal contagion
#             default_ver(function): condition for horizontal contagion
#             tau_hor(float): horizontal threshold for contagion
#             tau_ver(float): vertical threshold for contagion
#             type(str): layer if instregth of layer is taken into account
#                         or global is it is instrength over all layers
#     Returns:
#     """
#
#     assert(A_list[0].shape[0] == C.shape[0])
#     assert(len(A_list) == C.shape[1])
#
#     # getting the number of countries and layers
#     n_layers = len(A_list)
#     n_countries = A_list[0].shape[0]
#     # max number of interations is n x l since it is deterministic percolation
#     n_sim = n_layers * n_countries
#
#     #setting initial conditions and vectors of stored information
#     seed_country = np.where(states_original[:, :] == 1)[0]
#     seed_layer = np.where(states_original[:, :] == 1)[1]
#     state_history = np.zeros([n_sim, n_countries, n_layers])
#     hor_contagion = np.zeros([n_sim])
#     ver_contagion = np.zeros([n_sim])
#     mix_contagion = np.zeros([n_sim])
#     state_history[0, :, :] = copy.deepcopy(states_original)
#
#     # setting vectors of partially stored information
#     defaulted_assets_hor = np.zeros([n_countries, n_layers])
#     # nodes of the same country have == vertical defaulted assets
#     defaulted_assets_ver = np.zeros([n_countries])
#     states = copy.deepcopy(states_original)
#
#     # open file in which to save results
#     if save_csv == True:
#         f = open(save_name + ".csv", "w")
#         f.write("time step,layer,country,contagion type\n")
#
#     # setting the instrenth vector
#     instregth = np.zeros([n_countries, n_layers])
#     for l in range(n_layers):
#         #TODO check correct axis
#         #instregth[:, l] = np.sum(A_list[l], axis=1)
#         instregth[:, l] = np.sum(A_list[l], axis=0)
#
#     # running simulation, max time steps is n * l.
#     for t in range(1, n_layers*n_countries):
#         # add list entry where index of countries will be stored
#         #country_first_contagion.append([])
#         # get intralayer(horizontal) defaulted assets
#         for l in range(n_layers):
#             # indegree contagion thus, sum_j s_jA_ji; s vec and Adjmat of layer l
#             defaulted_assets_hor[:, l] = states[:, l].dot(A_list[l])
#
#         # get interlayer(vertical) defaulted assets
#         for c in range(n_countries):
#             # for each country look at the sum of their defaulted counterparts
#             defaulted_assets_ver[c] = sum(states[c, :].dot(C.T))
#             # same as C_array.dot(states[0,:])
#             # NOTE "self-loop" is considered but since if s=1, it is already
#             # defaulted we don't care.
#             # Also we use np.multiply since C_list[l] is a vector
#
#         # iterate over layers and (solvent) nodes to update their status
#         for l in range(n_layers):
#             solvent_nodes = np.where(states[:, l] == 0)[0]
#             for solvent in solvent_nodes:
#                 # variable that checks if contagion is both hor and ver
#                 mix = 0
#                 # check condition for horizontal contagion
#                 if default_hor(solvent, l, defaulted_assets_hor[:, l], instregth, tau_hor):
#                     # if default, update status and add to hor count
#                     states[solvent, l] = 1
#                     hor_contagion[t] += 1
#                     mix += 1
#                     if save_csv == True:
#                         # if a list of country names is given use it
#                         if country_names != 0:
#                             country = country_names[solvent]
#                         # otherwise save the index of country
#                         else:
#                             country = (str)
#                         f.write(str(t) + "," + str(l) + "," + country + \
#                                 "," + "horizontal" + "\n")
#
#                 if default_ver(solvent, l, defaulted_assets_ver, instregth, tau_ver, type_ver):
#                     # if default, update status and add to ver count
#                     states[solvent, l] = 1
#                     ver_contagion[t] += 1
#                     mix += 1
#                     if save_csv == True:
#                         # if a list of country names is given use it
#                         if country_names != 0:
#                             country = country_names[solvent]
#                         else:
#                         # otherwise save the index of country
#                             country = str(solvent)
#                         f.write(str(t) + "," + str(l) + "," + country + \
#                                 "," + "vertical" + "\n")
#                 if mix == 2:
#                     mix_contagion[t] += 1
#
#         # record the state history
#         state_history[t, :, :] = copy.deepcopy(states)
#
#         # if nothing happend in two time steps, then break
#         if ver_contagion[t - 2] == ver_contagion[t] and \
#             hor_contagion[t - 2]  == hor_contagion[t]:
#             t_end = t
#             break
#
#     if save_csv == True:
#         f.close()
#
#     return state_history[:t_end, :, :], hor_contagion[:t_end],\
#             ver_contagion[:t_end], mix_contagion[:t_end], t_end




def make_supra_coup(list_adj, list_vector):
    #TODO make function accoridng to matrix of coupling (for multiplex networks)
    """
    Might be OUTDATED

    Makes supradjacency matrix from list of adjacency matrices and coupling
    vectors
    Args:   list_adj(list of np array): list of adj mat
            list_vector(list of np array): list of coupling vectors
    Returns:SupAM(np matrix): supradjacency matrix"""

    #getting dimensions (layers and nodes)
    n_layers = len(list_adj)
    dim = list_adj[0].shape[0]
    # assert(len(list_vector) == dim)
    # start list where matrices will stack
    sup_list = []
    #iterate over layers to start making the matrix
    for i in range(n_layers):
        row = []
        for j in range(n_layers):
            if i == j:
                # adjacency matrix of layer
                row.append(list_adj[i])
            else:
                # coupling between layers
                row.append(np.diag(list_vector[j]))
        # stack the row of adj matrices
        sup_list.append(row)
        # convert into matrix
        SupAM = np.bmat(sup_list)
    return SupAM
