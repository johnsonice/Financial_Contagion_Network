
import numpy as np
import math
import networkx as nx
import copy
import itertools
import scipy.stats
from matplotlib import pylab as plt
from scipy.sparse import linalg as LA
from scipy import linalg as la
import matplotlib.gridspec as gridspec

class MultiplexNet:
    ###########################
    # --- initializing multiplex --- #
    ###########################
    def __init__(self, dataframe=None, adj_list=None, supra_adj=None, \
                 nodedict=None,layerdict=None, nodesmap="inoutedges", \
                 coupling_type="riskflow", nettype="multiplex"):
        """
        Args:
            dataframe(DataFrame): Pandas dataframe with the network information
            adj_list(np matrix): Class can also take a list of adjacency
                matrices to map onto multiplex
            nodedc(dict): nodes and names
            layerdc(dict): layers and names
            nodesmap(str): determines if only strongly connected component 
                is maped
            coupling_type(str): type of coupling between layers
            type(str): define if the network is a multiplex or a multilayer
                
        """
        
        if not isinstance(dataframe, type(None)):
            
            if nettype == "multiplex":
           
                # Define nodes and layers
                self.nodes = self.define_nodesandlayers(dataframe, nodesmap)[0]
                self.layers = self.define_nodesandlayers(dataframe, nodesmap)[1]
                self.countries = self.define_nodesandlayers(dataframe, nodesmap)[2]
                self.sectors = self.define_nodesandlayers(dataframe, nodesmap)[3]
                
                self.nnodes = len(self.nodes)
                self.nlayers = len(self.layers)
                
                # Make a graph for each layer in the adjacency matrix
                self.graph_list = self.get_graphsandadjmatriceslist(dataframe)[0]
                self.adj_matrix_list = self.get_graphsandadjmatriceslist(dataframe)[1]
                
                # Define coupling between layers
                self.coupling = self.coupling_list(coupling_type)
                self.supra_adj_matrix = self.make_supra_adjacency(self.adj_matrix_list\
                                                                 ,self.coupling)
            

        elif not isinstance(adj_list, type(None)):
            assert(isinstance(adj_list, type([])))
            self.adj_matrix_list = adj_list
            self.nodes = nodedict
            self.layers = layerdict
            self.nnodes = len(self.nodes)
            self.nlayers = len(self.layers)
            # reverse mappings
            self.countries  = dict(map(reversed, nodedict.items()))
            self.sectors = dict(map(reversed, layerdict.items()))
            
            # Define coupling between layers
            self.coupling = self.coupling_list(coupling_type)
            self.supra_adj_matrix = self.make_supra_adjacency(self.adj_matrix_list\
                                                             ,self.coupling)
            

        elif not isinstance(supra_adj, type(None)):
            assert(len(nodedict)*len(layerdict) == supra_adj.shape[0])
            self.supra_adj_matrix = supra_adj
            self.nodes = nodedict
            self.layers = layerdict
            # reverse mappings
            self.countries  = dict(map(reversed, nodedict.items()))
            self.sectors = dict(map(reversed, layerdict.items()))
            # # ndoes and lay
            self.nnodes = len(self.nodes)
            self.nlayers = len(self.layers)
            # get adjacency matrix of layers using supradajacency
            self.adj_matrix_list = []
            for l in range(self.nlayers):
                self.adj_matrix_list.append(self.supra_adj_matrix[l*\
                            self.nnodes:(l+1)*self.nnodes, l*self.nnodes:(l+1)\
                            *self.nnodes])
            
            
            
        
    def define_nodesandlayers(self, dataframe, nodesconsidered):
        """ Uses dataframe to obtain the nodes and layers. The mutliplex is 
        node align, i.e. all countries are in all layers even if the do not 
        have assets in all layers. There are two ways of mapping the data 
        into a multiplex, "all" considers all countries that have at least one
        incoming or outgoing link in any layer and "inoutedges" considers only
        countries that have at least one incoming and at least one outgoing 
        link in at least one layer. 
        """
        # NODES. Get the countries reported in data
        countries_out = list(dataframe["country"].value_counts().keys())
        countries_in = list(dataframe["counterpart"].value_counts().keys())
        
        if nodesconsidered == "all":
            countries = list(set().union(countries_out , countries_in ))
            countries.sort()
        elif nodesconsidered == "inoutedges":
            countries = list(set(countries_out).intersection(set(countries_in)))
            countries.sort()
        
        # LAYERS. Get the layers reported in data
        layers = list(dataframe.columns)
        layers.remove("country")
        layers.remove("counterpart")
        # Get dictionaries where you input the index of country and get name bk
        dict_index_country = dict(enumerate(countries))
        dict_index_layer = dict(enumerate(layers))
        # Get dictionaries where you input name of country/layer and get index
        dict_country_index  = dict(map(reversed, dict_index_country.items()))
        dict_country_layer  = dict(map(reversed, dict_index_layer.items()))
        return dict_index_country, dict_index_layer, dict_country_index,\
                dict_country_layer
    
    def define_nodesandlayers_multilayer(self, dataframe, nodesconsidered):
        """ 
        For mutliplex dataframe uses
        Uses dataframe to obtain the nodes and layers. The mutliplex is 
        node align, i.e. all countries are in all layers even if the do not 
        have assets in all layers. There are two ways of mapping the data 
        into a multiplex, "all" considers all countries that have at least one
        incoming or outgoing link in any layer and "inoutedges" considers only
        countries that have at least one incoming and at least one outgoing 
        link in at least one layer. 
        """
        # NODES. Get the countries reported in data
        countries_out = list(dataframe["country"].value_counts().keys())
        countries_in = list(dataframe["counterpart"].value_counts().keys())
        
        if nodesconsidered == "all":
            countries = list(set().union(countries_out , countries_in ))
            countries.sort()
        elif nodesconsidered == "inoutedges":
            countries = list(set(countries_out).intersection(set(countries_in)))
            countries.sort()
        
        # LAYERS. Get the layers reported in data
        #TODO below for multilayer
        layer_out = set(dataframe["country_sector"].value_counts().index.tolist())
        layer_in = set(dataframe["counterpart_sector"].value_counts().index.tolist())
        new_set = layer_out.intersection(layer_in)
        layers = list(new_set)
        # Get dictionaries where you input the index of country and get name bk
        dict_index_country = dict(enumerate(countries))
        dict_index_layer = dict(enumerate(layers))
        # Get dictionaries where you input name of country/layer and get index
        dict_country_index  = dict(map(reversed, dict_index_country.items()))
        dict_country_layer  = dict(map(reversed, dict_index_layer.items()))
        return dict_index_country, dict_index_layer, dict_country_index,\
                dict_country_layer    
    
    def get_graphsandadjmatriceslist(self, dataframe):
        """Takes dataframe, the predefines nodes and layers and maps the 
        multiplex network
        Returns
            G_listlist(list)): list of graphs, one graph per layer
            A_listlist(list)): list of matrices, one matrix per layer.
                                matrices are Scipy sparse matrices
        """
        # start by having a graph for each layer with n nodes.
        G_list = []
        for l in range(self.nlayers):
            G = nx.DiGraph()
            G.add_nodes_from(range(self.nnodes))
            G_list.append(G)
        
        # dictionaries with input= country(layer) name, output = index
        country_index  = dict(map(reversed, self.nodes.items()))

        for index, row in dataframe.iterrows():
            country = row["country"]
            counterpart = row["counterpart"]
            # if countries are in network (recall nodes can be discareded 
            # when using inoutedges)
            if country in country_index and counterpart in country_index:
                country_id = country_index[country]
                counterpart_id = country_index[counterpart]
                for l in range(self.nlayers):
                    lname = self.layers[l]
                    asset = row[lname]
                    if asset > 0:
    #                if asset.notnull() and asset > 0:    
                        G_list[l].add_edge(country_id, counterpart_id,\
                                       weight=asset)
                                
        A_list = [np.array(nx.to_numpy_matrix(G)) for G in G_list]
        
        return G_list, A_list
    
#    def get_graphsandadjmatriceslist_multilayer(self, dataframe):
#        """ As before but now give other two lists that define the interlayer 
#        adjacency matrix and graphs. This ones are quasi bipartite, bipartite
#        but with self-loops
#        Takes dataframe, the predefines nodes and layers and maps the 
#        multiplex network
#        Returns
#            G_list(list)): list of graphs, one graph per layer
#            A_list(list)): list of matrices, one matrix per layer.
#                                matrices are Scipy sparse matrices
#            G_interlayer_list(list):
#        """
#        # make a graph for each layer with n nodes.
#        G_list = []
#        for l in range(self.nlayers):
#            G = nx.DiGraph()
#            G.add_nodes_from(range(self.nnodes))
#            G_list.append(G)
#        # Also make a graph for each interlayer, they are l*l - l,
#        G_interlayer_list = []
#        for l in range(int(self.nlayers * (self.nlayers - 1))):
#            G = nx.DiGraph()
#            G.add_nodes_from(range(self.nnodes))
#            G_interlayer_list.append(G)
#        
#        # dictionaries with input= country(layer) name, output = index
#        country_index  = dict(map(reversed, self.nodes.items()))
#
#        for index, row in dataframe.iterrows():
#            country = row["country"]
#            counterpart = row["counterpart"]
#            # if countries are in network (recall nodes can be discareded 
#            # when using inoutedges)
#            if country in country_index and counterpart in country_index:
#                country_id = country_index[country]
#                counterpart_id = country_index[counterpart]
#                # if assets are within the same layer
#                if row["country_sector"] == row["counterpart_sector"]:
#                    l = self.layers[row["country_sector"]]
#                    # TODO make it flexible on year
#                    asset = row["2016"]
#                    if asset > 0:
#                        G_list[l].add_edge(country_id, counterpart_id,\
#                                           weight=asset)
#                else:
#                    layer_out = self.layers[row["country_sector"]]
#                    layer_int = self.layers[row["counterpart_sector"]]
#                    asset = row["2016"]
#                    if asset > 0:
#                        G_interlayer_list[].add_edge(country_id, counterpart_id,\
#                                           weight=asset)
##                    for l in range(self.nlayers):
##                        lname = self.layers[l]
##                        asset = row[lname]
##                        if asset > 0:
##        #                if asset.notnull() and asset > 0:    
##                            G_list[l].add_edge(country_id, counterpart_id,\
##                                           weight=asset)
#                                
#        A_list = [np.array(nx.to_numpy_matrix(G)) for G in G_list]
#        A_interlayer_list = [np.array(nx.to_numpy_matrix(G)) for G in G_interlayer_list]
#        
#        return G_list,G_interlayer_list, A_list, A_interlayer_list
    
                
    def coupling_list(self, coupling_type):
        """ Function that defined the coupling depending on the specified
        coupling type. Note coupling elements are vectors since we assume 
        multiplex network.
        """
        if type(coupling_type) == list:
            for element in coupling_type:
                assert(type(element) == np.ndarray)
                assert(element.shape == (self.nnodes,))
            coupling = coupling_type
        elif coupling_type == "riskflow":
            # list with instrength of each node for each layer
            coupling = [ am.sum(axis=0) for am in self.adj_matrix_list]
        elif coupling_type == "cashflow":
            # list with outstrength of each node for each layer
            coupling = [ am.sum(axis=1) for am in self.adj_matrix_list]
        return coupling
   
    def make_supra_adjacency(self, list_adjmat, list_coup):
        """ makes the supradjacency matrix from the adj mat of each layer
        and the coupling vector
        """

        sup_list = []
        for i in range(self.nlayers):
            row = []
            for j in range(self.nlayers):
                if i == j:
                    row.append(list_adjmat[i])
                else:
                    row.append(np.diag(list_coup[j]))
            sup_list.append(row)
            
        return np.array(np.bmat(sup_list))
    def make_supra_adjacency_multilayer(self, list_adjmat, list_coup):
        """ makes the supradjacency matrix from the adj mat of each layer
        and the coupling vector
        """

        sup_list = []
        for i in range(self.nlayers):
            row = []
            for j in range(self.nlayers):
                if i == j:
                    row.append(list_adjmat[i])
                else:
                    row.append(np.diag(list_coup[j]))
            sup_list.append(row)
            
        return np.array(np.bmat(sup_list))
     
    # =========================================================================
    # Centrality measures
    # =========================================================================
       
    def add_centrality(self, all_centrality):
        """ Function that appropriately adds the centrality
        """
        centrality = {}
        for i in range(self.nnodes):
            cent = 0
            for l in range(self.nlayers):
                cent += all_centrality[i + self.nnodes*l]
            if type(cent) == np.ndarray:
                cent = float(cent[0])
            centrality[i] = cent
        return centrality
    
    def pagerank(self):
        """Compute the Multiplex pagerank of all countries. returned as vector
        """
        G_supra = nx.from_numpy_array(self.supra_adj_matrix, \
                                      create_using=nx.DiGraph())
        all_pagerank = nx.pagerank(G_supra)
        pagerank = self.add_centrality(all_pagerank)
        
        ## add into a dictionary
        pagerank_dc = {}
        for i in range(self.nnodes):
            pagerank_dc[i] = pagerank[i]
        return pagerank_dc
    
    def hubs_and_auth(self):
        """
        """
        # making the hubs and auths matrices
        A_hub = []
        A_auth = []
        for am in self.adj_matrix_list:
            new_hub = np.dot(am, am.transpose())
            new_auth = np.dot(am.transpose(), am)
            A_hub.append(new_hub) 
            A_auth.append(new_auth)
        
        coup_list = []
        for c in self.coupling:
            coup_list.append(c*c)
            
        supra_hub = self.make_supra_adjacency(list_adjmat=A_hub,\
                                              list_coup=coup_list)
        supra_auth = self.make_supra_adjacency(list_adjmat=A_auth,\
                                              list_coup=coup_list)
        
        hub_eig = LA.eigs(supra_hub, k=1, which='LM')[1]
        auth_eig = LA.eigs(supra_auth, k=1, which='LM')[1]
        hubs = self.add_centrality(hub_eig)
        authorities = self.add_centrality(auth_eig)
        # take positive value
        if sum(authorities) < 0:
            authorities = -1*authorities
        if sum(hubs) < 0:
            hubs = -1*hubs
        # add results into dictionary
        hubs_dc = {}
        auth_dc = {}
        for i in range(self.nnodes):
            hubs_dc[i] = hubs[i]
            auth_dc[i] = authorities[i]
        return hubs_dc, auth_dc
    
    def aggregate_katz(self, weights=None):
        # for an aggregate net the adjacency matrix = supradj
        # TODO perhaps normalize
        G = nx.from_numpy_array(self.supra_adj_matrix, create_using=nx.DiGraph())
        beta = {}
        if type(weights) == type(None):
            beta = 1.0
        else:
            for i in range(self.nnodes):
                beta[i] = weights[i]
        # define damping factor (recomend > 0.1, but 0.1 is safe for convergence)
        eigs = la.eigvals(self.supra_adj_matrix)
        eigs.sort()
        alpha = 1/eigs[-1].real
        # rank nodes with katz
        return nx.katz_centrality_numpy(G, alpha=alpha, beta=beta, weight="weight")
        
    def multirank_G(self, z, direction="Debtor"):
        """function that update G value
        """
        G = np.zeros([self.nnodes, self.nnodes])
        for l in range(self.nlayers):
            if direction == "Debtor":
                G = G + self.adj_matrix_list[l] * z[l]
            elif direction == "Creditor":
                G = G + self.adj_matrix_list[l].T * z[l]
            else:
                print("ERROR direction is not Debtor or Creditor")
        return G
    
    def multirank_z(self, x, W, B_in, gamma=1, a=1, s=1):
        """function that update z value
        """
        z = (W.sum())**a * (B_in.dot(x**(s*gamma)))**s
        z = z / z.sum()
        return z
    
    def multirank(self, alpha=0.85, gamma=1, a=1, s=1, tolerance=0.001, \
                  mode="pagerank", v_katz=None, direction="Debtor"):
        """ Computes multirank for nodes and layers. Can do it for PageRank
        or Katz centrality as indicated by mode.
        """
        W = np.array([A.sum() for A in self.adj_matrix_list])
        B_in = np.zeros([self.nlayers, self.nnodes])
        for i in range(self.nlayers):
            # Depenging on direction use B_in or B_out
            if direction == "Debtor":
                # sum over columns sum_j A_{ji}
                B_in[i, :] = self.adj_matrix_list[i].sum(axis=1)
            elif direction == "Creditor":
                # sum over columns sum_j A_{ij}
                # In principle we should name it B_out
                B_in[i, :] = self.adj_matrix_list[i].sum(axis=0)
        # compute alpha for katz by the eigenvalue of the layers
        if mode == "katz" or mode == "katz_weighted":
            max_eig = 0.000001
            for l in range(self.nlayers):
                eigs= la.eigvals(self.adj_matrix_list[l])
                eigs.sort()
                eig = eigs[-1].real
                if eig > max_eig:
                    max_eig = eig
            alpha_katz = 1 / (self.nlayers * max_eig)
        # initial iteration conditions
        z = np.ones(self.nlayers)
        G = self.multirank_G(z)
        v = np.where((G + G.T).sum(axis=0) > 0, 1, 0)
        x = v/np.sum(v)
        x_old = np.inf * np.ones(self.nnodes)
        
        count = 0
        # start iterating until convergence
        while np.linalg.norm(x - x_old) > tolerance*np.linalg.norm(x):
            count+=1
            x_old = x[:]
            # recalculate G from z
            G = self.multirank_G(z)
            
           # recalcualte variables that depend on G outdegree
            v = np.where((G + G.T).sum(axis=1) > 0, 1, 0)
            if mode == "pagerank":
                # vector incorporating heavyside function of G outdegree
                heaviside = np.where(G.sum(axis=1) > 0, 1, 0)
                # making diag matrix of outdegree of G, inputing 1 if 0 outdeg
                D = np.diag(np.ones(self.nnodes)/(G.sum(1)+(G.sum(1)==0)))
                # Updating centrality measure
                # Term for teleportation excluding isolated nodes,thus heavysi
                beta = np.dot(1 - alpha*heaviside, x/np.sum(v))
                # Centrality is two terms: moving among neighbors + teleport      
                x = alpha * np.dot(np.dot(G.T, D), x) + beta*v
            elif mode == "katz":
                x = alpha_katz * np.dot(G.T, x) + v
            elif mode == "katz_weighted":
                x = alpha_katz * np.dot(G.T, x) + v_katz
            else:
                print("warning")

            # Recalculate layer influence
            z = self.multirank_z(x, W, B_in, gamma=gamma, a=a, s=s)

            if count > 10000:
                print("Did not converge ")
                break
        multirank ={}
        layer_rank ={}
        for i in range(self.nnodes):
            multirank[i] = x[i]
        for l in range(self.nlayers):
            layer_rank[l] = z[l]
        return multirank, layer_rank
    
    
    def dbr_v_mono(self):
        """Computes the relative value vector for a monoplex network
        """
        if len(self.adj_matrix_list) > 1:
            raise TypeError("monoplex debtrank takes one adjacency matrix")
        L = self.adj_matrix_list[0].sum(axis=1)
        v = L/L.sum()
        return v

    def dbr_W_mono(self, equity, a):
        """ Computes the impact matrix for a monoplex network
        """
        if len(self.adj_matrix_list) > 1:
            raise TypeError("monoplex debtrank takes one adjacency matrix")
        
        if min(equity) <= 0:
            print("WARNING Equity of all countries must be positive "+\
                  "inputing 1 when not positive")
            equity = np.maximum(np.ones(self.nnodes), equity)

        adj_mat = self.adj_matrix_list[0]
        D = np.diag(1 / equity)
        W = np.minimum(np.ones([self.nnodes, self.nnodes]), \
                       a*adj_mat.T.dot(D))
        return W
            

    def dbr_v_mx(self):
        """Computes the value matrix for a multiplex network
        returns array of dim (nl,)
        """
        # getting relative economic value of node-layers
        v = self.adj_matrix_list[0].sum(axis=1)
        for l in range(1, self.nlayers):
            v = np.concatenate((v, self.adj_matrix_list[l].sum(axis=1)), \
                               axis=0)
        v = v / v.sum()
        return v
    
    def dbr_W_intra(self, alpha, beta, equity, a=0.2):
        """function that calculates the impact matrix within one layer. 
        Needed
        for multiplex debtrank needed to compute the multiplex debtrank
        """
        D = np.diag(1 / equity[beta]) 
        w_temp = a*self.adj_matrix_list[alpha].T.dot(D)
        W = np.minimum(np.ones([self.nnodes, self.nnodes]), w_temp)
        return W

    def dbr_W_inter(self, alpha, beta, equity, b=1):
        """function that caclulates the impact matrix between layers. 
        Since network is multiplex W_inter is diagonal
        """
        L = self.adj_matrix_list[alpha].sum(axis=1)
        d = 1 / equity[beta] 
        w_diag = np.minimum(np.ones(self.nnodes), b*np.multiply(L, d))
#        W = np.min(np.ones([self.nnodes, self.nnodes]), gamma * \
#                   self.adj_matrix_list[alpha].T.dot(D))
        W = np.diag(w_diag)
        return W
    
    def dbr_W_mx(self, equity, a, b):
        """Computed the impact matrix for a multiplex network, uses
        functions W_inter and W_intra to map impact within and between layers
        """
        n = self.nnodes
        W = np.zeros([n*self.nlayers, n*self.nlayers])
        for alpha,beta in itertools.product([l for l in range(self.nlayers)],\
                                            [l for l in range(self.nlayers)]):
            # in the diagonal include intra-layer impact matrix
            if alpha == beta:
                W[n*alpha:n*(alpha+1), n*beta:n*(beta+1)] = self.dbr_W_intra(\
                  alpha, beta, equity, a=a)
            # in the off diagonal use inter-layer impact matrix
            else:
                W[n*alpha:n*(alpha+1), n*beta:n*(beta+1)] = self.dbr_W_inter(\
                  alpha, beta, equity, b=b)
        return W
        
    
    def compute_debtrank(self, equity, W, v, seeds, psi=1, per_layer=False):
        """Computes debtrank, either monoplex or multiplex, for given seeds. 
        """
        # Set initial states
        if per_layer:
            nlayers = 1
        else:
            nlayers = self.nlayers
        h = np.zeros(self.nnodes*nlayers)
        s = np.zeros(self.nnodes*nlayers)
        for i in seeds:
            h[i] = psi
            s[i] = 1
            h_old = h[:]
            s_old = s[:]    
        # initially defaulted assets
        h1v = h.dot(v)
        # Compute debtrank for given seeds 
        count = 0
        while 1 in s:
            # kronecker delta so that only distressed nodes impact
            delta = np.where(s == 1, 1, 0)
            h = np.minimum(np.ones(self.nnodes*nlayers), h_old + \
                           W.T.dot(np.multiply(h_old, delta)))
            # update status
            for j in range(self.nnodes*nlayers):
                if s_old[j] == 1:
                    s[j] = 2
                elif s_old[j] == 0 and h[j] > 0:
                    s[j] = 1
            # keep record of status
            h_old = h[:]    
            s_old = s[:]
            # avoid inf loop (this if should never happen)
            if count > self.nnodes*nlayers:
                print("WARNING more time steps that possible")
                break
            count += 1
        # return debtrank
        debtrank = h.dot(v) - h1v
        return debtrank
            
    def debtrank_monoplex(self, equity, seeds=None,a=2.0, psi=1):
        """Compute debtrank for a monoplex network
        equity(array): array with the equity of each country
        seeds(list): list of seeds, if None debtrank is computed for all
                    countries
        a(float): parameter for impact matrix
        psi(float): initial loss of assets for seeds
        """
        # Check it is monoplex
        if len(self.adj_matrix_list) > 1:
            raise TypeError("monoplex debtrank takes one adjacency matrix")
        # Check that equity is not zero, if so then set to one
        if min(equity) <= 0:
            print("WARNING Equity of all countries must be positive "+\
                  "inputing 1 when not positive")
            equity = np.maximum(np.ones(self.nnodes), equity)
        # Compute monoplex impact matrix and vector of values   
        W = self.dbr_W_mono(equity, a=a)
        v = self.dbr_v_mono()
        
        if seeds == None:
            seeds = [i for i in range(self.nnodes)]
        # start dictionary of seeds and debtrank
        debtrank = {}
        for sd in seeds:
            debtrank[sd] = self.compute_debtrank(equity, W, v, [sd], psi=psi)
            
        return debtrank
         
    def debtrank_multiplex(self, equity, seeds=None, a=2.0, b=1.0, psi=1):
        """Compute debtrank for a multiplex network
        equity(array): array with the equity of each country
        seeds(list): list of seeds, if None debtrank is computed for all
                    countries. If not none each element should be a list with
                    country index and layer index
        a(float): parameter for impact matrix within layers
        b(float): parameter for impact matrix between layers
        psi(float): initial loss of assets for seeds
        
        NOTE so far multiplex debtrank is the sum of the debtrank obtained
        by summing the debtrank of the country when initialized in each
        layer separetely
        """ 
        if len(self.adj_matrix_list) != self.nlayers:
            raise TypeError("must give one adjacency matrix per layer")
        for l in range(self.nlayers):
            if min(equity[l]) <= 0:
                print("WARNING Equity of all countries must be positive "+\
                  "inputing 1 when not positive")
            equity[l] = np.maximum(np.ones(self.nnodes), equity[l])
        # get impact matrix and value vector
        W = self.dbr_W_mx(equity, a=a, b=b)
        v = self.dbr_v_mx()
        

        debtrank = {}
        # TODO fix this, are seeds lists of lists or just list?
        if type(seeds) != type(None):
        # if no seeds are  not nodes traslate from c,l format to 1dim format    
            translated_seeds = []
            for country, layer in seeds:
                translated_seeds.append(country + layer*self.nnodes)
            for sd, cl in enumerate(seeds):
                # compute debtrank of seed (sd in 1dim format)
                db = self.compute_debtrank(equity, W, v, sd, a=a, b=b, psi=psi)
                # add to the dictionary the c,l format and the debtrank
                debtrank[cl] = db
        # if seeds are nor specified do it for every country in every layer
        # to give the rank of a country add them up
        if seeds == None:
            for i in range(self.nnodes):
                db = 0
                for l in range(self.nlayers):
                    # compute debtrank for node i in layer l
                    sd = i + l*self.nnodes
                    db += self.compute_debtrank(equity, W, v, [sd], psi=psi)
                debtrank[i] = db
        return debtrank
    
    def debtrank_multirank(self, equity, seeds=None, a=2.0, b=1.0, psi=1):
        """Compute debtrank for a multiplex network. It computes it
        individually for each layer and then averages using multirank weights
        for layers. 
        equity(array): array with the equity of each country
        seeds(list): list of seeds, if None debtrank is computed for all
                    countries. If not none each element should be a list with
                    country index and layer index
        a(float): parameter for impact matrix within layers
        b(float): parameter for impact matrix between layers
        psi(float): initial loss of assets for seeds
        
        NOTE so far multiplex debtrank is the sum of the debtrank obtained
        by summing the debtrank of the country when initialized in each
        layer separetely
        """ 
        z = np.array(list(self.multirank()[1].values()))
        z_weight = z / sum(z)
        
        if len(self.adj_matrix_list) != self.nlayers:
            raise TypeError("must give one adjacency matrix per layer")
        # for each layer compute debtrank
        if seeds == None:
            seeds = [i for i in range(self.nnodes)]
        
        debtrank = {}
        for l in range(self.nlayers):
            if min(equity[l]) <= 0:
                print("WARNING Equity of all countries must be positive "+\
                  "inputing 1 when not positive")
                equity = np.maximum(np.ones(self.nnodes), equity[l])
            
            adj_mat = self.adj_matrix_list[l]
            D = np.diag(1 / equity[l])
            W = np.minimum(np.ones([self.nnodes, self.nnodes]), \
                   a*adj_mat.T.dot(D)) 
            v = self.adj_matrix_list[l].sum(axis=1)
            v = v / v.sum()
            
            for i in seeds:
                db = self.compute_debtrank(equity[l], W, v, [i], per_layer=True)
                db = db*z_weight[l]
                if l == 0:
                    debtrank[i] = db
                else:
                    debtrank[i] += db

        return debtrank
        
    def one_seed_debtrank_monoplex(self, seeds, equity, psi=1, a=0.2):
        """Fast way of computing the debtrank in monoplex network,
        without having to speicy L, D, W, etc.
        """
        if len(self.adj_matrix_list) > 1:
            raise TypeError("monoplex debtrank takes one adjacency matrix")
            
        if min(equity) <= 0:
            print("WARNING Equity of all countries must be positive "+\
                  "inputing 1 when not positive")
            equity = np.maximum(np.ones(self.nnodes), equity)
        if seeds == None:
            seeds = [i for i in range(self.nnodes)]

        adj_mat = self.adj_matrix_list[0]
        # Define matrices of impact and economic value
        L = adj_mat.sum(axis=1)
        D = np.diag(1 / equity)
        W = np.minimum(np.ones([self.nnodes, self.nnodes]), a*adj_mat.T.dot(D))
        v = L/L.sum()
        #I = W.dot(v)
        
        # get debtrank for all nodes and append in dictionary
        debtrank  = {}
        for sd_idx, sds in enumerate(seeds):
            # initial conditions
            h = np.zeros(self.nnodes)
            s = np.zeros(self.nnodes)
            for i in sds:

                h[i] = psi
                s[i] = 1
            h_old = h[:]
            s_old = s[:]    
            # initially defaulted assets
            h1v = h.dot(v)
                
            # Iterate until the states of nodes reach steady state
            count = 0
            print("s = ", s)
            while 1 in s:
                # kronecker delta so that only distressed nodes impact
                delta = np.where(s == 1, 1, 0)
                h = np.minimum(np.ones(self.nnodes), h_old + \
                               W.T.dot(np.multiply(h_old, delta)))

                for j in range(self.nnodes):
                    if s_old[j] == 1:
                        s[j] = 2
                    elif s_old[j] == 0 and h[j] > 0:
                        s[j] = 1
                
                h_old = h[:]    
                s_old = s[:]

                if count > self.nnodes:
                    print("WARNING more time steps that possible")
                    break
                count += 1
                # add results into dictionary
            debtrank[sd_idx] = h.dot(v) - h1v
        return debtrank
   
    
    def default_cascade_rank(self, tau_hor, tau_ver, recovery_increment=0.01):
        """function that ranks countries according to the size of the default
        
        """
        v = self.dbr_v_mx()
        safe_assets = 0 # used when some coutnries cant default
        cascade_rank = {}    
        for i in range(self.nnodes):
            # consider countries that never default
            if type(tau_hor) == np.array:
                # if threshold is country dependent
                if tau_hor.shape == (self.nnodes,):
                    if tau_hor[i] == 1:
                        cascade_rank[i] = 0
                        # record the safe assets (of all layers)
                        for l in range(self.nlayers):
                            safe_assets += v[i + l * self.nnodes]
                    else:
                        seed = [[i, l] for l in range(self.nlayers)]
                        
                        states, hor, ver, mix, t_end = self.contagion_thershold(\
                                    seed, tau_hor, tau_ver, save_csv=False)
                        cascade_rank[i] = v.dot(states.flatten())
                if tau_hor.shape == (self.nnodes, self.nlayers):
                    if tau_hor[i] == 1:
                        cascade_rank[i] = 0
                        # record the safe assets
                        for l in range(self.nlayers):
                            safe_assets += v[i]
            # if homogenous threshold no point in having default free
            else:
                seed = [[i, l] for l in range(self.nlayers)]
                
                states, hor, ver, mix, t_end = self.contagion_thershold(\
                            seed, tau_hor, tau_ver, save_csv=False)
                cascade_rank[i] = v.dot(states.flatten())
          
        # renomarlize rank when there are safe assets
        for i in range(self.nnodes):
            # TODO check this should always be less than 1
            cascade_rank[i] = cascade_rank[i] / (1 - safe_assets)
        cascade_rank_tie_break = {}
        for i in range(self.nnodes):
            # if country is causing full default use recovery
            if 1 - cascade_rank[i] < 0.0001:
                seed = [[i, l] for l in range(self.nlayers)]
            
                cascade = 1
                # increasing CAR of countries by x percent
                x = 0
                while cascade > 0.9:
               
                    # assume extra amount of car
                    tau_hor_x = tau_hor * (1 + x)
                    tau_ver_x = tau_ver * (1 + x)
                    states, hor, ver, mix, t_end = self.contagion_thershold(\
                            seed, tau_hor_x, tau_ver_x, save_csv=False)
                    # compute the monetary amount lost
                    cascade = v.dot(states.flatten())
                    # TODO check renormalization
                    cascade = cascade / (1 - safe_assets)
                    x += recovery_increment
#
                cascade_rank_tie_break[i] = 1 + x
            else:
                cascade_rank_tie_break[i] = cascade_rank[i]
        
        return cascade_rank, cascade_rank_tie_break
    
#     # TODO function that ranks by number of defaulted countires 
#    def asset_ratio_rank(self, tau_hor, tau_ver, recovery_increment=0.01):
#        """function that ranks countries according to the size of the default
#        
#        """
#        asset_ratio = {}
#        v = self.adj_matrix_list[0].sum(axis=1)
#        for l in range(1, self.nlayers):
#            v = np.concatenate((v, self.adj_matrix_list[l].sum(axis=1)), \
#                               axis=0)
#        v = v / v.sum()
#        
#        return cascade_rank, cascade_rank_tie_break
        
    
    def check_dataframe_format(self, dataframe):
        """ Check that the dataframe has the appropriate format
        """
        # check that the assets of one country in another are only reported in
        # one row
        for c1 in self.nodes.values():
            for c2 in self.nodes.values():
                assert(len(dataframe[(dataframe["country"] == c1) &\
                          (dataframe["counterpart"] == c2)]) <= 1)
    
# =============================================================================
#     Contigent Claims
# =============================================================================
    def compute_d2(self, assets, DB, r, sigma, tau=1):
        """Computes d2, from the Black Scholes Eqs. Later used to compute
        probability of default
        A(array) (nl,) array
        """
        # we use mask array to avoid error when taking log of assets that are 0
        log_a_db = np.ma.log(assets/DB)
        d2 = (log_a_db  + (r - 0.5*sigma**2)*tau)/(sigma * \
                   np.sqrt(tau))
        # if there was a problem with the log, substitue with negative large #
        d2.filled(-1e10)
#        print("a/db = ", (assets[self.countries["France"]]/DB[self.countries["France"]]))
#        print("d2 = ", d2[self.countries["France"]] )
        return d2


    def p_default(self, d2, r, sigma):
        """ give probability of default for a country-sector given d2
        """
        return  scipy.stats.norm.cdf(-d2, loc=0, scale=1)
        
    def update_contingent_claims_matrix(self, A, p, exch_rate=1.0):
        """
        WARNING be careful you don't update the actual supra adjacency matrix
        """
#        print("p = ", p[:10] )
        return A * (1-p) * exch_rate
    def contigent_claims_spread(self, car, r=1, sigma=0.5, tau=1, T=10, \
                              exch_rate=1.0, defaulted_country=None, \
                              defaulted_sector=None, shock_size=1):
        """
        if T = None, returns rank until tolerance is met
        """
        # adapt car to multilayer if needed
        if car.shape[0] == self.nnodes:
            ext_car = np.concatenate((car, car), axis=None)
            for i in range(1,self.nlayers-1):
                ext_car = np.concatenate((ext_car, car), axis=None)
        else:
            ext_car = car
        # record the initial value of assets
        foreign_assets = self.supra_adj_matrix.sum(axis=1)
        # consider buffer reserves
        reserve = ext_car * foreign_assets
        initial_assets =  foreign_assets + reserve
        # use car to compute distress barrier
        B = (1 - ext_car) * foreign_assets
        
        
        # make a compy of supraadj matrix which will decay in value
        A = copy.deepcopy(self.supra_adj_matrix)
        # record lost assets
        assets_lost = np.zeros(T + 1)
        # wipe out the assets of the defaulted country
        if type(defaulted_country) != type(None):
            
            defaulted_nodelayer = self.countries[defaulted_country] * (1 + \
                                                self.sectors[defaulted_sector])
            
            assets_lost[0] = shock_size*A[:,defaulted_nodelayer].sum()
            A[:, defaulted_nodelayer] = (1 - shock_size) * \
                A[:, defaulted_nodelayer]
        else:
            assets_lost[0] = 0
        # assets is sum over columns plus the CAR
        assets = A.sum(axis=1) + reserve


        for t in range(1,T+1):
            # compute parameters needed to update claims
            d2 = self.compute_d2(assets, B, r, sigma, tau)
            p = self.p_default(d2, r, sigma)
            #print("p ", p)
            A = self.update_contingent_claims_matrix(A,p, exch_rate)
            # assets is sum over columns
            assets = A.sum(axis=1) + reserve
            # compute the loss of the whole network
            loss = (initial_assets - assets).sum()
            assets_lost[t] = loss
            
            
        return assets_lost
        
# =============================================================================
#     Contagion implementation
# =============================================================================
    
        
    def make_initial_states(self, seeds):
        # WARNING don't use with random seeds, activated nodes are activated in
        # all layers
        """ function that received a list of nodes/countries which are
        seeds for contagion and returns the array(n x l) of initial states
        for both the aggregate and the multiplexexi
        seeds(list of list): node-layers lists that are seeds of contagion
        seed_names(list): if not none should be a list of names of seed nodes
        seed_nodes(list): if not none, list of index of seed nodes
        seed_layers(list): if not none should be a list of layers of seed node.
        If None start in all seeds.
        """
        integer_seeds = copy.deepcopy(seeds)
        # Defining inverse dictionaries
        countries_idx = {v: k for k, v in self.nodes.items()}
        layers_idx = {v: k for k, v in self.layers.items()}
        # if seeds are strings convert to index
        for node_layer in integer_seeds:
            if isinstance(node_layer[0], str):
                node_layer[0] = countries_idx[node_layer[0]]
            elif not isinstance(node_layer[0], int):
                raise TypeError("seed must be a str or int")
            if isinstance(node_layer[1], str):
                node_layer[1] = layers_idx[node_layer[1]]
            elif not isinstance(node_layer[1], int):
                raise TypeError("seed must be a str or int")

        # Make the states array and add the seeds to defaulted state
        states = np.zeros([self.nnodes, self.nlayers])
        for seed in integer_seeds:
           # print("seed = ",seed)
            states[seed[0],seed[1]] = 1
        return states
    
    def threshold_rule(self, c, l, defaulted_assets, assets,\
                           tau):
        '''
        Evaluates contagion rule
        Args:   c(int): country index
                l(int): layer index
                defaulted_assets(1d-array):(horizontal) defaulted assests
                instrength_layer_list(2d-array): array of instr of node,layers
                tau(float or np array): if array must be nx1 or nxl
        Returns: (Bool): True if default conditon is met, False otherwise'''
        if assets == 0:
            # if node has no indegree cannot default
            return False
        # homogeneous threshold
        if isinstance(tau, float):
            threshold = tau
        # heterogenous theshold
        elif isinstance(tau, np.ndarray):
            # country specific
            if tau.shape == (self.nnodes,):
                threshold = tau[c]
            # layer specific
            elif tau.shape == (self.nlayers,):
                threshold = tau[l]
            # country and layer specific
            elif tau.shape == (self.nnodes, self.nlayers):
                threshold = tau[c,l]
            else:
                raise ValueError('threshold has incorrect dimension')
        else:
             raise TypeError("threshold should be a float or numpy array.\
                             tau is currently: {}".format(type(tau)))
        # evaluating rule   
        if defaulted_assets / assets > threshold:
            return True
        else:
            return False
        
    def add_to_csv(self, file, t, solvent, l, direction):
        """
        Args:   file(file): csv file in which to write
                t(int): time step
                solvent(int): solvent node index
                l(int): layer index of solvent node
                direction(str): horizontal or vertical
                country_names(list or int): if list, names of countries
        """
        country = self.nodes[solvent]
        layer = self.layers[l]
        file.write(str(t) + "," + layer + "," + country + \
                "," + direction + "\n")
    def add_to_csv_all(self, file, t, solvent, l, direction, lost_assets):
        """
        Args:   file(file): csv file in which to write
                t(int): time step
                solvent(int): solvent node index
                l(int): layer index of solvent node
                direction(str): horizontal or vertical
                country_names(list or int): if list, names of countries
        """
        country = self.nodes[solvent]
        layer = self.layers[l]
        file.write(str(t) + "," + layer + "," + country + \
                "," + direction + "," + str(lost_assets)  + "\n")
        
    def one_step_contagion(self, states, t, defaulted_assets_hor, \
                           defaulted_assets_ver, stregth, tau_hor, tau_ver, \
                           hor_contagion, ver_contagion, mix_contagion, f, \
                           save_csv=False, save_allinfo=False):
        """
        Args:   states((n,l) nparray): states(0 or 1) of node-layers
                t(int): time step of the simulation
                defaulted_assets_hor((n,l)np array): defaulted assets of node-
                    layer in its respective layer
                defaulted_assets_hor((n)np array): defaultes assets of nodes 
                    due to defaults in other layer (same node)
                stregth(np array): nxl assets of a country in each layer
                tau_hor(float or np array): horizontal thresholds, can be same
                    for all, country, layer or countr-layer specific. 
                tau_ver(float or np array): vertical threshold, same flexbility
                    as horizontal
                hor_contagion((t_sim,)np array): array with number of horizontal
                    defaults at that time
                ver_contagion((t_sim,)np array): array with number of horizontal
                    defaults at that time
                default_rule(function): states rule of cotnagion
                f(file): csv file where to save data
                
        Returns:states((n,l) nparray): updates states of node-layers
                hor_contagion: updated counter of horizontal contagion
                ver_contagion: updated counter of vertical contagion
                mix_contagion: updated counter of mix contagion
        """
        for l in range(self.nlayers):
            solvent_nodes = np.where(states[:, l] == 0)[0]
            for country in solvent_nodes:
                # variable that checks if contagion is both hor and ver
                mix = 0
                # Horizontal contagion
                horizontal_assets_all = stregth[country, l]
                if self.threshold_rule(country, l, \
                                       defaulted_assets_hor[country, l], \
                                       horizontal_assets_all, tau_hor):

                    states[country, l] = 1
                    hor_contagion[t] += 1
                    mix += 0.5
                    if save_csv == True:
                        if save_allinfo:
                            self.add_to_csv_all(f, t, country, l, "horizontal", \
                                            defaulted_assets_hor[country, l]/horizontal_assets_all)
                        else:
                            self.add_to_csv(f, t, country, l, "horizontal")
                # Vertical contagion
                vertical_assets_all = sum(stregth[country, :])
                if self.threshold_rule(country, l, \
                                       defaulted_assets_ver[country], \
                                       vertical_assets_all, tau_ver):
                    states[country, l] = 1
                    ver_contagion[t] += 1
                    mix += 0.5
    
                    if save_csv == True:
                        if save_allinfo:
                            self.add_to_csv_all(f, t, country, l, "vertical", \
                                            defaulted_assets_ver[country]/vertical_assets_all)
                        else:
                            self.add_to_csv(f, t, country, l, "vertical")
                if mix == 1: #if it was both hor and vert then it is mixed
                    mix_contagion[t] += 1
                    if save_csv == True:
                        self.add_to_csv(f, t, country, l, "mix")
    
        return states, hor_contagion, ver_contagion, mix_contagion
    
    
        
    def contagion_thershold(self, seeds, tau_hor, tau_ver, \
                            save_csv=False, save_name="", full_history=False, \
                            save_allinfo=False, direction="assets"):
        """
        run contagion for a list of adjacency matrices and coupling.
        For simplest vertical contagion and horizontal contagion adj matrices in
        A_list and the Coupling C must be unweighted (i.e 0, 1's).
        Args:   
            seeds(list): list of node-layers to be seeds in contagion
            tau_hor(float or np array): horizontal thresholds, can be same
                    for all, country, layer or countr-layer specific. 
            tau_ver(float or np array): vertical threshold, same flexbility
                    as horizontal
            save_csv
            save_name
            direction(str): if direction is assets links should flow in the 
                direction of assets, i.e. i -> j implies i has assets on j.
                if direction is risk, then i -> j implies j has assets on i. 

        Returns:
            state_history(array): time, node, layer defaulted
            hor_contagion(array): number of countries that defaulted hor
            ver_contagion(array): number of countries that defaulted ver
            mix_contagion(array): number of countries that defaulted mix
            t_end(int): time when contagion stops
        """
        # Note this coupling is different from the ones used in centrality.
        # This coupling is for vertical contagion and related to in/outdegree.
        coupling_array = np.zeros([self.nnodes, self.nlayers])
        for l in range(self.nlayers):
            if direction == "risk":
                coupling_array[:, l] = np.sum(self.adj_matrix_list[l], axis=0)
            elif direction == "assets":
                coupling_array[:, l] = np.sum(self.adj_matrix_list[l], axis=1)
 
        initial_states = self.make_initial_states(seeds)
        # Max number of interations (deterministic percolation)
        n_sim = self.nnodes * self.nlayers
    
        # Define arrays that store the contagion information 
        hor_contagion = np.zeros([n_sim])
        ver_contagion = np.zeros([n_sim])
        mix_contagion = np.zeros([n_sim])
        if full_history:
            state_history = np.zeros([n_sim, self.nnodes, self.nlayers])
            state_history[0, :, :] = copy.deepcopy(initial_states)
    
        # Define arrays that change according to status of contagion
        states = copy.deepcopy(initial_states)
        # Horizontal defaulted assets depend on country and layer
        defaulted_assets_hor = np.zeros([self.nnodes, self.nlayers])
        # Nodes of the same country have same defaulted assets
        defaulted_assets_ver = np.zeros([self.nnodes])
    
        if save_csv == True:
            f = open(save_name + ".csv", "w")
            f.write("time step,layer,country,contagion type\n")
        else:
            f = None
    
        stregth = np.zeros([self.nnodes, self.nlayers])
        if direction == "risk":
            # (in)strength vector with which lost assests will be compared
            
            for l in range(self.nlayers):
                stregth[:, l] = np.sum(self.adj_matrix_list[l], axis=0)
        elif direction == "assets":
            # (out)strength vector with which lost assests will be compared
            for l in range(self.nlayers):
                stregth[:, l] = np.sum(self.adj_matrix_list[l], axis=1)
    
        # simulation starts
        for t in range(1, n_sim):
            # get intralayer(horizontal) defaulted assets
            for l in range(self.nlayers):
                if direction == "risk":
                    # defaulted assets of i are sum_j s_jA_ji for given layer
                    defaulted_assets_hor[:, l] = states[:, l].dot(\
                                                self.adj_matrix_list[l])
                if direction == "assets":
                    defaulted_assets_hor[:, l] = self.adj_matrix_list[l].dot(\
                                                states[:, l])
    
            # get interlayer(vertical) defaulted assets
            for c in range(self.nnodes):
                # Defaulted assets is the sum_alpha s_iC_ialpha
                defaulted_assets_ver[c] = sum(np.multiply(states[c, :], \
                                    coupling_array[c, :]))
                # Note matrix sum includes self loop but if s_i=1 we do not
                # consider the country as already defauled
    
            # run contagion for one time step
            states, hor_contagion, ver_contagion, mix_contagion = \
            self.one_step_contagion(states, t, defaulted_assets_hor, \
                               defaulted_assets_ver, stregth, tau_hor, \
                               tau_ver, hor_contagion, ver_contagion, \
                               mix_contagion, f, save_csv=save_csv, save_allinfo=save_allinfo)
                
            if full_history:
                state_history[t, :, :] = copy.deepcopy(states)
            # if nothing happend in two time steps, then break
            if ver_contagion[t - 2] == ver_contagion[t] == 0 and \
                hor_contagion[t - 2]  == hor_contagion[t] == 0:
                t_end = t
                break
    
        if save_csv == True:
            f.close()
        if full_history:
            return state_history[:t_end, :, :], hor_contagion[:t_end],\
                ver_contagion[:t_end], mix_contagion[:t_end], t_end
        else:
            return states, hor_contagion[:t_end],\
                ver_contagion[:t_end], mix_contagion[:t_end], t_end
            
    def contagion_thershold_fast(self, seeds, tau_hor, tau_ver, \
                            save_csv=False, save_name="", save_allinfo=False):
        """
        run contagion for a list of adjacency matrices and coupling.
        For simplest vertical contagion and horizontal contagion adj matrices in
        A_list and the Coupling C must be unweighted (i.e 0, 1's).
        Args:   A_list(list of np array): list with adjacency matrices of layers
                C(array): Coupling array, should be n_countries x n_layers
                states_original(array): initial states of counties 0 for liquid
                                        1 for defaulted
                default_hor(function): condition for horizontal contagion
                default_ver(function): condition for horizontal contagion
                tau_hor(float or np array): horizontal thresholds, can be same
                    for all, country, layer or countr-layer specific. 
                tau_ver(float or np array): vertical threshold, same flexbility
                    as horizontal
                type(str): layer if instregth of layer is taken into account
                            or global is it is instrength over all layers
    
                add_to_csv(functions): modifies csv file
        Returns:
            state_history(array): time, node, layer defaulted
            hor_contagion(array): number of countries that defaulted hor
            ver_contagion(array): number of countries that defaulted ver
            mix_contagion(array): number of countries that defaulted mix
            t_end(int): time when contagion stops
        
        """
        # TODO remove this function, it is now the same as before with
        # false default history
        # Note this coupling is different from the ones used in centrality.
        # This coupling is for vertical contagion and related to indegree.
        coupling_array = np.zeros([self.nnodes, self.nlayers])
        for l in range(self.nlayers):
            coupling_array[:, l] = np.sum(self.adj_matrix_list[l], axis=0)
 
        initial_states = self.make_initial_states(seeds)
        # Max number of interations (deterministic percolation)
        n_sim = self.nnodes * self.nlayers
    
        # Define arrays that store the contagion information 
        hor_contagion = np.zeros([n_sim])
        ver_contagion = np.zeros([n_sim])
        mix_contagion = np.zeros([n_sim])
    
        # Define arrays that change according to status of contagion
        states = copy.deepcopy(initial_states)
        # Horizontal defaulted assets depend on country and layer
        defaulted_assets_hor = np.zeros([self.nnodes, self.nlayers])
        # Nodes of the same country have same defaulted assets
        defaulted_assets_ver = np.zeros([self.nnodes])
    
        if save_csv == True:
            f = open(save_name + ".csv", "w")
            if save_allinfo:
                f.write("time step,layer,country,contagion type,lost assets,car\n")
            else:
                f.write("time step,layer,country,contagion type\n")
        else:
            f = None
    
        # (in)strength vector with which lost assests will be compared
        instregth = np.zeros([self.nnodes, self.nlayers])
        for l in range(self.nlayers):
            instregth[:, l] = np.sum(self.adj_matrix_list[l], axis=0)
    
        # simulation starts
        for t in range(1, n_sim):
            # get intralayer(horizontal) defaulted assets
            for l in range(self.nlayers):
                # defaulted assets of i are sum_j s_jA_ji for given layer
                defaulted_assets_hor[:, l] = states[:, l].dot(\
                                            self.adj_matrix_list[l])
    
            # get interlayer(vertical) defaulted assets
            for c in range(self.nnodes):
                # Defaulted assets is the sum_alpha s_iC_ialpha
                defaulted_assets_ver[c] = sum(np.multiply(states[c, :], \
                                    coupling_array[c, :]))
                # Note matrix sum includes self loop but if s_i=1 we do not
                # consider the country as already defauled
    
            # run contagion for one time step
            states, hor_contagion, ver_contagion, mix_contagion = \
            self.one_step_contagion(states, t, defaulted_assets_hor, \
                               defaulted_assets_ver, instregth, tau_hor, \
                               tau_ver, hor_contagion, ver_contagion, \
                               mix_contagion, f, save_csv=False, save_allinfo=save_allinfo)
                

            # if nothing happend in two time steps, then break
            if ver_contagion[t - 2] == ver_contagion[t] == 0 and \
                hor_contagion[t - 2]  == hor_contagion[t] == 0:
                t_end = t
                break
    
        if save_csv == True:
            f.close()
    
        return states, hor_contagion[:t_end],\
                ver_contagion[:t_end], mix_contagion[:t_end], t_end
                
# =============================================================================
#     relative value of a node
# =============================================================================
    def node_rel_value(self, value_df, value_name="gdp"):
        """Given a data frame with values for each node, e.g. gdp, computes
        the relative value to the total
        """
        dict_values = dict(zip(value_df.country,value_df[value_name]))
        # get the total ammount upon existing countries in network
        total = 0
        for country in self.countries:
            total += dict_values[country]
        # making dictionary or relative values
        dict_rel_value = {}
        for country in self.countries:
            dict_rel_value[country] = dict_values[country]/total
            
        return dict_rel_value
        
# =============================================================================
#     Contagion tree
# =============================================================================
    def contagion_tree(csv_file):
        """Function that read a contagion files and saves an edgelist of the
        directed acyclic graph
        """
    
# =============================================================================
# Functions to explore results
# =============================================================================

def plot_network_contagion_comparison(network_list, seeds_list,\
                                  tau_list, tau_fix,x_array, tau_type="horizontal",\
                                    networks_names=None, \
                                   default_criteria="full", \
                                   save_name=None, fig_title=""):
    # TODO seed should be a list for each network, in case we want 
    # to compare between different seeds

    """ Given a list of networks, seeds and thresholds plots the size of the
    contagion in each network. One (or both)thresholds vary while one 
    (or none) remains fixed
    while the other threshold varies according to the list.
    Args:
        network_list(list): list of MutiplexNet objects
        seed_country(list): list or lists, one list per network with the seed
            country layers
        tau_list(list): list of thresholds
        tau_fix(float): threshold that remains fixed
        x_array(array/list): array plotted in x axis
        tau_type(str): determines if a threshold (hor or ver) remains fixed
        if "both" then both threshold vary and none remains fixed
        networks_names(list): name of each network to include as labels
        default_criteria(str): determines whether it count a default as a country 
        defaulted in all layers or only in one
    
    """
    # figure details
    fontsize_ticks = 30
    fontsize_axis = 36
    fontsize_title = 40
    fontsize_legend = 36
    
    f = plt.figure(figsize=(10,10))
    f.subplots_adjust(hspace=0.2)
    gs = gridspec.GridSpec(1, 1)
    ax = plt.subplot(gs[0])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(labelsize=fontsize_ticks)
    
    ax.axvline(100, color="red",linestyle="--", linewidth=3,label="real CAR")
    
    ndefaults = []
    for i, net in enumerate(network_list):
        ndefaults.append([])
        for tau in tau_list:
            # Run contagion simulation with the appropriate thresholds
            if tau_type == "horizontal":
                # note seed_list index corresponds to index of network
                finalstate, hor_default, ver_default, mix_default, t_sim=\
                net.contagion_thershold_fast(seeds_list[i],\
                tau, tau_fix)
            elif tau_type == "vertical":
                 finalstate, hor_default, ver_default, mix_default, t_sim=\
                net.contagion_thershold_fast(seeds_list[i], \
                tau_fix, tau)
            elif tau_type == "both":
                finalstate, hor_default, ver_default, mix_default, t_sim=\
                net.contagion_thershold_fast(seeds_list[i], \
                tau, tau)
            else:
                raise ValueError("tau must be 'horizontal' or 'vertical'")
            # Append results (according to defualt criteria) into list 
            if default_criteria == "full":
#                ndefaults[i].append(np.sum(finalstate)/net.nlayers)
                ndefaults[i].append(math.floor(np.sum(finalstate)/net.nlayers))
            elif default_criteria == "partial":
                ndefaults.append(np.count_nonzero(\
                                 np.sum(finalstate, axis=1)))
            else:
                raise ValueError("default_criteria is not 'full' or 'partial'")
        # plot with the according label
        
        if isinstance(networks_names, type(None)):
            net_label = str(i)
        else:
            net_label = networks_names[i]

        if type(x_array) == list:
            x_array = np.array(x_array)
        ax.plot(100*x_array, ndefaults[i], "o-", linewidth=5, markersize=15, label=net_label)
        
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=fontsize_legend)
    # Set charateristics of the overall plot
    #plt.xlim([0, 1])
    plt.xlabel("CAR (as % of real CAR)", fontsize=fontsize_axis)
    plt.ylabel("Number of " + default_criteria + " defaults", fontsize=fontsize_axis)
    plt.title(fig_title, fontsize=fontsize_title)   
    if isinstance(save_name, str):
        plt.savefig(save_name, bbox_inches="tight")
    plt.show()
                
    
def plot_network_contagion_comparison_value(df_value, network_list, seeds_list,\
                                  tau_list, tau_fix,x_array, tau_type="horizontal",\
                                    networks_names=None, \
                                   default_criteria="full", \
                                   save_name=None, fig_title=""):
    # TODO seed should be a list for each network, in case we want 
    # to compare between different seeds

    """ Given a list of networks, seeds and thresholds plots the size of the
    contagion in each network. One (or both)thresholds vary while one 
    (or none) remains fixed
    while the other threshold varies according to the list.
    Args:
        network_list(list): list of MutiplexNet objects
        seed_country(list): list or lists, one list per network with the seed
            country layers
        tau_list(list): list of thresholds
        tau_fix(float): threshold that remains fixed
        x_array(array/list): array plotted in x axis
        tau_type(str): determines if a threshold (hor or ver) remains fixed
        if "both" then both threshold vary and none remains fixed
        networks_names(list): name of each network to include as labels
        default_criteria(str): determines whether it count a default as a country 
        defaulted in all layers or only in one
    
    """
    value_array = np.fromiter(network_list[0].node_rel_value(df_value).values(), \
                              dtype=float)
    
    # figure details
    fontsize_ticks = 30
    fontsize_axis = 36
    fontsize_title = 40
    fontsize_legend = 36
    
    f = plt.figure(figsize=(10,10))
    f.subplots_adjust(hspace=0.2)
    gs = gridspec.GridSpec(1, 1)
    ax = plt.subplot(gs[0])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(labelsize=fontsize_ticks)
    
    ax.axvline(100, color="red",linestyle="--", linewidth=3,label="real CAR")
    
    ndefaults = []
    for i, net in enumerate(network_list):
        ndefaults.append([])
        for tau in tau_list:
            # Run contagion simulation with the appropriate thresholds
            if tau_type == "horizontal":
                # note seed_list index corresponds to index of network
                finalstate, hor_default, ver_default, mix_default, t_sim=\
                net.contagion_thershold_fast(seeds_list[i],\
                tau, tau_fix)
            elif tau_type == "vertical":
                 finalstate, hor_default, ver_default, mix_default, t_sim=\
                net.contagion_thershold_fast(seeds_list[i], \
                tau_fix, tau)
            elif tau_type == "both":
                finalstate, hor_default, ver_default, mix_default, t_sim=\
                net.contagion_thershold_fast(seeds_list[i], \
                tau, tau)
            else:
                raise ValueError("tau must be 'horizontal' or 'vertical'")
            # Append results (according to defualt criteria) into list 
            if default_criteria == "full":
                defaults = np.zeros(net.nnodes)
                for k in range(net.nnodes):
                    if sum(finalstate[k]) == net.nlayers:
                        defaults[k] = 1
                # make in percentage
                ndefaults[i].append(100*defaults.dot(value_array))
            elif default_criteria == "partial":
                # TODO repeat for partial defaults
                ndefaults.append(np.count_nonzero(\
                                 np.sum(finalstate, axis=1)))
            else:
                raise ValueError("default_criteria is not 'full' or 'partial'")
        # plot with the according label
        
        if isinstance(networks_names, type(None)):
            net_label = str(i)
        else:
            net_label = networks_names[i]

        if type(x_array) == list:
            x_array = np.array(x_array)
        ax.plot(100*x_array, ndefaults[i], "o-", linewidth=5, markersize=15, label=net_label)
        
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=fontsize_legend)
    # Set charateristics of the overall plot
    #plt.xlim([0, 1])
    plt.xlabel("CAR (as % of real CAR)", fontsize=fontsize_axis)
    plt.ylabel("Defaults in GDP (%)", fontsize=fontsize_axis)
    plt.title(fig_title, fontsize=fontsize_title)   
    if isinstance(save_name, str):
        plt.savefig(save_name, bbox_inches="tight")
    plt.show()