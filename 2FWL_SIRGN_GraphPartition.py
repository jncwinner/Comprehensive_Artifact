import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import math
import networkx as nx
from sklearn.decomposition import PCA

########################################################################################################################
#   A SIR-GN approach using 2-dimensional Folklore Weisfeiler lehman with Structural Graph Partitioning                #
########################################################################################################################
class loader:
    
    def __init__(self):
        self.countID = 0
        self.G = {}
        self.co = {}
        self.revco = {}
        self.graoh_indic = {}
        self.node_label = {}

    def nodeID(self, x, nl):
        if x not in self.co:
            self.co[x] = self.countID
            self.node_label[self.co[x]] = nl
            self.countID = self.countID + 1
            self.revco[self.co[x]] = x
        return self.co[x]

    def read(self,data, gi, nl, el):
        x=data.values
        for a in range(x.shape[0]):
            i=self.nodeID(x[a,0], nl[x[a,0]])
            j=self.nodeID(x[a,1], nl[x[a,1]])
            self.graoh_indic[i] = gi[x[a, 0]-1]
            self.graoh_indic[j] = gi[x[a, 1]-1]
            w=(el[a])
            self.addEdge((i,j), w)

    def read2(self,data, nl, el):
        x=data.values
        for a in range(x.shape[0]):
            i=self.nodeID(x[a,0], nl[x[a,0]])
            j=self.nodeID(x[a,1], nl[x[a,1]])
            w=(el[a])
            self.addEdge((i,j), w)

    def storeEmb(self, file, data):
        file1 = open(file, 'w')
        print("\ndatashape")
        print(data.shape)
        for a in range(data.shape[0] -1 ):
            s = '' + str(int(self.revco[a]))
            for b in range(data.shape[1]):
                s += ' ' + str(data[a, b])
            file1.write(s + "\n")
        file1.close()

    def fixG(self):
        for g in range(len(self.G)):
            self.G[g] = np.array([x for x in self.G[g]])

    def addEdge(self,s,w):
        (l1,l2)=s
        if l1 not in self.G:
            self.G[l1]={}
        if l2 not in self.G:
            self.G[l2]={}
        self.G[l1][l2]=w
        self.G[l2][l1]=w

class loader2:
    def __init__(self):
        self.countID = 0
        self.G = {}
        self.co = {}
        self.revco = {}

    def nodeID(self, x):
        if x not in self.co:
            self.co[x] = self.countID
            self.countID = self.countID + 1
            self.revco[self.co[x]] = x
        return self.co[x]

    def read(self, file):
        x = pd.read_csv(file, sep=',').values
        for a in range(x.shape[0]):
            i = self.nodeID(x[a, 0])
            j = self.nodeID(x[a, 1])
            self.addEdge((i, j))
        self.fixG()

    def read2(self, file):
        x = pd.read_csv(file, sep=' ').values
        print(x)

        for a in range(x.shape[0]):
            i = self.nodeID(x[a, 0])
            j = self.nodeID(x[a, 1])
            self.addEdge((i, j))
        self.fixG()

    def storeEmb(self, file, data):
        file1 = open(file, 'w')
        for a in range(data.shape[0]):
            s = '' + str(int(self.revco[a]))
            for b in range(data.shape[1]):
                s += ' ' + str(data[a, b])
            file1.write(s + "\n")
        file1.close()

    def fixG(self):
        for g in range(len(self.G)):
            self.G[g] = np.array([x for x in self.G[g]])

    def addEdge(self, s):
        (l1, l2) = s
        if l1 not in self.G:
            self.G[l1] = set()
        if l2 not in self.G:
            self.G[l2] = set()
        self.G[l1].add(l2)
        self.G[l2].add(l1)

def normalize(emb1):
    er=0.000000001
    m=np.min(emb1,axis=0)
    M=np.max(emb1,axis=0)
    vec=np.abs(M-m)
    m1=m[vec>=er]
    emb=np.ones(emb1.shape)
    emb[:,vec>=er]=(emb1[:,vec>=er]-m1)/(M[vec>=er]-m1)
    emb[:,vec<er]=1
    return emb

def normalizeRow(emb1):
    er=0.000000001
    M=emb1.max(axis=1)
    m=emb1.min(axis=1)
    vec=np.abs(M-m)
    m2=m[vec>=er]
    m1=m2.reshape(m2.shape[0],1)
    emb=np.ones(emb1.shape)
    emb[vec>=er,:]=(emb1[vec>=er,:]-m1)/(M[vec>=er].reshape(m2.shape[0],1)-m1)
    emb[vec<er,:]=1/emb1.shape[1]
    su=emb.sum(axis=1)
    emb2=np.ones(emb1.shape)
    vec1=np.abs(su)
    vec2=su[vec1>=er]
    emb2[vec1>=er,:]=emb[vec1>=er,:]/vec2.reshape(vec2.shape[0],1)
    emb2[vec1<er,:]=1/emb1.shape[1]
    return emb2

def getnumber1(emb1):
    emb=normalize(emb1)
    ss=set()
    for x in range(emb.shape[0]):
        sd=''
        for y in range(emb.shape[1]):
            sd+=','+str(round(emb[x,y],6))
        ss.add(sd)
    return len(ss)

def getnumber(emb1):
    emb=normalize(emb1)
    dic={}
    count=0
    for x in range(emb.shape[0]):
        h=dic
        for y in range(emb.shape[1]):
            sd=str(round(emb[x,y],6))
            if sd not in h:
                if y==emb.shape[1]-1:
                    h[sd]=count
                    count+=1
                else:
                    h[sd]={}
            h=h[sd]
    return count

def convert_to_networkx_graph(G):
    nx_graph = nx.Graph()
    for node, edges in G.items():
        for dest, weight in edges.items():
            nx_graph.add_edge(node, dest, weight=weight)
    return nx_graph

def convert_to_networkx_graph_noweight(G):
    nx_graph = nx.Graph()
    for node, edges in G.items():
        for dest, weight in edges.items():
            nx_graph.add_edge(node, dest)
    return nx_graph

########################################################################################################################
#   Create_clusters is a method which will recursively call itself until the requested number of graph partitions have #
#   been created                                                                                                       #
########################################################################################################################

def create_clusters(sorted_emb2, target_size, max_partitions):

    if max_partitions < 1:
        raise ValueError("Target size must be at least 1.")

    total_graph = nx.Graph()
    for edge_list in sorted_emb2:
        for e1, e2 in edge_list:
            total_graph.add_edge(e1, e2)

    if max_partitions == 1:
        return [total_graph]

    mid = len(sorted_emb2) // 2
    edges_1 = sorted_emb2[:mid]
    edges_2 = sorted_emb2[mid:]
    nx_graph1 = nx.Graph()
    nx_graph2 = nx.Graph()

    for edge_list in edges_1:
        for e1, e2 in edge_list:
            nx_graph1.add_edge(e1, e2)
    for edge_list in edges_2:
        for e1, e2 in edge_list:
            nx_graph2.add_edge(e1, e2)
            
    ccnum1 = nx.number_connected_components(nx_graph1)
    ccnum2 = nx.number_connected_components(nx_graph2)
    highmid = math.ceil(max_partitions/2)
    lowmid =  math.floor(max_partitions/2)

    if ((max_partitions == 2) and (ccnum1 < target_size and ccnum2 < target_size)):
        return [nx_graph1, nx_graph2]
    elif ccnum1 > target_size and ccnum2 < target_size:
        return [nx_graph2] + create_clusters(sorted_emb2[:mid],target_size, max_partitions-1)
    elif ccnum2 > target_size and ccnum1 < target_size:
        return [nx_graph1] + create_clusters(sorted_emb2[mid:],target_size, max_partitions-1)
    else:
        return create_clusters(sorted_emb2[:mid],target_size, lowmid)+ create_clusters(sorted_emb2[mid:],target_size, highmid)

def hashsamestruct2(emb1):
    print("hashsamestruct2")
    emb1 = normalize(emb1)
    emb_edges = {}
    for x1 in range(emb1.shape[0]):
        for x2 in l2.G[x1]:
            str1 = str((emb1[x1] + emb1[x2]).round(decimals=5, out=None))
            if str1 in emb_edges.keys():
                emb_edges[str1].append([x1,x2])
            else:
                emb_edges[str1] = [[x1,x2]]
    return emb_edges


def getroundemb(emb1):
    emb1=normalize(emb1)
    for x in range(emb1.shape[0]):
        for y in range(emb1.shape[1]):
            emb1[x,y]=round(emb1[x,y],6)
    return emb1

########################################################################################################################
#   Partition_graph is a method which will find significant strucutural information in the graph and preform           #
#   the requested number of splits on the graph with limiting the loss of structural information.                      #
########################################################################################################################

def partition_graph(G, G2, sirgnsavefilename, n, num_partitions):
    #apply SirGN
    siremb = sirGN(G2, n)
    l.storeEmb(sirgnsavefilename, siremb)
    # group edges in groups of same structure using the embeddings from SIRGN
    emb2 = hashsamestruct2(siremb)

    # Create the max target size for each partition from the number of connected components divided by the partitions
    nx_graph = convert_to_networkx_graph_noweight(G)
    target_size = round(nx.number_connected_components(nx_graph)/num_partitions)
    
    # Calculate the PageRank sum for each group of node pairs
    page_rank = nx.pagerank(nx_graph)
    hashgroupscore = {}
    for group_key, group in emb2.items():
        # Sum PageRank scores for all node pairs in the group
        group_score = sum(
            sum(page_rank[int(node)] for node in nodepair)
            for nodepair in group
        )
        hashgroupscore[group_key] = group_score
    # Sort group keys by their PageRank score in descending order
    sorted_pr_scores = sorted(hashgroupscore, key=lambda k: hashgroupscore[k], reverse=True)

    # Reconstruct emb2 in the sorted order
    sorted_emb2 = {key: emb2[key] for key in sorted_pr_scores}

    print("generating subgraphs")

    # Run Create_clusters to preform the binary search on the list of sorted values and find the best split
    subgraphs = create_clusters(list(sorted_emb2.values()), target_size, num_partitions)

    return subgraphs


def FWL_SIRGN_GraphPartition(G,G2, sirgnsavefilename, n, nl, num_parts ,iter=100):

    # using connected components to create splits upon the grpups of clusters produced
    cc = partition_graph(G, G2,sirgnsavefilename, n, num_parts)
    
    # remmap the new subgraphs to add the edge weights back into the graph
    nx_graph = convert_to_networkx_graph(G)
    subgraphs = [nx_graph.subgraph(c).copy() for c in cc]

########################################################################################################################
#   Start FWL_SIRGN and pass the list of subgraphs and n,nodelist, and iterations to preform                          #
########################################################################################################################
    subgraph_finalemb = FWL_SIRGN(subgraphs, n, nl, iter)

    return subgraph_finalemb


def sirGN(G,n,iter=100):
    nv=len(G)
    degree=np.array([[G[x].shape[0]] for x in range(nv)])
    emb=np.hstack([degree,np.zeros((nv,n-1))])
    count=getnumber(emb)
    embs=[emb]
    print('number of different groups',count)
    for i in range(iter):
        print(i)
        scaler = MinMaxScaler()
        emb1=scaler.fit_transform(emb)
        kmeans = KMeans(n_clusters=n, random_state=1).fit(emb1)
        val=kmeans.transform(emb1)
        M=val.max(axis=1)
        m=val.min(axis=1)
        subx=(M.reshape(nv,1)-val)/(M-m).reshape(nv,1)
        su=subx.sum(axis=1)
        subx=subx/su.reshape(nv,1)
        hh=[subx[G[i],:].sum(axis=0) if len(G[i])>0 else np.zeros(n) for  i in range(nv)]
        emb2=np.vstack(hh)
        d=getnumber(emb2)
        print('number of different groups',d)
        if count >=d:
            break
        else:
            count=d
            emb=emb2
            embs.append(emb2)
    return emb

def FWL_SIRGN(G, n, nl, iter=100):
    nv = len(nl)    # nv is the number of nodes in the entire graph
    n1l = []        #n1l is the list of node lables
    Gkeys = list(range(len(G))) #Gkeys is a list of the graph partitions index
    Gkeys2=[]   #Gkeys2 is the list of the node list in each graph partition
    zerolab = np.zeros(len(nl[1]))
    for key in Gkeys:
        Gkeys2.append(list(G[key].nodes()))
########################################################################################################################
#   Initialize the embeddings matrix and the node/edge lables                                                          #
########################################################################################################################

    el = np.zeros((nv * nv, len(list(G[0][Gkeys2[0][0]].items()) [0][1]['weight'])))    #el is the list of edge lables
    for x in range(nv):
        for y in range(nv):
            if (x == y):
                n1l.append(nl[x])
            else:
                n1l.append(zerolab)
    el = np.array(el)
    n1l = np.array(n1l)
    el = el.reshape((nv * nv,len(list(G[0][Gkeys2[0][0]].items()) [0][1]['weight'])))
    n1l = n1l.reshape((nv * nv, len(nl[0])))

    emb = np.hstack([np.ones((nv * nv, 1)), np.zeros((nv * nv, n * n - 1))])
    for g in Gkeys:
        for x in Gkeys2[g]:
            for neighbor, attributes in G[g][x].items():
                emb[x * nv + neighbor, 0] = 0
                emb[x * nv + neighbor, 1] = 1
                el_array = attributes['weight']
                weight_length = len(el_array)
                el[x * nv + neighbor, :weight_length] = el_array[:weight_length]
    labtotal = np.column_stack([n1l, el])
    emb = np.hstack([emb, labtotal])

########################################################################################################################
#   Start the FWL_SIRGN iterations on each graph partition                                                            #
########################################################################################################################
    
    print("entering iteration")
    for i in range(iter):
        print(f"iter: {i}")
        emb = normalize(emb)
        emb = PCA(n_components=int(n / 2)).fit_transform(emb)
        emb = np.hstack((emb, -emb))
        emb = normalizeRow(emb)
        emb2 = np.zeros((nv * nv, n * n))
        for g in Gkeys:
            for x in Gkeys2[g]:
                for y in Gkeys2[g]:
                    intersetz = set([h for h in np.hstack([list(G[g][x].keys()), list(G[g][x].keys())])]).union([x, y])
                    for z in intersetz:
                        if z in Gkeys2[g]:
                            emb2[x * nv + y] += np.matmul(
                                emb[nv * z + y].reshape((n, 1)),
                                emb[nv * y + z].reshape((1, n))).flatten()
        emb = np.hstack([emb2, labtotal])
    return emb[np.array([nv*x+x for x in range(nv)])]



if __name__ == "__main__":
    # Process data into format to fit the loaders
    data = pd.read_csv('data/MUTAG/MUTAG_A.txt', header=None, names=['src', 'trg'])
    edgelabels = pd.read_csv('data/MUTAG/MUTAG_edge_labels.txt', header=None, names=['edgelab'])
    nodelabels = pd.read_csv('data/MUTAG/MUTAG_node_labels.txt', header=None, names=['nodelab'])
    graphind = pd.read_csv('data/MUTAG/MUTAG_graph_indicator.txt', header=None)
    
    gindc = graphind.values
    
    nl = nodelabels.values
    enc = OneHotEncoder(handle_unknown='ignore')
    nodelabs = enc.fit_transform(nl).toarray()
    nldict = {}
    for i in range(1, len(nodelabs) + 1):
        nldict[i] = nodelabs[i - 1]
    
    el = edgelabels.values
    enc2 = OneHotEncoder(handle_unknown='ignore')
    edgelabs = enc2.fit_transform(el).toarray()
    
    l = loader()
    l.read(data, gindc, nldict, edgelabs)
    
    l2 = loader2()
    l2.read('data/MUTAG/MUTAG_A.txt')
    
    ####################################################################################################################
    # Run FWL_SIRGN approach with graph partition                                                                     #
    # Paramiters: (Graph1, Graph2, filename, n, node_labels, partitions, iterations)                                   #
    # -Graph1: The primary graph to preform the graph partition on and returns the 2FWL_SIR-GN embeddings.             #
    # -Graph2: The second graph is the same but is format friendly for SIR-GN to operate on for pre-embeddings.        #
    # -Filename: This is the path to save the pre-embeddings from SIR-GN for decreased time in future operations.      #
    # -N: This is the number of hops away from each node to calculate.                                                 #
    # -Node_labels: This is a list of all the nodes in the graph and their corresponding labels.                       #
    # -Partitions: This is the number of partitions to split the graph into to increase performance                    #
    # -Iterations: This is the number of iterations to do the 2FWL-SIRGN algorithm on the graph                        #
    ####################################################################################################################
    sirgnsavefilename = "SIRGN_embeddings_MUTAG_10_20_10.txt" #save path for SIRGN pre-embeddings
    finalemb = FWL_SIRGN_GraphPartition(l.G, l2.G, sirgnsavefilename, 10, l.node_label, 20, 10)
    l.storeEmb('FWL_SIRGN_GRAPH_PARTITION_MUTAG_10_20_10.txt',finalemb) #Save embeddings to filepath, final_embeddings