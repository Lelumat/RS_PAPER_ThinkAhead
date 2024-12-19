
import json
import numpy as np
import networkx as nx 
import copy
from scipy.special import softmax
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib.colors as mplcl
import sys

def _all_simple_paths_graph(G, source, targets, cutoff, depth):
    visited = dict.fromkeys([source])
    stack = [iter(G[source])]
    while stack:
        children = stack[-1]
        child = next(children, None)
        if child is None:
            stack.pop()
            visited.popitem()
        elif len(visited) < cutoff:
            if child in visited:
                continue
            depth_equality = len(targets & (set(visited.keys()) | {child})) == depth
            if depth_equality:#child in targets:
                yield list(visited) + [child]
            visited[child] = None
            #depth_inequality = len(targets & set(visited.keys())) < depth+1
            if len(targets - set(visited.keys())) > len(targets)-depth:  # expand stack until find all targets
                stack.append(iter(G[child]))
            else:
                visited.popitem()  # maybe other ways to child
        else:  # len(visited) == cutoff:
            #(targets & (set(children) | {child})) - set(visited.keys()) tartget vicini
            #non visitati
            for target in (targets & (set(children) | {child})) - set(visited.keys()):
                    if len(targets & (set(visited.keys()) | {target}))==depth:
                        yield list(visited) + [target]
            stack.pop()
            visited.popitem()


class Agent:
    
    def __init__(self, BETA = -5, valueType = None, graph = None):
        
        #self.planningDepth = 1 #depth
        self.BETA =BETA
        self.ChosenPath = None
        self.PossiblePaths = None
        self.PossiblePathsProbabilities = None
        self.nodeValue = None
        self.nodePreferences = None
        self.valueType = valueType
        self.t = 0.5
        self.laplacianMatrix = None
        self.nodePreferences = self.setNodePreferences(graph)

    def setNodePreferences(self, graph = None):
        self.nodeValue = {node: 1. if graph.nodes[node]["reward"] == 1 else 0. for i, node in enumerate(graph.nodes())}
        if self.valueType == "betweenness":
            self.nodeValue = nx.betweenness_centrality(graph)
        if self.valueType == "Laplacian":
                self.laplacianMatrix = nx.laplacian_matrix(graph).toarray()
                #self.nodeValue = np.dot(self.laplacianMatrix, np.array(list(self.nodeValue.values())))
                #make a dictionary
                #Find eigenvalues and eigenvectors of the Laplacian matrix
                w, v = np.linalg.eigh(self.laplacianMatrix)
                alpha = 1 #parameter to be tuned
                t = self.t #parameter to be tuned
                self.nodeValue = np.dot(v, np.dot(np.diag(np.exp(-alpha*w*t)), np.dot(v.T, np.array(list(self.nodeValue.values())))))
                self.nodeValue = {list(graph.nodes())[i]:self.nodeValue[i] for i in range(len(self.nodeValue))}
        #For each node in the graph, we will assign a preference for neighboring nodes based on the value map
        self.nodePreferences = {node:{} for node in graph.nodes()}
        #for each node in the graph, we will assign a preference for neighboring nodes based on the value map
        for node in graph.nodes():
            neighbors = list(graph.neighbors(node))
            #if the node has no neighbors, we assign a preference of 0
            if len(neighbors) == 0:
                self.nodePreferences[node] = 0.
            else:
                #we assign a preference of 1 to the neighbors of the node
                for neighbor in neighbors:
                    self.nodePreferences[node][neighbor] = self.nodeValue[neighbor]
        return self.nodePreferences

    def computePaths(self, CurrentGraph, CurrentPosition, targets, depth, cutoff, easyEnding = None, collectedRewards = None, totalReward = None):
        self.PossiblePaths = list(
            _all_simple_paths_graph(
                CurrentGraph,
                CurrentPosition,
                targets, 
                cutoff,
                depth
            ))
        
        #Impose hard constraint
        if easyEnding!= None:
            if len(collectedRewards) + depth < totalReward:
                #Drop all paths that contain the easyEnding
                self.PossiblePaths = [path for path in self.PossiblePaths if easyEnding not in path]  

    def selfAvoidingRandomWalk(self, CurrentGraph, CurrentPosition, nSteps):
        
        #print("Random Walk")
        #print("CurrentPosition: ", CurrentPosition)
        self.PossiblePaths = []
        possiblePath = [CurrentPosition]
        for _ in range(nSteps):
            #print("CurrentPosition: ", CurrentPosition)
            #Get the neighbors of the current position
            neighbors = list(CurrentGraph.neighbors(CurrentPosition))
            #Check if the neighbors are already in the path
            neighbors = [n for n in neighbors if n not in possiblePath]
            #print("     Neighbors: ", neighbors)
            if neighbors == []:
                #print("     No neighbors")
                #If there are no neighbors, the path is over
                if len(possiblePath) > 1:
                    self.PossiblePaths.append(possiblePath)
                    break
                else:
                    self.PossiblePaths.append(possiblePath[1:])
                    break

            #possiblePath.append(CurrentPosition)
            #print("possiblePath: ", possiblePath)
            #Choose a random neighbor
            probabilities = np.array([self.nodePreferences[CurrentPosition][n] for n in neighbors])
            probabilities += 10**-10
            probabilities = probabilities/np.sum(probabilities)
            ########nextPosition = neighbors[np.random.choice(len(neighbors))]
            nextPosition = neighbors[np.random.choice(len(neighbors), p = probabilities)]
            #Update the current position
            CurrentPosition = nextPosition
            #print("nextPosition: ", CurrentPosition)
            #Update the path
            possiblePath.append(CurrentPosition)   
            if CurrentGraph.nodes()[CurrentPosition]["reward"]:
                #If I met a reward, the path is over
                self.PossiblePaths.append(possiblePath)
                #print("Reached reward possiblePath: ", possiblePath)
                break
              

    def getPossiblePathsProbabilities(self):

        beta = self.BETA
        lenghts = np.array([len(path) for path in self.PossiblePaths])
        if self.PossiblePaths == []:
            self.PossiblePathsProbabilities = 1
        else:
            self.PossiblePathsProbabilities = softmax(beta*lenghts)
        
    def choosePaths(self):
        #print("PossiblePaths: ", self.PossiblePaths)
        if self.PossiblePaths != []:
            return self.PossiblePaths[
                np.random.choice(a = range(len(self.PossiblePaths)), p = self.PossiblePathsProbabilities)]
        else:
            return []
    def renderValueMap(self, graph = None):
        pos = {node:node for node in graph.nodes()}
        node_color = [self.nodeValue[n] for n in graph.nodes()]
        plt.figure()
        nx.draw_networkx(graph, pos = pos, node_color = node_color, node_size = 50, edge_color = 'gray', alpha = 1, cmap = "coolwarm")
        plt.axis("equal")
        plt.show()

class Environment:
    
    def __init__(self):
        
        self.OriginalGraph = None
        self.CurrentGraph = None
        self.Start = None
        self.CurrentPosition = None
        self.CurrentPath = []
        self.OriginalRewards = None
        self.CollectedRewards = None
        self.RemainingRewards = None
        self.node_to_id = None
        self.id_to_node = None
        self.easyEnding = None

    def initializeEnv(self, pathGraph):
        with open(pathGraph, 'r') as f:
            data = json.load(f)
        nodes = {}
        links = {}
        for n in data.get("nodes"):
            nodes[n.get("ID")] = n
        for l in data.get("links"):
            links[l.get("SOURCE"), l.get("TARGET")] = 1
        nodes_dict = {key:(nodes[key]["X"], nodes[key]["Y"]) for key in nodes.keys()}
        node_list = list(nodes_dict.values())
        link_list = [(nodes_dict[n[0]], nodes_dict[n[1]]) for n in links.keys()]
        col = {nodes_dict[i]:r["reward"] for i,r in enumerate(nodes.values())}
        g = nx.Graph()
        g.add_nodes_from(node_list)
        g.add_edges_from(link_list)
        nx.set_node_attributes(g, col, "reward")
        
        self.OriginalGraph = copy.deepcopy(g)
        self.getEasyEnding()
        self.CurrentGraph = copy.deepcopy(g)
        self.getGraphDictionaries()
        self.CurrentPosition = self.id_to_node[data["start"]]
        self.Start = copy.deepcopy(self.CurrentPosition)
        self.CurrentPath = [self.Start]
        self.getOriginalRewards()
        self.getCollectedRewards()
        self.getRemainingRewards()
        
    def getEasyEnding(self):
        #Check if there is a node with reward 1 that has degree 1:
        for node in self.OriginalGraph.nodes():
            if self.OriginalGraph.nodes()[node]["reward"] == 1:
                if self.OriginalGraph.degree(node) == 1:
                    self.easyEnding = node

    def getQuiteEasyending(self):
        #For all the nodes with reward 1, check the degree of the neighbors
        for node in self.OriginalGraph.nodes():
            if self.OriginalGraph.nodes()[node]["reward"] == 1:
                if self.OriginalGraph.degree(node) - len([n  for n in self.OriginalGraph.neighbors(node) if self.OriginalGraph.degree(n) == 1]) == 1:
                    #The node has only one neighbor with degree 1 except for one.
                    #The current node cannot be a neighbor of the easyEnding
                    if self.CurrentPosition in self.OriginalGraph.neighbors(node):
                        continue
                    else:   
                        self.easyEnding = node
                        break
                    
    def getGraphDictionaries(self):
        self.node_to_id = {node:i for i,node in enumerate(list(self.CurrentGraph.nodes()))}
        self.id_to_node = {i:node for i,node in enumerate(list(self.CurrentGraph.nodes()))}
        
    def getCurrentGraph(self):
        self.CurrentGraph = self.CurrentGraph.subgraph(self.OriginalGraph.nodes()- self.CurrentPath)
        
    def getOriginalRewards(self):
        self.OriginalRewards  = [n for n in self.CurrentGraph.nodes() if self.CurrentGraph.nodes()[n]["reward"]]
        
    def getCollectedRewards(self):
        self.CollectedRewards = [n for n in self.CurrentPath if n in self.OriginalRewards]
    
    def getRemainingRewards(self):
        self.RemainingRewards = list(set(self.OriginalRewards)-set(self.CollectedRewards))
    
    def Render(self, path = None):
        
        my_cmap = mplcl.ListedColormap(['blue','red','black','yellow'])
        col = [self.OriginalGraph.nodes()[n]["reward"] for n in self.OriginalGraph.nodes()]
        col[self.node_to_id[self.CurrentPosition]] = 3
        currentpathid = [self.node_to_id[node] for node in self.CurrentPath[:-1]]
        col = [c if ind not in currentpathid else 2 for ind, c in enumerate(col) ]
        nx.draw_networkx(
                    self.OriginalGraph,
                    pos = {n:n for n in self.OriginalGraph.nodes()},
                    cmap=my_cmap,
                    node_color = col,
                    with_labels = False
                )
        #Draw the path
        if path != None:
            nx.draw_networkx_edges(
                self.OriginalGraph,
                pos = {n:n for n in self.OriginalGraph.nodes()},
                edgelist = [(path[i], path[i+1]) for i in range(len(path)-1)],
                edge_color = "k"
            )
        plt.axis("Equal")
        plt.show()
        
    def UpdatePosition(self, chosenPath):
        
        if chosenPath == []:
            #Task cannot be solved anymore (credo)
            solved = 0
            stat = self.reportStats()
            Rtot = stat[0]
            Ltot = stat[1]
            return solved, Rtot, Ltot
        
        self.CurrentPath += chosenPath[1:]
        self.CurrentPosition = chosenPath[-1]
        self.getCollectedRewards()
        self.getRemainingRewards()
        originalNodes = set(list(self.OriginalGraph.nodes()))
        currentPathNodes = set(self.CurrentPath[:-1])
        self.CurrentGraph = self.OriginalGraph.subgraph(list(originalNodes-currentPathNodes))
        term_cond = self.terminationCondition()
        if term_cond != "unfinished":
            #If the task is finished, return the stats
            return term_cond
        else:
            return None
        
    def terminationCondition(self):
        if len(self.CollectedRewards) == len(self.OriginalRewards):
            #print("Task risolto")
            solved = 1
            stat = self.reportStats()
            Rtot = stat[0]
            Ltot = stat[1]
            return solved, Rtot, Ltot
        else:
            return "unfinished"
        
    def reportStats(self):
        Rtot = len(self.CollectedRewards)
        Ltot = len(self.CurrentPath)
        
        return Rtot, Ltot

STATS = {}


#mapPath = "/Users/Mattia/Desktop/EYE_TRACKER/MAPS/"
#problemList = list(np.loadtxt("/Users/Mattia/Desktop/EYE_TRACKER/PIPELINE/problemList.txt", dtype = str))

mapPath = "/home/mattia/TA/MAPS/"
problemList = list(np.loadtxt("/home/mattia/TA/problemList.txt", dtype = str))

t1 = time.time()

TRIAL_INDEX = []
R = []

#Receive depth from the command line
DEPTH_MAX = int(sys.argv[1])
BETA = 100
IT_MAX = int(sys.argv[2])
epsilon = float(sys.argv[3])

#Import the result form previouis depth
if DEPTH_MAX > 3:
    dfPreviousDepth = pd.read_csv("PDLaplacian" +str(DEPTH_MAX-1) + "Epsilon" +str(epsilon) + "BETA" + str(BETA) + ".csv", index_col = 0)

for TrialIndex, name in enumerate(problemList):
    print("TrialIndex: ", TrialIndex)
    name = name[:-5]
    
    with open(mapPath + name, 'r') as f:
            data = json.load(f)
    
    N_REW = data["totalPoints"]
    N_NODES = data["sizeX"]*data["sizeY"]
    statsR = np.zeros(N_REW+1)

    if N_REW < DEPTH_MAX:
        for j in range(N_REW+1):
            statsR[j] = dfPreviousDepth.loc[TrialIndex,str(j)]
        TRIAL_INDEX.append(TrialIndex)
        R.append(list(statsR))
        continue

    for _ in range(IT_MAX):
        
        env = Environment()
        env.initializeEnv(mapPath + name)
        #check if there is an easy ending:
        env.getQuiteEasyending()
        agent = Agent(BETA = -BETA, valueType = "Laplacian", graph = env.CurrentGraph)
        agent.renderValueMap(env.CurrentGraph)
        
        output = None
        while output == None:
            if np.random.rand() < epsilon:
                agent.selfAvoidingRandomWalk(env.CurrentGraph, env.CurrentPosition, 48)
            else:
                agent.computePaths(env.CurrentGraph,
                                env.CurrentPosition,
                                set(env.RemainingRewards),
                                min(len(env.RemainingRewards), DEPTH_MAX),
                                48, 
                                env.easyEnding,
                                env.CollectedRewards,
                                N_REW  
                                )
            agent.getPossiblePathsProbabilities()
            agent.chosenPath = agent.choosePaths()
            #print("ChosenPath: ", agent.chosenPath)
            output = env.UpdatePosition(agent.chosenPath)
            if output:   
                statsR[output[1]] +=1
                if output[0] == 1:     
                    break
            env.Render(agent.chosenPath)
    #Get a distribution from statsR
    statsR = statsR/statsR.sum()

    TRIAL_INDEX.append(TrialIndex)
    R.append(list(statsR))


print(time.time()-t1)

#Save the results as a dataframe
df = pd.DataFrame(R, index = TRIAL_INDEX)
df.to_csv("./PDLaplacian" +str(DEPTH_MAX) + "Epsilon" +str(epsilon) + "BETA" + str(BETA) + ".csv")


