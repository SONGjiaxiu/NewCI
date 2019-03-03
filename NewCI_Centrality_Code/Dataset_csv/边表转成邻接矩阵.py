import collections
import numpy as np
np.set_printoptions(threshold=np.nan)
def load_graph(path):
    G = collections.defaultdict(dict)
    with open(path) as text:
        for line in text:
            vertices = line.strip().split(',')
            v_i = int(vertices[0])
            v_j = int(vertices[1])
            w = float(vertices[2])
            G[v_i][v_j] = w
            G[v_j][v_i] = w
    return G
#G = load_graph(r'C:\Python27\sjxwork\NewCI_Centrality_Code\Dataset_csv\Citeseer.csv')
# print G
def Data_Shape(Data):
    List_A = []
    List_B = []
    for row in range(Data.shape[0]):
        List_A.append(Data[row][0])
        List_B.append(Data[row][1])
    List_A = list(set(List_A))
    List_B = list(set(List_B))
    length_A = len(List_A)
    length_B = len(List_B)
    #print '    the length of dataset:'+str(Data.shape[0])
    #print '    the length of first cloum node:('+str(length_A)+')'
    #print '    the length of second cloum node:('+str(length_B)+')'
    MaxNodeNum =  int(max(max(List_A),max(List_B)))+1
    #print '    number of node:'+str(MaxNodeNum)
    return MaxNodeNum
def MatrixAdjacency(MaxNodeNum,Data):
    MatrixAdjacency = np.zeros([MaxNodeNum,MaxNodeNum])
    for col in range(Data.shape[0]):
        i = int(Data[col][0])
        j = int(Data[col][1])
        MatrixAdjacency[i,j] = 1
        MatrixAdjacency[j,i] = 1
    return MatrixAdjacency
NetData = np.loadtxt(r'C:\Python27\sjxwork\NewCI_Centrality_Code\Dataset_csv\citeseer.txt')
MaxNodeNum = Data_Shape(NetData)
MatrixAdjacency_Net = MatrixAdjacency(MaxNodeNum, NetData)
print MatrixAdjacency_Net[1][100]
file=open(r'C:\Python27\sjxwork\NewCI_Centrality_Code\Dataset_csv\citeseer_juzhen.txt',"w+")
for i in range(0,MaxNodeNum):
    for j in range(0,MaxNodeNum):
        file.write(str(MatrixAdjacency_Net[i][j])+' ')
    file.write('\n')
file.close()
