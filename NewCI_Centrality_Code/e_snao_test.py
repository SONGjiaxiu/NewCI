import snap


UGraph = snap.GenRndGnm(snap.PUNGraph, 20, 100)
file1 = open(r"C:\Users\Administrator\Desktop\3th_result\mianyitu\email\G_dege1.csv", "w+" )

G_edgeS=[] 

for EI in UGraph.Edges():
    #print EI.GetSrcNId(), EI.GetDstNId(),1
    G_edge=(EI.GetSrcNId(), EI.GetDstNId())
    G_edgeS.append(list(G_edge))

print G_edgeS
for i in G_edgeS:
    print i[0],i[1]
    file1.write(str(i[0])+','+str(i[1])+','+str(1)+'\n')
