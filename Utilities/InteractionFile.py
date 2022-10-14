import numpy as np

names_of_atoms  = np.array(['NiUp','NiDw'])


NeighbNum=2

Jij=np.zeros([len(names_of_atoms),len(names_of_atoms),NeighbNum])

Jij[0,0,0]=13.6056980659*1.432*0.001
Jij[0,1,0]=-13.6056980659*1.432*0.001
Jij[1,0,0]=-13.6056980659*1.432*0.001
Jij[1,1,0]=13.6056980659*1.432*0.001

Jij[0,0,1]=13.6056980659*1.432*0.001
Jij[0,1,1]=-13.6056980659*1.432*0.001
Jij[1,0,1]=-13.6056980659*1.432*0.001
Jij[1,1,1]=13.6056980659*1.432*0.001

for k in range(NeighbNum):
  for i in range(len(names_of_atoms)):
    for j in range(len(names_of_atoms)):
      print(names_of_atoms[i], names_of_atoms[j], k, Jij[i,j,k])