import numpy as np
import sys

ML = int(sys.argv[1])

atoms=np.array([])
for num,i in enumerate(range(ML)):
  if i==0:
    atoms=np.append(atoms,[0.0 ,0.0, 2.0])
  elif i==ML-1:
    atoms=np.append(atoms,[0.0 ,0.0, 2.0])
  else:
    atoms=np.append(atoms,[0.0 ,0.0, 1.0])


atoms=atoms.reshape(int(len(atoms)/3),3)

for i in range(len(atoms)):
  print(atoms[i,0], atoms[i,1],atoms[i,2])
# atoms=atoms.reshape((4*ML)+(8*ML),4)

# print(a,0.0,0.0)
# print(0.0,a,0.0)
# print(0.0,0.0,(a/(2*supercell))*ML)

