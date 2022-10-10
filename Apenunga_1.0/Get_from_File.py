from Hamilt_Func import phase_factor
import numpy as np
import sys
import sympy as sp

def get_theta(mag):
    return sp.acos(mag[2]/np.linalg.norm(mag))

def get_phi(mag):
    if mag[0]>0:
        return sp.atan(mag[1]/mag[0])
    if mag[0]<0 and mag[1]>=0:
        return sp.atan(mag[1]/mag[0]) + sp.pi
    if mag[0]<0 and mag[1]<0:
        return sp.atan(mag[1]/mag[0]) - sp.pi
    if mag[0]==0 and mag[1]>0:
        return sp.pi/2
    if mag[0]==0 and mag[1]<0:
        return -sp.pi/2
    if mag[0]==0 and mag[1]==0:
        return 0

def read_file(fileName):
    f = open(fileName)

    Svalflag=False
    cellflag=False
    atomsflag=False
    kpathflag=False
    Aniflag=False

    unitcell=np.array([])
    S=[]
    SubLattices={}
    SubLatticesName={}

    Anisotropies={}

    atoms=np.array([])
    qpts=np.array([])

    M=0
    zNum=[]

    listofEELS=[]
    EELS_count=0
    AniCount=0

    for l in f.readlines():
        Temp = l.split()
        if len(Temp)!=0:

    #### Read cell block
            if Temp[0]=='#' and Temp[1]=='END' and Temp[2]=='INPUT_CELL_CARD':
                cellflag=False
            if cellflag:
                unitcell=np.append(unitcell,Temp)
            if (Temp[0]=='#' and Temp[1]=='INPUT_CELL_CARD'):
                cellflag=True
    ##### end read cell block

    #### Read spin size block
            if Temp[0]=='#' and Temp[1]=='END' and Temp[2]=='INPUT_SValues' :
                Svalflag=False
            if Svalflag:
                SubLattices[str(M)]={'theta':get_theta(np.array(Temp[1:]).astype(float)),\
                                       'phi':get_phi(np.array(Temp[1:]).astype(float))}
                S=np.append(S,np.linalg.norm(np.array(Temp[1:]).astype(float)))
                SubLatticesName[Temp[0]]=M
                M+=1
            if (Temp[0]=='#' and Temp[1]=='INPUT_SValues'):
                Svalflag=True
    ##### end spin size block

    #### Read atoms block
            if Temp[0]=='#' and Temp[1]=='END' and Temp[2]=='INPUT_ATOMS_CARD_FRAC' :
                atomsflag=False
            if atomsflag:
                convTemp=np.append(np.array(Temp[1:]).astype(float),SubLatticesName[Temp[0]])
                atoms = np.append(atoms,convTemp)
                zNum.append(Temp[-1])
            if (Temp[0]=='#' and Temp[1]=='INPUT_ATOMS_CARD_FRAC'):
                atomsflag=True
    ##### end atoms block

    #### Read anisotropies block
            if len(Temp)!=0:
                if Temp[0]=='#' and Temp[1]=='END' and Temp[2]=='INPUT_LOCAL_ANISOTROPIES' :
                    Aniflag=False
                if Aniflag:
                    Anisotropies[str(AniCount)]={'modulus':(1e-3)*np.linalg.norm(np.array(Temp).astype(float),ord=1) , \
                                                'theta':get_theta(np.array(Temp).astype(float)),\
                                                'phi':get_phi(np.array(Temp).astype(float))}
                    AniCount+=1
                if (Temp[0]=='#' and Temp[1]=='INPUT_LOCAL_ANISOTROPIES'):
                    Aniflag=True
    ##### end anisotropies block            
            
    #### Read kpath block
            if Temp[0]=='#' and Temp[1]=='END' and Temp[2]=='KPATH_CARD' :
                kpathflag=False
            if kpathflag:
                if Temp[-1]=='*':
                    listofEELS.append(EELS_count)
                    convTemp=np.array(Temp[:-1]).astype(float)
                    qpts = np.append(qpts,convTemp)
                    EELS_count+=1
                else:
                    convTemp=np.array(Temp).astype(float)
                    qpts = np.append(qpts,convTemp)
                    EELS_count+=1
            if (Temp[0]=='#' and Temp[1]=='KPATH_CARD'):
                kpathflag=True
    ##### end kpath block



    ##### Reshape unitcell and atoms
    unitcell=unitcell.reshape(3,3).astype(float)
    atoms=atoms.reshape(int(len(atoms)/4),4).astype(float)
    qpts=qpts.reshape(int(len(qpts)/3),3).astype(float)

    zNum=[float(i) for i in zNum]
    zNum=[*set(zNum)]
    ML=len(zNum)

    return unitcell, M, SubLattices, S, atoms, ML, qpts,listofEELS, Anisotropies

def get_Neigb(M,Jmatrix,unitcell,atomsFrac,q,NeighbNum,ML):

    ## Convert the fractional cell to coordinate distances.

    temp = np.dot(atomsFrac[:,:3],unitcell)  # Atoms with actual distances

    atoms = np.column_stack((temp,atomsFrac[:,-1]))  # Atoms with actual size with character (M)

    ## At each layer we only want to account for one of each possible interaction 
    ## Set an empty array to store the supercell of atoms

    atomsSuper=np.array([])

    zaxisArray=np.unique(atoms[:,2])
    print(zaxisArray)
    # Get the number of atoms types per layer

    layerCounter=0
    test=0
    numofAtomtypesinLayer=np.zeros([ML])
    numofAtominLayer=np.zeros([ML])
    Mlist=[]

    for i in atoms:
        j3=np.where(zaxisArray==i[2])[0][0]
        if i[2]!=layerCounter:
            layerCounter=i[2]
            Mlist=[]
        numofAtominLayer[j3]+=1
        
        
        if i[3] not in Mlist:
            numofAtomtypesinLayer[j3]+=1
            Mlist.append(i[3])
        
    ## Create the super cell, by multiplying in the x and y directions ###########

    for i in range(-2,3):
        for j in range(-2,3):
            temp = atoms + np.append(np.sum(((np.array([i,j,0])*unitcell)),axis=1),0)
            atomsSuper = np.append(atomsSuper,temp)

    atomsSuper = atomsSuper.reshape(int(len(atomsSuper)/4),4) 
    
    LayerDict={}

    JLayerDict={}

    for i in range(len(zaxisArray)):
        NeibDict={}
        for j in range(NeighbNum):
            NeibDict[str(j+1)]={'Para':{'LayerofNeib':np.nan,
                                        'PhaseMatrix':np.zeros([M,M,len(q)], dtype=complex),
                                        'NeigbMatrix':np.zeros([M,M], dtype=complex)},
                                'PerpUp':{'LayerofNeib':np.nan,
                                        'PhaseMatrix':np.zeros([M,M,len(q)], dtype=complex),
                                        'NeigbMatrix':np.zeros([M,M], dtype=complex)},
                                'PerpDw':{'LayerofNeib':np.nan,
                                        'PhaseMatrix':np.zeros([M,M,len(q)], dtype=complex),
                                        'NeigbMatrix':np.zeros([M,M], dtype=complex)}}
        LayerDict[str(i)]=NeibDict

    for i in range(len(zaxisArray)):
        NeibDict={}
        for j in range(NeighbNum):
            NeibDict[str(j+1)]={'Para':{'LayerofNeib':np.nan,
                                        'Jmatrix':np.zeros([M,M,len(q)], dtype=complex),
                                        'NeigbMatrix':np.zeros([M,M], dtype=complex)},
                                'PerpUp':{'LayerofNeib':np.nan,
                                        'Jmatrix':np.zeros([M,M,len(q)], dtype=complex),
                                        'NeigbMatrix':np.zeros([M,M], dtype=complex)},
                                'PerpDw':{'LayerofNeib':np.nan,
                                          'Jmatrix':np.zeros([M,M,len(q)], dtype=complex),
                                          'NeigbMatrix':np.zeros([M,M], dtype=complex)}}
        JLayerDict[str(i)]=NeibDict 

    for AtomNum in range(len(atoms)):
        
        #print('Atoms',atoms[AtomNum])
        
        j3=np.where(zaxisArray==atoms[AtomNum,2])[0][0]

        DistancesTemp = atomsSuper[:,:-1]-atoms[AtomNum,:-1]   # We get the distances of every atom
        Distances = np.sqrt(np.sum(DistancesTemp**2,axis=1))   # in the atom with respect to the atom analysed 

        arr1inds = Distances.argsort()                         # Get the indexes that sort the distances 
        Distances = Distances[arr1inds]                        # Sort the distances  
        sorted_Vectors = atomsSuper[arr1inds,:]                # Sort the supercell atoms (closest first)    

        # Now we create a final array with the information about the neighbours of this atom in question
        # [Atoms position(xyz), SubLattice, Distance to the atom in question]

        FinalNeihb = np.column_stack((sorted_Vectors[:,:-1]-atoms[AtomNum,:-1],sorted_Vectors[:,-1],Distances))[1:]

        DistancesUnique=np.asarray(sorted(set(Distances)))               # Remove repeated distances
        DistancesUnique=np.unique(DistancesUnique.round(decimals=6))     # Round them

        # We need to go over the atoms, and at each layer count only each pair of interaction once.
        # First let's write a disctionary with the J, relative vector, 

        AtomIndex=int(atoms[AtomNum,-1])
        #print(AtomIndex)

        for i in FinalNeihb: 
            
            ## Phase

            NeibPara=np.zeros([M,M,len(q)], dtype=complex)

            NeibPerpPlus=np.zeros([M,M,len(q)], dtype=complex)

            NeibPerpMinus=np.zeros([M,M,len(q)], dtype=complex)

            
            ## Jij

            JNeibPara=np.zeros([M,M,len(q)], dtype=complex)

            JNeibPerpPlus=np.zeros([M,M,len(q)], dtype=complex)

            JNeibPerpMinus=np.zeros([M,M,len(q)], dtype=complex)
            # Get the layer of the current neighbour

            try:
                j3Neib=np.where(zaxisArray==atoms[AtomNum,2]+i[2])[0][0]
            except:
                continue
            
            CurrentNeib=np.where(DistancesUnique==i[-1].round(decimals=6))[0][0]

            if CurrentNeib>NeighbNum:
                continue
            
            
            #print('Neighbour',i)
            Temp = np.array([np.insert(i[:3],0,Jmatrix[int(atoms[AtomNum,-1]),int(i[-2]),CurrentNeib-1])])
            # Temp stores information that is used to calculate the phase factor
            # [[Relative Distance X,Relative Distance Y, Relative Distance Z]]

            
            
            if j3==j3Neib:

                NeibPara[int(i[-2]),AtomIndex]+=phase_factor(q, Temp)
                JNeibPara[int(i[-2]),AtomIndex]+=Temp[0,0]

                #if M>1:
                #    NeibPara[AtomIndex,int(i[-2])]+=phase_factor(q, Temp)
                #    JNeibPara[AtomIndex,int(i[-2])]+=Temp[0,0] 
                #print(j3,CurrentNeib)
                LayerDict[str(j3)][str(CurrentNeib)]['Para']['LayerofNeib']=j3Neib
                JLayerDict[str(j3)][str(CurrentNeib)]['Para']['LayerofNeib']=j3Neib
                
                LayerDict[str(j3)][str(CurrentNeib)]['Para']['PhaseMatrix']+=NeibPara
                JLayerDict[str(j3)][str(CurrentNeib)]['Para']['Jmatrix']+=JNeibPara
                
                LayerDict[str(j3)][str(CurrentNeib)]['Para']['NeigbMatrix'][int(i[3]),int(AtomIndex)]=(numofAtominLayer[j3]/numofAtomtypesinLayer[j3])**-1
                JLayerDict[str(j3)][str(CurrentNeib)]['Para']['NeigbMatrix'][int(i[3]),int(AtomIndex)]=(numofAtominLayer[j3]/numofAtomtypesinLayer[j3])**-1
                
            if j3>j3Neib:
                NeibPerpMinus[int(i[-2]),AtomIndex]+=phase_factor(q,  Temp)
                JNeibPerpMinus[int(i[-2]),AtomIndex]+=Temp[0,0]

                #if M>1:
                #    NeibPerpMinus[AtomIndex,int(i[-2])]+=phase_factor(q,  Temp)
                #    JNeibPerpMinus[AtomIndex,int(i[-2])]+=Temp[0,0]

                LayerDict[str(j3)][str(CurrentNeib)]['PerpDw']['LayerofNeib']=j3Neib 
                JLayerDict[str(j3)][str(CurrentNeib)]['PerpDw']['LayerofNeib']=j3Neib
                
                LayerDict[str(j3)][str(CurrentNeib)]['PerpDw']['PhaseMatrix']+=NeibPerpMinus
                JLayerDict[str(j3)][str(CurrentNeib)]['PerpDw']['Jmatrix']+=JNeibPerpMinus
                
                LayerDict[str(j3)][str(CurrentNeib)]['PerpDw']['NeigbMatrix'][int(i[3]),int(AtomIndex)]=(numofAtominLayer[j3]/numofAtomtypesinLayer[j3])**-1
                JLayerDict[str(j3)][str(CurrentNeib)]['PerpDw']['NeigbMatrix'][int(i[3]),int(AtomIndex)]=(numofAtominLayer[j3]/numofAtomtypesinLayer[j3])**-1
                
            if j3<j3Neib:
                NeibPerpPlus[int(i[-2]),AtomIndex]+=phase_factor(q, Temp)
                JNeibPerpPlus[int(i[-2]),AtomIndex]+=Temp[0,0]

                #if M>1:
                #    NeibPerpPlus[AtomIndex,int(i[-2])]+=phase_factor(q, Temp)
                #    JNeibPerpPlus[AtomIndex,int(i[-2])]+=Temp[0,0]
                    
                LayerDict[str(j3)][str(CurrentNeib)]['PerpUp']['LayerofNeib']=j3Neib 
                JLayerDict[str(j3)][str(CurrentNeib)]['PerpUp']['LayerofNeib']=j3Neib
                
                LayerDict[str(j3)][str(CurrentNeib)]['PerpUp']['PhaseMatrix']+=NeibPerpPlus
                JLayerDict[str(j3)][str(CurrentNeib)]['PerpUp']['Jmatrix']+=JNeibPerpPlus
                
                LayerDict[str(j3)][str(CurrentNeib)]['PerpUp']['NeigbMatrix'][int(i[3]),int(AtomIndex)]=(numofAtominLayer[j3]/numofAtomtypesinLayer[j3])**-1
                JLayerDict[str(j3)][str(CurrentNeib)]['PerpUp']['NeigbMatrix'][int(i[3]),int(AtomIndex)]=(numofAtominLayer[j3]/numofAtomtypesinLayer[j3])**-1

       
    return LayerDict, JLayerDict