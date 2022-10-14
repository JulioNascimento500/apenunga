import numpy as np
import sympy as sp

###############################################
#
# Apenunga code Hamiltonian functions
# Author: Julio do Nascimento 
# Date: 10/08/2022
#
###############################################



## The use of scipy is important to avoid roundoff errors, for more info:
## https://stackoverflow.com/questions/18646477/why-is-sin180-not-zero-when-using-python-and-numpy

#SubLattices = {"1":{'theta':0,'phi':0},"2":{'theta':sp.pi,'phi':0}}

#################################################
# Functions that represent the v basis, in terms of the 
# ladder operators.
#
###############################################

    
# First term increases sublattice second increases J3

##########################################
#
# Function defining the tranformation for angles of diffent atoms in the magnetic unitcell.
#
#######################################
def scos(x): return sp.N(sp.cos(x))
def ssin(x): return sp.N(sp.sin(x))

def Fzz(Dict,r,s):
    r = str(r)
    s = str(s)
    
    SinProd=ssin(Dict[r]['theta'])*ssin(Dict[s]['theta'])
    CosProd=scos(Dict[r]['theta'])*scos(Dict[s]['theta'])
    
    SinDiff=ssin(Dict[r]['phi']-Dict[s]['phi'])
    CosDiff=scos(Dict[r]['phi']-Dict[s]['phi'])
    
    return float(SinProd*CosDiff + CosProd)

def G1(Dict,r,s):
    r = str(r)
    s = str(s)
    
    SinProd=ssin(Dict[r]['theta'])*ssin(Dict[s]['theta'])
    CosProd=scos(Dict[r]['theta'])*scos(Dict[s]['theta'])
    
    SinDiff=ssin(Dict[r]['phi']-Dict[s]['phi'])
    CosDiff=scos(Dict[r]['phi']-Dict[s]['phi'])
    
    CosPlus=scos(Dict[r]['theta']) + scos(Dict[s]['theta'])
    
    return float((CosProd + 1)*CosDiff + SinProd - 1.0j*SinDiff*CosPlus)

def G2(Dict,r,s):
    r = str(r)
    s = str(s)

    SinProd=ssin(Dict[r]['theta'])*ssin(Dict[s]['theta'])
    CosProd=scos(Dict[r]['theta'])*scos(Dict[s]['theta'])
    
    SinDiff=ssin(Dict[r]['phi']-Dict[s]['phi'])
    CosDiff=scos(Dict[r]['phi']-Dict[s]['phi'])
    
    CosMinus=scos(Dict[r]['theta']) - scos(Dict[s]['theta'])
    
    return float((CosProd - 1)*CosDiff + SinProd - 1.0j*SinDiff*CosMinus)

def phase_factor(q, neighList):

    Gamma = np.array([neighList[:,0]])*np.exp(-1.0j*np.matmul(q,neighList[:,1:].T))

    return np.sum(Gamma, axis=1)

# def phase_factor(q, neighList):

#     Gamma = np.exp(-1.0j*np.dot(q,neighList.T))

#     return np.sum(Gamma)

def Hamiltonian(ML,M,NeighbNum,S,JLayerDict,LayerDict,SubLattices,qpts):

    H=np.zeros([2*M*ML,2*M*ML,qpts], dtype=complex,order='F')
    k=0
    numofatomsperlayer=1

    for i in range(1,NeighbNum+1):
        for j3 in range(ML):
            for r in range(M):
                for s in range(M):
                    
                    j3st=(M*j3)
                    # Parallel
                    
#                     nN=JLayerDict[str(j3)][str(i)]['Para']['LayerofNeib']
                    
#                     if not np.isnan(nN):
                    
                    JTermPara=JLayerDict[str(j3)][str(i)]['Para']['Jmatrix'][r,s]*         \
                              JLayerDict[str(j3)][str(i)]['Para']['NeigbMatrix'][r,s]

                    PhaseTermPara=LayerDict[str(j3)][str(i)]['Para']['PhaseMatrix'][r,s]*  \
                                  LayerDict[str(j3)][str(i)]['Para']['NeigbMatrix'][r,s]

                    H[j3st+r,j3st+r,:]+=S[s]*JTermPara*Fzz(SubLattices,r,s)
                    H[j3st+s,j3st+s,:]+=S[r]*JTermPara*Fzz(SubLattices,r,s)
                    
                    H[j3st+r+(M*ML),j3st+r+(M*ML),:]+=S[s]*JTermPara*Fzz(SubLattices,r,s)
                    H[j3st+s+(M*ML),j3st+s+(M*ML),:]+=S[r]*JTermPara*Fzz(SubLattices,r,s)

                    H[j3st+r+(M*ML),j3st+s+(M*ML),:]-=(np.sqrt(S[r]*S[s])/2)*PhaseTermPara*np.conj(G1(SubLattices,r,s))
                    H[j3st+r+(M*ML),j3st+s,:]-=(np.sqrt(S[r]*S[s])/2)*PhaseTermPara*G2(SubLattices,r,s)
                    H[j3st+r,j3st+s,:]-=(np.sqrt(S[r]*S[s])/2)*np.conj(PhaseTermPara)*G1(SubLattices,r,s)
                    H[j3st+r,j3st+s+(M*ML),:]-=(np.sqrt(S[r]*S[s])/2)*np.conj(PhaseTermPara)*np.conj(G2(SubLattices,r,s))
                    
                    H[j3st+r,j3st+s,:]-=(np.sqrt(S[r]*S[s])/2)*PhaseTermPara*np.conj(G1(SubLattices,r,s))
                    H[j3st+r,j3st+s+(M*ML),:]-=(np.sqrt(S[r]*S[s])/2)*PhaseTermPara*G2(SubLattices,r,s)
                    H[j3st+r+(M*ML),j3st+s+(M*ML),:]-=(np.sqrt(S[r]*S[s])/2)*np.conj(PhaseTermPara)*G1(SubLattices,r,s)
                    H[j3st+r+(M*ML),j3st+s,:]-=(np.sqrt(S[r]*S[s])/2)*np.conj(PhaseTermPara)*np.conj(G2(SubLattices,r,s))

                    
                    # Perp Up
                    
                    nN=JLayerDict[str(j3)][str(i)]['PerpUp']['LayerofNeib']*M
                    
                    
                    if not np.isnan(nN):

                        JTermPerpUp=JLayerDict[str(j3)][str(i)]['PerpUp']['Jmatrix'][r,s]*         \
                                  JLayerDict[str(j3)][str(i)]['PerpUp']['NeigbMatrix'][r,s]

                        PhaseTermPerpUp=LayerDict[str(j3)][str(i)]['PerpUp']['PhaseMatrix'][r,s]*  \
                                        LayerDict[str(j3)][str(i)]['PerpUp']['NeigbMatrix'][r,s]                        
                        
                        H[j3st+r,j3st+r,:]+=S[s]*JTermPerpUp*Fzz(SubLattices,r,s)
                        H[nN+s,nN+s,:]+=S[r]*JTermPerpUp*Fzz(SubLattices,r,s)

                        H[j3st+r+(M*ML),j3st+r+(M*ML),:]+=S[s]*JTermPerpUp*Fzz(SubLattices,r,s)
                        H[nN+s+(M*ML),nN+s+(M*ML),:]+=S[r]*JTermPerpUp*Fzz(SubLattices,r,s)

                        H[j3st+r+(M*ML),nN+s+(M*ML),:]-=(np.sqrt(S[r]*S[s])/2)*PhaseTermPerpUp*np.conj(G1(SubLattices,r,s))
                        H[j3st+r+(M*ML),nN+s,:]-=(np.sqrt(S[r]*S[s])/2)*PhaseTermPerpUp*G2(SubLattices,r,s)
                        H[j3st+r,nN+s,:]-=(np.sqrt(S[r]*S[s])/2)*np.conj(PhaseTermPerpUp)*G1(SubLattices,r,s)
                        H[j3st+r,nN+s+(M*ML),:]-=(np.sqrt(S[r]*S[s])/2)*np.conj(PhaseTermPerpUp)*np.conj(G2(SubLattices,r,s))

                        H[j3st+r,nN+s,:]-=(np.sqrt(S[r]*S[s])/2)*PhaseTermPerpUp*np.conj(G1(SubLattices,r,s))
                        H[j3st+r,nN+s+(M*ML),:]-=(np.sqrt(S[r]*S[s])/2)*PhaseTermPerpUp*G2(SubLattices,r,s)
                        H[j3st+r+(M*ML),nN+s+(M*ML),:]-=(np.sqrt(S[r]*S[s])/2)*np.conj(PhaseTermPerpUp)*G1(SubLattices,r,s)
                        H[j3st+r+(M*ML),nN+s,:]-=(np.sqrt(S[r]*S[s])/2)*np.conj(PhaseTermPerpUp)*np.conj(G2(SubLattices,r,s))
                                                           
                                                           
                    # Perp Down
                    
                    nN=JLayerDict[str(j3)][str(i)]['PerpDw']['LayerofNeib']*M
                    
                    if not np.isnan(nN):
                        
                        JTermPerpDw=JLayerDict[str(j3)][str(i)]['PerpDw']['Jmatrix'][r,s]*         \
                                    JLayerDict[str(j3)][str(i)]['PerpDw']['NeigbMatrix'][r,s]

                        PhaseTermPerpDw=LayerDict[str(j3)][str(i)]['PerpDw']['PhaseMatrix'][r,s]*  \
                                        LayerDict[str(j3)][str(i)]['PerpDw']['NeigbMatrix'][r,s]   
                        
                        H[j3st+r,j3st+r,:]+=S[s]*JTermPerpDw*Fzz(SubLattices,r,s)
                        H[nN+s,nN+s,:]+=S[r]*JTermPerpDw*Fzz(SubLattices,r,s)
                                                           
                        H[j3st+r+(M*ML),j3st+r+(M*ML),:]+=S[s]*JTermPerpDw*Fzz(SubLattices,r,s)
                        H[nN+s+(M*ML),nN+s+(M*ML),:]+=S[r]*JTermPerpDw*Fzz(SubLattices,r,s)

                        H[j3st+r+(M*ML),nN+s+(M*ML),:]-=(np.sqrt(S[r]*S[s])/2)*PhaseTermPerpDw*np.conj(G1(SubLattices,r,s))
                        H[j3st+r+(M*ML),nN+s,:]-=(np.sqrt(S[r]*S[s])/2)*PhaseTermPerpDw*G2(SubLattices,r,s)
                        H[j3st+r,nN+s,:]-=(np.sqrt(S[r]*S[s])/2)*np.conj(PhaseTermPerpDw)*G1(SubLattices,r,s)
                        H[j3st+r,nN+s+(M*ML),:]-=(np.sqrt(S[r]*S[s])/2)*np.conj(PhaseTermPerpDw)*np.conj(G2(SubLattices,r,s))

                        H[j3st+r,nN+s,:]-=(np.sqrt(S[r]*S[s])/2)*PhaseTermPerpDw*np.conj(G1(SubLattices,r,s))
                        H[j3st+r,nN+s+(M*ML),:]-=(np.sqrt(S[r]*S[s])/2)*PhaseTermPerpDw*G2(SubLattices,r,s)
                        H[j3st+r+(M*ML),nN+s+(M*ML),:]-=(np.sqrt(S[r]*S[s])/2)*np.conj(PhaseTermPerpDw)*G1(SubLattices,r,s)
                        H[j3st+r+(M*ML),nN+s,:]-=(np.sqrt(S[r]*S[s])/2)*np.conj(PhaseTermPerpDw)*np.conj(G2(SubLattices,r,s))

                    
                


    #H = H/(4*1.375)
    H = H/(4)
    #print('Before')
    #print(H[:,:,0])
    H[(M*ML):,(M*ML):,:] = -H[(M*ML):,(M*ML):,:]
    
    H[(M*ML):,:(M*ML),:] = -H[(M*ML):,:(M*ML),:]
    #print('After')
    #print(H[:,:,0])

    return H

def Ax(Dict,r,KDict):
    eta=KDict['theta']
    delta=KDict['phi']
    theta=Dict[str(r)]['theta']
    phi=Dict[str(r)]['phi']
    return(ssin(eta)*scos(theta)*scos(delta-phi) -  \
           scos(eta)*ssin(theta))

def Ay(Dict,r,KDict):
    eta=KDict['theta']
    delta=KDict['phi']
    theta=Dict[str(r)]['theta']
    phi=Dict[str(r)]['phi']
    return(ssin(eta)*ssin(delta-phi))
           
def Az(Dict,r,KDict):
    eta=KDict['theta']
    delta=KDict['phi']
    theta=Dict[str(r)]['theta']
    phi=Dict[str(r)]['phi']
    return(ssin(eta)*ssin(theta)*scos(delta-phi) -  \
           scos(eta)*scos(theta))

def HamiltonianK(ML,M,S,SubLattices,Anisotropies,qpts):

    H=np.zeros([2*M*ML,2*M*ML,qpts], dtype=complex,order='F')

    for j3 in range(ML):
        for r in range(M):
           
                j3st=(M*j3)
            
                
           
                H[j3st+r,j3st+r,:] = -float( (Anisotropies[str(j3)]['modulus']*S[r]/2)* \
                                      ((Ax(SubLattices,r,Anisotropies[str(j3)])**2) + \
                                      (Ay(SubLattices,r,Anisotropies[str(j3)])**2) - \
                                    (2*Az(SubLattices,r,Anisotropies[str(j3)])**2)))
           
                H[j3st+r+(M*ML),j3st+r+(M*ML),:] = -float((Anisotropies[str(j3)]['modulus']*S[r]/2)* \
                                                    ((Ax(SubLattices,r,Anisotropies[str(j3)])**2) + \
                                                    (Ay(SubLattices,r,Anisotropies[str(j3)])**2) - \
                                                  (2*Az(SubLattices,r,Anisotropies[str(j3)])**2)) )
           
                H[j3st+r,j3st+r+(M*ML),:] = -float((Anisotropies[str(j3)]['modulus']*S[r]/2)* \
                    (Ax(SubLattices,r,Anisotropies[str(j3)]) + 1.0j*Ay(SubLattices,r,Anisotropies[str(j3)]))**2 )
                                                  
           
                H[j3st+r+(M*ML),j3st+r,:] = -float((Anisotropies[str(j3)]['modulus']*S[r]/2)* \
                    (Ax(SubLattices,r,Anisotropies[str(j3)]) - 1.0j*Ay(SubLattices,r,Anisotropies[str(j3)]))**2 )

    H = H/(2)

    H[(M*ML):,(M*ML):,:] = -H[(M*ML):,(M*ML):,:]
    
    H[(M*ML):,:(M*ML),:] = -H[(M*ML):,:(M*ML),:]


    return H