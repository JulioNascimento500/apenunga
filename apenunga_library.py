"""Definition of the Inputs class.

This module defines the central object in the calcualation the Inputs
object.
"""
import numpy as np
import sympy as sym
from ase import Atoms, Atom
from ase.data import covalent_radii
from ase.calculators.neighborlist import NeighborList
from scipy.linalg import eig, inv,lu
import matplotlib.pyplot as plt
from numba import njit
from functools import wraps
import time
from numba import njit



def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper

def get_theta(mag):
    return sym.acos(mag[2]/np.linalg.norm(mag))

def get_phi(mag):
    if mag[0]>0:
        return sym.atan(mag[1]/mag[0])
    if mag[0]<0 and mag[1]>=0:
        return sym.atan(mag[1]/mag[0]) + sym.pi
    if mag[0]<0 and mag[1]<0:
        return sym.atan(mag[1]/mag[0]) - sym.pi
    if mag[0]==0 and mag[1]>0:
        return sym.pi/2
    if mag[0]==0 and mag[1]<0:
        return -sym.pi/2
    if mag[0]==0 and mag[1]==0:
        return 0

def phase_factor(q, neighList):

    Gamma = np.array([neighList[:,0]])*np.exp(-1.0j*np.matmul(q,neighList[:,1:].T))

    return np.sum(Gamma, axis=1)

##########################################
#
# Function defining the tranformation for angles of diffent atoms in the magnetic unitcell.
#
#######################################

def scos(x): return sym.N(sym.cos(x))

def ssin(x): return sym.N(sym.sin(x))

def get_theta(mag):
    return sym.acos(mag[2]/np.linalg.norm(mag))

def get_phi(mag):
    if mag[0]>0:
        return sym.atan(mag[1]/mag[0])
    if mag[0]<0 and mag[1]>=0:
        return sym.atan(mag[1]/mag[0]) + sym.pi
    if mag[0]<0 and mag[1]<0:
        return sym.atan(mag[1]/mag[0]) - sym.pi
    if mag[0]==0 and mag[1]>0:
        return sym.pi/2
    if mag[0]==0 and mag[1]<0:
        return -sym.pi/2
    if mag[0]==0 and mag[1]==0:
        return 0


def Fzz(Atoms,r,s):
        
    theta_r = Atoms.aseatoms.get_initial_magnetic_moments()[r,1]
    phi_r = Atoms.aseatoms.get_initial_magnetic_moments()[r,2]
    
    theta_s = Atoms.aseatoms.get_initial_magnetic_moments()[s,1]
    phi_s = Atoms.aseatoms.get_initial_magnetic_moments()[s,2]
    
    SinProd=ssin(theta_r)*ssin(theta_s)
    CosProd=scos(theta_r)*scos(theta_s)
    
    CosDiff=scos(phi_r-phi_s)
    
    return float(SinProd*CosDiff + CosProd)


def G1(Atoms,r,s):
        
    theta_r = Atoms.aseatoms.get_initial_magnetic_moments()[r,1]
    phi_r = Atoms.aseatoms.get_initial_magnetic_moments()[r,2]
    
    theta_s = Atoms.aseatoms.get_initial_magnetic_moments()[s,1]
    phi_s = Atoms.aseatoms.get_initial_magnetic_moments()[s,2]
    
    SinProd=ssin(theta_r)*ssin(theta_s)
    CosProd=scos(theta_r)*scos(theta_s)
    
    SinDiff=ssin(phi_r-phi_s)
    CosDiff=scos(phi_r-phi_s)
    
    CosPlus=scos(theta_r) + scos(theta_s)
    
    return float(((CosProd + 1)*CosDiff) + SinProd - (1.0j*SinDiff*CosPlus))


def G2(Atoms,r,s):
       
    theta_r = Atoms.aseatoms.get_initial_magnetic_moments()[r,1]
    phi_r = Atoms.aseatoms.get_initial_magnetic_moments()[r,2]
    
    theta_s = Atoms.aseatoms.get_initial_magnetic_moments()[s,1]
    phi_s = Atoms.aseatoms.get_initial_magnetic_moments()[s,2]

    SinProd=ssin(theta_r)*ssin(theta_s)
    CosProd=scos(theta_r)*scos(theta_s)
    
    SinDiff=ssin(phi_r-phi_s)
    CosDiff=scos(phi_r-phi_s)
    
    CosMinus=scos(theta_r) - scos(theta_s)
    
    return float(((CosProd - 1)*CosDiff) + SinProd - (1.0j*SinDiff*CosMinus))


def Ax(Atoms,Anisotropy,r):
    eta=Anisotropy[r,1]
    delta=Anisotropy[r,2]
    theta = Atoms.aseatoms.get_initial_magnetic_moments()[r,1]
    phi = Atoms.aseatoms.get_initial_magnetic_moments()[r,2]
    return float(ssin(eta)*scos(theta)*scos(delta-phi) -  \
           scos(eta)*ssin(theta))

def Ay(Atoms,Anisotropy,r):
    eta=Anisotropy[r,1]
    delta=Anisotropy[r,2]
    theta = Atoms.aseatoms.get_initial_magnetic_moments()[r,1]
    phi = Atoms.aseatoms.get_initial_magnetic_moments()[r,2]
    return float(ssin(eta)*ssin(delta-phi))
           
def Az(Atoms,Anisotropy,r):
    eta=Anisotropy[r,1]
    delta=Anisotropy[r,2]
    theta = Atoms.aseatoms.get_initial_magnetic_moments()[r,1]
    phi = Atoms.aseatoms.get_initial_magnetic_moments()[r,2]
    return float(ssin(eta)*ssin(theta)*scos(delta-phi) -  \
           scos(eta)*scos(theta))

def Get_atom(Dict,AtomNum):
    return [i for i in Dict if Dict[i]==AtomNum]

class Input:
  '''Inputs Object:
  
  
  '''
  def __init__(self,fileName,zperiodic=False,step_size=0.01,neib=None):

    Svalflag=False
    cellflag=False
    atomsflag=False
    kpathflag=False
    Aniflag=False
    Jijflag=False
    jsflag=False
    ksflag=False

    cell=np.array([])
    self.aseatoms=None

    self.S=[]
    self.MagMoms=np.array([])

    self.Anisotropies=None

    self.atoms=np.array([])
    self.qpts=np.array([])
    self.JMatrix_overide=np.array([])
    self.ML = None
    self.NumNei = None
    self.M=0
    self.zNum=[]
    AniCount=0
    self.Js=None
    self.masterflag='False'
    self.a=None

    pi=sym.pi

    self.zperiodic=zperiodic

    f = open(fileName)
    for l in f.readlines():
        Temp = l.split()
        if len(Temp)!=0:
    #### Master flag 
            if Temp[0].lower()=='masterflag':
                self.masterflag = Temp[1]



    #### Read cell block
            if Temp[0]=='#' and Temp[1].lower()=='end' and Temp[2].lower()=='input_cell_card':
                cellflag=False
            if cellflag:
                cell=np.append(cell,Temp)
            if (Temp[0]=='#' and Temp[1].lower()=='input_cell_card'):
                cellflag=True
    ##### end read cell block

    #### Read atoms block
            if Temp[0]=='#' and Temp[1].lower()=='end' and Temp[2].lower()=='input_atoms_card_frac' :
                atomsflag=False
            if atomsflag:

                atomRead=np.array([Temp[0],float(Temp[1]),float(Temp[2]),float(Temp[3])])
                
                MagMoms=np.array([float(Temp[4]),eval(Temp[5]),eval(Temp[6])])

                self.atoms = np.append(self.atoms,atomRead)
                self.MagMoms = np.append(self.MagMoms,MagMoms)

                self.zNum.append(Temp[3])
            if (Temp[0]=='#' and Temp[1].lower()=='input_atoms_card_frac'):
                atomsflag=True
    ##### end atoms block

    #### Read  Number of neighbours
            if Temp[0].lower()=='neighbour' and neib==None:
                self.NumNei=int(Temp[1])
    

            if neib!=None:
                self.NumNei=neib
    #### end Number of neighbours block

    #### Read Ks
            if Temp[0]=='#' and Temp[1].lower()=='end' and Temp[2].lower()=='input_ks' :
                ksflag=False
                
            if ksflag:
                #print(self.atoms)
                self.Anisotropies=np.tile(np.array([float(Temp[0]),eval(Temp[1]),eval(Temp[2])]),(int(len(self.atoms)/4),1))

            if (Temp[0]=='#' and Temp[1].lower()=='input_ks'):
                ksflag=True

    #### end Ks

    #### Read anisotropies block
            if len(Temp)!=0:
                if Temp[0]=='#' and Temp[1].lower()=='end' and Temp[2].lower()=='input_local_anisotropies' :
                    Aniflag=False
                if Aniflag:
                    print(Temp)
                    AniCount=int(Temp[0])
                    self.Anisotropies[AniCount,0]=float(Temp[1])
                    self.Anisotropies[AniCount,1]=eval(Temp[2])
                    self.Anisotropies[AniCount,2]=eval(Temp[3])
                if (Temp[0]=='#' and Temp[1].lower()=='input_local_anisotropies'):

                    Aniflag=True
    ##### end anisotropies block    


    #### Read Js
            if Temp[0]=='#' and Temp[1].lower()=='end' and Temp[2].lower()=='input_js' :
                jsflag=False
                
            if jsflag:
                if count==0:
                    L=int(np.sqrt(len(Temp)))
                    if L==1:
                        self.Js=np.array([np.reshape(Temp,(L,L))]).astype(float)
                        count+=1
                    else:
                        self.Js=np.reshape(Temp,(L,L,1)).astype(float)
                        count+=1
                else:
                    L=int(np.sqrt(len(Temp)))
                    self.Js=np.dstack((self.Js,np.reshape(Temp,(L,L,1)).astype(float)))

            if (Temp[0]=='#' and Temp[1].lower()=='input_js'):
                jsflag=True
                count=0

    #### end Js

    #### Read Jij block
            if Temp[0]=='#' and Temp[1].lower()=='end' and Temp[2].lower()=='input_local_jij' :
                Jijflag=False                
            if Jijflag:
                self.JMatrix_overide.append([float(x) for x in Temp])
            if (Temp[0]=='#' and Temp[1].lower()=='input_local_jij'):
                Jijflag = True
                self.JMatrix_overide = []

    ##### end Jij size block       
            
    #### Read kpath block
            if Temp[0]=='#' and Temp[1].lower()=='end' and Temp[2].lower()=='kpath_card' :
                kpathflag=False
            if kpathflag:
                    convTemp = np.array(Temp).astype(float)
                    self.qpts = np.append(self.qpts,convTemp)
            if (Temp[0]=='#' and Temp[1].lower()=='kpath_card'):
                kpathflag=True
    ##### end kpath block



    ##### Reshape unitcell and atoms
    cell=cell.reshape(3,3).astype(float)
    self.atoms=self.atoms.reshape(int(len(self.atoms)/4),4)
    self.MagMoms=self.MagMoms.reshape(int(len(self.MagMoms)/3),3).astype(float)
    self.qpts=self.qpts.reshape(int(len(self.qpts)/3),3).astype(float)

    self.a=cell[0,0]

    #print(self.zNum)

    self.zNum=[float(i) for i in self.zNum]
    self.N_dict = {i:list(self.zNum).count(i) for i in self.zNum}
    self.zNum=np.array([*sorted(set(self.zNum))])
    self.ML=len(self.zNum)

    Atoms_Positions=[]

    for num,atom in enumerate(self.atoms):
        Atoms_Positions.append(Atom(atom[0],np.dot(atom[1:4].astype(float),cell) ,magmom=self.MagMoms[num]))

    self.aseatoms=Atoms(Atoms_Positions,cell=cell)
    self.aseatoms.set_pbc((True, True, zperiodic))

    def get_distance(self,bondpair):
        Pos_Atom_1 = self.aseatoms.positions[bondpair[0]]
        Pos_Atom_2 = self.aseatoms.positions[bondpair[1]]+np.dot(bondpair[2],self.aseatoms.cell)
        distanceVector = Pos_Atom_2 - Pos_Atom_1
        return np.linalg.norm(distanceVector)

  #@profile
  def get_Neigb(self,step_size):
    #print(step_size)
    Lists_of_Neigbours=[]
    List_of_bondpairs=[]

    radius=0

    ## Get all neighbours with increasing radius until read get far enought to fufill
    ## the required furthest neighbour.

    while len(List_of_bondpairs)<self.NumNei:
        radius+=step_size
        cutoffs = radius * (covalent_radii[self.aseatoms.numbers]/covalent_radii[self.aseatoms.numbers])

        
        nl = NeighborList(cutoffs=cutoffs, self_interaction=False,bothways=True)
        nl.update(self.aseatoms)
        bondpairs = []
        for a in range(len(self.aseatoms)):
            indices, offsets = nl.get_neighbors(a)

            bondpairs.extend([(a, a2, offset)
                            for a2, offset in zip(indices, offsets)])
        if len(bondpairs)!=0: 
            if len(bondpairs) not in List_of_bondpairs:
                List_of_bondpairs.append(len(bondpairs))
                Lists_of_Neigbours.append(bondpairs)

    ## Remove duplicates            
                
    for i in range(len(List_of_bondpairs)-1,0,-1):
        t_delete=[]
        for num,bond1 in enumerate(Lists_of_Neigbours[i]): 
            for bond2 in Lists_of_Neigbours[i-1]:
                if bond1[0]==bond2[0] and bond1[1]==bond2[1] and (bond1[2]==bond2[2]).all():
                    t_delete.append(num)
                    
        Lists_of_Neigbours[i]=list(np.delete(np.array(Lists_of_Neigbours[i],dtype=object), t_delete,axis=0))     

    return Lists_of_Neigbours



  #@profile
  def Hamiltonian(self,step_size=0.01,hermitian=True):

    Lists_of_Neigbours=self.get_Neigb(step_size)

    LayersDictionary={}
    for i in range(len(self.zNum)):
        LayersDictionary[i]=np.array([])

    for num,i in enumerate(self.atoms):
        Temp=np.where((self.zNum==float(i[-1])))[0][0]
        LayersDictionary[Temp]=np.append(LayersDictionary[Temp],self.aseatoms.get_initial_magnetic_moments()[num])

    for i in range(len(self.zNum)):
        Term1=LayersDictionary[i]
        LayersDictionary[i]=np.reshape(Term1,(len(Term1)//3,3))
        LayersDictionary[i]=np.unique(LayersDictionary[i], axis=0)

    def get_distance_vector(Input,bondpair):
        Pos_Atom_1 = Input.aseatoms.positions[bondpair[0]]
        Pos_Atom_2 = Input.aseatoms.positions[bondpair[1]]+np.dot(bondpair[2],Input.aseatoms.cell)
        distanceVector = Pos_Atom_2 - Pos_Atom_1
        #distanceVector[-1] = 0
        return distanceVector

    def get_distance(Input,bondpair):
            Pos_Atom_1 = Input.aseatoms.positions[bondpair[0]]
            Pos_Atom_2 = Input.aseatoms.positions[bondpair[1]]+np.dot(bondpair[2],self.aseatoms.cell)
            distanceVector = Pos_Atom_2 - Pos_Atom_1
            return np.linalg.norm(distanceVector)

    a=self.aseatoms.cell[0,0]
    b=self.aseatoms.cell[1,1]
    c=self.aseatoms.cell[2,2]


    self.orientationEach = np.array([LayersDictionary[i][j] for i in LayersDictionary.keys() for j,value in enumerate(LayersDictionary[i])])

    self.Total_types=np.unique(self.aseatoms.get_initial_magnetic_moments(),axis=0)


    self.M_list=[len(LayersDictionary[key]) for key in LayersDictionary.keys()]

    M_types=[]

    for key in LayersDictionary.keys():
        M_types.append([np.where((self.Total_types==item).all(axis=1))[0][0] for item in LayersDictionary[key]])


    N_list=[self.N_dict[key] for key in self.N_dict.keys()]

    list_Distances=[]
    for num,j in enumerate(Lists_of_Neigbours): 
        for bondpair in j: 
            list_Distances.append(round(get_distance(self,bondpair),5)) 
   

    list_Distances=np.array(list(set(list_Distances)))

    M=len(self.Total_types)

    self.M_list=list(np.array(self.M_list))

    ML=len(self.M_list)

    H_main=np.zeros([sum(self.M_list),sum(self.M_list),len(self.qpts)],dtype=complex)
    H_main1=np.zeros([sum(self.M_list),sum(self.M_list),len(self.qpts)],dtype=complex)
    H_off1=np.zeros([sum(self.M_list),sum(self.M_list),len(self.qpts)],dtype=complex)
    H_off2=np.zeros([sum(self.M_list),sum(self.M_list),len(self.qpts)],dtype=complex)
    H_final=np.zeros([2*sum(self.M_list),2*sum(self.M_list),len(self.qpts)],dtype=complex)


    q = np.copy(self.qpts)
    q[:,2]=0

    nw_length=[len(term) for term in M_types]

    S=self.aseatoms.get_initial_magnetic_moments()[:,0]

    for num,j in enumerate(Lists_of_Neigbours):
        #print(j)
        for i in j:
            
            DistanceInd=np.where(list_Distances==round(get_distance(self,i),5))[0][0]
            
            ## Layer of atoms 
            Layer1=np.where((self.zNum==float(self.atoms[i[0],-1])))[0][0]
            Layer2=np.where((self.zNum==float(self.atoms[i[1],-1])))[0][0]

  
            r=np.where((self.Total_types==self.aseatoms.get_initial_magnetic_moments()[i[0]]).all(axis=1))[0][0]
            s=np.where((self.Total_types==self.aseatoms.get_initial_magnetic_moments()[i[1]]).all(axis=1))[0][0]

            #print(i,r,s,round(get_distance(self,i),5))
            
            if self.masterflag.lower()=='true':
                JMatrixValue=self.Js[0,0,DistanceInd]
            else:
                JMatrixValue=self.Js[r,s,DistanceInd]

            sumnw_length_i=sum(nw_length[:Layer1])
            sumnw_length_j=sum(nw_length[:Layer2])

            Mi=M_types[Layer1]
            Mj=M_types[Layer2]

            rn=Mi.index(r)
            sn=Mj.index(s)

            Sr=(S[(sumnw_length_i)+rn])
            Ss=(S[(sumnw_length_j)+sn])

            ######################
            z=1*(self.M_list[Layer1]/N_list[Layer1])
            Gamma=(np.exp(-1.0j*np.dot(get_distance_vector(self,i),np.transpose(2*np.pi*(q/np.array([a,a,a]))))))*(self.M_list[Layer1]/N_list[Layer1])
            FzzM=Fzz(self,i[0],i[1])
            G1M=G1(self,i[0],i[1])
            G2M=G2(self,i[0],i[1])



            H_main[(sumnw_length_i)+rn,(sumnw_length_i)+rn,:]+=z*JMatrixValue*(Sr)*FzzM

            H_main[(sumnw_length_j)+sn,(sumnw_length_j)+sn,:]+=z*JMatrixValue*(Ss)*FzzM                       


            H_main[(sumnw_length_i)+rn,(sumnw_length_j)+sn,:]-=JMatrixValue*    \
            + (((np.sqrt((Sr*Ss))/2)*(Gamma*G1M))  \
            +  ((np.sqrt((Sr*Ss))/2)*(np.conj(Gamma)*np.conj(G1M))))

            ################################
            H_main1[(sumnw_length_i)+rn,(sumnw_length_i)+rn,:]+=z*JMatrixValue*(Sr)*FzzM

            H_main1[(sumnw_length_j)+sn,(sumnw_length_j)+sn,:]+=z*JMatrixValue*(Ss)*FzzM                           


            H_main1[(sumnw_length_i)+rn,(sumnw_length_j)+sn,:]-=JMatrixValue*    \
            + (((np.sqrt((Sr*Ss))/2)*(np.conj(Gamma)*np.conj(G1M)))  \
            +  ((np.sqrt((Sr*Ss))/2)*(Gamma*G1M)))

            
            #################################
            H_off1[(sumnw_length_i)+rn,(sumnw_length_j)+sn,:]-=JMatrixValue*    \
            + (((np.sqrt((Sr*Ss))/2)*(Gamma*np.conj(G2M)))  \
            +  ((np.sqrt((Sr*Ss))/2)*(np.conj(Gamma)*G2M))) 

            
            ################################
            H_off2[(sumnw_length_i)+rn,(sumnw_length_j)+sn,:]-=JMatrixValue*    \
            + (((np.sqrt((Sr*Ss))/2)*(np.conj(Gamma)*G2M))  \
            +  ((np.sqrt((Sr*Ss))/2)*(Gamma*np.conj(G2M)))) 
        
                                  
                
    if hermitian:
        for i in range(len(self.qpts)):
            H_final[:,:,i]=np.block([[H_main[:,:,i],H_off1[:,:,i]],
                                    [H_off2[:,:,i],H_main1[:,:,i]]])
    else:
        for i in range(len(self.qpts)):
            H_final[:,:,i]=np.block([[H_main[:,:,i],-H_off1[:,:,i]],
                                    [H_off2[:,:,i],-(H_main1[:,:,i])]])
            
    H_final=H_final/4
    

    return H_final


  def HamiltonianK(self,step_size=0.01,hermitian=True):

    LayersDictionary={}
    for i in range(len(self.zNum)):
        LayersDictionary[i]=np.array([])

    for num,i in enumerate(self.atoms):
        Temp=np.where((self.zNum==float(i[-1])))[0][0]
        LayersDictionary[Temp]=np.append(LayersDictionary[Temp],self.aseatoms.get_initial_magnetic_moments()[num])

    for i in range(len(self.zNum)):
        Term1=LayersDictionary[i]
        LayersDictionary[i]=np.reshape(Term1,(len(Term1)//3,3))
        LayersDictionary[i]=np.unique(LayersDictionary[i], axis=0)

    self.M_list=[len(LayersDictionary[key]) for key in LayersDictionary.keys()]

    N_list=[self.N_dict[key] for key in self.N_dict.keys()]

    self.Total_types=np.unique(self.aseatoms.get_initial_magnetic_moments(),axis=0)

    M=len(self.Total_types)

    ML=len(self.M_list)

    S=self.aseatoms.get_initial_magnetic_moments()[:,0]

    H_main=np.zeros([M*ML,M*ML,len(self.qpts)],dtype=complex)
    H_off1=np.zeros([M*ML,M*ML,len(self.qpts)],dtype=complex)
    Hk=np.zeros([2*M*ML,2*M*ML,len(self.qpts)],dtype=complex)

    for i in range(ML):
        Mi=self.M_list[i]
        for r in range(Mi):
                    Sr=S[(Mi*i)+r]
                    ######################
                    AxTerm=Ax(self,self.Anisotropies,(Mi*i)+r)
                    AyTerm=Ay(self,self.Anisotropies,(Mi*i)+r)
                    AzTerm=Az(self,self.Anisotropies,(Mi*i)+r)


                    H_main[(Mi*i)+r,(Mi*i)+r,:]-=self.Anisotropies[(Mi*i)+r,0] * ( (Sr*(AxTerm**2)) + (Sr*(AyTerm**2)) - (2*(Sr**2)*(AzTerm**2)) )

                    #################################
                    H_off1[(Mi*i)+r,(Mi*i)+r,:]-=(self.Anisotropies[(Mi*i)+r,0]/2)*((Sr/2)*(AxTerm**2) -(Sr/2)*(AyTerm**2) +1.0j*(Sr/2)*(AxTerm*AyTerm) +1.0j*(Sr/2)*(AyTerm*AxTerm))
                                                               
                                                                        

                
            
    if hermitian:
        for i in range(len(self.qpts)):
            Hk[:,:,i]=np.block([[H_main[:,:,i],H_off1[:,:,i]],
                                [np.conj(H_off1[:,:,i]),H_main[:,:,i]]])
    else:
        for i in range(len(self.qpts)):
            Hk[:,:,i]=np.block([[H_main[:,:,i],-H_off1[:,:,i]],
                                [np.conj(H_off1[:,:,i]),-(H_main[:,:,i])]])    

    return Hk

def diagonalize_function(Hamiltonian):

    #eVals_full=np.array([],dtype=complex)
    #eVecs_fullL=np.array([],dtype=complex)
    #eVecs_fullR=np.array([],dtype=complex)

    n=len(Hamiltonian[:,:,0])//2

    nx=np.shape(Hamiltonian)[0]
    ny=np.shape(Hamiltonian)[1]
    nz=np.shape(Hamiltonian)[2]
    
    eVals_full=np.zeros((nz,nx),dtype=complex)
    eVecs_fullL=np.zeros((nx,ny,nz),dtype=complex)
    eVecs_fullR=np.zeros((nx,ny,nz),dtype=complex)
    
    for i in range(len(Hamiltonian[0,0,:])):

        # if np.allclose(np.transpose(np.conjugate(Hamiltonian[:,:,i])),Hamiltonian[:,:,i],atol=1e-10):
        #         #Chloensky decomposition
        #     L, U = lu(Hamiltonian[:,:,i],permute_l=True)

        #     N=np.block([[np.diag(np.ones(n)),np.zeros([n,n])],
        #                 [np.zeros([n,n]),-np.diag(np.ones(n))]])

        #     C=np.transpose(U).dot(N).dot(L)
        # else:
        #     C=Hamiltonian[:,:,i]

        eVals,eVecsL,eVecsR =  eig(Hamiltonian[:,:,i], left=True, right=True)
        idx = eVals.argsort()[::1] 
        #print(idx)
        
        
        eVals = eVals[idx]
        eVecsL = eVecsL[:,idx]
        eVecsR = eVecsR[:,idx]
    
        eVals_full[i,:]=eVals
        eVecs_fullL[:,:,i]=eVecsL
        eVecs_fullR[:,:,i]=eVecsR      
        
        # if i==0:
        #     eVals_full=eVals
        #     eVecs_fullL=eVecsL
        #     eVecs_fullR=eVecsR
        # else:
        #     eVals_full=np.vstack((eVals_full,eVals))#np.append(eVals_full,eVals)
        #     eVecs_fullL=np.dstack((eVecs_fullL,eVecsL))
        #     eVecs_fullR=np.dstack((eVecs_fullR,eVecsR))

    return eVals_full,eVecs_fullL,eVecs_fullR