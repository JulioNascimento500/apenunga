#!/usr/bin/env python3

from Get_from_File import *
import Hamilt_Func as HF
import magnons
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from ase.lattice import BCC
from pandas import *
from ase.cell import Cell

import argparse
parser = argparse.ArgumentParser()

#-T Temperatures -u USERNAME -p PASSWORD -size 20
parser.add_argument("-T", "--temperatures", help="Temperatures")
parser.add_argument("-f", "--filename", help="FileName")
parser.add_argument("-n", "--numberofneighbours", help="numberofneighbours")
#parser.add_argument("-k", "--klist", help="klist")

args = parser.parse_args()

fileName=args.filename

NeighbNum=int(args.numberofneighbours)

Temperatures = [float(x) for x in args.temperatures.split(',')]

unitcell, M, SubLattices, S, atoms, ML,qpts,listofEELS,Anisotropies = read_file(fileName)

print(listofEELS)

temp = np.dot(atoms[:,:3],unitcell)


for T in Temperatures:

    Jmatrix=np.zeros([M,M,NeighbNum])

    Jmatrix[0,0,:]=13.6056980659*1.432*0.001
    Jmatrix[0,1,:]=-13.6056980659*1.432*0.001
    # Jmatrix[0,2,:]=-13.6056980659*1.432*0.001
    Jmatrix[1,0,:]=-13.6056980659*1.432*0.001
    Jmatrix[1,1,:]=13.6056980659*1.432*0.001
    # Jmatrix[1,2,:]=-13.6056980659*1.432*0.001
    # Jmatrix[2,0,:]=-13.6056980659*1.432*0.001
    # Jmatrix[2,1,:]=-13.6056980659*1.432*0.001
    # Jmatrix[2,2,:]=13.6056980659*1.432*0.001


    # Jmatrix[0,0,:]=13.6056980659*1.432*0.001
    # Jmatrix[0,1,:]=-13.6056980659*1.432*0.001
    # Jmatrix[1,0,:]=-13.6056980659*1.432*0.001
    # Jmatrix[1,1,:]=13.6056980659*1.432*0.001


    # for i in range(M):
    #     for j in range(M):
    #         if i!=j:
    #             Jmatrix[i,j,:]=-13.6056980659*1.432*0.001
    #         else:
    #             Jmatrix[i,j,:]=13.6056980659*1.432*0.001
    #### Required terms for now ##############
    a=unitcell[0,0]
    b=unitcell[1,1]                          #
    c=unitcell[1,1]#7.23997#unitcell[2,2]
    klist = "GNPGH"## "GNPGH"  "LGWLKG"             #
    lat = BCC(a)                              #
    points = lat.get_special_points()         #
    Spacing=400                               #
    ##########################################

    # ##### Required terms for now ##############
    # a=unitcell[0,0]                          #
    # cell = Cell.fromcellpar(unitcell)
    # klist = args.klist#"GNPGH"## "GNPGH"                 #
    # lat = cell.get_bravais_lattice()          #
    # points = lat.get_special_points()         #
    # Spacing=len(qpts)                             #
    # ###########################################

    LayerDict, JLayerDict = get_Neigb(M,Jmatrix,unitcell,atoms,2*np.pi*qpts/(np.array([a,b,c])),NeighbNum,ML)

    H = HF.Hamiltonian(ML,M,NeighbNum,S,JLayerDict,LayerDict,SubLattices,len(qpts))

    H += HF.HamiltonianK(ML,M,S,SubLattices,Anisotropies,len(qpts))
    #set_option('display.max_rows', 500)
    #print(DataFrame(H[:,:,0]))

    Emin=0.001
    Emax=1.0

    Egrid=300
    Qgrid=len(qpts)

    plotValues=np.zeros((Egrid,Qgrid),dtype=complex,order='F')

    w=np.zeros((2*M*ML,Qgrid),dtype=complex)
    v=np.zeros((2*M*ML,2*M*ML,Qgrid),dtype=complex)


    for i in range(Qgrid):
        w[:,i],v[:,:,i] = np.linalg.eig(H[:,:,i])

        w=np.asfortranarray(w)
        v=np.asfortranarray(v)

    magnons.magnons_function(2*M*ML,Egrid,Qgrid,2*np.pi*qpts/(np.array([a,b,c])),v,v,w,plotValues,Emin, Emax,a,T)

    plotValues=abs(plotValues)

    #####################################
    #    Ploting part
    #####################################

    plotValues= np.where(plotValues < 0.001, 0.001, plotValues)

    if len(listofEELS) != 0:
        for i in listofEELS:
            plt.plot(plotValues[i,:])

        plt.show()

    label_ticks=[]

    for i in klist:
        if i=='G':
            label_ticks.append(r'$\Gamma$')
        else:
            label_ticks.append(i)

    normal_ticks=[]

    for idx,i in enumerate(qpts):
        for key in klist:
            if np.all(i==points[key]):
                normal_ticks.append(idx)

    normal_ticks=sorted(set(normal_ticks))

    for key in [points[k] for k in klist]:
        print(key)

    fig, ax = plt.subplots(1,1)

    Q = np.linspace(0,Spacing,Qgrid)

    omega = np.linspace(Emin,Emax,Egrid)

    pcm = ax.pcolor(Q, omega, plotValues,
                    norm=colors.LogNorm(vmin=1e2, vmax=1e7),#vmin=100, vmax=2000 #vmin=1e2, vmax=1e7
                    cmap="jet")
    fig.colorbar(pcm, ax=ax)
    
    ax.tick_params(labelsize=13) 
    ax.set_xticks(normal_ticks)
    ax.set_xticklabels(label_ticks, fontsize=13)

    for i in normal_ticks:
        ax.axvline(x=i,color='k',linestyle='--',linewidth=0.8)

    ax.set_xlabel(r'$\vec{q}$',fontsize=15 )
    ax.set_ylabel(r'Energy(eV)',fontsize=15)

    plt.tight_layout()
    plt.savefig(fileName[:-2]+str(T)+'AniSurfFirst'+str(Anisotropies[str(0)]['modulus'])+'AniSurfLast'+str(Anisotropies[str(ML-1)]['modulus'])+'.png')

