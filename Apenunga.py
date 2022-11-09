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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-T", "--temperatures", help="Temperatures")
    parser.add_argument("-f", "--filename", help="FileName")
    parser.add_argument("-n", "--numberofneighbours", help="numberofneighbours")
    parser.add_argument("-E", "--EnergyGrid", help="EnergyGrid")


    args = parser.parse_args()

    fileName=args.filename

    Egrid=args.EnergyGrid

    NeighbNum=int(args.numberofneighbours)

    Temperatures = [float(x) for x in args.temperatures.split(',')]

    unitcell, M, SubLattices, S, atoms, ML,qpts,listofEELS,Anisotropies,Jmatrix = read_file(fileName)

    print(listofEELS)

    temp = np.dot(atoms[:,:3],unitcell)


    for T in Temperatures:

        a=unitcell[0,0]
        b=unitcell[1,1]
        c=unitcell[1,1]

        LayerDict, JLayerDict = get_Neigb(M,Jmatrix,unitcell,atoms,2*np.pi*qpts/(np.array([a,b,c])),NeighbNum,ML)

        H = HF.Hamiltonian(ML,M,NeighbNum,S,JLayerDict,LayerDict,SubLattices,len(qpts))

        H += HF.HamiltonianK(ML,M,S,SubLattices,Anisotropies,len(qpts))

        Qgrid=len(qpts)

        plotValues=np.zeros((Egrid,Qgrid),dtype=complex,order='F')

        w=np.zeros((2*M*ML,Qgrid),dtype=complex)
        v=np.zeros((2*M*ML,2*M*ML,Qgrid),dtype=complex)


        for i in range(Qgrid):
            w[:,i],v[:,:,i] = np.linalg.eig(H[:,:,i])

            w=np.asfortranarray(w)
            v=np.asfortranarray(v)

        magnons.magnons_function(2*M*ML,Egrid,Qgrid,2*np.pi*qpts/(np.array([a,b,c])),v,v,w,plotValues,0.0, 0.5,a,T)

        plotValues=abs(plotValues)

        np.save(fileName[:-2]+str(T)+'AniSurfFirst'+str(Anisotropies[str(0)]['modulus'])+'AniSurfLast'+str(Anisotropies[str(ML-1)]['modulus']),plotValues)
