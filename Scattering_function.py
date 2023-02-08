from numba import njit,jit
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import copy
import sys

import magnons

def cleanup(array):
    error_array=np.finfo(float).eps * np.sqrt(np.sum(array**2))
    print(error_array)
    return(np.where(abs(array)<error_array,0,array ))

cmap = cm.get_cmap('jet')
rgba = cmap(0.0)
cmap.set_bad('white',1.)

def U(T,P):
    return np.stack(np.array([[np.cos(T)*np.cos(P),np.cos(T)*np.sin(P),-np.sin(T)],
                     [-np.sin(P),np.cos(P),np.zeros([len(P)])],
                     [np.sin(T)*np.cos(P),np.sin(T)*np.sin(P),np.cos(P)]]))




def V(magmoms):

    Vplus=U(magmoms[:,1],magmoms[:,2])[0,:] + 1.0j*U(magmoms[:,1],magmoms[:,2])[1,:]
    Vminus=U(magmoms[:,1],magmoms[:,2])[0,:] - 1.0j*U(magmoms[:,1],magmoms[:,2])[1,:]

    return Vplus,Vminus

def Get_PlotValues(Input,Egrid,minY,maxY,eVecsL,eVecsR,eVals,T,delta,alpha,beta):

  N = len(eVals[0,:])//2
  print(N)
  N1 = len(Input.zNum)
  print(N1)
  Qgrid = len(Input.qpts)

  omegaX=np.linspace(minY,maxY,Egrid,dtype=complex)

  S=Input.aseatoms.get_initial_magnetic_moments()[:,0]

  Vplus,Vminus=V(Input.orientationEach)

  print(np.shape(Vplus))

  a=Input.aseatoms.cell[0,0]
  b=Input.aseatoms.cell[1,1]
  c=Input.aseatoms.cell[2,2]

  Total=[]

  for numi,i in enumerate(Input.M_list):
    for j in range(i):
      Total.append(Input.zNum[numi]*c)
    
  Total=np.array(Total)

  A = (np.zeros(len(Total)))
  A = np.vstack((A,np.zeros(len(Total))))
  A = np.vstack((A,Total))
  A = np.transpose(A)


  exp_sum_j = np.exp(-1.0j*np.dot(2*np.pi*Input.qpts/np.array([a,a,a]),np.transpose(A)))

  print('DELTA',delta)
  exp_sum_j = cleanup(exp_sum_j)

  
  plotValues= np.zeros((Egrid,Qgrid),dtype=complex,order='F')

  magnons.magnons_function(Qgrid,Egrid,plotValues,N,2*N ,T,eVecsR,eVals,exp_sum_j,omegaX,delta,Vplus,Vminus,alpha,beta)

  print(plotValues)

  plotValues=(1/(2*len(Input.zNum)))*plotValues
  return plotValues

def print_suit_scattering_function(Input,plotValues,klist,lat,Egrid,EMIN,EMAX,dim='meV',path=True,vmin=1e-2, vmax=1e4,cmap='jet',ymin=0,ymax=500,color='log',):
  plt.rcParams['figure.dpi'] = 600
  if path:
      label_ticks=[]
      points = lat.get_special_points()
      for i in klist:
          if i=='G':
              label_ticks.append(r'$\Gamma$')
          else:
              label_ticks.append(i)

      normal_ticks=[]

      for idx,i in enumerate(Input.qpts):
          for key in klist:
              if np.all(i==points[key]):
                  normal_ticks.append(idx)

      normal_ticks=sorted(set(normal_ticks))

  plotValues1 = abs(plotValues)

  fig, ax = plt.subplots(1,1)

  if color=='log':
    pcm = ax.pcolor(np.linspace(0,len(plotValues1[0,:]),len(plotValues1[0,:]))+1, np.linspace(EMIN,EMAX,Egrid), plotValues1,
                    norm=colors.LogNorm(vmin=vmin, vmax=vmax),#vmin=100, vmax=2000 #vmin=1e2, vmax=1e7
                    cmap=cmap)
  else:
    pcm = ax.pcolor(np.linspace(0,len(plotValues1[0,:]),len(plotValues1[0,:]))+1, np.linspace(EMIN,EMAX,Egrid), plotValues1,
                    vmin=vmin, vmax=vmax, #vmin=1e2, vmax=1e7
                    cmap=cmap)      

  if path:
      for i in normal_ticks:
          ax.axvline(x=i,color='k',linestyle='--',linewidth=0.8)
      ax.set_xticks(normal_ticks)
      ax.set_xticklabels(label_ticks)
  ax.set_xlabel(r'$q$',fontsize=15)
  ax.set_ylabel(r'$Energy ($'+dim+'$)$',fontsize=15)

  ax.set_ylim((EMIN,EMAX))
  fig.colorbar(pcm, ax=ax)
  ax.set_facecolor(rgba)
  plt.tight_layout()
  plt.plot()