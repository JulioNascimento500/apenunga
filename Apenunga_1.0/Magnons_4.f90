subroutine magnons_function(N,Egrid,Qgrid,qpts,EigVecL,EigVecR,EigVals,plotValues,minY,maxY,a,T)

  use constants                      !Minimal useage where practical 
  !use omp_lib

  implicit none
  !### Reals ###
  real(kind=dp)       ::  stepsize, omegaN,total, trial, average_Jij,minY,maxY

  real(kind=dp), intent(in)     :: a
  real(kind=dp), intent(in)     :: T

  real                :: start_time, end_time

  !### Integer ###
  integer             :: b,info,i,j,k,stat,next
  integer,intent(in)  :: N
  integer,intent(in)  :: Qgrid,Egrid

  !### Arrays ##

  complex(kind=8), dimension(N,N,Qgrid), intent(in)     :: EigVecL(N,N,Qgrid) 
  complex(kind=8), dimension(N,N,Qgrid), intent(in)     :: EigVecR(N,N,Qgrid) 
  complex(kind=8), dimension(N,Qgrid), intent(in)     :: EigVals(N,Qgrid) 


  real(kind=dp), dimension(Qgrid,3), intent(in)     :: qpts(Qgrid,3)

  complex(kind=8), dimension(Egrid,Qgrid),intent(inout) :: plotValues(Egrid,Qgrid) 
  
  real(kind=dp), dimension(:),allocatable        :: WI,WR ,XL,XR, qx, omegaX
  complex(kind=8), dimension(:),allocatable        :: WORK

  complex(kind=8), dimension(:),allocatable        :: W
  
  
  integer, dimension(:),allocatable              :: RWORK
  complex(kind=8), dimension(:,:), allocatable  :: exp_sum_j 

  complex(kind=8), dimension(:,:), allocatable  :: P,Q,L
  
  complex(kind=8), dimension(:,:),allocatable        :: VL
  complex(kind=8), dimension(:,:),allocatable        :: VR


  !real(kind=dp), dimension(1:N),intent(in)        :: Jij(N)
  !real(kind=dp), dimension(1:N),intent(in)        :: K_ani(N)
  
  
  
  
  !#### f2py intent block ####
  !f2py intent(in)       :: N
  !f2py intent(in)       :: Egrid
  !f2py intent(in)       :: Qgrid
  !f2py intent(in)       :: a
  !f2py intent(in)       :: T


  !f2py intent(in)       :: EigVecL
  !f2py depend(N,N,Qgrid)  :: EigVecL

  !f2py intent(in)       :: EigVecR
  !f2py depend(N,N,Qgrid)  :: EigVecR

  !f2py intent(in)       :: EigVals
  !f2py depend(N,Qgrid)  :: EigVals

  !f2py intent(inout) :: plotValues
  !f2py depend(Egrid,Qgrid)   :: plotValues

  !f2py intent(in) :: qpts
  !f2py depend(Qgrid)   :: qpts

  b=0.0_dp
  average_Jij=0.04_dp!sum(Jij)/real(N,kind=dp)
  
  !
  !###############################

  !#### Allocate arrays ####
  !
  allocate(P(N,N), stat=stat)
  if (stat/=0) stop 'Error allocating P array'
  !
  allocate(Q(N,N), stat=stat)
  if (stat/=0) stop 'Error allocating Q array'
  !
  allocate(L(N,N), stat=stat)
  if (stat/=0) stop 'Error allocating L array'
  !
  allocate(W(N), stat=stat)
  if (stat/=0) stop 'Error allocating W array'
  !
  allocate(WI(N), stat=stat)
  if (stat/=0) stop 'Error allocating W array'
  !
  allocate(VL(N,N), stat=stat)
  if (stat/=0) stop 'Error allocating W array'
  !
  allocate(VR(N,N), stat=stat)
  if (stat/=0) stop 'Error allocating W array'
  !
  allocate(WORK(10*N), stat=stat)
  if (stat/=0) stop 'Error allocating W array'
  !
  !allocate(WORK(1+ (6*N) + 2*(N)**2), stat=stat)
  !if (stat/=0) stop 'Error allocating W array'
  !
  allocate(RWORK((2*N)), stat=stat)
  if (stat/=0) stop 'Error allocating W array'
  !
  allocate(exp_sum_j(Qgrid,N), stat=stat)
  if (stat/=0) stop 'Error allocating W array'
  !
  allocate(omegaX(Egrid), stat=stat)
  if (stat/=0) stop 'Error allocating W array'
  !
  !
  !#########################

  !#### Allocate arrays ####
  !
  ! allocate(P(N,N), stat=stat)
  ! if (stat/=0) stop 'Error allocating P array'
  ! !
  ! allocate(Q(N,N), stat=stat)
  ! if (stat/=0) stop 'Error allocating Q array'
  ! !
  ! allocate(L(2*N,2*N), stat=stat)
  ! if (stat/=0) stop 'Error allocating L array'
  ! !
  ! allocate(W(2*N), stat=stat)
  ! if (stat/=0) stop 'Error allocating W array'
  ! !
  ! allocate(EigVec(2*N,2*N,Qgrid), stat=stat)
  ! if (stat/=0) stop 'Error allocating L array'
  ! !
  ! allocate(EigVals(2*N,Qgrid), stat=stat)
  ! if (stat/=0) stop 'Error allocating W array'
  ! !
  ! allocate(WORK(1+ (6*2*N) + 2*(2*N)**2), stat=stat)
  ! if (stat/=0) stop 'Error allocating W array'
  ! !
  ! allocate(IWORK(3+ (5*2*N)), stat=stat)
  ! if (stat/=0) stop 'Error allocating W array'
  ! !
  ! allocate(exp_sum_j(Qgrid,N), stat=stat)
  ! if (stat/=0) stop 'Error allocating W array'
  ! !
  ! allocate(omegaX(Egrid), stat=stat)
  ! if (stat/=0) stop 'Error allocating W array'
  !
  !
  !#########################


  do i=1,Egrid
    stepsize=(maxY-minY)/real(Egrid,kind=dp)
    omegaX(i)= minY + (stepsize*(i-1))
  end do


  !print *, Hamiltonian(:,:,2) 


  ! print *, (EigVec(:,:,1)==EigVec(:,:,2))

  do j=1,N
    do i=1,Qgrid
      exp_sum_j(i,j)=exp(-cmplx(0.0_dp,dot_product(qpts(i,:),(/0.0_dp*a,0.0_dp*a,0.25_dp*a/))))
    end do
  end do

  do i=1,Qgrid
    do j=1,Egrid
      total=0.0_dp
      do k=1,N
        omegaN=EigVals(k,i) !2*k - 1 

        XL = EigVecL(1:int(N/2),k,i) + EigVecL(int(N/2)+1:N,k,i) 
        XR = EigVecR(1:int(N/2),k,i) + EigVecR(int(N/2)+1:N,k,i) 
        ! total = total + &
        ! ((abs(dot_product(exp_sum_j(i,:),X))**2.0_dp) * &
        ! (boson_dist(omegaX(j),omegaN,T) + 1.0_dp) &
        ! * broadening(omegaX(j),omegaN,average_Jij))

        total = total + &
        (((conjg(dot_product(exp_sum_j(i,:),XL))*dot_product(exp_sum_j(i,:),XR))) * &
        (boson_dist(omegaX(j),omegaN,T) + 1.0_dp) &
        * broadening(omegaX(j),omegaN,average_Jij))
        !print *, omegaX(j)

      plotValues(j,i)=total
      end do
    end do
  end do

  print *, plotValues(1,1)

  open(unit=10, file='outfile.magnon', status='replace', iostat=stat)
  if (stat /= 0) stop 'Error opening outfile.magnon array'
  write (10,*) Egrid, Qgrid
  do i=1,Qgrid
    write (10,*) plotValues(:,i)/(2.0_dp*real(N,kind=dp))
    !print *, '################################'
  end do

  CLOSE ( unit=10, STATUS='KEEP') 
contains

  real(kind=dp) function sum_j(X,q,N)
  !
  !################
  ! This function calculates 
  ! the summation over the eigenvectors multiplied by a phase
  !
  ! In : X vector from the Confined magnons paper
  ! In : wave vector q
  !
  ! Out: Final Summation
  !
  !################
  implicit none
  real(kind=dp), intent(in)   :: q
  real(kind=dp), dimension(N) :: X
  complex(kind=dp)            :: sum_j_comp
  integer                     :: N,j
  sum_j_comp=(0.0_dp,0.0_dp)

      do j=1,N
        sum_j_comp = sum_j_comp + exp(complex(0.0_dp,-q*real(j,kind=dp)))*X(j)
      end do
      

  sum_j=abs(sum_j_comp)

  end function sum_j
  
  real(kind=dp) function boson_dist(omega,omegaN,T)
  !
  !################
  ! This function calculates 
  ! The boson distribution
  !
  ! In : Given omega (omega)
  ! In : Particular Eigenvalue of the system (omegaN)
  ! In : Temperature (T)
  ! In : Size of array X (N)
  !
  ! Out: Value for the distribution in omega
  !
  !################    
  implicit none
  real(kind=dp), intent(in)   :: omega,omegaN,T
  real(kind=dp)               :: Kb
 
    kb=8.617333e-5_dp


    boson_dist = 1/(exp(((omega-omegaN))/(Kb*T)) - 1.0_dp)
  
  end function boson_dist

  real(kind=dp) function broadening(omega,omegaN,Jij)
  !
  !################
  ! This function calculates 
  ! The boson distribution
  !
  ! In : Given omega (omega)
  ! In : Particular Eigenvalue of the system (omegaN)
  ! In : Temperature (T)
  ! 
  !
  ! Out: Value for the broadening in omega
  !
  !################    
  use constants
  
  implicit none
  
  real(kind=dp)         :: delta,Jij,omega,omegaN

    delta=0.2_dp*abs(Jij)*0.5_dp
    broadening = (1.0_dp/sqrt(2.0_dp*pi*(delta**2.0_dp)))*exp(-(((omega-omegaN)**2.0_dp)/(2.0_dp*(delta**2.0_dp))))

  end function broadening

end
