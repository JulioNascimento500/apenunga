gfortran -c constants.f90

f2py -L/usr/local/lib -lblas -L/usr/local/lib -llapack --f90flags='-fopenmp' -lgomp -c Magnons_4.f90 -m magnons
