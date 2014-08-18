blackScholes
============
to plot data in the outputs file, run:
gnuplot
then at the prompt type:
splot 'filename.txt' using 1:2:3 with points palette pointsize 1 pointtype 1

make -f sMake makes the sequential solver
make -f tMake makes the threaded solver
make makes the cuda solver with parallel matrix ops 

Dependencies for the cuda solver are the cuda library, and cula (the free version)
Dependencies for the other stuff is the armadillo c++ library

