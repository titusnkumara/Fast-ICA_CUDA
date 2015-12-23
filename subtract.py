import numpy
import sys

cpu = numpy.loadtxt(sys.argv[1])
gpu = numpy.loadtxt(sys.argv[2])

sub = abs(cpu)-abs(gpu)

numpy.savetxt("cmpre.txt",sub,fmt='%f')
