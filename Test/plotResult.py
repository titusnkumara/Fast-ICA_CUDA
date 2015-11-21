"""
=====================================
Plotting data
=====================================

insert outputfile and block

"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
import sys



#load text to array

data = np.loadtxt(sys.argv[1])
print "size is  :", data.shape
X = data.T
print data

try:
    blocking  = str(sys.argv[2])
except:
    blocking = 'noblock'

###############################################################################
# Plot results

plt.figure()

models = [X]
names = ['ICA recovered signals']

for ii, (model, name) in enumerate(zip(models, names), 1):
    plt.title(name)
    for sig in model.T:
        plt.plot(sig)
        plt.subplots_adjust(top=.25)
plt.show(block=(blocking=='block'))
