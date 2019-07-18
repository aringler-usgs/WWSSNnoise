#!/usr/bin/env python
import glob
import numpy as np
import sys
import matplotlib.pyplot as plt

debug = True
files = glob.glob('ALQ_*.csv')

if debug:
    print(files)

fig = plt.figure(1)
f2 = open('table1.csv', 'w')
mags = []
for curfile in files:
    t =[]
    m =[]
    with open(curfile,"r") as f:
        next(f)
        for line in f:
            t.append(float(line.split(',')[0]))
            m.append(float(line.split(',')[1]))
    t = np.asarray(t)
    m = np.asarray(m)
    m -= m[0]
    val = max(abs(m))*10. - min(abs(m))*10.
    
    if val > 100:
        val /= 2.
    Kc=0.137
    G = 0.137
    ic = 0.04
    mag = val*Kc/(G*ic)
    curfile = curfile.replace('.csv', '')
    curfile = curfile.replace('_', ', ')
    curfile = curfile[:-2] + ':' + curfile[-2:]
    mags.append(int(mag))
    f2.write(curfile + ', ' + str(int(mag)) + '\n')

f2.close()

mags = np.asarray(mags)

print('Here is the mean: ' + str(np.mean(mags)))
print('Here is the std:' + str(np.std(mags)))

