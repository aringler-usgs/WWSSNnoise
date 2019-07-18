#!/usr/bin/env python
from obspy.core import UTCDateTime
import matplotlib.pyplot as plt
import numpy as np
f = open('ALQ_Results.csv','r')

times = []
ws = []
ls = []
wg = []
lg = []
sig2 = []
resi = []

for line in f:
    line = line.split(', ')
    time = UTCDateTime('19' + line[3] + '-' + line[1] + '-' + line[2] + 'T00:00:00')
    times.append(time.year + float(time.julday)/365.25)
    ws.append(float(line[5]))
    ls.append(float(line[6]))
    wg.append(float(line[7]))
    lg.append(float(line[8]))
    sig2.append(float(line[9]))
    resi.append(float(line[11]))
    
    
    
f.close()

import matplotlib as mpl
# Importing and applying font
mpl.rc('font', family = 'serif')
mpl.rc('font', serif = 'Times') 
#mpl.rc('text', usetex = True)
mpl.rc('font',size=18)

fig = plt.figure(1, figsize=(12,18))

plt.subplot(6,1,1)
plt.title('Step Calibration Parameters for ALQ LPZ')
ws = np.asarray(ws)
ws[(ws >= 1.)] = 1.

plt.plot(times, ws,'.', markersize=12)
plt.plot([0, 2000.], [(2.*np.pi/30.), (2.*np.pi/30.)], linewidth=2.)
plt.ylabel('${\omega}_s$ (rad.)')
plt.xlim((min(times), max(times)))

plt.text(1960, 1, '(a)')

ls = np.asarray(ls)
ls[(ls >= 5.)] = 5.
plt.subplot(6,1,2)
plt.plot(times, ls,'.', markersize=12)
plt.plot([0, 2000.], [1.96, 1.96], linewidth=2.)
plt.ylabel('${\lambda}_s$')
plt.xlim((min(times), max(times)))

plt.text(1960, 5, '(b)')
wg = np.asarray(wg)
wg[(wg >= .5)] = .5
plt.subplot(6,1,3)
plt.plot(times, wg,'.', markersize=12)
plt.plot([0, 2000.], [(2.*np.pi/98.1), (2.*np.pi/98.1)], linewidth=2.)
plt.ylabel('${\omega}_g$ (rad.)')
plt.xlim((min(times), max(times)))

plt.text(1960, 0.5, '(c)')
lg = np.asarray(lg)
lg[(lg >= 5.)] = 5.
plt.subplot(6,1,4)
plt.plot(times, lg,'.', markersize=12)
plt.plot([0, 2000.], [0.95, 0.95], linewidth=2.)
plt.ylabel('${\lambda}_g$')
plt.xlim((min(times), max(times)))

plt.text(1960, 5, '(d)')
sig2 = np.asarray(sig2)
sig2[(sig2 <= -0.5)] = -0.5 
plt.subplot(6,1,5)
plt.plot(times, sig2,'.', markersize=12)
plt.plot([0, 2000.], [0.0392, 0.0392], linewidth=2.)
plt.ylabel('${\sigma}^2$')
plt.xlim((min(times), max(times)))
plt.xlabel('Time (year)')
plt.text(1960, 0.6, '(e)')

resi = np.array(resi)
resi[(resi>= 20.)] = 20.
plt.subplot(6,1,6)
plt.plot(times, resi,'.', markersize=12)
plt.ylabel('Error (%)')
plt.xlim((min(times), max(times)))
plt.xlabel('Time (year)')
plt.text(1960, 20., '(f)')


print(np.mean(resi))
print(np.median(resi))
print(np.percentile(resi,79))




plt.savefig('ALQ.jpg', format='JPEG')
plt.show()
