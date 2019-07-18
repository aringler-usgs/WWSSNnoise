#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import periodogram, detrend
from obspy.core import read
from obspy.signal.filter import bandpass

import matplotlib as mpl
# Importing and applying font
mpl.rc('font', family = 'serif')
mpl.rc('font', serif = 'Times') 
mpl.rc('text', usetex = True)
mpl.rc('font',size=18)

t =[]
m = []
#with open("micro_01_01_71_1645.csv","r") as f:
    #next(f)
    #for line in f:
        #t.append(float(line.split(',')[0]))
        #m.append(float(line.split(',')[1]))
#t = np.asarray(t)
#m = np.asarray(m)*5.
#m = detrend(m, type='linear')
#t -= min(t)
#f = interp1d(t, m)
#tgood = np.linspace(0., max(t), 1000.)
#mgood = f(tgood)
#f, p = periodogram(mgood, fs=1., nfft=2**8, scaling='spectrum')


#plt.figure(1)
#plt.semilogx(f, 10.*np.log10(p))
#plt.show()

#import matplotlib as mpl
## Importing and applying font
#mpl.rc('font', family = 'serif')
#mpl.rc('font', serif = 'Times') 
##mpl.rc('text', usetex = True)
#mpl.rc('font',size=18)


## in A
#ic = 0.04

## in N/A
#G = 0.1036

#Kc = 0.137

## Peak of pulse in mm
#p = 80.

#mag = p*Kc/(G*ic)

#print('Here is the mag: ' + str(mag))
#m /= mag
#mgood /= mag
#m /= 100.
#mgood /= 100.
mag = 2339.
st = read('19640801_1717_19640801_1816.SAC')
t = np.arange(0., st[0].stats.npts)/50.
m = st[0].data
fig = plt.figure(2, figsize=(12,12))
m2 = bandpass(m, 0.1, 1., 2)
#plt.title('ALQ LPZ 08 01 1964 17:77 Mag. x' + str(int(round(mag,0))))
plt.subplot(2,1,1)
plt.title('ALQ LPZ 08 01 1964 17:17 Mag. x' + str(int(round(mag,0))))
plt.plot(t,m, label='Raw Digitized')
plt.text(-350, 17, '(a)')
plt.ylabel('Amplitude (pixels)')
plt.xlabel('Time (s)')
plt.legend()
plt.xlim(min(t), max(t))
plt.subplot(2,1,2)
plt.plot(t,m2, label='0.1 s to 1 s Bandpass')
plt.ylim((-0.5,0.5))
plt.text(-350, 0.45, '(b)')
plt.legend()
#plt.plot(tgood,mgood, label='Interpolated')
plt.xlim(min(t), max(t))
plt.ylabel('Amplitude (pixels)')
plt.xlabel('Time (s)')
#plt.legend()
plt.savefig('noise.jpg', format='JPEG')
plt.show()
