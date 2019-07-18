#!/usr/bin/env python
from scipy.signal import freqs
import numpy as np
import matplotlib.pyplot as plt
from obspy.core import UTCDateTime

f = open('ALQ_Results.csv','r')

import matplotlib as mpl
# Importing and applying font
mpl.rc('font', family = 'serif')
mpl.rc('font', serif = 'Times') 
#mpl.rc('text', usetex = True)
mpl.rc('font',size=18)

times = []
ws = []
ls = []
wg = []
lg = []
sig2 = []
resi = []

fig = plt.figure(1, figsize=(12,12))
idx = 0
for line in f:
    line = line.split(', ')
    time = UTCDateTime('19' + line[3] + '-' + line[1] + '-' + line[2] + 'T00:00:00')
    times.append(time.year + float(time.julday)/365.25)
    ws = float(line[5])
    ls = float(line[6])
    wg = float(line[7])
    lg = float(line[8])
    sig2 = float(line[9])
    #resi.append(float(line[11]))
    gain =1.
    num = [gain, 0.]
    den = [1., 2*ls*ws+2*lg*wg, ws**2 + wg**2 +4.*ls*ws*lg*wg*(1-sig2), 2*ls*ws*wg**2 + 2*lg*wg*ws**2, (ws**2)*(wg**2)]
    w, h = freqs(num, den, worN= np.logspace(-3, 1, 1000))
    h /= np.abs(h[np.argmin(w-.1)])
    plt.subplot(2,1,1)
    plt.title('Response Curves for ALQ LPZ')
    #plt.semilogx(1./w, 10.*np.log10(abs(h)), color='C0')
    plt.xlabel('Period (s)')
    plt.ylabel('Relative Amplitude (dB)')
    plt.xlim((1., 100.))
    plt.subplot(2,1,2)
    #plt.semilogx(1./w, np.unwrap(np.angle(h))*180./np.pi, color='C0')
    
    plt.xlim((1., 100.))
    if idx == 0:
        plt.subplot(2,1,1)
        plt.semilogx(1./w, 10.*np.log10(abs(h)), color='C0', label='Estimated Response') 
        plt.xlabel('Period (s)')
        plt.ylabel('Relative Amplitude (dB)')
        plt.xlim((1., 100.))
        plt.subplot(2,1,2)
        plt.semilogx(1./w, np.unwrap(np.angle(h))*180./np.pi, color='C0', label='Estimated Response')
        plt.xlabel('Period (s)')
        plt.ylabel('Phase (degrees)')
    else:
        plt.subplot(2,1,1)
        plt.semilogx(1./w, 10.*np.log10(abs(h)), color='C0') 
        plt.xlabel('Period (s)')
        plt.ylabel('Relative Amplitude (dB)')
        plt.xlim((1., 100.))
        plt.subplot(2,1,2)
        plt.semilogx(1./w, np.unwrap(np.angle(h))*180./np.pi, color='C0')
        plt.xlabel('Period (s)')
        plt.ylabel('Phase (degrees)')
    idx =1
    
    
ws = 2*np.pi/30.
ls = 1.96
wg = 2.*np.pi/98.1
lg = 0.95
sig2 =  0.0392
num = [gain, 0.]
den = [1., 2*ls*ws+2*lg*wg, ws**2 + wg**2 +4.*ls*ws*lg*wg*(1-sig2), 2*ls*ws*wg**2 + 2*lg*wg*ws**2, (ws**2)*(wg**2)]
w, h = freqs(num, den, worN= np.logspace(-3, 1, 1000))
h /= np.abs(h[np.argmin(w-0.1)])
plt.subplot(2,1,1)
plt.semilogx(1./w, 10.*np.log10(abs(h)), color='C1', linewidth=3, label='Nominal')
plt.text(0.5, 20., '(a)')
plt.legend(loc=4)
plt.ylim((-30, 20.))
plt.subplot(2,1,2)
plt.semilogx(1./w, np.unwrap(np.angle(h))*180./np.pi, color='C1', linewidth=3, label='Nominal')
plt.text(0.5, 100., '(b)')
plt.legend(loc=4)
#plt.show()
f.close()
plt.savefig('Summary_resps.jpg', format='JPEG')



