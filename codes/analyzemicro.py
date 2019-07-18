#!/usr/bin/env python
import glob
import sys
import scipy.signal
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from obspy.signal.invsim import simulate_seismometer
from obspy.signal.filter import bandpass
from obspy.core import read
from scipy.signal import periodogram, detrend, welch
from obspy.signal.spectral_estimation import get_nlnm, get_nhnm

debug = True

import matplotlib as mpl
# Importing and applying font
mpl.rc('font', family = 'serif')
mpl.rc('font', serif = 'Times') 
mpl.rc('text', usetex = True)
mpl.rc('font',size=18)

def get_response(mon, day, year, debug = False):
    if debug:
        print(mon)
        print(day)
        print(year)
    f = open('ALQR_Results.csv','r')
    for line in f:
        line = line.replace(' ','')
        if debug:
            print(line)
        line = line.split(',')
        
        if (line[1] == mon) and (line[2] == day) and (line[3] == year):
            if debug:
                print('We have a match')
            
            
            ws = float(line[5])
            ls = float(line[6])
            wg = float(line[7])
            lg = float(line[8])
            sig2 = float(line[9])
            
            den = [1., 2*ls*ws+2*lg*wg, ws**2 + wg**2 +4.*ls*ws*lg*wg*(1-sig2), 2*ls*ws*wg**2 + 2*lg*wg*ws**2, (ws**2)*(wg**2)]
            f.close()
            break
            
    f = open('table1.csv','r')
    for line in f:
        line = line.replace(' ','')
        line = line.split(',')
        if (line[1] == mon) and (line[2] == day) and (line[3] == year):    
            gain = float(line[5])
            f.close()
            break
    num = [gain, 0.]
    z, p, k = scipy.signal.tf2zpk(num, den)
    paz = {}
    paz['poles'] = p
    paz['zeros'] = z
    paz['sensitivity'] = k
    paz['gain']=1.
    return paz

wwLP = [-100., -130., -150., -140.]


wwLPP =[5., 15., 150., 1500. ]
wwLPP = np.asarray(wwLPP)
wwLP = np.asarray(wwLP)
wwLP = 10**(wwLP/10.)*wwLPP
wwLP = 10.*np.log10(wwLP)
print(wwLP)

wwSP = [-120., -110.]
wwSP = np.asarray(wwSP)

wwSPP = [.3, 25.,]
wwSPP = np.asarray(wwSPP)
wwSP = 10**(wwSP/10.)*wwSPP
wwSP = 10.*np.log10(wwSP)


micros = glob.glob('*.SAC')

first = True
for curfile in micros:
    
    label = curfile.split('_')
    label = label[0]
    year = label[2:4]
    mon = label[4:6]
    day = label[6:8]
    print(year + ' ' + mon + ' ' + day)
    #mon, day, year = label[2], label[3], label[4]
    try:
        paz = get_response(mon, day, year)
        if debug:
            print(paz)
    except:
        print('Can not find response')
        continue
    t, m = [], []
    #with open(curfile,"r") as f:
    #    next(f)
    #    for line in f:
    #        t.append(float(line.split(',')[0]))
    #        m.append(float(line.split(',')[1]))
    st = read(curfile)
    print(st)
    st.decimate(2)
    #st.decimate(5)
    t = np.arange(0., st[0].stats.npts)/50.
    print(len(t))
    
    #st.simulate(paz_remove=paz)
    #
    m = st[0].data/(85./.3)
    print(len(m))
    t = np.asarray(t)
    m = np.asarray(m)
    
    #m = scipy.signal.detrend(m, type='linear')
    t -= min(t)
    #tgood = np.linspace(0., max(t), max(t)*10)
    #f = interp1d(t, m)
    mgood = m
    f, p = welch(mgood, fs=st[0].stats.sampling_rate, nfft=2**16, noverlap=2**8, nperseg=2**16)
    num, den = scipy.signal.zpk2tf(paz['zeros'], paz['poles'],paz['sensitivity'])
    
    w, h = scipy.signal.freqs(num, den, worN= 2.*np.pi*f)
    h *= (2.*np.pi*f)**0

    p = 10.*np.log10(p/np.abs(h)**2)
    p = p[1:]
    f = f[1:]
    if np.isnan(np.min(p)):
        continue
    if p[np.argmin(f-1.)] >= -50.:
        continue
    
    
    if not 'powerArray' in vars():
        powerArray = np.asarray(p)
    else:
        powerArray = np.vstack((powerArray,np.asarray(p)))
    #p -= p[-1]
    print(p)
    plt.figure(1, figsize=(12,12))
    if first:
        plt.semilogx(1./f, p, 'C0', alpha=0.3, label='Digitized Trace')
        first= False
    else:
        plt.semilogx(1./f, p, 'C0', alpha=0.3)
    
    #plt.show()
    #mgood = f(tgood)
    
    
    #fig = plt.figure(1)
    #plt.subplot(2,1,1)
    #plt.plot(t,m)
    #plt.plot(t,mgood)
    #plt.subplot(2,1,2)
    #plt.plot(t,mgm)

    ##
    ##tgood = np.linspace(0., max(t), 1000.)
    
    #plt.show()
    #plt.clf()
    #sys.exit()

per, nlnm = get_nlnm()
per, nhnm = get_nhnm()
print(powerArray)
plt.semilogx(1./f, np.amin(powerArray,axis=0),'C1',label='Minimum of Digitized Traces')
plt.semilogx(wwLPP, wwLP,'C2', label='WWSSN LP Minimum', linewidth=3)
plt.semilogx(wwSPP, wwSP, 'C3', label='WWSSN SP Minimum', linewidth=3)
plt.semilogx(per, nhnm, 'C4', label='NHNM/NLNM', linewidth=3)
plt.semilogx(per, nlnm, 'C4', linewidth=3)
plt.legend(loc=8)
plt.xlim((1./20., 1000.))
plt.xlabel('Period (s)')
plt.ylabel('Power (dB rel. 1 $(m/s^2)^2/Hz)$)')
plt.savefig('PSD_plot.jpg', format='JPEG', dpi=400)
plt.show()

