#!/usr/bin/env python
from obspy.core import read, UTCDateTime
from obspy.signal import PPSD
from obspy.clients.fdsn.client import Client
import glob
import matplotlib.pyplot as plt
from obspy.signal.spectral_estimation import get_nlnm, get_nhnm
import matplotlib as mpl
mpl.rc('font', family='serif')
mpl.rc('font', serif='Times')
mpl.rc('text', usetex=True)
mpl.rc('font', size=18)

debug = True
sta = 'ALQ'

net = "DW"
stime= UTCDateTime('1981-001T00:00:00.0')
etime= UTCDateTime('1982-365T00:00:00.0')

#client=Client()
#inv = client.get_stations(network=net, station=sta, starttime=stime, endtime=etime, channel="*", level='response')

#for chan in ['LHZ', 'LHN', 'LHE', 'SHZ']:
    #files = glob.glob("/msd/DW_ALQ/*/*/_" + chan + '*')
    
    #for curfile in files:
        #st = read(curfile)
        #if 'ppsd' not in vars():
            #ppsd = PPSD(st[0].stats, inv, period_smoothing_width_octaves=0.5)
        #if debug:
            #print(curfile)
        
        
        #ppsd.add(st)
    #ppsd.save_npz(net + '_' + sta + '_' + chan + '.npz')
    #del ppsd
    ##ppsd.plot()
    

per, nlnm = get_nlnm()
per, nhnm = get_nhnm()
fig = plt.figure(1, figsize=(12,12))
for chan in ['LHZ', 'LHN', 'LHE']:
    ppsd = PPSD.load_npz(net + '_' + sta + '_' + chan + '.npz')
    #result = ppsd.psd_values
    #print(result)
    perLP, mean = ppsd.get_mean()
    plt.semilogx(perLP, mean, label=chan + ' Mean', linewidth=2)
    plt.xlim((2,100.))
plt.semilogx(per, nhnm, 'C4', label='NHNM/NLNM', linewidth=3)
plt.semilogx(per, nlnm, 'C4', linewidth=3)
plt.xlabel('Period (s)')

#plt.xlim((1./20., 1000.))
ppsd = PPSD.load_npz(net + '_' + sta + '_SHZ.npz')
perLP, mean = ppsd.get_percentile(92.)
plt.semilogx(perLP, mean, label='SHZ 92nd Percentile', linewidth=2)
plt.ylabel('Power (dB rel. 1 $(m/s^2)^2/Hz)$)')
plt.legend(loc=9)
plt.axvspan(0.001, 6, alpha=0.5, color='0.3')
plt.savefig('PSD_plot.jpg', format='JPEG', dpi=400)
plt.show()
