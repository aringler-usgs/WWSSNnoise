#!/usr/bin/env python
import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fmin, fminbound
from scipy.signal import step
import sys
import numpy as np

import matplotlib as mpl
# Importing and applying font
mpl.rc('font', family = 'serif')
mpl.rc('font', serif = 'Times') 
#mpl.rc('text', usetex = True)
mpl.rc('font',size=18)

files = glob.glob("ALQ*")

#files = [files[9]]

debug = True
ws = 2*np.pi/30.
ls = 1.96
wg = 2.*np.pi/98.1
lg = 0.95
sig2 =  0.0392

t = np.arange(0., 8*60., 0.1)



def cal_step(param, t):
    ws, ls, wg, lg, sig2, gain = param
    num = [gain, 0.]
    den = [1., 2*ls*ws+2*lg*wg, ws**2 + wg**2 +4.*ls*ws*lg*wg*(1-sig2), 2*ls*ws*wg**2 + 2*lg*wg*ws**2, (ws**2)*(wg**2)]
    _, steppulse = step((num,den), T=t)
    steppulse /= max(steppulse)
    return steppulse


vals = []



fg = open('Results.csv','w')

fig = plt.figure(1,figsize=(12,12))
for curfile in files:
    t, s =[], []
    with open(curfile) as f:
        next(f)
        for line in f:
            
            t.append(float(line.split(',')[0]))
            s.append(float(line.split(',')[1]))
    t = np.asarray(t)
    s = np.asarray(s)
    idx = np.argsort(t)
    t = t[idx]
    s = s[idx]
    
    t -= min(t)
    s /= max(np.abs(s))
    s -= s[0]
    if min(s) < -.1:
        s *= -1.
    s /= max(np.abs(s))
    def cost_function(param):
        val =  np.sum((s - cal_step(param, t))**2)
        print(val)
        return val
    
    #cost_function([ws, ls, wg, lg, sig2, 1.])
    x = fmin(cost_function, [ws, ls, wg, lg, sig2, 1.])
    #x = [ws, ls, wg, lg, sig2, 1.]
    print(x)
    vals.append(x)
    ss = cal_step(x, t)
    
    
    title2 = curfile.replace('.csv','')
    title2 = title2.split('_')
    titleplt = title2[0] + ' ' + title2[1] + ' ' + title2[2] + ' 19' + title2[3] + ' ' + title2[4][:2] + ':' + title2[4][2:]
    resi = np.sqrt(cost_function(x)/np.sum(s**2))*100.
    print(title2)

    fg.write(title2[0] + ', ' + title2[1] + ', ' + title2[2] + ', ' + title2[3] + ', ' + title2[4]
            + ', ' + str(x[0]) + ', ' + str(x[1]) + ', ' + str(x[2]) + ', ' + str(x[3]) + ', ' + str(x[4]) + ', ' + str(x[5]) + ', ' + str(resi) + '\n')
    plt.title(titleplt)
    plt.plot(t,ss, label='Synthetic')
    plt.plot(t,s, label='Data')
    plt.xlim((min(t), max(t)))
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (normalized)')
    plt.legend()
    plt.savefig('Check' + curfile.replace('csv','jpg'), format='JPEG')

    plt.clf()
    plt.close()

fg.close()

#fig = plt.figure(2)
#for idx, val in enumerate(vals):
    #for ind in range(5):
        #plt.subplot(5,1,ind+1)
        #plt.plot(idx, val[ind])
#plt.show()
    
