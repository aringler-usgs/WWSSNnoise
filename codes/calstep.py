#!/usr/bin/env python
import matplotlib.pyplot as plt
from scipy.signal import step
import matplotlib.image as mpimg
from scipy import ndimage
import numpy as np

limmin = 8



ws = 2*np.pi/28.
ls = 1.96
wg = 2.*np.pi/98.1
lg = 0.95
sig2 =  0.03


num = [4., 0.]
den = [1., 2*ls*ws+2*lg*wg, ws**2 + wg**2 +4.*ls*ws*lg*wg*(1-sig2), 2*ls*ws*wg**2 + 2*lg*wg*ws**2, (ws**2)*(wg**2)]


#zeros = [0.]
#poles = [-0.08168, -0.05223, -.3971+0.1409j, -.3971-0.1409j]
#gain = 9.95

t = np.arange(0., 8*60., 0.1)

_, steppulse = step((num,den), T=t)


img = mpimg.imread('080170_1352_0007_04.tif')


#def rgb2gray(rgb):
    #return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])


img = ndimage.rotate(img, 2.)

#print(img)


##steppulse 
#x =[]
#y =[]
#with open('010166_1500_0007_04.csv','r') as f:
    #next(f)
    #for line in f:
        #line = line.split(',')
        #x.append(float(line[0]))
        #y.append(float(line[1])*35.)

#x = np.asarray(x)
#x -= min(x)
#x *= max(x)/60.

fig = plt.figure(1)
plt.imshow(img, origin='lower', extent = [0,60*limmin,0,60*limmin])
plt.plot(t, steppulse, 'r')
#plt.plot(x,y)
plt.show()

