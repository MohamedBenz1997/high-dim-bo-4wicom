import math

import numpy as np
import tensorflow as tf
from config import  Config
import os

from LargeScaleGainClass import Large_Scale_Gain
import matplotlib.pyplot as plt
from matplotlib import transforms, pyplot

# base = pyplot.gca().transData
# rot = transforms.Affine2D().rotate_deg(90)

data = Large_Scale_Gain()
Azi_phi_deg = tf.constant([[0.0]],"float32")
Elv_thetha_deg = tf.constant([[range(3600)]],"float32")/10.

antenna_gain = data.Antenna_gain_3GPP(Azi_phi_deg,Elv_thetha_deg)
array_gain = data.array_gain
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

plt.plot(Elv_thetha_deg.numpy().flatten()*(2*math.pi)/360,(data.G_elv1).numpy().flatten())
plt.plot(Elv_thetha_deg.numpy().flatten()*(2*math.pi)/360,array_gain[0].numpy().flatten())
plt.plot(Elv_thetha_deg.numpy().flatten()*(2*math.pi)/360,data.G_Antenna.numpy().flatten())

ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_xticks(np.pi/180 * np.linspace(0,  360, 12, endpoint=False))
plt.legend(["Element gain","Array gain","Antenna gain"],loc="upper right")
dir = os.getcwd()
plt.title("Antenna and array patterns in Elevation for downtilt=12  and Azi=0   ")
plt.savefig(dir+'/results/elv_array_pattern_tilt.pdf',bbox_inches='tight')
plt.show()
#--- Azi

Azi_phi_deg = tf.constant([[range(360)]],"float32")-180
Elv_thetha_deg = tf.constant([[[102.0]*360]],"float32")

antenna_gain = data.Antenna_gain_3GPP(Azi_phi_deg,Elv_thetha_deg)
G_sector = data.sectoring(Azi_phi_deg,Elv_thetha_deg)
Azi_phi_sector = data.Azi_phi_sector
array_gain = data.array_gain
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

# plt.plot(Azi_phi_deg.numpy().flatten()*(2*math.pi)/360,antenna_gain.numpy().flatten())

for i in range(3):
    x=Azi_phi_sector[:,0,:].numpy().flatten()*(2*math.pi)/360
    # ind_sort = np.argsort(x)
    # x = x[ind_sort]
    y = G_sector[:,i,:].numpy().flatten()-array_gain.numpy().flatten()
    # y= y[ind_sort]
    plt.plot(x,y)
# plt.plot(Azi_phi_deg.numpy().flatten()*(2*math.pi)/360,array_gain.numpy().flatten())
# plt.legend(["Antenna gain","Array gain"])
dir = os.getcwd()
plt.title("Azimuth pattern for different sectors (down tilt=12) ")
plt.savefig(dir+'/results/azi_pattern_sectors_tilde.pdf',bbox_inches='tight')
plt.show()
