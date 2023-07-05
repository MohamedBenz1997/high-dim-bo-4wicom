# Generated code -- CC0 -- No Rights Reserved -- http://www.redblobgames.com/grids/hexagons/

from __future__ import division
from __future__ import print_function
import collections
import math
import matplotlib.pyplot as plt
from hexfunctions import *

sum =0
N = 2
T0 = []
for i in range(2*N+1):
    for j in range(2*N+1):
        for k in range(2*N+1):
            if (i-N)+(j-N)+(k-N)==0:
                h =Hex(i-N,j-N,k-N)
                T0.append(h)

T0_center = [0]*len(T0)
orgin = Point(0.0,0.0)
radius = 250
size = Point(2/math.sqrt(3.0)*radius,2/math.sqrt(3.0)*radius)
layout  = Layout(layout_flat,size,orgin)
fig, ax = plt.subplots(1, 1)
for i in range(len(T0)):
    T0_center[i] = hex_to_pixel(layout,T0[i])
    ax.plot(T0_center[i].x,T0_center[i].y,"ko")
    ax.text(T0_center[i].x,T0_center[i].y,str(i))
    corners= polygon_corners(layout, T0[i])
    x,y = get_corners(corners)
    ax.plot(x,y,'cyan')

c = Hex(0.0,0.0,0.0)
c_miror0 = Hex(2*N+1,-N-1,-N)
# c_miror0_pos = hex_to_pixel(layout,c_miror0)
rotate = hex_subtract(c_miror0,c)
c_miror = [0]*6
pos_miror =[0]*6
for i in range(6):
    rotate = hex_rotate_right(rotate)
    c_miror[i] = rotate
    pos_miror[i] = hex_to_pixel(layout,rotate)
    # ax.plot(pos_miror[i].x,pos_miror[i].y,'o')
    # ax.text(pos_miror[i].x,pos_miror[i].y,str(i))
colors = ["red", "blue" , "green", "orange", "purple",'brown']
for i in range(len(c_miror)):
    center_i = c_miror[i]
    for j in range(len(T0)):
        Ti_center_j = hex_add(center_i,T0[j])
        Ti_center_j_pos = hex_to_pixel(layout, Ti_center_j)
        ax.plot(Ti_center_j_pos.x, Ti_center_j_pos.y, "ko")
        ax.text(Ti_center_j_pos.x, Ti_center_j_pos.y, str(j))
        corners = polygon_corners(layout, Ti_center_j)
        x, y = get_corners(corners)
        ax.plot(x, y, colors[i])


plt.show()

