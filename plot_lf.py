import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import os
import numpy as np
from sliding_wd import sliding_window

datapath = '.../IASI/data_v3/20130817/'
parameter_list  = ['C_F', 'ZA', 'lat', 'lon', 'SP', 'SCG', 'L_F']
parameter = 'C_F'
index = np.argwhere(np.array(parameter_list)==parameter).squeeze()
files = [elem for elem in os.listdir(datapath) if parameter in elem]
datalist = []
for fil in files:
    datalist.append(np.load(datapath+fil))
data = np.vstack(datalist)


### Plotting

plt.hist(data[:,:,index].flatten(),20)
plt.savefig('hist_(1,1).png')

ws_list = [(5,5),(10,10),(30,30),(60,60)]
for elem in ws_list:
    plt.clf()
    blocks = sliding_window(data[:,:,-1], ws=elem)
    blocks = blocks.mean(axis=(2,3))
    print(blocks.size)
    plt.hist(blocks.flatten())
    plt.savefig('hist_'+str(elem)+'.png')
    
