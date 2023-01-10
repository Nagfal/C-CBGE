#python3  
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By  : Nagfal 
# Created Date: 8/1/2023 
# version ='1.1'
# ---------------------------------------------------------------------------
""" plot Fig. 4"""  

import xlrd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import numpy as np

#the data file name
data1 = xlrd.open_workbook('fig_plot\\C_N_full_fixEta.xls')

table1 = data1.sheets()[0]    

nrows = table1.nrows
ncols =  table1.ncols



font1 = {'family' : 'serif',

'weight' : 'normal',

'size'   : 26,

}

data = []
data_2 = []
for rowx in range(nrows):
    row_data = table1.row_values(rowx, start_colx=0, end_colx=None)
    data.append(np.array(row_data))


data = np.array(data)


x = np.array(list(range(nrows)))+1
y = list(range(ncols))
X, Y = np.meshgrid(x, y)

Z = data.T

Z_avg = Z




base_list = np.linspace(-2, 2, 40).tolist()
y_ticks = [pow(10,i) for i in base_list]

fig, ax = plt.subplots()
pcm = ax.pcolormesh(x, y_ticks, Z_avg)

line_sigma_x = x
# line_sigma_y = [0.5] * len(line_sigma_x)
line_sigma_y2 = [1.0] * len(line_sigma_x)

ax.plot(line_sigma_x,line_sigma_y2,'.-',color = 'red',label='$C=1.0$')
# ax.plot(line_sigma_x,line_sigma_y,'.-',color = 'orange',label='$C^2=1.0$')

plt.legend(prop={'family':'SimHei','size':24})
cb = fig.colorbar(pcm,ax = ax)
cb.ax.tick_params(labelsize=18)
cb.set_label('regret $\\bar{R}(T)$', fontdict = font1)
plt.yscale('log')
plt.ylabel('exploration parameter $C^2$',fontsize = 26)
plt.xlabel('agent number $|\mathcal{I}|$',fontsize = 26)
plt.tick_params(labelsize = 20)
plt.show()
print(data)