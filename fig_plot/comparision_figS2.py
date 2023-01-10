#python3  
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By  : Nagfal 
# Created Date: 8/1/2023 
# version ='1.1'
# ---------------------------------------------------------------------------
""" plot the sub figures in Fig. 2"""  

# from ossaudiodev import SNDCTL_COPR_RDATA
import xlrd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import numpy as np

#data file name
#example: 
# 'fig_plot\\cycle_s1.xls' cycle network
# 'fig_plot\\sto02_s1.xls' stochastic network with p_N = 0.2
#'fig_plot\\sto04_s1.xls' stochastic network with p_N = 0.4
data = xlrd.open_workbook('fig_plot\\sto04_s2.xls')



agent_number =50



if agent_number == 20:
    avg_table = 0
    std_table = 2
elif agent_number == 50:
    avg_table = 1
    std_table = 3
    
table1 = data.sheets()[avg_table] 
std_table = data.sheets()[std_table] 

nrows = table1.nrows
ncols =  table1.ncols



data = []

std_data = []


fig, ax = plt.subplots()
x = list(range(nrows-1))

for colx in range(0,ncols):
    col_data = table1.col_values(colx, start_rowx=1)
    data.append(col_data)
    std_col_data = std_table.col_values(colx, start_rowx=1)
    std_data.append(std_col_data)

ax.plot(x, data[0],label='BGE')
ax.plot(x, data[1],label='UCBGE')
ax.plot(x, data[2],label='DDUCB')
ax.plot(x, data[3],label='C-CBGE')


plt.fill_between(x,np.array(data[0])-std_data[0],np.array(data[0])+std_data[0],alpha = 0.2)
plt.fill_between(x,np.array(data[1])-std_data[1],np.array(data[1])+std_data[1],alpha = 0.2)
plt.fill_between(x,np.array(data[2])-std_data[2],np.array(data[2])+std_data[2],alpha = 0.2)
plt.fill_between(x,np.array(data[3])-std_data[3],np.array(data[3])+std_data[3],alpha = 0.2)

ax.set_xlim(0,10000)

ax.set_xlabel('round $t$',fontsize = 26)
ax.set_ylabel('average regret $\\bar{R}$',fontsize = 26)
ax.grid()
plt.legend(fontsize=20)
plt.tick_params(labelsize = 20)
plt.savefig('Fig2(f).png',bbox_inches='tight', dpi=800)
    # data.append(np.array(col_data))





pass

