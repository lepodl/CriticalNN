# -*- coding: utf-8 -*- 
# @Time : 2022/6/29 23:50 
# @Author : lepold
# @File : fig_plot.py

import numpy as np
import pandas as pd
from plotnine import *
from matplotlib import colors
import matplotlib.pyplot as plt
import skmisc
from skmisc.loess import loess as loess_klass

# file = np.load('../data/grid_search_d100_new.npz')
file = np.load('../data/grid_search_d100_new_new.npz')
# file = np.load('../data/grid_search_d500_new_new.npz')
mean_fr = file["mean_fr"]
cv = file["cv"]
pcc = file["pcc"]
ks = file["ks"]
cc = file["cc"]
ampa_contribution = file["ampa"]
gabaA_contribution = file["gabaA"]
fig, ax = plt.subplots(2, 2, figsize=(10, 10), dpi=100)
ax = ax.flatten()
x, y = mean_fr.shape

im = ax[0].imshow(mean_fr, cmap='jet', interpolation='gaussian')
fig.colorbar(im, ax=ax[0], shrink=0.6)
ax[0].grid(False)
ax[0].set_title("mean_fr")

im = ax[1].imshow(cv, cmap='jet', interpolation='gaussian')
fig.colorbar(im, ax=ax[1], shrink=0.6)
ax[1].grid(False)
ax[1].set_title("cv")



def arg_percentile(series, x):
    a, b = 0, 1
    while True:
        # m是a、b的终点
        m = (a+b)/2
        # 可以打印查看求解过程
        # print(np.percentile(series, 100*m), x)
        if np.percentile(series, 100*m) >= x:
            b = m
        elif np.percentile(series, 100*m) < x:
            a = m
        # 如果区间左右端点足够靠近，则退出循环。
        if np.abs(a-b) <= 0.000001:
            break
    return m

cc_data = cc.flatten()
percent1 = arg_percentile(cc_data, 1.0)
percent2 = arg_percentile(cc_data, 1.5)
first = np.round(255*percent1).astype(np.int32)
second = np.round(255*(percent2 - percent1)).astype(np.int32)
third = 256 - first - second
#colors2 = new_cmap(np.linspace(0, 1, first))
colors1 = plt.cm.seismic(np.linspace(0.1, 0.42, first))
# colors2 = np.ones((second, 4), dtype=np.float64)
colors2 = plt.cm.seismic(np.linspace(0.42, 0.58, second))
colors3 = plt.cm.seismic(np.linspace(0.58, 0.9, third))
cols = np.vstack((colors1, colors2,colors3))
mymap = colors.LinearSegmentedColormap.from_list('my_colormap', cols)
im = ax[3].imshow(cc, interpolation='gaussian', cmap='seismic')
fig.colorbar(im, ax=ax[3], shrink=0.6)
ax[3].grid(False)
ax[3].set_title("cc")

im = ax[2].imshow(pcc, cmap='jet', interpolation='gaussian')
fig.colorbar(im, ax=ax[2], shrink=0.6)
ax[2].grid(False)
ax[2].set_title("pcc")


for i in range(4):
    yticks = np.linspace(0, x, 4, endpoint=False, dtype=np.int8)
    ax[i].set_yticks(yticks)
    ax[i].set_yticklabels([f'{data:.3f}' for data in ampa_contribution[yticks]], rotation=60)
    xticks = np.linspace(0, y, 4, endpoint=False, dtype=np.int8)
    ax[i].invert_yaxis()
    ax[i].set_xticks(xticks)
    ax[i].set_xticklabels([f'{data:.2f}' for data in gabaA_contribution[xticks]], )
    ax[i].set_ylabel(r"$AMPA$")
    ax[i].set_xlabel(r"$GABA_{A}$")

gabaA_here = np.load('../notebook/critical_param.npy')[:130, 2]
file = np.load('../data/along_critical_line.npz')
ks_size = file['ks_size']
slop_size = file['exponent_size']
peak_freq = file['peak_power']
cc = file['cc']
df = pd.DataFrame({"x": gabaA_here, "ks":ks_size, "slope":slop_size, })
plot_loess=(ggplot( df, aes('x','ks')) +
geom_point(fill="black",colour="black",size=3,shape='o') +
geom_smooth(method = 'loess',span=0.4,se=True,colour="#00A5FF",fill="#00A5FF",alpha=0.2)+ #(f)
scale_y_continuous(breaks = np.arange(0, 126, 25))+
theme(
axis_title=element_text(size=18,face="plain",color="black"),
axis_text = element_text(size=16,face="plain",color="black"),
legend_position="none",
aspect_ratio =1,
figure_size = (5, 5),
dpi = 100))
plt.gcf().show()
print(plot_loess)
