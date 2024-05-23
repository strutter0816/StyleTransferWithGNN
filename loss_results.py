import os
import torch, time
import pandas as pd
import numpy as np
import pdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpathes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.lines as mlines

def show_results(save_name, path, names):
    pd0, pd_e = [],[]
    for i in range(len(path)):
        path[i] = os.path.join('experiments',path[i], 'loss.csv')
        pd0.append(pd.DataFrame(pd.read_csv(path[i])))
    
    colors = ['lightseagreen','gray', 'navy']
    lw = 1.0
    ls = ['-','--',':','-.']
    
    fig = plt.figure(figsize=(3,6))
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    for i in range(len(path)):
        y_c, y_s = [], []
        for k in range(len(pd0[i]['content_loss'])):
            y_c.append(float(pd0[i]['content_loss'].values[k][7:13]))
            y_s.append(float(pd0[i]['style_loss'].values[k][7:13]))
        plt.plot(y_c, color=colors[0], label=names[i]+': Content Loss', linewidth=lw, linestyle=ls[i])
        plt.plot(y_s, color=colors[1], label=names[i]+': Style Loss', linewidth=lw, linestyle=ls[i])
        plt.legend(fontsize=8) 
    
    plt.savefig(save_name+'.jpg', bbox_inches = 'tight', pad_inches=0.1)
    plt.savefig(save_name+'.pdf', bbox_inches = 'tight', pad_inches=0.1)


if __name__=='__main__':
    show_results('losses',['graph_adain_s2_v1', 'baseline_adain'], ['Graph','Baseline'])