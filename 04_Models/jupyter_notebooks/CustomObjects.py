#!/usr/bin/python3

import pickle
import re
import math
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import scipy.stats as stats
import scipy.spatial as sp
import scipy.cluster.hierarchy as hc

from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer, RobustScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import os
import sys
import sklearn
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

from keras.utils import plot_model






#-------------------------------------------------------- CONFIG PLOTS

BACKGROUND_COL = 'white'
AXIS_COLOR     = '#DBD7D2'
LINES_COLOR    = '#DBD7D2'
sns.set_style("whitegrid", {'grid.linestyle': '--',
                            'grid.color': LINES_COLOR, 
                            'axes.edgecolor': AXIS_COLOR,
                            'axes.facecolor':BACKGROUND_COL,
                            'figure.facecolor':BACKGROUND_COL,
                            })
plt.rcParams['axes.facecolor'] = BACKGROUND_COL
#
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Set DPI for all figures output in the notebooks
plt.rcParams['figure.dpi'] = 36 # I set a low dpi to make the ipynb files smaller 

# Set DPI for saving figures
plt.rcParams['savefig.dpi'] = 250

#--------------------------------------------------------   VARIABLES 

CT_LIST = ['ESC', 'MES', 'CP', 'CM']
HM_LIST = ['H3K4me3', 'H3K27ac', 'H3K27me3',  'RNA']
PREFIXES = [HM + '_' + CT for HM in HM_LIST for CT in CT_LIST]


MARKER_GENES = {'ESC': ['Nanog','Pou5f1','Sox2','Dppa5a'],
                'MES': ['Mesp1','T', 'Vrtn','Dll3'],
                'CP':  ['Gata5', 'Tek','Sox18','Lyl1',],
                'CM':  ['Actn2', 'Coro6','Myh6','Myh7'],
                }

MARKER_GENES_EXT = {'ESC': ['Nanog','Pou5f1','Sox2','L1td1','Dppa5a','Tdh','Esrrb','Lefty1','Zfp42','Sfn','Lncenc1','Utf1'],
                    'MES': ['Mesp1','Mesp2','T', 'Vrtn','Dll3','Dll1', 'Evx1','Cxcr4','Pcdh8','Pcdh19','Robo3','Slit1'],
                    'CP':  ['Sfrp5', 'Gata5', 'Tek','Hbb-bh1','Hba-x', 'Pyy','Sox18','Lyl1','Rgs4','Igsf11','Tlx1','Ctse'],
                    'CM':  ['Nppa','Gipr', 'Actn2', 'Coro6', 'Col3a1', 'Bgn','Myh6','Myh7','Tnni3','Hspb7' ,'Igfbp7','Ndrg2'],
                    }

SEL_GENES = {**MARKER_GENES_EXT, 
                'PRC1': ['Ring1','Pcgf1','Pcgf2','Pcgf3','Pcgf5','Pcgf5','Pcgf6',
                        'Cbx2','Cbx4','Cbx6','Cbx7','Cbx8','Phc1','Phc2','Phc3','Scmh1','Rybp','Yaf2','Kdm2b','Wdr5',
                        #'Pcgf4',
                        ],
                'PRC2': ['Ezh1', 'Ezh2', 'Suz12', 'Eed', 'Rbbp4', 'Rbbp7', 'Jarid2', 'Aebp2', 'Epop', 'Lcor'],
                'C11':['4930519K11Rik', 'Bhlhe22', 'Kcnma1', 'S1pr5']
                }

try:
    with open(f'../data/gene_sets/gene_sets_dict.pkl', 'rb') as f:
        GENE_SETS = pickle.load(f)
except FileNotFoundError:
    print('Gene sets not found. Please run the notebook to generate the gene sets.')

try:
    with open(f'../data/gene_sets/gonzalez_dict.pkl', 'rb') as f:
        GONZALEZ = pickle.load(f)
except FileNotFoundError:
    print('Gonzalez annotations not found. Please run the notebook to generate the gene sets.')
    
try:
    with open(f'../data/Morey/ESC_MES_FC25_Morey.pkl', 'rb') as f:
            ESC_MES_FC25_Morey = pickle.load(f)
except FileNotFoundError:
    print('ESC_MES_FC25_Morey not found. Please run the notebook to generate the gene sets.')

#-------------------------------------------------------- Discrete colors dict

HM_COL_DICT = {'H3K4me3': '#f37654','H3K27ac': '#b62a77','H3K27me3': '#39A8AC','RNA':'#ED455C'}
CT_COL_DICT = {'ESC': '#355070', 'MES': '#6d597a', 'CP': '#b56576','CM': '#eaac8b'}
CT_COL_DICT= {'ESC': '#405074',
                'MES': '#7d5185',
                'CP': '#c36171',
                'CM': '#eea98d',}
SET_COL_DICT= {'training':'#97DA58','validation':'#9b58da','test':'#DA5A58'}
GONZALEZ_COL_DICT= {'Active': '#E5AA44','Bivalent': '#7442BE'}

#-------------------------------------------------------- FUNCTIONS


def subscript_get(string):
    return f"$_{{\\mathrm{{{string}}}}}$"


def violins(DF,COL_DICT,SAVEFIG,TITLE='',SAT=0.8,X_LAB='AVG', LOG_SCALE=False, DPI=150,WIDTH_FACTOR=1):
    ''' 
    Plot violin plots for multiple columns of a DataFrame.

    Parameters:
        DF (pandas DataFrame): The DataFrame containing the data to plot.
        COL_DICT (dict): Dictionary mapping column names to colors for plotting.
        SAVEFIG (str): Filepath to save the resulting plot.
        TITLE (str, optional): Title for the plot. Default is an empty string.
        SAT (float, optional): Saturation parameter for the plot. Default is 0.8.
        X_LAB (str, optional): Label for the x-axis. Default is 'AVG'.
        LOG_SCALE (bool, optional): Flag indicating whether to use logarithmic scale. Default is False.
    '''
    MAPPED_COL = list(map(lambda x: next((v for k, v in COL_DICT.items() if k in x), None), DF.columns))
    plt.figure(figsize=(6,len(DF.columns)*0.4))
    if LOG_SCALE: 
        plt.figure(figsize=(6*WIDTH_FACTOR,len(DF.columns)*0.4))
        plt.suptitle(TITLE,fontsize=16)
        plt.subplot(1,2,1)
    else:
        plt.figure(figsize=(3*WIDTH_FACTOR,len(DF.columns)*0.4))
        plt.title(TITLE,fontsize=16)
    #
    sns.violinplot(DF, palette=MAPPED_COL,orient='h',saturation=SAT, inner=None,width=0.9,
                linewidth=0,);
    sns.boxplot(DF, palette=MAPPED_COL,orient='h',saturation=SAT,
                    width=0.2,linewidth=1,showcaps=1,
                    medianprops={"color": "w", "linewidth": 1},
                    boxprops=dict(facecolor="black",alpha=0.4),
                    whiskerprops=dict(color="black", alpha=0.4),
                    flierprops = dict(marker='.',markerfacecolor='grey', markersize=1,alpha=0.5));
    sns.despine(left=1,bottom=1)
    plt.xlabel(X_LAB);
    #
    if LOG_SCALE:
        plt.subplot(1,2,2)
        sns.violinplot(np.log10(DF), palette=MAPPED_COL,orient='h',saturation=SAT, width=0.9,inner=None,
                    linewidth=0,);
        sns.boxplot(np.log10(DF), palette=MAPPED_COL,orient='h',saturation=SAT,
                        width=0.2,linewidth=1,showcaps=1,
                        medianprops={"color": "w", "linewidth": 1},
                        boxprops=dict(facecolor="black",alpha=0.4),
                        whiskerprops=dict(color="black", alpha=0.4),
                        flierprops = dict(marker='.',markerfacecolor='grey', markersize=1,alpha=0.5)).set_yticklabels([])
        plt.xlabel(f'log10({X_LAB})');
        sns.despine(left=0,bottom=1)
    plt.savefig(SAVEFIG, format="png", bbox_inches="tight", dpi=DPI);

def violins_train_test(X_train, X_test, SET_COL_DICT, INPUT_NAME,MODE='validation', TITLE = 'Feature distribution of train and test set',SAT=0.8,X_LABEL='Z-score'):
    train_data = X_train.melt(var_name='Feature', value_name='Value', ignore_index=False)
    train_data['Set'] = 'training'
    test_data = X_test.melt(var_name='Feature', value_name='Value', ignore_index=False)
    test_data['Set'] = MODE
    plot_data = pd.concat([train_data, test_data])
    # 
    plt.figure(figsize=(5,len(X_train.columns)*0.3))
    sns.violinplot(y='Feature', x='Value', hue='Set', data=plot_data,  palette=SET_COL_DICT, 
                inner=None,legend=0, split=1,linewidth=0, width=0.9, saturation=SAT,)
    plt.xticks(rotation=90)
    plt.title(TITLE)
    plt.xlabel(X_LABEL)
    plt.ylabel('')
    plt.tight_layout()
    sns.despine(left=1)
    plt.savefig(f'../figures/split/violins_{INPUT_NAME}_{MODE}.pdf', format="pdf", bbox_inches="tight");


def freq_ct_max(DF,CT_COL_DICT,SAVEFIG,TITLE='Freq. of each CT(max) in XX',SAT=1):
    CT_MAX = np.argmax(DF,axis=1)
    _ , STABLE_counts = np.unique(CT_MAX, return_counts=True)
    CT_MAX = pd.DataFrame({'CT': DF.columns, 'Frequency': STABLE_counts})
    MAPPED_COL = list(map(lambda x: next((v for k, v in CT_COL_DICT.items() if k in x), None), DF.columns))
    plt.figure(figsize=(4, 2))
    sns.barplot(x='Frequency', hue='CT', data=CT_MAX, dodge=1, palette=MAPPED_COL, saturation=SAT,legend=0)
    subscript = subscript_get('max')
    plt.ylabel(f"CT{subscript_get('max')}")
    plt.xlabel('# of genes')
    sns.despine()
    plt.title(TITLE)
    plt.savefig(SAVEFIG, format="pdf", bbox_inches="tight");

def color_tick_by_HM_HEATMAP(HM_LIST, HM_COL_DICT, HEATMAP,FONTSIZE=16,ROTATE=True):
    """
    Change the color of x and y tick labels in a heatmap based on partial matches with histone marks.

    Parameters:
        HM_LIST (list): List of histone marks.
        HM_COL_DICT (dict): Dictionary mapping histone marks to colors.
        HEATMAP (Axes object): The Axes object containing the heatmap with x and y tick labels to be colored.
        FONTSIZE (int, optional): Font size for the tick labels. Default is 16.
        ROTATE (bool, optional): Whether to rotate the tick labels. Default is True.
    """
    for label in HEATMAP.ax_heatmap.axes.get_xticklabels():
        text = label.get_text()
        for mark in HM_LIST:
            if mark in text:
                label.set_color(HM_COL_DICT.get(mark, 'black'))  # Default to black if mark not found
                label.set_fontsize(FONTSIZE)
                if ROTATE:
                    label.set_rotation(45)
                    label.set_horizontalalignment('right')
    for label in HEATMAP.ax_heatmap.axes.get_yticklabels():
        text = label.get_text()
        for mark in HM_LIST:
            if mark in text:
                label.set_color(HM_COL_DICT.get(mark, 'black'))  # Default to black if mark not found
                label.set_fontsize(FONTSIZE)

def color_tick_by_HM_BOXPLOT(HM_LIST, HM_COL_DICT, BOXPLOT):
    """
    Change the color of y tick labels in a boxplot based on partial matches with histone marks.

    Parameters:
        HM_LIST (list): List of histone marks.
        HM_COL_DICT (dict): Dictionary mapping histone marks to colors.
        BOXPLOT (Axes object): The Axes object containing the boxplot with y tick labels to be colored.
    """
    for label in BOXPLOT.get_xticklabels():
        text = label.get_text()
        for mark in HM_LIST:
            if mark in text:
                label.set_color(HM_COL_DICT.get(mark, 'black'))  # Default to black if mark not found
                label.set_rotation(90)

def assign_palette(PALETTE, DICT):    
    palette = sns.color_palette(PALETTE, len(GENE_SETS)).as_hex()
    COL_DICT = {}
    for key, color in zip(DICT.keys(), palette):
        COL_DICT[key] = color
    return COL_DICT




def get_color_list(FEATURE_NAMES,COL_DICT):
    color_list=[]
    for FEATURE in FEATURE_NAMES:
        for key, value in COL_DICT.items():
            if key in FEATURE:
                color_list.append(value)
    return color_list

def force_square(x,fw=8,fh=8):
    dx, dy = np.diff(x.ax_heatmap.transData.transform([[0, 1], [1, 0]]), axis=0).squeeze()
    if dx > dy:
        fh = fh * dx / dy
    else:
        fw = fw * dy / dx
    x.fig.set_size_inches(fw, fh)

def export_legend(COL_DICT, filename, MK_SIZE=8, TITLE=None):
    plt.figure(figsize=(4, 2))

    custom_legend = [plt.Line2D([0], [0], marker='o', color='w', label=ct, markerfacecolor= color, markersize = MK_SIZE) for ct, color in COL_DICT.items()]
    legend = plt.legend(handles=custom_legend,framealpha = 1 ,title=TITLE)
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, format='pdf', bbox_inches=bbox)
    plt.grid(False)
    sns.despine()
    
    # Hide tick labels
    plt.xticks([])
    plt.yticks([])

def corr_clustering(DF,HM_COL_DICT,CT_COL_DICT,SAVEFIG,TITLE=None,CORR_METHOD='kendall',LINK_METHOD='average',FW=8,FH=8):
    CORR = DF.corr(method = CORR_METHOD)
    DF_dism = 1 - CORR   # distance matrix
    linkage = hc.linkage(sp.distance.squareform(DF_dism), method=LINK_METHOD, optimal_ordering=1)
    #
    FEAT_COL = get_color_list(CORR.columns, HM_COL_DICT)
    CT_COL = get_color_list(CORR.columns, CT_COL_DICT)
    COL_GROUP = pd.DataFrame({'Cell Type':CT_COL, 'Histone mark':FEAT_COL}, index=CORR.columns)
    plt.figure()
    x = sns.clustermap(CORR,row_linkage=linkage, col_linkage=linkage
                    ,dendrogram_ratio=0.1,cmap='RdBu_r',row_cluster=1,col_cluster=1,vmin=-1, vmax=1,
                    cbar_kws={"label": "Kendall's τ coeff."},
                    linewidths=0.5, linecolor='black',col_colors=COL_GROUP,yticklabels=1,colors_ratio=0.02)
    #x.ax_row_dendrogram.set_visible(False) #suppress row dendrogram
    color_tick_by_HM_HEATMAP(HM_COL_DICT.keys(),HM_COL_DICT,x,ROTATE=True,FONTSIZE=7)
    force_square(x, fw=FW, fh=FH)
    x.ax_cbar.set_position([-0.2, 0.4, 0.02, 0.2])
    plt.suptitle(TITLE,size=15,y=1.1)
    #
    plt.savefig(SAVEFIG, format="pdf", bbox_inches="tight");

def RNA_gene_CT(MAIN,GENE_LIST,CT_LIST,CT_COL_DICT,SAVE_PREFIX):
    plt.figure(figsize=(len(GENE_LIST)*3,3))
    for i,GENE_NAME in enumerate(GENE_LIST):
        CT_REG = '|'.join(CT_LIST)
        Series = MAIN.loc[GENE_NAME]
        CT = Series.index.str.extract(f'({CT_REG})')[0]
        REP = Series.index.str.extract(f'(\d)')[0]
        DF = pd.DataFrame({'FPKMs':Series.values, 'CT':CT, 'REP':REP})
        #
        #plt.figure(figsize=(3,3))
        plt.subplot(1,len(GENE_LIST),i+1)
        plt.title(GENE_NAME,pad=20)
        sns.stripplot(data=DF,x='CT',y='FPKMs',hue='CT',palette=CT_COL_DICT,
                    s=12, alpha=1, legend=False,linewidth=0)
        sns.lineplot(data=DF,x='CT',y='FPKMs', err_style=None,
                    color='black',linewidth=1, dashes=(2, 2))
        sns.despine(left=1,bottom=0,top=1)
        plt.xlabel('')
        plt.yticks(  np.round( [ 0, max(DF['FPKMs']) ], 1 )  )
        plt.ylim([0,  max(DF['FPKMs'])*1.1])
        if i>0:plt.ylabel('')
        plt.tight_layout()
    plt.savefig(f'../figures/gene_trend/{SAVE_PREFIX}.pdf', format="pdf", bbox_inches="tight")
    

def calculate_mean_features(DF, FEATURE_PREFIXES):
    """
    Calculate mean features based on the given column prefixes.

    Parameters:
        DF (pandas.DataFrame): DataFrame containing the original data.
        FEATURE_PREFIXES (list): List of column prefixes for which mean features need to be calculated.

    Returns:
        pandas.DataFrame: DataFrame with mean features added.
    """
    AVG_COL_DF = pd.DataFrame()  # Initialize empty DataFrame
    for PREFIX in FEATURE_PREFIXES:
        columns = DF.filter(regex=f'^{PREFIX}_').columns
        if len(columns) != 0:
            AVG_COL_DF[PREFIX] = DF[columns].mean(axis=1)
        print(f'{list(columns)} -> {PREFIX} COLUMNS ')
    return AVG_COL_DF





#----------------------------------------------------------- PCA, t-SNE, UMAP for Sample QC




def RNA_PCA(DATA, CT_REG, CT_COL_DICT,SAVE_PREFIX,comp3=None):
    N_COMP=3
    pca = PCA(n_components=N_COMP)
    pca.fit(DATA)
    PCA_VAR = (pca.explained_variance_ratio_*100)
    PCA_VAR = [f'PCA{x} ({int(PCA_VAR[x-1])}%)' for x in range(1, N_COMP + 1)]
    PCA_DF = pd.DataFrame(pca.transform(DATA),columns=PCA_VAR)
    #
    PCA_DF['CT']    = DATA.index.str.extract(CT_REG)
    PCA_DF['REP']   = DATA.index.str.extract(r'(_\d)')
    PCA_DF['REP']   = PCA_DF['REP'].str.extract(r'(\d)')
    #
    i=1
    legend=False
    kwargs={'edgecolor':"grey", 'linewidth':1}
    plt.figure(figsize = (12,6))
    plt.suptitle(SAVE_PREFIX)
    if comp3 == None: comp3 = 1
    else:             comp3 = 2
    for comp in range(0,comp3):
        plt.subplot(1,2,i)
        ax=sns.scatterplot(data = PCA_DF, x=PCA_VAR[comp], y=PCA_VAR[1], hue='CT',palette=CT_COL_DICT, s=130, legend=legend,
                       **kwargs)
        sns.despine(left=1,bottom=1)
        if legend: sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        plt.axis('equal')
        i=i+1
        legend=True
    plt.savefig(f'../figures/preprocessing/PCA/RNA_{SAVE_PREFIX}_PCA.pdf', format="pdf", bbox_inches="tight")

def CHIP_PCA(DATA, HM_REG, MARKER_LIST, CT_REG, CT_COL_DICT,SAVE_PREFIX,):
    N_COMP=3
    pca = PCA(n_components=N_COMP)
    pca.fit(DATA)
    PCA_VAR = (pca.explained_variance_ratio_*100)
    PCA_VAR = [f'PCA{x} ({int(PCA_VAR[x-1])}%)' for x in range(1, N_COMP + 1)]
    PCA_DF = pd.DataFrame(pca.transform(DATA),columns=PCA_VAR)
    #
    PCA_DF['HM']    = DATA.index.str.extract(HM_REG)
    PCA_DF['CT']    = DATA.index.str.extract(CT_REG)
    PCA_DF['REP']   = DATA.index.str.extract(r'(_\d)')
    PCA_DF['REP']   =   PCA_DF['REP'].str.extract(r'(\d)')
    #
    i=1
    kwargs={'edgecolor':"grey", 'linewidth':1}
    legend=False
    sns.set_style('whitegrid')
    plt.figure(figsize = (6,6))
    plt.suptitle(SAVE_PREFIX)
    ax=sns.scatterplot(data = PCA_DF, x=PCA_VAR[0], y=PCA_VAR[1], hue='CT',palette=CT_COL_DICT, s=130, style='HM',markers=MARKER_LIST, legend=legend,
                       **kwargs)
    sns.despine(left=1,bottom=1)
    if legend: sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.axis('equal')
    i=i+1
    legend=True
    plt.savefig(f'../figures/preprocessing/PCA/CHIP_{SAVE_PREFIX}_PCA.pdf', format="pdf", bbox_inches="tight")

def CHIP_PCA_HM(DATA, HM_LIST, MARKER_LIST, CT_REG, CT_COL_DICT,SAVE_PREFIX,):
    legend=False
    sns.set_style('whitegrid')
    plt.figure(figsize = (6*len(HM_LIST),6))
    plt.suptitle(SAVE_PREFIX)
    for i,HM in enumerate(HM_LIST):
        if i+1 == len(HM_LIST):legend=True
        X = DATA.filter(regex=HM).transpose()
        pca = PCA(n_components=2)
        pca = pca.fit(X)
        PCA_VAR = (pca.explained_variance_ratio_*100)
        PCA_VAR = [f'PCA{x} ({int(PCA_VAR[x-1])}%)' for x in (1, 2)]
        PCA_DF = pd.DataFrame(pca.transform(X),columns=PCA_VAR)
        HM_REG = '|'.join(HM_LIST)
        PCA_DF['HM']    = X.index.str.extract(f'({HM_REG})')
        PCA_DF['CT']    = X.index.str.extract(CT_REG)
        PCA_DF['REP']   = X.index.str.extract(r'(_\d)')
        PCA_DF['REP']   =   PCA_DF['REP'].str.extract(r'(\d)')
        
        #
        kwargs={'edgecolor':"grey", 'linewidth':1}
        plt.subplot(1,len(HM_LIST),i+1)
        plt.title(HM)
        ax=sns.scatterplot(data = PCA_DF, x=PCA_VAR[0], y=PCA_VAR[1], hue='CT',palette=CT_COL_DICT, s=130,legend=legend,style='REP',**kwargs)
        sns.despine(left=1,bottom=1)
        if legend:sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        plt.axis('equal')
    plt.savefig(f'../figures/preprocessing/PCA/CHIP_{SAVE_PREFIX}_PCA.pdf', format="pdf", bbox_inches="tight")

def RNA_UMAP(DATA, CT_REG, CT_COL_DICT,SAVE_PREFIX):
    N_COMP=2
    UMAP_trasformer = umap.UMAP(n_neighbors=2,
                                min_dist=1,
                                n_components=N_COMP,
                                metric='euclidean',
                                densmap=True
                                )
    
    UMAP_cols = [f'UMAP{x}' for x in range(1, N_COMP + 1)]
    UMAP_DF = pd.DataFrame(UMAP_trasformer.fit_transform(DATA.transpose()),columns=UMAP_cols)
    UMAP_DF['CT'], UMAP_DF['REP'] = DATA.columns.str.extract(CT_REG), DATA.columns.str.extract(r'(_\d)')
    UMAP_DF['REP'] =                UMAP_DF['REP'].str.extract(r'(\d)')
    kwargs={'edgecolor':"grey", 'linewidth':1}
    plt.figure(figsize = (6,6))
    plt.suptitle(SAVE_PREFIX)
    x = sns.scatterplot(data = UMAP_DF, x='UMAP1', y='UMAP2', hue='CT',palette=CT_COL_DICT, s=130,**kwargs)
    sns.despine(left=1,bottom=1)
    sns.move_legend(x,loc="upper left", bbox_to_anchor=(1, 1))
    plt.axis('equal')
    plt.savefig(f'../figures/preprocessing/PCA/RNA_{SAVE_PREFIX}_UMAP.pdf', format="pdf", bbox_inches="tight")


def RNA_tSNE(DATA, CT_REG, CT_COL_DICT,SAVE_PREFIX):
    N_COMP=2
    tSNE = TSNE(perplexity=2)
    tSNE_col = [f'tSNE{x}' for x in range(1, N_COMP + 1)]
    tSNE_DF = pd.DataFrame(tSNE.fit_transform(DATA.transpose()),columns=tSNE_col)
    tSNE_DF['CT'], tSNE_DF['REP'] = DATA.columns.str.extract(CT_REG), DATA.columns.str.extract(r'(_\d)')
    tSNE_DF['REP'] =                tSNE_DF['REP'].str.extract(r'(\d)')
    kwargs={'edgecolor':"grey", 'linewidth':1}
    plt.figure(figsize = (6,6))
    plt.suptitle(SAVE_PREFIX)
    x = sns.scatterplot(data = tSNE_DF, x='tSNE1', y='tSNE2', hue='CT',palette=CT_COL_DICT, s=130,**kwargs)
    sns.despine(left=1,bottom=1)
    sns.move_legend(x,loc="upper left", bbox_to_anchor=(1, 1))
    plt.axis('equal')
    plt.savefig(f'../figures/preprocessing/PCA/RNA_{SAVE_PREFIX}_tSNE.pdf', format="pdf", bbox_inches="tight")

def CHIP_UMAP(DATA, HM_REG, MARKER_LIST, CT_REG, CT_COL_DICT,SAVE_PREFIX):
    N_COMP=2
    UMAP_trasformer = umap.UMAP(
                                n_neighbors=2,
                                #min_dist=1,
                                spread=4,
                                n_components=N_COMP,
                                #metric='canberra',
                                densmap=True
                                )
    
    UMAP_cols = [f'UMAP{x}' for x in range(1, N_COMP + 1)]
    UMAP_DF = pd.DataFrame(UMAP_trasformer.fit_transform(DATA.transpose()),columns=UMAP_cols)
    UMAP_DF['HM']    =              DATA.columns.str.extract(HM_REG)
    UMAP_DF['CT'], UMAP_DF['REP'] = DATA.columns.str.extract(CT_REG), DATA.columns.str.extract(r'(_\d)')
    UMAP_DF['REP'] =                UMAP_DF['REP'].str.extract(r'(\d)')
    plt.figure(figsize = (6,6))
    plt.suptitle(SAVE_PREFIX)
    kwargs={'edgecolor':"grey", 'linewidth':1}
    x = sns.scatterplot(data = UMAP_DF, x='UMAP1', y='UMAP2', hue='CT',palette=CT_COL_DICT, s=130, style='HM',markers=MARKER_LIST,**kwargs)
    sns.despine(left=1,bottom=1)
    sns.move_legend(x,loc="upper left", bbox_to_anchor=(1, 1))
    plt.axis('equal')
    plt.savefig(f'../figures/preprocessing/PCA/CHIP_{SAVE_PREFIX}_UMAP.pdf', format="pdf", bbox_inches="tight")

def CHIP_tSNE(DATA, HM_REG, MARKER_LIST, CT_REG, CT_COL_DICT,SAVE_PREFIX):
    N_COMP=2
    tSNE = TSNE(perplexity=2)
    tSNE_col = [f'tSNE{x}' for x in range(1, N_COMP + 1)]
    tSNE_DF = pd.DataFrame(tSNE.fit_transform(DATA.transpose()),columns=tSNE_col)
    tSNE_DF['HM']    =              DATA.columns.str.extract(HM_REG)
    tSNE_DF['CT'], tSNE_DF['REP'] = DATA.columns.str.extract(CT_REG), DATA.columns.str.extract(r'(_\d)')
    tSNE_DF['REP'] =                tSNE_DF['REP'].str.extract(r'(\d)')
    kwargs={'edgecolor':"grey", 'linewidth':1}
    plt.figure(figsize = (6,6))
    plt.suptitle(SAVE_PREFIX)
    x = sns.scatterplot(data = tSNE_DF, x='tSNE1', y='tSNE2', hue='CT',palette=CT_COL_DICT, s=130, style='HM',markers=MARKER_LIST,**kwargs)
    sns.despine(left=1,bottom=1)
    sns.move_legend(x,loc="upper left", bbox_to_anchor=(1, 1))
    plt.axis('equal')
    plt.savefig(f'../figures/preprocessing/PCA/CHIP_{SAVE_PREFIX}_tSNE.pdf', format="pdf", bbox_inches="tight")




def PREPROCESS_DATA(MAIN):
    """
    Preprocess the input DataFrame with various scaling methods.

    Parameters:
        DF (DataFrame): The input DataFrame.
        FEATURE_PREFIX (str): Partial string to match columns for preprocessing.

    Returns:
        dict: Dictionary where keys are preprocessing method names and values are the corresponding DataFrames.
    """

    # Select columns matching the FEATURE_PREFIX
    DF = MAIN.copy()

    # Log transformation
    LOG_DF = DF.copy()
    LOG_DF.iloc[:,:] = np.log10(LOG_DF.iloc[:,:])

    # Log MinMax scaler
    LOG_MINMAX_DF = LOG_DF.copy()
    LOG_MINMAX_DF.iloc[:,:] = MinMaxScaler().fit_transform(LOG_MINMAX_DF.iloc[:,:])
    
    # Log Robust Scaler
    LOG_ROBUST = LOG_DF.copy()
    SCALER = RobustScaler(with_centering=False, quantile_range=(25, 75))
    LOG_ROBUST.iloc[:,:] = SCALER.fit_transform(LOG_ROBUST.iloc[:,:])
    
    # Log StdScaler   
    LOG_STD_DF = LOG_DF.copy()
    LOG_STD_DF.iloc[:,:] = StandardScaler(with_mean=True).fit_transform(LOG_STD_DF.iloc[:,:])


    # StdScaler 
    STD_DF = DF.copy()
    STD_DF.iloc[:,:] = StandardScaler().fit_transform(STD_DF.iloc[:,:])
    #STD_DF.iloc[:,:] = np.log10(STD_DF.iloc[:,:].replace(0,0.01))

    # Quantile uniform
    QUN_DF = DF.copy()
    SCALER = QuantileTransformer(output_distribution='uniform')
    QUN_DF.iloc[:,:] = SCALER.fit_transform(QUN_DF.iloc[:,:])
    
    # Quantile normal
    QNORM_DF = DF.copy()
    SCALER = QuantileTransformer(output_distribution='normal')
    QNORM_DF.iloc[:,:] = SCALER.fit_transform(QNORM_DF.iloc[:,:])
    
    # Quantile normal + MinMax
    QNORM_MINMAX_DF = QNORM_DF.copy()
    SCALER = MinMaxScaler()
    QNORM_MINMAX_DF.iloc[:,:] = SCALER.fit_transform(QNORM_MINMAX_DF.iloc[:,:])
    
    # Normalizer
    NORM= DF.copy()
    NORM.iloc[:,:] = Normalizer().fit_transform(NORM.iloc[:,:])

    # Log Norm   
    LOG_NORM_DF = LOG_DF.copy()
    LOG_NORM_DF.iloc[:,:] = Normalizer().fit_transform(LOG_NORM_DF.iloc[:,:])

    # Define preprocessing methods and corresponding DataFrames
    DF_TRANSFORMED = {
        'original': DF,
        'log': LOG_DF,
        'log_MinMax': LOG_MINMAX_DF,
        'log_Robust': LOG_ROBUST,
        'log_StdScaler': LOG_STD_DF,
        'StdScaler': STD_DF,
        'QntUni': QUN_DF,
        'QntNorm': QNORM_DF,
        'QntNorm_MinMax': QNORM_MINMAX_DF,
        'Normalizer': NORM,
        'log_Normalizer':LOG_NORM_DF
    }
    
    
    return DF_TRANSFORMED




def plot_history(history=None,train_col='green',val_col='magenta'):
    """
    Plot training/validation loss and RMSE.

    Parameters:
        history (keras.callbacks.History): History object containing training/validation metrics.
        train_col (str): Color for training loss and RMSE.
        val_col (str): Color for validation loss and RMSE.
    """
    plt.plot(history.history['root_mean_squared_error']    ,linestyle='solid',  color=train_col, label = "train_RMSE")
    plt.plot(history.history['val_root_mean_squared_error'],linestyle='solid',  color=val_col, label = "val_RMSE")
    
    val_rmse = history.history['val_root_mean_squared_error'][-1]
    
    plt.title(f'Training history (Test. RMSE={round(val_rmse,3)})')
    plt.xlabel("Epoch #")
    plt.ylabel("RMSE")
    plt.legend()


def violins_error(R_DF,ERROR_COL,SET_COL_DICT,SAVEFIG,SAT=0.8,VIOLIN=True):
    plt.figure(figsize=(4,2))
    if VIOLIN:
        sns.violinplot(y='SET', x=ERROR_COL, data=R_DF,orient='h', hue='SET', palette=SET_COL_DICT, 
                        #common_norm=1,density_norm='count',
                        saturation=SAT, width=0.9,inner=None,linewidth=0, cut=0,alpha=0.7);
    sns.boxplot(y='SET', x=ERROR_COL, data=R_DF,orient='h', hue='SET', palette=SET_COL_DICT, 
                            width=0.2,linewidth=1,showcaps=1,
                            showfliers=0, 
                            medianprops={"color": "w", "linewidth": 1},
                            boxprops=dict(facecolor="black",alpha=0.4),
                            whiskerprops=dict(color="black", alpha=0.4),
                            flierprops = dict(marker='.',markerfacecolor='grey', markersize=1,alpha=0.5))
    sns.despine(left=1,bottom=1)
    plt.ylabel('')
    plt.savefig(SAVEFIG, format="png", bbox_inches="tight");


def get_color_list(FEATURE_NAMES,COL_DICT):
    color_list=[]
    for FEATURE in FEATURE_NAMES:
        for key, value in COL_DICT.items():
            if key in FEATURE:
                color_list.append(value)
    return color_list

def force_square(x,fw=8,fh=8):
    dx, dy = np.diff(x.ax_heatmap.transData.transform([[0, 1], [1, 0]]), axis=0).squeeze()
    if dx > dy:
        fh = fh * dx / dy
    else:
        fw = fw * dy / dx
    x.fig.set_size_inches(fw, fh)

def FC_heatmaps(DF,R_DF,GENE_TO_PLOT,FEATURE_PREF,HM_COL_DICT,DIR_FIG,fc_cmap='seismic_r',TITLE='',vmin=-5,vmax=5,H=0.5, W=1.75):
    DF = DF.loc[DF.index.intersection(GENE_TO_PLOT)]
    R_DF = R_DF.loc[R_DF.index.intersection(GENE_TO_PLOT)]
    # Reorder based on GENE_TO_PLOT
    DF = DF.reindex(GENE_TO_PLOT)
    R_DF = R_DF.reindex(GENE_TO_PLOT)
    FEAT_REG = '|'.join(FEATURE_PREF)
    R_FC = R_DF.filter(regex=FEAT_REG).filter(regex='FC')
    FC   = DF.filter(regex=FEAT_REG).filter(regex='FC')
    for X, title in zip([FC,R_FC], [f'{TITLE} Original FCs',f'{TITLE} Reconstructed FCs']):
        col_colors=pd.DataFrame({' ':get_color_list(X.columns,HM_COL_DICT)},index=X.columns)
        x = sns.clustermap(X,  cmap=fc_cmap,
                            center=0,vmin=vmin,vmax=vmax,
                            col_cluster=False,row_cluster=False, 
                            cbar_kws={ 'orientation':'horizontal'},
                            linecolor='white',
                            linewidth=0.1, 
                            col_colors=col_colors,
                            figsize=(2+( W * 3 ), (H * (X.shape[0]+3) )),
                            colors_ratio=(0.025,0.06*(6/X.shape[0])),
                            xticklabels=X.columns
                            );
        x.ax_heatmap.set_yticklabels(X.index, rotation=0)
        x.ax_heatmap.set_xticklabels(X.columns, size=8,rotation=60,horizontalalignment='right')
        x.ax_heatmap.set_ylabel('')
        #force_square(x,fw=W,fh=H)
        #cbar
        x.ax_cbar.set_position([0, 0.5, 0.1, 0.02])
        x.ax_cbar.set_title('Z-score(log2FC)')
        x.ax_cbar.tick_params(axis='y', length=10)
        plt.suptitle(title,size=20)
        plt.savefig(f'{DIR_FIG}FC_heatmaps_{TITLE}_{title}.pdf',bbox_inches="tight")

def X_heatmaps(DF,R_DF,GENE_TO_PLOT,FEATURE_PREF,SET_COL_DICT,HM_COL_DICT,CT_COL_DICT,DIR_FIG,cmap='mako',TITLE='', relative_range=False,vmin= -2.5, vmax= 2.5 , H=0.5, W=1.75):
    df = DF.loc[DF.index.intersection(GENE_TO_PLOT)]
    r_df = R_DF.loc[R_DF.index.intersection(GENE_TO_PLOT)]
    # Reorder based on GENE_TO_PLOT
    df = df.reindex(GENE_TO_PLOT)
    r_df = r_df.reindex(GENE_TO_PLOT)
    FEAT_REG = '|'.join(FEATURE_PREF)
    SET = r_df['SET']
    R_X = r_df.filter(regex=FEAT_REG).filter(regex='^(?!.*FC).*$')
    X   = df.filter(regex=FEAT_REG).filter(regex='^(?!.*FC).*$')
    assert (X.columns == R_X.columns).all()
    if relative_range:
        if X.min().min() < R_X.min().min(): vmin = X.min().min()
        else: vmin = R_X.min().min()
        if X.max().max() > R_X.max().max(): vmax = X.max().max()
        else: vmax = R_X.max().max()
    for X, title in zip([X,R_X], [f'{TITLE} Original',f'{TITLE} Reconstructed']):
        col_colors=pd.DataFrame({   ' ':get_color_list(X.columns,HM_COL_DICT),
                                    '  ':  get_color_list(X.columns,CT_COL_DICT),
                                    '   ':np.repeat('white', len(X.columns))},index=X.columns)
        row_colors= pd.DataFrame({'SET':get_color_list(SET, SET_COL_DICT)},index=X.index)

        x = sns.clustermap(X,  cmap=cmap,
                            center=None,vmin=vmin,vmax=vmax,
                            col_cluster=False,row_cluster=False, 
                            cbar_kws={ 'orientation':'horizontal'},
                            linecolor='white',
                            linewidth=0, 
                            col_colors=col_colors,
                            row_colors=row_colors,
                            figsize=(2+( W * 3 ), (H * (X.shape[0]+3) )),
                            colors_ratio=(0.025,0.06*(6/X.shape[0]))
                            );
        x.ax_heatmap.set_yticklabels(X.index, rotation=0)
        x.ax_heatmap.set_xticklabels('')
        x.ax_heatmap.set_ylabel('')
        #force_square(x,fw=W,fh=H)
        #cbar
        x.ax_cbar.set_position([0, 0.5, 0.1, 0.02])
        x.ax_cbar.set_title('Z-score(log(x))')
        x.ax_cbar.tick_params(axis='y', length=10)
        plt.suptitle(title,size=20)

        plt.savefig(f'{DIR_FIG}X_heatmaps_{TITLE}_{title}.pdf',bbox_inches="tight")
        
def X_clustermaps(DF,R_DF,GENE_TO_PLOT,FEATURE_PREF,SET_COL_DICT,HM_COL_DICT,CT_COL_DICT,DIR_FIG,cmap='mako',TITLE='', relative_range=False,vmin= -2.5, vmax= 2.5 , H=0.15, W=1.2):
    df = DF.loc[DF.index.intersection(GENE_TO_PLOT)]
    r_df = R_DF.loc[R_DF.index.intersection(GENE_TO_PLOT)]
    FEAT_REG = '|'.join(FEATURE_PREF)
    SET = r_df['SET']
    R_X = r_df.filter(regex=FEAT_REG).filter(regex='^(?!.*FC).*$')
    X   = df.filter(regex=FEAT_REG).filter(regex='^(?!.*FC).*$')
    assert (X.columns == R_X.columns).all()
    if relative_range:
        if X.min().min() < R_X.min().min(): vmin = X.min().min()
        else: vmin = R_X.min().min()
        if X.max().max() > R_X.max().max(): vmax = X.max().max()
        else: vmax = R_X.max().max()
    for df, title in zip([X,R_X], [f'{TITLE} Original',f'{TITLE} Reconstructed']):
        col_colors=pd.DataFrame({   ' ':get_color_list(df.columns,HM_COL_DICT),
                                    '  ':  get_color_list(df.columns,CT_COL_DICT),
                                    '   ':np.repeat('white', len(df.columns))},index=df.columns)
        row_colors= pd.DataFrame({'SET':get_color_list(SET, SET_COL_DICT)},index=df.index)

        x = sns.clustermap(df,  cmap=cmap,
                            center=None,vmin=vmin,vmax=vmax,
                            col_cluster=False, z_score=False,
                            
                            row_cluster=True,
                            metric='correlation',
                            method='average',
                            yticklabels=list(df.index),
                            
                            cbar_kws={ 'orientation':'horizontal'},
                            #cbar=False,
                            linecolor='white',
                            linewidth=0, 
                            col_colors=col_colors,
                            row_colors=row_colors,
                            figsize=(2+( W * 3 ), (H * (df.shape[0]+3) )),
                            colors_ratio=(0.025,0.06*(6/df.shape[0]))
                            );
        #x.ax_heatmap.set_yticklabels(labels=list(df.index), rotation=0, fontdict={'fontsize':4})
        plt.setp(x.ax_heatmap.yaxis.get_majorticklabels(), fontsize=8)

        x.ax_heatmap.set_xticklabels('')
        x.ax_heatmap.set_ylabel('')
        #force_square(x,fw=W,fh=H)
        #cbar
        # x.ax_cbar.set_position([0, 0.5, 0.1, 0.02])
        # x.ax_cbar.set_title('Z-score(log(x))')
        # x.ax_cbar.tick_params(axis='y', length=10)
        plt.suptitle(title,size=20)

        plt.savefig(f'{DIR_FIG}X_clustermaps_{TITLE}_{title}.pdf',bbox_inches="tight")
        

def Sc_selected(R_DF,GENE_TO_PLOT,ERROR_COL,TITLE,DIR_FIG):
    X = R_DF.loc[R_DF.index.intersection(GENE_TO_PLOT)]
    plt.figure(figsize=(3,1.5))
    sns.violinplot(x = R_DF[ERROR_COL],width=0.9,inner=None,linewidth=0,color='#D3DEDB')
    sns.boxplot(x=R_DF[ERROR_COL],orient='h',  
                                width=0.1,linewidth=1,showcaps=1,
                                medianprops={"color": "w", "linewidth": 1},
                                boxprops=dict(facecolor="black",alpha=0.4),
                                whiskerprops=dict(color="black", alpha=0.4),
                                flierprops = dict(marker='.',markerfacecolor='grey', markersize=1,alpha=0.5))
    sns.stripplot(data=X,x = ERROR_COL,hue='SET',legend=0,palette=SET_COL_DICT, size=7)
    sns.despine(left=1,bottom=1)
    #plt.title(TITLE)
    plt.savefig(f'{DIR_FIG}{ERROR_COL}_{TITLE}.pdf', format="pdf", bbox_inches="tight");







############################################################################################################
#------------------------------------------- LATENT SPACE ANALYSIS and CLUSTERING ------
############################################################################################################








def method_umap2D(DF, METHOD):
    # Extract the relevant columns for UMAP
    data_for_umap = DF.filter(regex=rf'(^{METHOD}\d)')
    umap_model = umap.UMAP(n_components=2,
                            random_state=42,
                            #min_dist=0.1,
                            #n_neighbors=30,
                            )
    umap_embedding = umap_model.fit_transform(data_for_umap)
    # Add the UMAP embedding to the dataframe
    DF[f"{METHOD}_UMAP1"] = umap_embedding[:,0]
    DF[f"{METHOD}_UMAP2"] = umap_embedding[:,1]
    return DF


def map_genes(gene, group_dict):
    for group, genes in group_dict.items():
        if gene in genes:
            return group
    return 'other'


def umap_2d_discrete(DF, METHOD, GROUP_DICT, NAME, DIR, COL_DICT=None, 
                                SIZE=75,
                                cat_order=None, 
                                REST_COL='grey', REST_SIZE=2, REST_ALPHA=0.3,MARKER='X'):
    
    DF[NAME] = DF.index.map(lambda gene: map_genes(gene, GROUP_DICT))
    
    if cat_order is None:
        cat_order = list(GROUP_DICT.keys())
        
    COL_DICT['other'] = REST_COL
    
    plt.figure(figsize=(8, 8))
    
    # Plot 'other' genes
    df_other = DF[DF[NAME] == 'other']
    plt.scatter(x=df_other[f'{METHOD}_UMAP1'], y=df_other[f'{METHOD}_UMAP2'],
                c=COL_DICT['other'], s=REST_SIZE, alpha=REST_ALPHA, label='Other Genes', linewidths=0)
    
    # Plot grouped genes
    for category in cat_order:
        df_highlight = DF[DF[NAME] == category]
        plt.scatter(x=df_highlight[f'{METHOD}_UMAP1'], y=df_highlight[f'{METHOD}_UMAP2'],
                    c=COL_DICT[category],  label=category, 
                    #edgecolors='#3B3C36',
                    edgecolors='white',
                    alpha=1,linewidths=0.5 , s=SIZE, marker=MARKER)
    
    # Add labels and title
    sub = subscript_get(f'{METHOD}')
    plt.xlabel(f'UMAP1 {sub}')
    plt.ylabel(f'UMAP2 {sub}')
    plt.title(NAME)
    #plt.legend(title='',)
    plt.grid(False)
    sns.despine()
    
    # Hide tick labels
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f"{DIR}UMAP2D_{METHOD}_{NAME}.png", dpi=300)
    

from mpl_toolkits.axes_grid1 import make_axes_locatable


def umap_2d_continuous(DF, METHOD, VALUE_COLUMN, DIR, SIZE=4, COLORMAP='magma', VMIN=None, VMAX=None, PERCENTILES=None):
    """
    Plots a 2D UMAP scatter plot with points colored by continuous values.

    Parameters:
    - DF: DataFrame containing the data.
    - METHOD: String representing the dimensionality reduction method (e.g., 'VAE').
    - VALUE_COLUMN: String representing the column with continuous values for coloring.
    - DIR: String representing the directory to save the plot.
    - SIZE: Integer for the size of the main points.
    - COLORMAP: String representing the colormap to use for continuous values.
    """
    if PERCENTILES:
            VMIN = np.percentile(DF[VALUE_COLUMN], PERCENTILES[0])
            VMAX = np.percentile(DF[VALUE_COLUMN], PERCENTILES[1])
    plt.figure(figsize=(6, 6))
    plt.suptitle(VALUE_COLUMN)
    # Create the main plot
    ax = plt.gca()
    scatter = ax.scatter(x=DF[f'{METHOD}_UMAP1'], y=DF[f'{METHOD}_UMAP2'],
                            c=DF[VALUE_COLUMN], s=SIZE, cmap=COLORMAP, alpha=0.8,  linewidths=0,
                            vmin=VMIN, vmax=VMAX)

    # Create a new axis for the colorbar above the main plot
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('top', size='1%', pad=0.05)

    # Create the colorbar
    cbar = plt.colorbar(scatter, cax=cax, orientation='horizontal')
    #cbar.set_label(VALUE_COLUMN, fontsize=18)  # Bigger label

    # Set colorbar ticks to show only the extremes and the center
    ticks = [VMIN,  VMAX]
    if VMIN <0 and VMAX > 0:
        ticks = [VMIN, 0,  VMAX]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f'{tick:.1f}' for tick in ticks])  # Format tick labels
    cbar.ax.tick_params(labelsize=12)  # Smaller tick labels
    cax.xaxis.set_ticks_position('top')
    cax.xaxis.set_label_position('top')

    # Add labels and title
    sub = subscript_get(f'{METHOD}')
    ax.set_xlabel(f'UMAP1 {sub}')
    ax.set_ylabel(f'UMAP2 {sub}')
    ax.set_title('')  # Remove title as the colorbar is above the plot
    ax.grid(False)
    sns.despine()

    # Hide tick labels
    ax.set_xticks([])
    ax.set_yticks([])

    # Save the plot
    plt.savefig(f"{DIR}UMAP2D_{METHOD}_{VALUE_COLUMN}.png", dpi=300)
    plt.show()


def umap_2d_continuous_grid(DF, METHOD, VALUE_COLS, DIR, NAME, SIZE=4, 
                            COLORMAP='magma', VMIN=None, VMAX=None, 
                            COLORMAP_DICT=None,PERCENTILES=None, CENTER=False):
    """
    Plots a grid of 2D UMAP scatter plots with points colored by continuous values.

    Parameters:
    - DF: DataFrame containing the data.
    - METHOD: String representing the dimensionality reduction method (e.g., 'VAE').
    - VALUE_COLS: List of strings representing the columns with continuous values for coloring.
    - DIR: String representing the directory to save the plot.
    - SIZE: Integer for the size of the main points.
    - COLORMAP: String representing the colormap to use for continuous values.
    """
    
    num_plots = len(VALUE_COLS)
    grid_size = math.ceil(math.sqrt(num_plots))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 6, grid_size * 6))
    axes = axes.flatten()
    sub = subscript_get('6D')
    sub2 = subscript_get('2D')
    plt.suptitle(f'{METHOD}{sub} --> UMAP{sub2}')

    for i, VALUE_COLUMN in enumerate(VALUE_COLS):
        #
        if COLORMAP_DICT is not None:
            for key, value in COLORMAP_DICT.items():
                if key in VALUE_COLUMN:
                    COLORMAP = value
                    break
        # Ensure the values are numeric
        try:
            values = pd.to_numeric(DF[VALUE_COLUMN])
        except ValueError:
            raise ValueError(f"Column {VALUE_COLUMN} contains non-numeric values.")
        
        if PERCENTILES:
            VMIN = np.percentile(values, PERCENTILES[0])
            VMAX = np.percentile(values, PERCENTILES[1])
        if CENTER:
            # Center the colormap at 0 if VMIN is negative and VMAX is positive
            abs_max = max(abs(VMIN), abs(VMAX))
            VMIN = -abs_max
            VMAX = abs_max
        ax = axes[i]
        scatter = ax.scatter(x=DF[f'{METHOD}_UMAP1'], y=DF[f'{METHOD}_UMAP2'],
                            c=values, s=SIZE, cmap=COLORMAP, alpha=0.8, linewidths=0,
                            vmin= VMIN, vmax = VMAX)

        # Create a new axis for the colorbar above the main plot
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('top', size='1%', pad=0.05)

        # Create the colorbar
        cbar = plt.colorbar(scatter, cax=cax, orientation='horizontal')

        # Set colorbar ticks to show only the extremes and the center
        ticks = [VMIN, VMAX]
        if VMIN < 0 and VMAX > 0:
            ticks = [VMIN, 0, VMAX]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f'{tick:.1f}' for tick in ticks])  # Format tick labels
        cbar.ax.tick_params(labelsize=12)  # Smaller tick labels
        cbar.set_label(VALUE_COLUMN, fontsize=16)  # Bigger label
        cax.xaxis.set_ticks_position('top')
        cax.xaxis.set_label_position('top')

        # Add labels and title
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title('')
        ax.grid(False)
        sns.despine()

        # Hide tick labels
        ax.set_xticks([])
        ax.set_yticks([])

    # Remove unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout
    plt.tight_layout()
    plt.savefig(f"{DIR}UMAP2D_{METHOD}_grid_{NAME}.png", dpi=150)
    
    
    
def assign_palette(PALETTE, DICT):    
    palette = sns.color_palette(PALETTE, len(GENE_SETS)).as_hex()
    COL_DICT = {}
    for key, color in zip(DICT.keys(), palette):
        COL_DICT[key] = color
    return COL_DICT


##########################################################################################
#------------------------------------------------ CLUSTERING -----------------------------
#########################################################################################


from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

def gmm_silhouette_score(estimator, X):
    labels = estimator.predict(X)
    if len(set(labels)) > 1:  # Check if there is more than one cluster
        return silhouette_score(X, labels)
    else:
        return -1  # Invalid silhouette score when only one cluster is found

def GMM_grid_search(MAIN_DF, ORIGINAL_DF, METHODS, max_k=10, verbose=False):
    DF_DICT = {}
    for METHOD in METHODS:
        param_grid = {
            "n_components": range(10, max_k + 1, 5),
            "covariance_type": ["spherical", "tied", "diag", "full"],}
        results = []
        if METHOD == 'original':
            X = ORIGINAL_DF
        else:
            X = MAIN_DF.filter(regex=rf'(^{METHOD}\d)')
            print(list(X.columns))
        
        for n_components in param_grid['n_components']:
            for cov_type in param_grid['covariance_type']:
                print(f'Running on  {METHOD}     k= {n_components}   cov= {cov_type}')
                gmm = GaussianMixture(n_components=n_components, 
                                        covariance_type=cov_type, 
                                        max_iter=200,
                                        n_init=5,
                                        init_params='k-means++',)
                gmm.fit(X)
                bic = gmm.bic(X)
                aic = gmm.aic(X)
                silhouette = gmm_silhouette_score(gmm, X)
                results.append({
                    'n_components': n_components,
                    'covariance_type': cov_type,
                    '-BIC': -bic,
                    '-AIC': -aic,
                    'Silhouette': silhouette
                })
        
        DF_DICT[METHOD] = pd.DataFrame(results)
    return DF_DICT

def plot_scores(df,METHOD):
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    scores = ['-BIC', '-AIC', 'Silhouette']
    colors = sns.color_palette('husl', n_colors=len(df['covariance_type'].unique()))
    plt.suptitle(f'{METHOD} GMM GridSearch Scores')
    for i, score in enumerate(scores):
        if i==0:legend=True
        else:legend=False
        ax = axes[i]
        sns.lineplot(data=df, x='n_components', y=score, hue='covariance_type', palette=colors, ax=ax,legend=legend)
        ax.set_ylabel(score)
        ax.set_xlabel('# of clusters')
        #ax.set_title(f'{score}')
    sns.despine()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{DIR_FIG}GMM_opt_{METHOD}.pdf', format="pdf", bbox_inches="tight");



def GMM_train(MAIN_DF,ORIGINAL_DF, METHODS, best_k_dict,best_cov_dict):
    for METHOD in METHODS:
        model = GaussianMixture(n_components=best_k_dict[METHOD],
                                covariance_type=best_cov_dict[METHOD],
                                random_state=42,
                                max_iter=100,
                                n_init = 10,
                                )
        if METHOD == 'original': 
            X = ORIGINAL_DF
        else: 
            X= MAIN_DF.filter(regex=rf'(^{METHOD}\d)')
        model.fit(X)
        MAIN_DF[f'GMM_{METHOD}_{best_k_dict[METHOD]}']  = model.predict(X)
        

def umap_2d_clusters(DF, METHOD, best_k_dict, DIR, CMAP='Spectral', SIZE=4, REST_SIZE=2, REST_ALPHA=0.8):
    """
    Plots a 2D UMAP scatter plot with points colored by GMM cluster labels.

    Parameters:
    - DF: DataFrame containing the data.
    - METHOD: String representing the dimensionality reduction method (e.g., 'VAE').
    - best_k_dict: Dictionary containing the best number of clusters (k) for each method.
    - DIR: String representing the directory to save the plot.
    - CMAP: String representing the colormap to use for cluster colors.
    - SIZE: Integer for the size of the main points.
    - REST_SIZE: Integer for the size of the other points.
    - REST_ALPHA: Float for the alpha value of the other points.
    """

    GMM_LABELS = f'GMM_{METHOD}_{best_k_dict[METHOD]}'
    DF[GMM_LABELS] = DF[GMM_LABELS].astype(str)
    sorted_labels = sorted(DF[GMM_LABELS].unique(), key=lambda x: int(x))
    DF[GMM_LABELS] = pd.Categorical(DF[GMM_LABELS], categories=sorted_labels, ordered=True)
    
    RAINBOW = sns.color_palette(CMAP, n_colors=best_k_dict[METHOD]).as_hex()
    COL_DICT = dict(zip(sorted_labels, RAINBOW))
    
    if METHOD == 'original':
        PLOT_METHOD = 'VAE'
    else:
        PLOT_METHOD = METHOD
    
    plt.figure(figsize=(8, 8))
    
    # Plot all points with their respective cluster colors
    for label in sorted_labels:
        df_cluster = DF[DF[GMM_LABELS] == label]
        plt.scatter(x=df_cluster[f'{PLOT_METHOD}_UMAP1'], y=df_cluster[f'{PLOT_METHOD}_UMAP2'],
                    c=COL_DICT[label], s=SIZE, alpha=REST_ALPHA, label=f'Cluster {label}', linewidths=0)
    
    # Add labels and title
    sub = subscript_get(f'{METHOD}')
    plt.xlabel(f'UMAP1 {sub}')
    plt.ylabel(f'UMAP2 {sub}')
    plt.title(f'{METHOD} Clusters k={best_k_dict[METHOD]}')
    plt.grid(False)
    sns.despine()
    
    # Hide tick labels
    plt.xticks([])
    plt.yticks([])

    # Add legend
    plt.legend(title='Clusters', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Save the plot
    plt.savefig(f"{DIR}UMAP2D_{METHOD}_k_{best_k_dict[METHOD]}.png", dpi=300, bbox_inches='tight')
    plt.show()







def plot_boxplots_clusters(df,CLUSTER_COL, FEATURE_PREFIXES, HM_COL_DICT=None,X_LINE=0, TITLE=None, X_LAB = 'Z-score'):
    # Get unique GMM_LABELS
    unique_labels = df[CLUSTER_COL].sort_values().unique()
    num_labels = len(unique_labels)
    VMIN,VMAX= df[FEATURE_PREFIXES].min().min(), df[FEATURE_PREFIXES].max().max()
    #
    MAPPED_COL=None
    if HM_COL_DICT != None:
        MAPPED_COL = list(map(lambda x: next((v for k, v in HM_COL_DICT.items() if k in x), None), FEATURE_PREFIXES))
    # Set up subplots
    if num_labels<6:num_labels=6
    fig, axes = plt.subplots(num_labels//5+1, 5, figsize=(11, 4 + (num_labels//5)*len(FEATURE_PREFIXES)*0.2))
    #
    plt.suptitle(TITLE, fontsize=20,y=1.02)
    # Iterate over each unique value of GMM_LABELS
    for idx, label in enumerate(unique_labels):
        # Subset the dataframe for the current label
        subset_df = df[df[CLUSTER_COL] == label]
        num_genes = subset_df.shape[0]
        # Melt the dataframe to long format for seaborn
        melted_df = subset_df.melt(id_vars=CLUSTER_COL, value_vars=FEATURE_PREFIXES, var_name='HM')
        # Plot the boxplot
        col_index = idx % 5
        row_index = idx // 5

        sns.violinplot(ax=axes[row_index,col_index], y='HM', x='value', data=melted_df, hue='HM',palette=MAPPED_COL,saturation=0.8,
                        #common_norm=1,density_norm='area', 
                        width=0.7,inner=None,linewidth=0, cut=1);
        plt.setp(axes[row_index,col_index].collections, alpha=.8)


        sns.boxplot(ax=axes[row_index,col_index], y='HM', x='value', data=melted_df, hue='HM',palette=MAPPED_COL,
                        width=0.3,linewidth=1,showcaps=1,
                        medianprops={"color": "w", "linewidth": 1},
                        boxprops=dict(facecolor="black",alpha=0.4),
                        whiskerprops=dict(color="black", alpha=0.5),
                        flierprops = dict(marker='.',markerfacecolor='grey', markersize=1, alpha=0.5))
        # Set title and labels
        axes[row_index,col_index].set_title(f'C{label} ({num_genes})',size=15)
        axes[row_index,col_index].set_ylabel('')  
        axes[row_index,col_index].axvline(x=X_LINE, color='black', linestyle='-',linewidth=0.5)

        # Set y limits equal for all plots
        axes[row_index,col_index].set_xlim(VMIN,VMAX)  # Adjust the limits as per your data range
        if row_index == (num_labels//5)-1:
            axes[row_index,col_index].set_xlabel(X_LAB,fontsize=8)
        else:        
            #axes[row_index,col_index].set_xticklabels('')
            axes[row_index,col_index].set_xlabel('')
        # Remove y-axis label for all but the first plot
        if col_index != 0: axes[row_index,col_index].set_yticklabels('')
        
        # Change the size of xticklabels
        for tick in axes[row_index, col_index].get_xticklabels():
            tick.set_fontsize(8)  # Change xtick label size
    idx+=1
    col_index+=1
    row_index = idx // 5
    plt.tight_layout()
