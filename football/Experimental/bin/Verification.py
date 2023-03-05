#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 19:46:29 2022

@author: robertmegnia
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import r2_score
import numpy as np
import os
import warnings
basedir = os.path.dirname(os.path.abspath(__file__))
datadir = f"{basedir}/../data"
warnings.simplefilter("ignore")
os.chdir(f"{basedir}/../..")
def mse(a, b):
    x = round(np.mean((a - b) ** 2), 2)
    return x


def mae(a, b):
    rms = round(np.mean(np.abs(a - b)), 2)
    return rms

def scatterPlot():
    fig = plt.figure()
    y_pred = self.pred
    y_test = self.y_test
    m = self.method
    pos = self.pos
    ms = mse(y_pred, y_test)
    ma = mae(y_pred, y_test)
    r2 = r2_score(y_test, y_pred)
    r2 = np.round(r2, 2)
    plt.scatter(y_pred, y_test, color="gray")
    plt.title(f"{m} {pos} R2={r2} MSE={ms} MAE={ma}")
    plt.xlabel("projected")
    plt.ylabel("actual")
    plt.xlim(0, np.nanmax(y_test))
    plt.ylim(0, np.nanmax(y_test))
    sns.regplot(y_pred, y_test, scatter=True, order=1)
    plt.show()
    
def weeklyVerificationPlots(week,season):
    fig,ax=plt.subplots(1)
    df=pd.read_csv(
        f'{datadir}/Projections/{2022}/WeeklyProjections/{season}_Week{week}_Projections_verified.csv')
    df=df[df.wProjection>5]

    df.rename({'wProjection':"Rob's Proj",
               'FP_Proj':'FantasyPros',
               'FP_Proj2':'Fantasy Pros Expert Consensus',
               'RG_projection':'RotoGrinders',
               'GI':'GridIronAI',
               "NicksAgg":"Nick's Proj"},axis=1,inplace=True)
    qb_sources=["Rob's Proj",
                "FantasyPros",
                "Fantasy Pros Expert Consensus",
                "RotoGrinders",
                "GridIronAI",
                "Nick's Proj"]
    qb=df
    qb=qb[qb[qb_sources].isna().any(axis=1)==False]
    for source in qb_sources:
        qb[f'{source} mae']=qb.apply(lambda x: mae(x.DKPts,x[source]),axis=1)
    mae_df = qb.mean()[qb[[a+' mae' for a in qb_sources]].columns]
    mae_df.name='MAE'
    mae_df.sort_values(inplace=True)
    # mae_df=mae_df.to_frame().sort_values(by='MAE')
    ax = mae_df.plot.bar(rot=45)
    low=round(mae_df.min()-0.5,0)
    high=round(mae_df.max()+0.5,0)
    ax.set_ylim(low,high)
    ax.set_ylabel('Mean Absolute Error',fontsize=18)
    ax.set_yticks(np.arange(low,high,0.2))
    ax.set_title(f'Week {week} Verification (DraftKings) N={len(qb)}',fontsize=18)
    ax.figure.set_figheight(10)
    ax.figure.set_figwidth(20)
    ax.figure.savefig('/Users/robertmegnia//Desktop/Verification.png',dpi=200)
    plt.show()
    