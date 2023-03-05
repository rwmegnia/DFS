#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 11:55:53 2022

@author: robertmegnia
"""
import os
import pandas as pd
import numpy as np
import pickle
basedir = os.path.dirname(os.path.abspath(__file__))
datadir = f'{basedir}/../../data'
projdir=f'{datadir}/Projections'
etcdir=f'{basedir}/../../etc'
os.chdir(f'{basedir}/../')
from ResearchProjections.config.MLModel_config import KnownFeatures
from Models.ModelDict import *
def getScaledProjection(df):
    for pos in df.position.unique():
        model_dict=ModelRanksDict[pos]
        for method in model_dict.keys():
            model=model_dict[method]
            df.loc[df.position==pos,f'ScaledProj_{method}']=model.predict(df.loc[df.position==pos,['ConsensusRank']+KnownFeatures])
        df['ScaledProj']=df[['ScaledProj_'+method for method in model_dict.keys()]].mean(axis=1)
    df.drop(['ScaledProj_'+method for method in model_dict.keys()],axis=1,inplace=True)
    return df

def getConsensusRanking(df):
    '''
    Rank players with each of the follow metrics and take the average
    to get a consensus ranking
    
    Projection
    TopDown
    Stochastic
    ML
    DC_proj
    PMMRank
    UpsideProb
    LeverageScore

    '''
    metrics=['Projection','TopDown','Stochastic','ML','DC_proj','PMM','UpsideProb']
    for metric in metrics:
        df[f'{metric}Rank']=df.groupby(['position','week','season'])[metric].apply(lambda x: x.rank(ascending=False,method='min'))
    df['Consensus']=df[[metric+'Rank' for metric in metrics]].mean(axis=1)
    df['ConsensusRank']=df.groupby(['position','week','season']).Consensus.apply(lambda x: x.rank(method='min'))    
    df['Rank']=df.groupby(['position','week','season']).DKPts.apply(lambda x: x.rank(ascending=False,method='min'))    
    df.drop(['Consensus']+[metric+'Rank' for metric in metrics],axis=1,inplace=True)
    df=getScaledProjection(df)
    return df