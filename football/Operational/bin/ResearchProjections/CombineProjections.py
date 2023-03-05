#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 17:03:15 2022

@author: robertmegnia

Make a final prediction by combining the following

Stochastic projections
ML Projections
Top Down Projections
DepthChart Projections
"""
import os
import pandas as pd
import numpy as np
import pickle
from Utils import *
basedir = os.path.dirname(os.path.abspath(__file__))
datadir = f'{basedir}/../../data'
projdir=f'{datadir}/Projections'
etcdir=f'{basedir}/../../etc'

#%% 
start_week=1
start_season=2022
end_season=2022
proj_frames=[]
for season in range(start_season,end_season+1):
    print(season)
    if season==2020:
        end_week=17
    else:
        end_week=18
    for week in range(5,10):
        print(week)
        ml=pd.read_csv(f'{projdir}/{season}/ML/{season}_Week{week}_MLProjections2.csv')
        top_down=pd.read_csv(f'{projdir}/{season}/TopDown/{season}_Week{week}_TopDownProjections.csv')
        dc=pd.read_csv(f'{projdir}/TeamDepthChartProjections/{season}_Week{week}_TeamDepthChartProjections.csv')
        ml=ml.merge(top_down[['gsis_id','week','season','TopDown']],on=['gsis_id','season','week'],how='left')
        ml.drop(['Stochastic','Floor','Ceiling','UpsideProb','RosterPercent'],axis=1,errors='ignore',inplace=True)
        rookies=pd.read_csv(f'{projdir}/{season}/Rookies/{season}_Week{week}_RookieProjections.csv')
        stochastic=pd.read_csv(f'{projdir}/{season}/Stochastic/{season}_Week{week}_StochasticProjections.csv')
        ml=ml.merge(stochastic[['Stochastic','Median','Floor','Ceiling','UpsideProb','UpsideScore','gsis_id','week','season']],on=['gsis_id','week','season'],how='left')
        if len(rookies)!=0:
            rookies=rookies[[c for c in rookies.columns if c in ml.columns]]
            projections=pd.concat([ml,rookies]).reset_index(drop=True)
            projections.loc[projections.TopDown.isna()==True,'TopDown']=projections.loc[projections.TopDown.isna()==True][['Stochastic','ML','PMM']].mean(axis=1)
            projections=projections[projections.salary.isna()==False]
            projections['depth_team']=projections.groupby(['team','position']).salary.apply(lambda x: x.rank(ascending=False,method='first')).astype(int)
            projections.drop(projections[(projections.position=='QB')&(projections.depth_team!=1)].index,inplace=True)
            projections=pd.concat([projections.groupby('gsis_id').apply(lambda x: merge_dc_projections(x,dc))])
            projections.loc[projections.DC_proj.isna()==True,'DC_proj']= projections.loc[projections.DC_proj.isna()==True][['Stochastic','ML','PMM']].mean(axis=1)
            projections['Projection']=projections[['Median','ML','TopDown','DC_proj','PMM']].mean(axis=1)
            projections=getOwnership(projections)
            projections=getImpliedOwnership(projections)
            # projections=getConsensusRanking(projections)
            projections.to_csv(f'{projdir}/{season}/All/{season}_Week{week}_Projections.csv',index=False)
            proj_frames.append(projections)
        else:
            ml.loc[ml.TopDown.isna()==True,'TopDown']=ml.loc[ml.TopDown.isna()==True][['Stochastic','ML','PMM']].mean(axis=1)            
            ml.depth_team=ml.depth_team.astype(int)
            ml=pd.concat([ml.groupby('gsis_id').apply(lambda x: merge_dc_projections(x,dc))])
            ml.loc[ml.DC_proj.isna()==True,'DC_proj']=ml.loc[ml.DC_proj.isna()==True][['Stochastic','ML','PMM']].mean(axis=1)
            ml['Projection']=ml[['Median','ML','TopDown','DC_proj','PMM']].mean(axis=1)
            ml=getOwnership(ml)
            ml=getImpliedOwnership(ml)
            # ml=getConsensusRanking(ml)
            ml.to_csv(f'{projdir}/{season}/All/{season}_Week{week}_Projections.csv',index=False)
            proj_frames.append(ml)


        

        