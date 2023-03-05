#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 19:08:41 2022

@author: robertmegnia
"""

import os
import pandas as pd
import warnings
basedir = os.path.dirname(os.path.abspath(__file__))
etcdir = f'{basedir}/../../etc'
datadir= f'{basedir}/../../data'
warnings.simplefilter('ignore')
os.chdir(f'{basedir}/../..')
share_projections=[]
team_projections=[]
for season in range(2022,2023):
    for week in range(1,10):
        if (week==18)&(season==2020):
            continue
        share_df=pd.read_csv(f'{datadir}/Projections/PlayerShareProjections/{season}_Week{week}_PlayerShareProjections2.csv')
        share_df=share_df[['full_name','position','gsis_id','week','season','team','rushing_DKPts_share','receiving_DKPts_share','DKPts']] 
        share_df['DepthChart']=share_df.groupby(['position','team','week','season']).DKPts.apply(lambda x: x.rank(ascending=False,method='min'))
        share_df.loc[(share_df.position=='QB')&(share_df.DepthChart==1),'proj_passing_DKPts_share']=1.
        share_df.loc[(share_df.position!='QB')|(share_df.DepthChart!=1),'proj_passing_DKPts_share']=0
        team_df=pd.read_csv(f'{datadir}/Projections/TeamProjections/{season}_Week{week}_TeamProjections2.csv')
        top_down=share_df.merge(team_df[['team_fpts',
                                        'pass_attempt',
                                        'rush_attempt',
                                        'passing_fpts',
                                        'rushing_fpts','receiving_fpts','game_id','week','season','opp','team']]
                                        ,on=['week','season','team'],how='left')
        top_down['TopDown']=(top_down.proj_passing_DKPts_share*top_down.passing_fpts)+(top_down.rushing_DKPts_share*top_down.rushing_fpts)+(top_down.receiving_DKPts_share*top_down.receiving_fpts)
        top_down.to_csv(f'{datadir}/Projections/{season}/TopDown/{season}_Week{week}_TopDownProjections2.csv',index=False)
        