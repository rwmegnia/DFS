#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 13:58:43 2022

@author: robertmegnia
"""
import pandas as pd
import numpy as np
df=pd.read_csv('/Volumes/XDrive/DFS/football/Experimental/data/game_logs/2022/2022_Offense_GameLogs.csv')
# Player Stats Dicts
QB_STATS={'pass_yards':np.sum,
 'pass_air_yards':np.sum,
 'pass_td':np.sum,
 'pass_att':np.sum,
 'pass_cmp':np.sum,
 'int':np.sum,
 'sacks':np.sum,
 'qb_hit':np.sum,
 'rush_yards':np.sum,
 'rush_td':np.sum,
 'rush_att':np.sum,
 'rush_redzone_looks':np.sum,
 'rush_value':np.sum,
 'pass_yds_per_att':np.mean,
 'rush_yds_per_att':np.mean,
 'passer_rating':np.mean}

RB_STATS={'rush_yards':np.mean,
          'rush_td':np.sum,
          'rush_att':np.mean,
          'rush_share':np.mean,
          'rush_redzone_looks':np.sum,
          'rush_value':np.mean,
          'rec_yards':np.mean,
          'rec_air_yards':np.mean,
          'yac':np.sum,
          'rec_td':np.sum,
          'targets':np.mean,
          'rec':np.mean,
          'target_share':np.mean,
          'wopr':np.mean,
          'target_value':np.mean,
          'air_yards_share':np.mean,
          'rec_redzone_looks':np.sum,
          'Usage':np.mean,
          'HVU':np.mean,
          'rush_yds_per_att':np.mean,
          'PPO':np.mean,
          'HV_PPO':np.mean,
          'rush_yards_share':np.mean,
          'offensive_snapcount_percentage':np.mean
          }
WR_STATS={
          'rec_yards':np.mean,
          'rec_air_yards':np.mean,
          'adot':np.mean,
          'yac':np.sum,
          'rec_td':np.sum,
          'targets':np.mean,
          'rec':np.mean,
          'target_share':np.mean,
          'wopr':np.mean,
          'target_value':np.mean,
          'air_yards_share':np.mean,
          'rec_redzone_looks':np.sum,
          'PPO':np.mean,
          'rec_yards_share':np.mean,
          'exDKPts':np.mean,
          'poe':np.mean,
          'offensive_snapcount_percentage':np.mean
          }

# Def Stat Sticks
QB_STATS_allowed={'pass_yards':np.sum,
                 'pass_air_yards':np.sum,
                 'pass_td':np.sum,
                 'int':np.sum,
                 'sacks':np.sum,
                 'qb_hit':np.sum,
                 'pass_yds_per_att':np.mean,
                 'passer_rating':np.mean}

RB_STATS_allowed={'rush_yards':np.mean,
          'rush_td':np.sum,
          'rec_yards':np.mean,
          'yac':np.sum,
          'rec_td':np.sum,
          'rec':np.mean,
          'rush_yds_per_att':np.mean,
          'PPO':np.mean,
          }
WR_STATS_allowed={
          'rec_yards':np.mean,
          'rec_air_yards':np.mean,
          'adot':np.mean,
          'yac':np.sum,
          'rec_td':np.sum,
          'targets':np.mean,
          'rec':np.mean,
          'PPO':np.mean,
          'exDKPts':np.mean,
          'poe':np.mean,
          }
qbs=df[(df.position=='QB')&(df.depth_team==1)].groupby(['full_name','gsis_id']).agg(QB_STATS)
for column in qbs.columns:
    qbs[f'{column}_rank']=qbs[column].rank(ascending=False,method='min')
    
rbs=df[df.position=='RB'].groupby(['full_name','gsis_id']).agg(RB_STATS)
for column in rbs.columns:
    rbs[f'{column}_rank']=rbs[column].rank(ascending=False,method='min')
    
wrs=df[df.position=='WR'].groupby(['full_name','gsis_id']).agg(WR_STATS)
for columns in wrs.columns:
    wrs[f'{column}_rank']=wrs[column].rank(ascending=False,method='min')

tes=df[df.position=='TE'].groupby(['full_name','gsis_id']).agg(WR_STATS)
for columns in tes.columns:
    tes[f'{column}_rank']=tes[column].rank(ascending=False,method='min')
    
    
qbs_def=df[(df.position=='QB')&(df.depth_team==1)].groupby('opp').agg(QB_STATS_allowed)
for column in qbs_def.columns:
    qbs_def[f'{column}_rank']=qbs_def[column].rank(ascending=False)