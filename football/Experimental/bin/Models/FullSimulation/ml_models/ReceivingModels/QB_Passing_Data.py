#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 11:32:23 2023

@author: robertmegnia
"""

import pandas as pd
import numpy as np
def getPasserRating(cmp, td, pass_int, yds, att):
    a = ((cmp / att) - 0.3) * 5
    b = ((yds / att) - 3) * 0.25
    c = (td / att) * 20
    d = 2.375 - ((pass_int / att) * 25)
    terms = {"a": a, "b": b, "c": c, "d": d}
    top = 0
    for term in terms.keys():
        if terms[term] > 2.375:
            terms[term] = 2.375
        elif terms[term] < 0:
            terms[term] = 0
        top += terms[term]
    return (top / 6) * 100
# Read in play by play data and filter
# to pass players where a QB threw the ball

df=pd.read_csv('/Volumes/XDrive/DFS/football/Experimental/bin/Models/FullSimulation/pbp_data/2016_2022_pbp_data.csv')
df=df[df.passer_position=='QB']
df=df[df.play_type=='pass']

# Create column for completed air yards
df.loc[df.complete_pass==1,'completed_air_yards']=df.loc[df.complete_pass==1].air_yards

# Adjust pass_length column
df.loc[df.air_yards<7.5,'pass_length']='short'
df.loc[(df.air_yards>=7.5)&(df.air_yards<15),'pass_length']='medium'
df.loc[df.air_yards>=15,'pass_length']='deep'

# Create EPA columns
df.loc[df.pass_length=='short','short_pass_epa']=df.loc[df.pass_length=='short'].epa
df.loc[df.pass_length=='short','short_pass']=1
df.loc[(df.pass_length=='short')&(df.complete_pass==1),'complete_short_pass']=1
df.loc[df.pass_length=='medium','medium_pass_epa']=df.loc[df.pass_length=='medium'].epa
df.loc[df.pass_length=='medium','medium_pass']=1
df.loc[(df.pass_length=='medium')&(df.complete_pass==1),'complete_medium_pass']=1

df.loc[df.pass_length=='deep','deep_pass_epa']=df.loc[df.pass_length=='deep'].epa
df.loc[df.pass_length=='deep','deep_pass']=1
df.loc[(df.pass_length=='deep')&(df.complete_pass==1),'complete_deep_pass']=1

# Individual QBs
qbs = df.groupby(['game_id','passer_player_id','posteam','week','season']).agg(
                {'air_yards':np.sum,
                'interception':np.sum,
                'completed_air_yards':np.sum,
                'epa':np.sum,
                'air_epa':np.sum,
                'short_pass_epa':np.sum,
                'medium_pass_epa':np.sum,
                'deep_pass_epa':np.sum,
                'short_pass':np.sum,
                'complete_short_pass':np.sum,
                'medium_pass':np.sum,
                'complete_medium_pass':np.sum,
                'deep_pass':np.sum,
                'complete_deep_pass':np.sum,
                'pass_attempt':np.sum,
                'complete_pass':np.sum,
                'passing_yards':np.sum,
                'pass_touchdown':np.sum,
                'week':'first',
                'season':'first'})
qbs['air_yards_efficiency']=qbs.completed_air_yards/qbs.air_yards
qbs['air_yards_per_attempt']=qbs.air_yards/qbs.pass_attempt
qbs['completion_percentage']=qbs.complete_pass/qbs.pass_attempt
qbs['short_completion_percentage']=qbs.complete_short_pass/qbs.short_pass
qbs['medium_completion_percentage']=qbs.complete_medium_pass/qbs.medium_pass
qbs['deep_completion_percentage']=qbs.complete_deep_pass/qbs.deep_pass
qbs['epa']=qbs.epa/qbs.pass_attempt
qbs['air_epa']=qbs.air_epa/qbs.pass_attempt
qbs['short_pass_epa']=qbs.short_pass_epa/qbs.short_pass
qbs['medium_pass_epa']=qbs.medium_pass_epa/qbs.medium_pass
qbs['deep_pass_epa']=qbs.deep_pass_epa/qbs.deep_pass
qbs['int_rate']=qbs.interception/qbs.pass_attempt
qbs['passer_rating']=qbs.apply(lambda x: getPasserRating(x.complete_pass,x.pass_touchdown,x.interception,x.passing_yards,x.pass_attempt),axis=1)
QB_STATS=['epa',
 'air_epa',
 'short_pass_epa',
 'medium_pass_epa',
 'deep_pass_epa',
 'air_yards_efficiency',
 'air_yards_per_attempt',
 'completion_percentage',
 'short_completion_percentage',
 'medium_completion_percentage',
 'deep_completion_percentage',
 'int_rate',
 'passer_rating']
qbs[QB_STATS]=qbs.groupby('passer_player_id').apply(lambda x: x.ewm(min_periods=1,span=8).mean().shift()[QB_STATS])
for stat in QB_STATS:
    qbs.rename({stat:f'qb_ewm_{stat}'},axis=1,inplace=True)
qbs = qbs[['qb_ewm_'+stat for stat in QB_STATS]]
qbs.to_csv('QB_Passing_Model_Data.csv')
#%%
# Team QBs
team_qbs = df.groupby(['game_id','posteam','week','season']).agg(
                {'air_yards':np.sum,
                'interception':np.sum,
                'completed_air_yards':np.sum,
                'epa':np.sum,
                'air_epa':np.sum,
                'short_pass_epa':np.sum,
                'medium_pass_epa':np.sum,
                'deep_pass_epa':np.sum,
                'short_pass':np.sum,
                'complete_short_pass':np.sum,
                'medium_pass':np.sum,
                'complete_medium_pass':np.sum,
                'deep_pass':np.sum,
                'complete_deep_pass':np.sum,
                'pass_attempt':np.sum,
                'complete_pass':np.sum,
                'passing_yards':np.sum,
                'pass_touchdown':np.sum,
                'week':'first',
                'season':'first'})
team_qbs['air_yards_efficiency']=team_qbs.completed_air_yards/team_qbs.air_yards
team_qbs['air_yards_per_attempt']=team_qbs.air_yards/team_qbs.pass_attempt
team_qbs['completion_percentage']=team_qbs.complete_pass/team_qbs.pass_attempt
team_qbs['short_completion_percentage']=team_qbs.complete_short_pass/team_qbs.short_pass
team_qbs['medium_completion_percentage']=team_qbs.complete_medium_pass/team_qbs.medium_pass
team_qbs['deep_completion_percentage']=team_qbs.complete_deep_pass/team_qbs.deep_pass
team_qbs['epa']=team_qbs.epa/team_qbs.pass_attempt
team_qbs['air_epa']=team_qbs.air_epa/team_qbs.pass_attempt
team_qbs['short_pass_epa']=team_qbs.short_pass_epa/team_qbs.short_pass
team_qbs['medium_pass_epa']=team_qbs.medium_pass_epa/team_qbs.medium_pass
team_qbs['deep_pass_epa']=team_qbs.deep_pass_epa/team_qbs.deep_pass
team_qbs['int_rate']=team_qbs.interception/team_qbs.pass_attempt
team_qbs['passer_rating']=team_qbs.apply(lambda x: getPasserRating(x.complete_pass,x.pass_touchdown,x.interception,x.passing_yards,x.pass_attempt),axis=1)

team_qbs=team_qbs.groupby('posteam').apply(lambda x: x.ewm(min_periods=1,span=8).mean().shift())
team_qbs=team_qbs.add_prefix('team_qb_ewm_')
team_qbs=team_qbs[ 
                    ['team_qb_ewm_epa',
                     'team_qb_ewm_air_epa',
                     'team_qb_ewm_short_pass_epa',
                     'team_qb_ewm_medium_pass_epa',
                     'team_qb_ewm_deep_pass_epa',
                     'team_qb_ewm_air_yards_efficiency',
                     'team_qb_ewm_air_yards_per_attempt',
                     'team_qb_ewm_completion_percentage',
                     'team_qb_ewm_short_completion_percentage',
                     'team_qb_ewm_medium_completion_percentage',
                     'team_qb_ewm_deep_completion_percentage',
                     'team_qb_ewm_int_rate',
                     'team_qb_ewm_passer_rating'
                     ]
                  ]
team_qbs.to_csv('QB_Team_Passing_Model_Data.csv')
#%% Redzone Stats
df=df[df.yardline_100<=20]
# Individual QBs
qbs = df.groupby(['game_id','passer_player_id','posteam','week','season']).agg(
                {'air_yards':np.sum,
                'interception':np.sum,
                'completed_air_yards':np.sum,
                'epa':np.sum,
                'air_epa':np.sum,
                'short_pass_epa':np.sum,
                'medium_pass_epa':np.sum,
                'deep_pass_epa':np.sum,
                'short_pass':np.sum,
                'complete_short_pass':np.sum,
                'medium_pass':np.sum,
                'complete_medium_pass':np.sum,
                'deep_pass':np.sum,
                'complete_deep_pass':np.sum,
                'pass_attempt':np.sum,
                'complete_pass':np.sum,
                'passing_yards':np.sum,
                'pass_touchdown':np.sum,
                'week':'first',
                'season':'first'})
qbs['air_yards_efficiency']=qbs.completed_air_yards/qbs.air_yards
qbs['air_yards_per_attempt']=qbs.air_yards/qbs.pass_attempt
qbs['completion_percentage']=qbs.complete_pass/qbs.pass_attempt
qbs['short_completion_percentage']=qbs.complete_short_pass/qbs.short_pass
qbs['medium_completion_percentage']=qbs.complete_medium_pass/qbs.medium_pass
qbs['deep_completion_percentage']=qbs.complete_deep_pass/qbs.deep_pass
qbs['epa']=qbs.epa/qbs.pass_attempt
qbs['air_epa']=qbs.air_epa/qbs.pass_attempt
qbs['short_pass_epa']=qbs.short_pass_epa/qbs.short_pass
qbs['medium_pass_epa']=qbs.medium_pass_epa/qbs.medium_pass
qbs['deep_pass_epa']=qbs.deep_pass_epa/qbs.deep_pass
qbs['int_rate']=qbs.interception/qbs.pass_attempt
qbs['passer_rating']=qbs.apply(lambda x: getPasserRating(x.complete_pass,x.pass_touchdown,x.interception,x.passing_yards,x.pass_attempt),axis=1)
QB_STATS=['epa',
          'air_epa',
 'short_pass_epa',
 'medium_pass_epa',
 'deep_pass_epa',
 'air_yards_efficiency',
 'air_yards_per_attempt',
 'completion_percentage',
 'short_completion_percentage',
 'medium_completion_percentage',
 'deep_completion_percentage',
 'int_rate',
 'passer_rating']
qbs=qbs.groupby('passer_player_id').apply(lambda x: x.ewm(min_periods=1,span=8).mean().shift()[QB_STATS])
for stat in QB_STATS:
    qbs.rename({stat:f'qb_redzone_ewm_{stat}'},axis=1,inplace=True)
qbs = qbs[['qb_redzone_ewm_'+stat for stat in QB_STATS]]
qbs.to_csv('QB_Redzone_Passing_Model_Data.csv')
#%%
# Team QBs
team_qbs = df.groupby(['game_id','posteam','week','season']).agg(
                {'air_yards':np.sum,
                'interception':np.sum,
                'completed_air_yards':np.sum,
                'epa':np.sum,
                'air_epa':np.sum,
                'short_pass_epa':np.sum,
                'medium_pass_epa':np.sum,
                'deep_pass_epa':np.sum,
                'short_pass':np.sum,
                'complete_short_pass':np.sum,
                'medium_pass':np.sum,
                'complete_medium_pass':np.sum,
                'deep_pass':np.sum,
                'complete_deep_pass':np.sum,
                'pass_attempt':np.sum,
                'complete_pass':np.sum,
                'passing_yards':np.sum,
                'pass_touchdown':np.sum,
                'week':'first',
                'season':'first'})
team_qbs['air_yards_efficiency']=team_qbs.completed_air_yards/team_qbs.air_yards
team_qbs['air_yards_per_attempt']=team_qbs.air_yards/team_qbs.pass_attempt
team_qbs['completion_percentage']=team_qbs.complete_pass/team_qbs.pass_attempt
team_qbs['short_completion_percentage']=team_qbs.complete_short_pass/team_qbs.short_pass
team_qbs['medium_completion_percentage']=team_qbs.complete_medium_pass/team_qbs.medium_pass
team_qbs['deep_completion_percentage']=team_qbs.complete_deep_pass/team_qbs.deep_pass
team_qbs['epa']=team_qbs.epa/team_qbs.pass_attempt
team_qbs['air_epa']=team_qbs.air_epa/team_qbs.pass_attempt
team_qbs['short_pass_epa']=team_qbs.short_pass_epa/team_qbs.short_pass
team_qbs['medium_pass_epa']=team_qbs.medium_pass_epa/team_qbs.medium_pass
team_qbs['deep_pass_epa']=team_qbs.deep_pass_epa/team_qbs.deep_pass
team_qbs['int_rate']=team_qbs.interception/team_qbs.pass_attempt
team_qbs['passer_rating']=team_qbs.apply(lambda x: getPasserRating(x.complete_pass,x.pass_touchdown,x.interception,x.passing_yards,x.pass_attempt),axis=1)

team_qbs=team_qbs.groupby('posteam').apply(lambda x: x.ewm(min_periods=1,span=8).mean().shift())
team_qbs=team_qbs.add_prefix('team_qb_redzone_ewm_')
team_qbs=team_qbs[ 
                    ['team_qb_redzone_ewm_epa',
                     'team_qb_redzone_ewm_air_epa',
                     'team_qb_redzone_ewm_short_pass_epa',
                     'team_qb_redzone_ewm_medium_pass_epa',
                     'team_qb_redzone_ewm_deep_pass_epa',
                     'team_qb_redzone_ewm_air_yards_efficiency',
                     'team_qb_redzone_ewm_air_yards_per_attempt',
                     'team_qb_redzone_ewm_completion_percentage',
                     'team_qb_redzone_ewm_short_completion_percentage',
                     'team_qb_redzone_ewm_medium_completion_percentage',
                     'team_qb_redzone_ewm_deep_completion_percentage',
                     'team_qb_redzone_ewm_int_rate',
                     'team_qb_redzone_ewm_passer_rating'
                     ]
                  ]
team_qbs.to_csv('QB_Team_Redzone_Passing_Model_Data.csv')
