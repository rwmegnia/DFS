#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 05:31:26 2021

@author: robertmegnia
"""
import pandas as pd
import os
basedir = os.path.dirname(os.path.abspath(__file__))
datadir = f'{basedir}/../../../../data/'
os.chdir(f'{basedir}/../')
from team_db_config import QB_STATS, RB_STATS, REC_STATS
db=pd.read_csv(f'{datadir}/game_logs/Full/Offense_Database.csv')
#%%
QB=db[db.position=='QB']
RB=db[db.position=='RB']
WR=db[db.position=='WR']
TE=db[db.position=='TE']
def getQBData(df,defense=False):
    if defense==True:
        team_column='opp'
    else:
        team_column='team'
    qb_frame=pd.concat(df.apply(lambda x: QB[(QB[team_column]==x.team)&(QB.season==x.season)&(QB.week==x.week)&(QB.depth_team==1)],axis=1).values)[QB_STATS]
    if defense==True:
        qb_frame.rename({'team':'opp','opp':'team'},axis=1,inplace=True)
        qb_frame.drop('opp',axis=1)
        qb_frame.set_index(['team','season','week'],inplace=True)
        qb_frame=qb_frame.add_prefix('QB1_')
        qb_frame=qb_frame.add_suffix('_allowed')
    else:
        qb_frame.drop('opp',axis=1)
        qb_frame.set_index(['team','season','week'],inplace=True)
        qb_frame=qb_frame.add_prefix('QB1_')
    df=df.set_index(['team','season','week']).join(qb_frame) 
    return df.reset_index()

def getRBData(df,dc,defense=False):
    if defense==True:
        team_column='opp'
    else:
        team_column='team'
    rb_frame=pd.concat(df.apply(lambda x: RB[(RB[team_column]==x.team)&(RB.season==x.season)&(RB.week==x.week)&(RB.depth_team==dc)],axis=1).values)[RB_STATS]
    if defense==True:
        rb_frame.rename({'team':'opp','opp':'team'},axis=1,inplace=True)
        rb_frame.drop('opp',axis=1)
        rb_frame.set_index(['team','season','week'],inplace=True)
        rb_frame=rb_frame.add_prefix(f'RB{dc}_')
        rb_frame=rb_frame.add_suffix('_allowed')
    else:
        rb_frame.drop('opp',axis=1)
        rb_frame.set_index(['team','season','week'],inplace=True)
        rb_frame=rb_frame.add_prefix(f'RB{dc}_')
    df=df.set_index(['team','season','week']).join(rb_frame) 
    return df.reset_index()

def getWRData(df,dc,defense=False):
    if defense==True:
        team_column='opp'
    else:
        team_column='team'
    wr_frame=pd.concat(df.apply(lambda x: WR[(WR[team_column]==x.team)&(WR.season==x.season)&(WR.week==x.week)&(WR.depth_team==dc)],axis=1).values)[REC_STATS]
    if defense==True:
        wr_frame.rename({'team':'opp','opp':'team'},axis=1,inplace=True)
        wr_frame.drop('opp',axis=1)
        wr_frame.set_index(['team','season','week'],inplace=True)
        wr_frame=wr_frame.add_prefix(f'WR{dc}_')
        wr_frame=wr_frame.add_suffix('_allowed')
    else:
        wr_frame.drop('opp',axis=1)
        wr_frame.set_index(['team','season','week'],inplace=True)
        wr_frame=wr_frame.add_prefix(f'WR{dc}_')
    df=df.set_index(['team','season','week']).join(wr_frame) 
    return df.reset_index()

def getTEData(df,dc,defense=False):
    if defense==True:
        team_column='opp'
    else:
        team_column='team'
    te_frame=pd.concat(df.apply(lambda x: TE[(TE[team_column]==x.team)&(TE.season==x.season)&(TE.week==x.week)&(TE.depth_team==dc)],axis=1).values)[REC_STATS]
    if defense==True:
        te_frame.rename({'team':'opp','opp':'team'},axis=1,inplace=True)
        te_frame.drop('opp',axis=1)
        te_frame.set_index(['team','season','week'],inplace=True)
        te_frame=te_frame.add_prefix(f'TE{dc}_')
        te_frame=te_frame.add_suffix('_allowed')
    else:
        te_frame.drop('opp',axis=1)
        te_frame.set_index(['team','season','week'],inplace=True)
        te_frame=te_frame.add_prefix(f'TE{dc}_')
    df=df.set_index(['team','season','week']).join(te_frame) 
    return df.reset_index()
