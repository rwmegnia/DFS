#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 10:02:35 2022

@author: robertmegnia
"""
import pandas as pd

def standardizeStats(x,window):
    averages = x.rolling(window=window*2, min_periods=window).mean()
    stds = x.rolling(window=window*2, min_periods=window).mean()
    x = (x - averages) / stds
    return x

def getAdjOppRank(x):
    mean = x.rolling(window=16,min_periods=1).mean().shift(1)
    return mean

def newOppRanks(df):
    df['game_date']=pd.to_datetime(df.game_date)
    df.sort_values(by='game_date',inplace=True)
    df=df[df.offensive_snapcount_percentage.isna()==False]
    df=df[df.offensive_snapcount_percentage>=0.33]
    df['DKPts_z']=df.groupby('gsis_id').DKPts.transform(standardizeStats,window=4)
    afpa = df.groupby(['opp','game_date','week','season','position'],as_index=False).DKPts_z.sum()
    afpa['afpa_per_game']=afpa.groupby(['opp','position']).DKPts_z.transform(getAdjOppRank)
    afpa['Adj_opp_Rank']=afpa.groupby(['week','season','position']).afpa_per_game.rank()
    return