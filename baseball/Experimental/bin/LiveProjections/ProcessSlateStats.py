#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 18:52:23 2022

@author: robertmegnia
"""
def processSlateStats(df):
    df['exWOBA_Diff']=df.exWOBA-df.wOBA
    df['split']=df.apply(lambda x: x.handedness+'_VS_'+x.opp_pitcher_hand,axis=1)
    df=df[[ 'full_name',
            'team',
            'game_location',
            'opp',
            'position',
            'order',
            'Salary',
            'split',
            'Projection',
            'RG_projection',
            'ownership_proj',
            'ConsRank',
            'Floor',
            'Ceiling',
            'Stochastic',
            'ML',
            'GB',
            'ScaledProj',
            'total_runs',
            'proj_runs',
            'moneyline',
            'exWOBA_Diff',
            'exWOBA',
            'wOBA',
            'ISO',
            'opp_pitcher_era',
            'SB_perAtBat',
            'K_prcnt',
            'oppK_prcnt',
            'era',
            'WHIP']]
    df=df.round(3)
    return df
    