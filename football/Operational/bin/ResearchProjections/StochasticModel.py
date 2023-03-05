#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 22:04:16 2022

@author: robertmegnia
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 08:59:54 2021

@author: robertmegnia
"""
import numpy as np
import pandas as pd
import pickle
from scipy.stats import norm
import os
import sys
import warnings
warnings.simplefilter('ignore')
basedir = os.path.dirname(os.path.abspath(__file__))
etcdir = f"{basedir}/../../etc"
datadir = f"{basedir}/../../data"
projdir = f"{datadir}/Projections"
from config.ColumnMappings import *
from getDKPts import getDKPts
from RosterUtils import *
os.chdir(f"{basedir}/../")
from Models.ModelDict import *
os.chdir("./ResearchProjections")
from ModelFunctions import OffenseStochastic,DefenseStochastic,QBStochastic,RBStochastic,WRStochastic
#%%

def rolling_average(df, window=8):
    return df.rolling(min_periods=1, window=window).mean()

# Build Stochastic Projections Database
datadir = f"{basedir}/../../data"
frames=[]
for season in range(2022, 2023):
    if season < 2021:
        end_week = 17
    else:
        end_week = 18
    print(season)
    for week in range(1, 10):
        print(week)
        proj_frames = []
        dst_db = pd.read_csv(
                f"{datadir}/game_logs/Full/DST_Database.csv"
            ).drop_duplicates()
        dst_db.game_date = pd.to_datetime(dst_db.game_date)
        game_date = dst_db[(dst_db.season == season) & (dst_db.week == week)].game_date.min()
        stats_df = dst_db[dst_db.game_date < game_date]
        stats_df.sort_values(by="game_date", inplace=True)
        weekly_proj_df = dst_db[(dst_db.season == season) & (dst_db.week == week)]
        dst_proj=DefenseStochastic(weekly_proj_df,stats_df)
        proj_frames.append(dst_proj)
            #
        off_db = pd.read_csv(
                                    f"{datadir}/game_logs/Full/Offense_Database.csv"
                                ).drop_duplicates()
        off_db.game_date = pd.to_datetime(off_db.game_date)
        off_db.sort_values(by=['game_date','team','position','salary','offensive_snapcount_percentage'],ascending=False,inplace=True)
        stats_df = off_db[(off_db.game_date < game_date)]
        # stats_df=stats_df[stats_df.season==2022]
        stats_df.sort_values(by="game_date", inplace=True)
        weekly_proj_df = off_db[(off_db.season == season) & 
                                (off_db.week == week)]
        # Start with QBs
        QBs=weekly_proj_df[(weekly_proj_df.position=='QB')&(weekly_proj_df.depth_team==1)]
        qb_stats=stats_df[(stats_df.position=='QB')&(stats_df.depth_team==1)&(stats_df.offensive_snapcount_percentage>0.85)]
        qb_proj=QBStochastic(QBs,qb_stats)
        
        # RBs
        RBs = weekly_proj_df[(weekly_proj_df.position=='RB')&(weekly_proj_df.depth_team<=2)]
        rb_stats=stats_df[(stats_df.position=='RB')&(stats_df.depth_team<=2)]
        rb_proj=RBStochastic(RBs, rb_stats)
        
        # WRs
        WRs = weekly_proj_df[(weekly_proj_df.position=='WR')&(weekly_proj_df.depth_team<=4)]
        wr_stats = stats_df[(stats_df.position=='WR')&(stats_df.depth_team<=4)&(stats_df.offensive_snapcount_percentage>=.25)]
        wr_proj=WRStochastic(WRs, wr_stats)
        # TEs
        TEs = weekly_proj_df[(weekly_proj_df.position=='TE')&(weekly_proj_df.depth_team<=2)]
        te_stats = stats_df[(stats_df.position=='TE')&(stats_df.depth_team<=2)&(stats_df.offensive_snapcount_percentage>=.25)]
        te_proj=OffenseStochastic(TEs, te_stats)
        # Drop irrelevant players
        proj_frames.append(qb_proj)
        proj_frames.append(rb_proj)
        proj_frames.append(wr_proj)
        proj_frames.append(te_proj)
        if len(proj_frames) == 0:
            continue
        proj_df = pd.concat(proj_frames)
        projdir = f"{datadir}/Projections"
        proj_df.to_csv(
            f"{projdir}/{season}/Stochastic/{season}_Week{week}_StochasticProjections.csv",
            index=False,
        )
        frames.append(proj_df)
df=pd.concat(frames)
#runfile('/Volumes/XDrive/DFS/football/Experimental/bin/ResearchProjections/StochasticModel.py', wdir='/Volumes/XDrive/DFS/football/Experimental/bin/ResearchProjections')