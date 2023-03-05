#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 20:12:31 2023

@author: robertmegnia
"""

import nfl_data_py as nfl
import pandas as pd
from sklearn.neural_network import MLPClassifier as NN
from sklearn.linear_model import LogisticRegression as LR
from datetime import datetime
import numpy as np
import warnings
import os
import pickle
warnings.simplefilter("ignore")
basedir = os.path.dirname(os.path.abspath(__file__))
model_dir = f"{basedir}/ml_models"
df=nfl.import_pbp_data([2022])
df['defteam_timeout']=df.defteam.shift(-1)
df['posteam_timeout']=df.posteam.shift(-1)
df.loc[df.timeout_team==df.posteam_timeout,'timeout_team']='posteam'
df.loc[df.timeout_team==df.defteam_timeout,'timeout_team']='defteam'
df.loc[df.timeout_team=='posteam','posteam_timeout']=1.0
df.loc[df.timeout_team!='posteam','posteam_timeout']=0.0
df.loc[df.timeout_team=='defteam','defteam_timeout']=1.0
df.loc[df.timeout_team!='defteam','defteam_timeout']=0.0


try:
    df["time"] = df.time.apply(
        lambda x: datetime.strptime(x, "%M:%S") if x is not None else x
    )
except TypeError:
    df["time"] = df.time.apply(
        lambda x: datetime.strptime(x, "%M:%S") if x is not np.nan else x
    )
df["end_time"] = df.time.shift(-1)
df["elapsed_time"] = (df.time - df.end_time).apply(
    lambda x: x.total_seconds()
)
df['clock_running'] = True

# Stop clock for incomplete pass
df.loc[(df.complete_pass==0)&
        (df.sack==0)&
        (df.play_type!='run'),'clock_running']=False

# Stop clock for running out of bounds play with 2 minutes left in first half
df.loc[(df.qtr==2)&
        (df.half_seconds_remaining<=120)&
        (df.play_type=='run')&
        (df.out_of_bounds==1),'clock_running']=False

# Stop clock for running out of bounds play with 5 minutes left in second half
df.loc[(df.qtr==4)&
        (df.half_seconds_remaining<=300)&
        (df.play_type=='run')&
        (df.out_of_bounds==1),'clock_running']=False
# Stop clock for receiving out of bounds play with 2 minutes left in first half
df.loc[(df.qtr==2)&
        (df.half_seconds_remaining<=120)&
        (df.complete_pass==1)&
        (df.out_of_bounds==1),'clock_running']=False
# Stop clock for receiving out of bounds play with 5 minutes left in second half
df.loc[(df.qtr==4)&
        (df.half_seconds_remaining<=300)&
        (df.complete_pass==1)&
        (df.out_of_bounds==1),'clock_running']=False

# Stop clock after any field goal atdft
df.loc[(df.field_goal_attempt==1),'clock_running']=False

# Stop clock after any punt
df.loc[(df.punt_attempt==1),'clock_running']=False

# Stop clock after any 4th down that doesn't end with a first down
df.loc[(df.down==4)&(df.first_down==0),'clock_running']=False

# Stop clock for a timeout
df.loc[(df.timeout==1),'clock_running']=False

# Stop clock after a score or turnover
df.loc[(df.touchdown==1)|
        (df.safety==1)|
        (df.interception==1)|
        (df.fumble_lost==1),'clock_running']=False

# Stop clock for kickoff
df.loc[df.play_type.isin(['kickoff',None]),'clock_running']=False
df['yardline']=(100-df.yardline_100).shift(-1)
df['clock_running']=df.clock_running.shift()
df['posteam_timeout']=df.posteam_timeout.astype(float)
df['defteam_timeout']=df.defteam_timeout.astype(float)
df['score_differential']=(df.posteam_score-df.defteam_score).shift(-1)
df['down']=df.down.shift(-1)
df['ydstogo']=df.ydstogo.shift(-1)
#%% First half posteam timeout

feats=['half_seconds_remaining',
     'down',
     'ydstogo',
     'yardline',
     'posteam_timeout']
subset=df[(df.qtr<=2)&(df.clock_running==True)][feats].dropna()
model = NN().fit(subset.drop('posteam_timeout',axis=1),subset.posteam_timeout)
pickle.dump(
            model,
            open(
                f"{model_dir}/TimeoutModels/FirstHalfPosteamTimeout_model.pkl",
                "wb",
            ),
        )
#%% First half posteam 2 minute drill timeout
feats=['half_seconds_remaining',
       'down',
       'ydstogo',
       'yardline',
       'posteam_timeout']
subset=df[(df.qtr==2)&
          (df.half_seconds_remaining<=120)&
          (df.clock_running==True)][feats].dropna()
model = NN().fit(subset.drop('posteam_timeout',axis=1),subset.posteam_timeout)
pickle.dump(
            model,
            open(
                f"{model_dir}/TimeoutModels/FirstHalfPosteam2MinTimeout_model.pkl",
                "wb",
            ),
        )
#%% First half defteam timeout


feats=['half_seconds_remaining',
       'down',
       'ydstogo',
       'yardline',
       'defteam_timeout']
subset=df[(df.qtr<=2)&
          (df.clock_running==True)][feats].dropna()
model = NN().fit(subset.drop('defteam_timeout',axis=1),subset.defteam_timeout)
pickle.dump(
            model,
            open(
                f"{model_dir}/TimeoutModels/FirstHalfDefteamTimeout_model.pkl",
                "wb",
            ),
        )
#%% First half defteam 2 minute drill timeout


feats=['half_seconds_remaining',
       'down',
       'ydstogo',
       'yardline',
       'defteam_timeout']
subset=df[(df.qtr==2)&
          (df.half_seconds_remaining<=120)&
          (df.clock_running==True)][feats].dropna()
model = NN().fit(subset.drop('defteam_timeout',axis=1),subset.defteam_timeout)
pickle.dump(
            model,
            open(
                f"{model_dir}/TimeoutModels/FirstHalfDefteam2MinTimeout_model.pkl",
                "wb",
            ),
        )
#%% Second half posteam timeout

feats=['half_seconds_remaining',
       'down',
       'ydstogo',
       'yardline',
       'posteam_timeout']
subset=df[(df.qtr>2)&
          (df.clock_running==True)][feats].dropna()
model = NN().fit(subset.drop('posteam_timeout',axis=1),subset.posteam_timeout)
pickle.dump(
            model,
            open(
                f"{model_dir}/TimeoutModels/SecondHalfPosteamTimeout_model.pkl",
                "wb",
            ),
        )
#%% Second half posteam 2 minute drill timeout
feats=['half_seconds_remaining',
       'down',
       'ydstogo',
       'yardline',
       'posteam_timeout']
subset=df[(df.qtr>3)&
          (df.half_seconds_remaining<=120)&
          (df.clock_running==True)][feats].dropna()
model = NN().fit(subset.drop('posteam_timeout',axis=1),subset.posteam_timeout)
pickle.dump(
            model,
            open(
                f"{model_dir}/TimeoutModels/SecondHalfPosteam2MinTimeout_model.pkl",
                "wb",
            ),
        )
#%% Second half defteam timeout


feats=['half_seconds_remaining',
       'down',
       'ydstogo',
       'yardline',
       'defteam_timeout']
subset=df[(df.qtr>2)&
          (df.clock_running==True)][feats].dropna()
model = NN().fit(subset.drop('defteam_timeout',axis=1),subset.defteam_timeout)
pickle.dump(
            model,
            open(
                f"{model_dir}/TimeoutModels/SecondHalfDefteamTimeout_model.pkl",
                "wb",
            ),
        )
#%% Second half defteam 2 minute drill timeout


feats=['half_seconds_remaining',
       'down',
       'ydstogo',
       'yardline',
       'defteam_timeout']
subset=df[(df.qtr>3)&
          (df.half_seconds_remaining<=120)&
          (df.clock_running==True)][feats].dropna()
model = NN().fit(subset.drop('defteam_timeout',axis=1),subset.defteam_timeout)
pickle.dump(
            model,
            open(
                f"{model_dir}/TimeoutModels/SecondHalfDefteam2MinTimeout_model.pkl",
                "wb",
            ),
        )