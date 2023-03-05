#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 07:14:17 2023

@author: robertmegnia
"""

import pandas as pd
import nfl_data_py as nfl
import os
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier as NN
import pickle
from datetime import datetime
import numpy as np

def fixDepthCharts(df):
    df.week-=1
    df.loc[(df.season<2021)&(df.week==18),'game_type']='WC'
    df.loc[(df.season<2021)&(df.week==19),'game_type']='DIV'
    df.loc[(df.season<2021)&(df.week==20),'game_type']='CON'
    df.loc[(df.season<2021)&(df.week==21),'game_type']='SB'
    df.loc[(df.season>=2021)&(df.week==19),'game_type']='WC'
    df.loc[(df.season>=2021)&(df.week==20),'game_type']='DIV'
    df.loc[(df.season>=2021)&(df.week==21),'game_type']='CON'
    df.loc[(df.season>=2021)&(df.week==22),'game_type']='SB'
    df=df[df.week!=0]
    return df
depth_charts = nfl.import_depth_charts(range(2016, 2023))
depth_charts=fixDepthCharts(depth_charts)
depth_charts.drop("depth_position", axis=1, inplace=True)
depth_charts.drop_duplicates(inplace=True)
depth_charts.club_code.replace({"OAK": "LV", "SD": "LAC"}, inplace=True)
if os.path.exists("../../pbp_data/2016_2022_pbp_data.csv") == False:
    temp = nfl.import_pbp_data(range(2016, 2023))
    # Merge in positions of passers, rushers, and receivers
    temp = temp.merge(
        depth_charts.rename(
            {
                "gsis_id": "passer_player_id",
                "club_code": "posteam",
                "position": "passer_position",
            },
            axis=1,
        )[["passer_player_id", "posteam", "week", "season", "passer_position"]],
        on=["posteam", "week", "season", "passer_player_id"],
        how="left",
    )
    temp.drop_duplicates(inplace=True)
    temp = temp.merge(
        depth_charts.rename(
            {
                "gsis_id": "rusher_player_id",
                "club_code": "posteam",
                "position": "rusher_position",
            },
            axis=1,
        )[["rusher_player_id", "posteam", "week", "season", "rusher_position"]],
        on=["posteam", "week", "season", "rusher_player_id"],
        how="left",
    )
    temp.drop_duplicates(inplace=True)
    temp = temp.merge(
        depth_charts.rename(
            {
                "gsis_id": "receiver_player_id",
                "club_code": "posteam",
                "position": "receiver_position",
            },
            axis=1,
        )[
            [
                "receiver_player_id",
                "posteam",
                "week",
                "season",
                "receiver_position",
            ]
        ],
        on=["posteam", "week", "season", "receiver_player_id"],
        how="left",
    )
    temp.drop_duplicates(inplace=True)
else:
    temp = pd.read_csv("../../pbp_data/2016_2022_pbp_data.csv")
try:
    temp["time"] = temp.time.apply(
        lambda x: datetime.strptime(x, "%M:%S") if x is not np.nan else x
    )
except:
    temp["time"] = temp.time.apply(
        lambda x: datetime.strptime(x, "%M:%S") if x is not None else x
    ) 
temp["end_time"] = temp.time.shift(-1)
temp["elapsed_time"] = (temp.time - temp.end_time).apply(
    lambda x: x.total_seconds()
)
temp['clock_running'] = True

# Stop clock for incomplete pass
temp.loc[(temp.complete_pass==0)&
        (temp.sack==0)&
        (temp.play_type!='run'),'clock_running']=False

# Stop clock for running out of bounds play with 2 minutes left in first half
temp.loc[(temp.qtr==2)&
        (temp.half_seconds_remaining<=120)&
        (temp.play_type=='run')&
        (temp.out_of_bounds==1),'clock_running']=False

# Stop clock for running out of bounds play with 5 minutes left in second half
temp.loc[(temp.qtr==4)&
        (temp.half_seconds_remaining<=300)&
        (temp.play_type=='run')&
        (temp.out_of_bounds==1),'clock_running']=False
# Stop clock for receiving out of bounds play with 2 minutes left in first half
temp.loc[(temp.qtr==2)&
        (temp.half_seconds_remaining<=120)&
        (temp.complete_pass==1)&
        (temp.out_of_bounds==1),'clock_running']=False
# Stop clock for receiving out of bounds play with 5 minutes left in second half
temp.loc[(temp.qtr==4)&
        (temp.half_seconds_remaining<=300)&
        (temp.complete_pass==1)&
        (temp.out_of_bounds==1),'clock_running']=False

# Stop clock after any field goal attempt
temp.loc[(temp.field_goal_attempt==1),'clock_running']=False

# Stop clock after any punt
temp.loc[(temp.punt_attempt==1),'clock_running']=False

# Stop clock after any 4th down that doesn't end with a first down
temp.loc[(temp.down==4)&(temp.first_down==0),'clock_running']=False

# Stop clock for a timeout
temp.loc[(temp.timeout==1),'clock_running']=False

# Stop clock after a score or turnover
temp.loc[(temp.touchdown==1)|
        (temp.safety==1)|
        (temp.interception==1)|
        (temp.fumble_lost==1),'clock_running']=False

# Stop clock for kickoff
temp.loc[temp.play_type.isin(['kickoff',None]),'clock_running']=False
temp.loc[temp.penalty_team==temp.posteam,'penalty_team']='posteam'
temp.loc[temp.penalty_team==temp.defteam,'penalty_team']='defteam'
#%% Prenap Penalty
pre_snap_penalties=['Delay of Game', 
                    'False Start', 
                    'Neutral Zone Infraction', 
                    'Encroachment',
                    None
                    ]

subset = temp[(temp.penalty_type.isin(pre_snap_penalties))&
              (temp.special_teams_play==0)&
              (temp.play_type.isin(['pass','run','no_play']))]

features=['penalty','qtr']

subset=subset[features].dropna()
model = NN().fit(subset[['qtr']],
                 subset.penalty)

pickle.dump(model,open('./ml_models/PenaltyModels/pre_snap_penalty_model.pkl','wb'))
#%% Presnap Penalty Team

subset = temp[(temp.penalty_type.isin(pre_snap_penalties))&
              (temp.penalty==1)&
              (temp.special_teams_play==0)&
              (temp.play_type.isin(['pass','run','no_play']))]

features=['down',
          'ydstogo',
          'qtr',
          'half_seconds_remaining',
          'penalty_type',
          'penalty_team']

subset=subset[features].dropna()
model = NN().fit(subset[features].drop(['penalty_type',
                                        'penalty_team'],axis=1),
                 subset.penalty_team)

pickle.dump(model,open('./ml_models/PenaltyModels/pre_snap_penalty_team_model.pkl','wb'))
#%% Presnap Defteam Penalty Type
subset = temp[(temp.penalty_type.isin(pre_snap_penalties))&
              (temp.penalty==1)&
              (temp.penalty_team=='defteam')&
              (temp.special_teams_play==0)&
              (temp.play_type.isin(['pass','run','no_play']))]

features=['down',
          'ydstogo',
          'qtr',
          'half_seconds_remaining',
          'penalty_type']

subset=subset[features].dropna()
model = NN().fit(subset[features].drop(['penalty_type'],axis=1),
                 subset.penalty_type)

pickle.dump(model,open('./ml_models/PenaltyModels/pre_snap_penalty_defteam_model.pkl','wb'))
#%% Presnap Defteam Penalty Elapsed Time
subset = temp[(temp.penalty_type.isin(pre_snap_penalties))&
              (temp.penalty==1)&
              (temp.penalty_team=='defteam')&
              (temp.special_teams_play==0)&
              (temp.play_type.isin(['pass','run','no_play']))]

features=['down',
          'ydstogo',
          'qtr',
          'half_seconds_remaining',
          'elapsed_time']

subset=subset[features].dropna()
model = NN().fit(subset[features].drop(['elapsed_time'],axis=1),
                 subset.elapsed_time)

pickle.dump(model,open('./ml_models/PenaltyModels/pre_snap_penalty_defteam_et_model.pkl','wb'))
#%% Presnap Posteam Penalty Type
subset = temp[(temp.penalty_type.isin(pre_snap_penalties))&
              (temp.penalty==1)&
              (temp.penalty_team=='posteam')&
              (temp.special_teams_play==0)&
              (temp.play_type.isin(['pass','run','no_play']))]

features=['down',
          'ydstogo',
          'qtr',
          'half_seconds_remaining',
          'penalty_type']

subset=subset[features].dropna()
model = NN().fit(subset[features].drop(['penalty_type'],axis=1),
                 subset.penalty_type)

pickle.dump(model,open('./ml_models/PenaltyModels/pre_snap_penalty_posteam_model.pkl','wb'))
#%% Presnap Posteam Penalty Elapsed Time
subset = temp[(temp.penalty_type.isin(pre_snap_penalties))&
              (temp.penalty==1)&
              (temp.penalty_team=='posteam')&
              (temp.special_teams_play==0)&
              (temp.play_type.isin(['pass','run','no_play']))]

features=['down',
          'ydstogo',
          'qtr',
          'half_seconds_remaining',
          'elapsed_time']

subset=subset[features].dropna()
model = NN().fit(subset[features].drop(['elapsed_time'],axis=1),
                 subset.elapsed_time)

pickle.dump(model,open('./ml_models/PenaltyModels/pre_snap_penalty_posteam_et_model.pkl','wb'))
#%% Penalty After Rushing Play

subset = temp[(temp.special_teams_play==0)&
              (temp.rush_attempt==1)]

features=['down',
          'ydstogo',
          'qtr',
          'half_seconds_remaining',
          'penalty']

subset=subset[features].dropna()
model = NN().fit(subset[features].drop(['penalty'],axis=1),
                 subset.penalty)

pickle.dump(model,open('./ml_models/PenaltyModels/rushing_penalty_model.pkl','wb'))
#%% Rushing Penalty Team
subset = temp[(temp.special_teams_play==0)&
              (temp.rush_attempt==1)&
              (temp.penalty==1)]

features=['down',
          'ydstogo',
          'qtr',
          'half_seconds_remaining',
          'penalty_team']

subset=subset[features].dropna()
model = NN().fit(subset[features].drop(['penalty_team'],axis=1),
                 subset.penalty_team)

pickle.dump(model,open('./ml_models/PenaltyModels/rushing_penalty_team_model.pkl','wb'))
#%% Defteam Rushing Penalty Type
subset = temp[(temp.special_teams_play==0)&
              (temp.rush_attempt==1)&
              (temp.penalty==1)&
              (temp.penalty_team=='defteam')]

features=['down',
          'ydstogo',
          'qtr',
          'half_seconds_remaining',
          'penalty_type']

subset=subset[features].dropna()
model = NN().fit(subset[features].drop(['penalty_type'],axis=1),
                 subset.penalty_type)

pickle.dump(model,open('./ml_models/PenaltyModels/rushing_penalty_defteam_model.pkl','wb'))
#%% Defteam Rushing Penalty Elapsed Time
subset = temp[(temp.special_teams_play==0)&
              (temp.rush_attempt==1)&
              (temp.penalty==1)&
              (temp.penalty_team=='defteam')]

features=['down',
          'ydstogo',
          'qtr',
          'half_seconds_remaining',
          'elapsed_time']

subset=subset[features].dropna()
model = NN().fit(subset[features].drop(['elapsed_time'],axis=1),
                 subset.elapsed_time)

pickle.dump(model,open('./ml_models/PenaltyModels/rushing_penalty_defteam_et_model.pkl','wb'))
#%% Posteam Rushing Penalty Type
subset = temp[(temp.special_teams_play==0)&
              (temp.rush_attempt==1)&
              (temp.penalty==1)&
              (temp.penalty_team=='posteam')]

features=['down',
          'ydstogo',
          'qtr',
          'half_seconds_remaining',
          'penalty_type']

subset=subset[features].dropna()
model = NN().fit(subset[features].drop(['penalty_type'],axis=1),
                 subset.penalty_type)

pickle.dump(model,open('./ml_models/PenaltyModels/rushing_penalty_posteam_model.pkl','wb'))
#%% Posteam Rushing Penalty Type
subset = temp[(temp.special_teams_play==0)&
              (temp.rush_attempt==1)&
              (temp.penalty==1)&
              (temp.penalty_team=='posteam')]

features=['down',
          'ydstogo',
          'qtr',
          'half_seconds_remaining',
          'elapsed_time']

subset=subset[features].dropna()
model = NN().fit(subset[features].drop(['elapsed_time'],axis=1),
                 subset.elapsed_time)

pickle.dump(model,open('./ml_models/PenaltyModels/rushing_penalty_posteam_et_model.pkl','wb'))
#%% Penalty After Incomplete Pass

subset = temp[(temp.special_teams_play==0)&
              (temp.pass_attempt==1)&
              (temp.complete_pass==0)]

features=['down',
          'ydstogo',
          'qtr',
          'half_seconds_remaining',
          'penalty']

subset=subset[features].dropna()
model = NN().fit(subset[features].drop(['penalty'],axis=1),
                 subset.penalty)

pickle.dump(model,open('./ml_models/PenaltyModels/incomplete_pass_penalty_model.pkl','wb'))

#%% Incomplete Pass Penalty Team
subset = temp[(temp.special_teams_play==0)&
              (temp.pass_attempt==1)&
              (temp.complete_pass==0)&
              (temp.penalty==1)]

features=['down',
          'ydstogo',
          'qtr',
          'half_seconds_remaining',
          'penalty_team']

subset=subset[features].dropna()
model = NN().fit(subset[features].drop(['penalty_team'],axis=1),
                 subset.penalty_team)

pickle.dump(model,open('./ml_models/PenaltyModels/incomplete_pass_penalty_team_model.pkl','wb'))
#%% Incomplete Pass Defteam Penalty
subset = temp[(temp.special_teams_play==0)&
              (temp.pass_attempt==1)&
              (temp.complete_pass==0)&
              (temp.penalty==1)&
              (temp.penalty_team=='defteam')]

features=['down',
          'ydstogo',
          'qtr',
          'half_seconds_remaining',
          'penalty_type']

subset=subset[features].dropna()
model = NN().fit(subset[features].drop(['penalty_type'],axis=1),
                 subset.penalty_type)

pickle.dump(model,open('./ml_models/PenaltyModels/incomplete_pass_penalty_defteam_model.pkl','wb'))
#%% Incomplete Pass Defteam Penalty Elapsed Time
subset = temp[(temp.special_teams_play==0)&
              (temp.pass_attempt==1)&
              (temp.complete_pass==0)&
              (temp.penalty==1)&
              (temp.penalty_team=='defteam')]

features=['down',
          'ydstogo',
          'qtr',
          'half_seconds_remaining',
          'elapsed_time']

subset=subset[features].dropna()
model = NN().fit(subset[features].drop(['elapsed_time'],axis=1),
                 subset.elapsed_time)

pickle.dump(model,open('./ml_models/PenaltyModels/incomplete_pass_penalty_defteam_et_model.pkl','wb'))
#%% Incomplete Pass Posteam Penalty
subset = temp[(temp.special_teams_play==0)&
              (temp.pass_attempt==1)&
              (temp.complete_pass==0)&
              (temp.penalty==1)&
              (temp.penalty_team=='posteam')]

features=['down',
          'ydstogo',
          'qtr',
          'half_seconds_remaining',
          'penalty_type']

subset=subset[features].dropna()
model = NN().fit(subset[features].drop(['penalty_type'],axis=1),
                 subset.penalty_type)

pickle.dump(model,open('./ml_models/PenaltyModels/incomplete_pass_penalty_posteam_model.pkl','wb'))
#%% Incomplete Pass Posteam Penalty Elapsed Time
subset = temp[(temp.special_teams_play==0)&
              (temp.pass_attempt==1)&
              (temp.complete_pass==0)&
              (temp.penalty==1)&
              (temp.penalty_team=='posteam')]

features=['down',
          'ydstogo',
          'qtr',
          'half_seconds_remaining',
          'elapsed_time']

subset=subset[features].dropna()
model = NN().fit(subset[features].drop(['elapsed_time'],axis=1),
                 subset.elapsed_time)

pickle.dump(model,open('./ml_models/PenaltyModels/incomplete_pass_penalty_posteam_et_model.pkl','wb'))
#%% Penalty After Complete Pass

subset = temp[(temp.special_teams_play==0)&
              (temp.pass_attempt==1)&
              (temp.complete_pass==1)]

features=['down',
          'ydstogo',
          'qtr',
          'half_seconds_remaining',
          'penalty']

subset=subset[features].dropna()
model = NN().fit(subset[features].drop(['penalty'],axis=1),
                 subset.penalty)

pickle.dump(model,open('./ml_models/PenaltyModels/complete_pass_penalty_model.pkl','wb'))

#%% Complete Pass Penalty Team
subset = temp[(temp.special_teams_play==0)&
              (temp.pass_attempt==1)&
              (temp.complete_pass==1)&
              (temp.penalty==1)]

features=['down',
          'ydstogo',
          'qtr',
          'half_seconds_remaining',
          'penalty_team']

subset=subset[features].dropna()
model = NN().fit(subset[features].drop(['penalty_team'],axis=1),
                 subset.penalty_team)

pickle.dump(model,open('./ml_models/PenaltyModels/complete_pass_penalty_team_model.pkl','wb'))
#%% Complete Pass Defteam Penalty
subset = temp[(temp.special_teams_play==0)&
              (temp.pass_attempt==1)&
              (temp.complete_pass==1)&
              (temp.penalty==1)&
              (temp.penalty_team=='defteam')]

features=['down',
          'ydstogo',
          'qtr',
          'half_seconds_remaining',
          'penalty_type']

subset=subset[features].dropna()
model = NN().fit(subset[features].drop(['penalty_type'],axis=1),
                 subset.penalty_type)

pickle.dump(model,open('./ml_models/PenaltyModels/complete_pass_penalty_defteam_model.pkl','wb'))
#%% Complete Pass Defteam Penalty Elapsed Time
subset = temp[(temp.special_teams_play==0)&
              (temp.pass_attempt==1)&
              (temp.complete_pass==1)&
              (temp.penalty==1)&
              (temp.penalty_team=='defteam')]

features=['down',
          'ydstogo',
          'qtr',
          'half_seconds_remaining',
          'elapsed_time']

subset=subset[features].dropna()
model = NN().fit(subset[features].drop(['elapsed_time'],axis=1),
                 subset.elapsed_time)

pickle.dump(model,open('./ml_models/PenaltyModels/complete_pass_penalty_defteam_et_model.pkl','wb'))
#%% Complete Pass Posteam Penalty
subset = temp[(temp.special_teams_play==0)&
              (temp.pass_attempt==1)&
              (temp.complete_pass==1)&
              (temp.penalty==1)&
              (temp.penalty_team=='posteam')]

features=['down',
          'ydstogo',
          'qtr',
          'half_seconds_remaining',
          'penalty_type']

subset=subset[features].dropna()
model = NN().fit(subset[features].drop(['penalty_type'],axis=1),
                 subset.penalty_type)

pickle.dump(model,open('./ml_models/PenaltyModels/complete_pass_penalty_posteam_model.pkl','wb'))
#%% Complete Pass Defteam Penalty Elapsed Time
subset = temp[(temp.special_teams_play==0)&
              (temp.pass_attempt==1)&
              (temp.complete_pass==1)&
              (temp.penalty==1)&
              (temp.penalty_team=='posteam')]

features=['down',
          'ydstogo',
          'qtr',
          'half_seconds_remaining',
          'elapsed_time']

subset=subset[features].dropna()
model = NN().fit(subset[features].drop(['elapsed_time'],axis=1),
                 subset.elapsed_time)

pickle.dump(model,open('./ml_models/PenaltyModels/complete_pass_penalty_posteam_et_model.pkl','wb'))