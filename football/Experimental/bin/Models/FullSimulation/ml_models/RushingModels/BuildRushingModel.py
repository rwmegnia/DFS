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
# %% Filter Plays to relevant rushing plays
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
#%%
pbp = temp[temp.n_offense.isna() == False]
pbp = pbp[
    pbp.play_type_nfl.isin(
        ["RUSH", "PASS", "FIELD_GOAL", "SACK", "XP_KICK", "PAT2"]
    )
]
pbp = pbp[
    (pbp.offense_personnel.str.contains("DL") == False)
    & (pbp.offense_personnel.str.contains("LB") == False)
    & (pbp.offense_personnel.str.contains("2 QB") == False)
    & (pbp.offense_personnel.str.contains("3 QB") == False)
    & (pbp.offense_personnel.str.contains("DB") == False)
    & (pbp.offense_personnel.str.contains("LS") == False)
    & (pbp.offense_personnel.str.contains("K") == False)
    & (pbp.offense_personnel.str.contains("P") == False)
]


#%% Filter PBP Data to RB/FB rush attempts
rushing = pbp[(pbp.rush_attempt == 1) & (pbp.qb_dropback == 0)]
rushing = rushing[(rushing.n_offense == 11) & (rushing.n_defense == 11)]
rushing = rushing[rushing.rusher_position.isin(["RB", "FB"])]
#%% Get Player ID and Position of every player on the field
for i in range(1, 12):
    print(i)
    rushing[f"offense_player_{i}_id"] = rushing.offense_players.apply(
        lambda x: x.split(";")[i - 1]
    )
    rushing = rushing.merge(
        depth_charts.rename(
            {
                "position": f"offense_player_{i}_position",
                "club_code": "posteam",
                "gsis_id": f"offense_player_{i}_id",
            },
            axis=1,
        )[
            [
                f"offense_player_{i}_id",
                "week",
                "season",
                "posteam",
                f"offense_player_{i}_position",
            ]
        ],
        on=[f"offense_player_{i}_id", "week", "season", "posteam"],
        how="left",
    )
    rushing.drop_duplicates(inplace=True)
    rushing[f"defense_player_{i}_id"] = rushing.defense_players.apply(
        lambda x: x.split(";")[i - 1]
    )
    rushing = rushing.merge(
        depth_charts.rename(
            {
                "position": f"defense_player_{i}_position",
                "club_code": "defteam",
                "gsis_id": f"defense_player_{i}_id",
            },
            axis=1,
        )[
            [
                f"defense_player_{i}_id",
                "week",
                "season",
                "defteam",
                f"defense_player_{i}_position",
            ]
        ],
        on=[f"defense_player_{i}_id", "week", "season", "defteam"],
        how="left",
    )
#%%
OL_Data = pd.read_csv("./OL_Data/OL_RushingData.csv")
RushingData = pd.read_csv("./Rushing_Data/RushingData.csv")
LB_Data = pd.read_csv("./LB_Data/LB_RushingStats.csv")
DL_Data = pd.read_csv("./DL_Data/DL_RushingStats.csv")
DB_Data = pd.read_csv("./DB_Data/DB_RushingStats.csv")
OL_Data = OL_Data[
    ["gsis_id", "posteam", "game_id", "line_epa", "adjusted_line_yards"]
]
RushingData = RushingData[
    ["gsis_id", "posteam", "game_id", "ypc", "open_field_ypc", "rush_epa"]
]
LB_Data = LB_Data[
    ["gsis_id", "defteam", "game_id", "def_second_level_rushing_yards"]
]
DB_Data = DB_Data[
    ["gsis_id", "defteam", "game_id", "def_open_field_rushing_yards"]
]
DL_Data = DL_Data[
    ["gsis_id", "defteam", "game_id", "def_line_epa", "def_adjusted_line_yards"]
]
#%%
pbp_rushing_frames = []
for i in range(1, 12):
    print(i)
    rushing_frame = rushing.groupby(
        [
            "game_id",
            "game_date",
            "week",
            "season",
            f"offense_player_{i}_id",
            f"offense_player_{i}_position",
            "play_id",
            "rusher_position",
            "posteam",
        ]
    ).mean()
    rushing_frame = rushing_frame.reset_index().rename(
        {
            f"offense_player_{i}_id": "gsis_id",
            f"offense_player_{i}_position": "position",
        },
        axis=1,
    )
    rushing_frame = rushing_frame.merge(
        OL_Data, on=["gsis_id", "game_id", "posteam"], how="left"
    )
    rushing_frame = rushing_frame.merge(
        RushingData, on=["gsis_id", "game_id", "posteam"], how="left"
    )
    pbp_rushing_frames.append(rushing_frame)
    rushing_frame = rushing.groupby(
        [
            "game_id",
            "game_date",
            "week",
            "season",
            f"defense_player_{i}_id",
            f"defense_player_{i}_position",
            "play_id",
            "rusher_position",
            "defteam",
        ]
    ).mean()
    rushing_frame = rushing_frame.reset_index().rename(
        {
            f"defense_player_{i}_id": "gsis_id",
            f"defense_player_{i}_position": "position",
        },
        axis=1,
    )
    rushing_frame = rushing_frame.merge(
        DL_Data, on=["gsis_id", "game_id", "defteam"], how="left"
    )
    rushing_frame = rushing_frame.merge(
        LB_Data, on=["gsis_id", "game_id", "defteam"], how="left"
    )
    rushing_frame = rushing_frame.merge(
        DB_Data, on=["gsis_id", "game_id", "defteam"], how="left"
    )
    rushing_frame.reset_index(drop=True, inplace=True)
    pbp_rushing_frames.append(rushing_frame)
#%%
df = pd.concat(pbp_rushing_frames)
df = df.groupby(["game_id", "play_id"]).mean()[
    [
        "rushing_yards",
        "ypc",
        "open_field_ypc",
        "rush_epa",
        "line_epa",
        "adjusted_line_yards",
        "def_line_epa",
        "def_adjusted_line_yards",
        "def_second_level_rushing_yards",
        "def_open_field_rushing_yards",
        "defenders_in_box",
    ]
]
df.dropna(inplace=True)
X_train, X_test, y_train, y_test = train_test_split(
    df.drop("rushing_yards", axis=1),
    df.rushing_yards,
    test_size=0.25,
    train_size=0.75,
    shuffle=True,
)
model = NN().fit(df.drop("rushing_yards", axis=1), df.rushing_yards)
pickle.dump(model, open("NN_RushingModel.pkl", "wb"))
#%% Redzone Rushing Model
df = pd.concat(pbp_rushing_frames)
df = df[(df.yardline_100) <= 20]
df = df.groupby(["game_id", "play_id"]).mean()[
    [
        "rushing_yards",
        "ypc",
        "open_field_ypc",
        "rush_epa",
        "line_epa",
        "adjusted_line_yards",
        "def_line_epa",
        "def_adjusted_line_yards",
        "def_second_level_rushing_yards",
        "def_open_field_rushing_yards",
        "defenders_in_box",
    ]
]
df.dropna(inplace=True)
X_train, X_test, y_train, y_test = train_test_split(
    df.drop("rushing_yards", axis=1),
    df.rushing_yards,
    test_size=0.25,
    train_size=0.75,
    shuffle=True,
)
model = NN().fit(df.drop("rushing_yards", axis=1), df.rushing_yards)
pickle.dump(model, open("NN_RedzoneRushingModel.pkl", "wb"))
#%% Fumble Lost Rush Model
model_features = [
    "rushing_yards",
    "down",
    "ydstogo",
    "yardline_100",
    "half_seconds_remaining",
]
et_rush_fumble_subset = temp[
    model_features
    + ["rush_attempt", "rusher_position", "fumble_lost", "elapsed_time"]
][(temp.rush_attempt == 1) & (temp.rusher_position != "QB")].dropna()
X_train, X_test, y_train, y_test = train_test_split(
    et_rush_fumble_subset[model_features],
    et_rush_fumble_subset.fumble_lost,
    test_size=0.25,
    shuffle=True,
)
model = NN().fit(X_train, y_train)
pickle.dump(model, open("NN_RushFumbleModel.pkl", "wb"))
#%% Fumble Lost Return Yards Model
model_features = [
    "rushing_yards",
    "down",
    "ydstogo",
    "yardline_100",
    "half_seconds_remaining",
]
rush_fumble_yards_subset = temp[
    model_features
    + [
        "rush_attempt",
        "rusher_position",
        "fumble_recovery_1_yards",
        "fumble_lost",
        "elapsed_time",
    ]
][
    (temp.rush_attempt == 1)
    & (temp.rusher_position != "QB")
    & (temp.fumble_lost == 1)
].dropna()
X_train, y_train = (
    rush_fumble_yards_subset[model_features],
    rush_fumble_yards_subset.fumble_recovery_1_yards,
)
pickle.dump(model, open("NN_RushFumbleReturnYardsModel.pkl", "wb"))

#%% Elapsed Time Rush Model
model_features = [
    "rushing_yards",
    "down",
    "ydstogo",
    "yardline_100",
    "half_seconds_remaining",
    'clock_running',
    'out_of_bounds'
]
for pos in ['QB','RB','WR']:
    et_rush_subset = temp[
        model_features + ["rush_attempt", "rusher_position", "elapsed_time"]
    ][(temp.play_type=='run')&(temp.rusher_position==pos)].dropna()
    X_train, X_test, y_train, y_test = train_test_split(
        et_rush_subset[model_features],
        et_rush_subset.elapsed_time,
        test_size=0.01,
        shuffle=True,
    )
    model = NN().fit(X_train, y_train)
    pickle.dump(model, open(f"NN_{pos}RushElapsedTime.pkl", "wb"))
#%% Elapsed Time Rush Model 4th Quarter
model_features = [
    "rushing_yards",
    "down",
    "ydstogo",
    "yardline_100",
    "half_seconds_remaining",
    'clock_running',
    'out_of_bounds'
]
for pos in ['WR','QB','RB']:
    print(pos)
    et_rush_subset = temp[
        model_features + ["rush_attempt", "rusher_position", "elapsed_time"]
    ][(temp.play_type == 'run')&(temp.rusher_position==pos)&(temp.qtr==4)].dropna()
    X_train, X_test, y_train, y_test = train_test_split(
        et_rush_subset[model_features],
        et_rush_subset.elapsed_time,
        test_size=0.01,
        shuffle=True,
    )
    model = NN().fit(X_train, y_train)
    pickle.dump(model, open(f"NN_4thQtr{pos}RushElapsedTime.pkl", "wb"))
#%% QB Kneel Elapsed Time
temp.loc[(temp.timeout_team==temp.posteam),'timeout_team_type']='posteam'
temp.loc[(temp.timeout_team==temp.defteam),'timeout_team_type']='defteam'

model_features = [
    "qtr"
]
qb_kneel_subset = temp[
        model_features + ["elapsed_time"]
    ][(temp.play_type == 'qb_kneel')&(temp.qtr==4)].dropna()
X_train, X_test, y_train, y_test = train_test_split(
        qb_kneel_subset[model_features],
        qb_kneel_subset.elapsed_time,
        test_size=0.01,
        shuffle=True,
    )
model = NN().fit(X_train, y_train)
pickle.dump(model, open(f"NN_4thQtrQBKneelElapsedTime.pkl", "wb"))
#%% Elapsed Time Rush Fumble Mode
model_features = [
    "rushing_yards",
    "down",
    "ydstogo",
    "yardline_100",
    "half_seconds_remaining",
    "fumble_recovery_1_yards",
]
et_rush_fumble_subset = temp[
    model_features
    + [
        "rush_attempt",
        "rusher_position",
        "elapsed_time",
        "fumble_recovery_1_yards",
    ]
][
    (temp.rush_attempt == 1)
    & (temp.fumble_lost == 1)
    & (temp.rusher_position != "QB")
].dropna()
X_train, y_train = (
    et_rush_fumble_subset[model_features],
    et_rush_fumble_subset.elapsed_time,
)
model = NN().fit(X_train, y_train)
pickle.dump(model, open("NN_RushFumbleElapsedTime.pkl", "wb"))
#%% Ran out of bounds model
model_features=[
    "rushing_yards",
    "down",
    "ydstogo",
    "yardline_100",
    "half_seconds_remaining",
    ]
for pos in ['QB','RB','WR']:
    ob_subset = temp[model_features+[
        'rush_attempt',
        'rusher_position',
        'out_of_bounds']][(temp.rush_attempt==1)&
                          (temp.rusher_position==pos)]
    X_train, y_train = (
        ob_subset[model_features],
        ob_subset.out_of_bounds,
    )
    X_train.dropna(inplace=True)
    y_train=y_train[y_train.index.isin(X_train.index)]
    model = NN().fit(X_train, y_train)
    pickle.dump(model, open(f"NN_{pos}RushOB.pkl", "wb"))
