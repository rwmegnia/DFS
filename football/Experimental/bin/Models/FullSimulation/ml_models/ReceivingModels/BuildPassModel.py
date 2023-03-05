#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 10:55:57 2023

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


depth_charts=pd.read_csv('../../depth_charts/2016_2022_Rosters.csv')

if os.path.exists("../../pbp_data/2016_2022_pbp_data.csv") == False:
    temp = nfl.import_pbp_data(range(2016, 2023))
    # Merge in positions of passers, rushers, and receivers
    temp = temp.merge(
        depth_charts.rename(
            {
                "gsis_id": "passer_player_id",
                "team": "posteam",
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
                "team": "posteam",
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
                "team": "posteam",
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
        lambda x: datetime.strptime(x, "%M:%S") if x is not None else x
    )
except TypeError:
    temp["time"] = temp.time.apply(
        lambda x: datetime.strptime(x, "%M:%S") if x is not np.nan else x
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
# %% Filter Plays to relevant passing plays
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
passing = pbp[(pbp.play_type == "pass")]
passing = passing[(passing.n_offense == 11) & (passing.n_defense == 11)]
passing = passing[passing.receiver_position.isin(["RB", "WR", "TE"])]
#%% Get Player ID and Position of every player on the field
for i in range(1, 12):
    print(i)
    passing[f"offense_player_{i}_id"] = passing.offense_players.apply(
        lambda x: x.split(";")[i - 1]
    )
    passing = passing.merge(
        depth_charts.rename(
            {
                "position": f"offense_player_{i}_position",
                "team": "posteam",
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
    passing.drop_duplicates(inplace=True)
    passing[f"defense_player_{i}_id"] = passing.defense_players.apply(
        lambda x: x.split(";")[i - 1]
    )
    passing = passing.merge(
        depth_charts.rename(
            {
                "position": f"defense_player_{i}_position",
                "team": "defteam",
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
OL_Data = pd.read_csv("./OL_Data/OL_PassingData.csv")
passingData = pd.read_csv("./QB_Passing_Model_Data.csv")
passingData.rename({"passer_player_id": "gsis_id"}, axis=1, inplace=True)
receivingData = pd.read_csv("./Receiving_Model_Data.csv")
receivingData.rename({"receiver_player_id": "gsis_id"}, axis=1, inplace=True)
Def_Data = pd.read_csv("./DefensivePassData.csv")
Def_Data = Def_Data.groupby(["gsis_id", "game_id"], as_index=False).first()
#%%
OL_Data = OL_Data[
    [
        "gsis_id",
        "posteam",
        "game_id",
        "OL_epa",
        "OL_short_target_epa",
        "OL_medium_target_epa",
        "OL_deep_target_epa",
    ]
]
passingData = passingData[
    [
        "gsis_id",
        "game_id",
        "posteam",
        "qb_ewm_epa",
        "qb_ewm_air_epa",
        "qb_ewm_short_pass_epa",
        "qb_ewm_medium_pass_epa",
        "qb_ewm_deep_pass_epa",
        "qb_ewm_air_yards_efficiency",
        "qb_ewm_air_yards_per_attempt",
        "qb_ewm_completion_percentage",
        "qb_ewm_short_completion_percentage",
        "qb_ewm_medium_completion_percentage",
        "qb_ewm_deep_completion_percentage",
        "qb_ewm_int_rate",
        "qb_ewm_passer_rating",
    ]
]
receivingData = receivingData[
    [
        "game_id",
        "gsis_id",
        "posteam",
        "rec_ewm_epa",
        "rec_ewm_yac_epa",
        "rec_ewm_yards_after_catch",
        "rec_ewm_short_target_epa",
        "rec_ewm_medium_target_epa",
        "rec_ewm_deep_target_epa",
        "rec_ewm_air_yards_efficiency",
        "rec_ewm_catch_rate",
        "rec_ewm_short_catch_rate",
        "rec_ewm_medium_catch_rate",
        "rec_ewm_deep_catch_rate",
        "rec_ewm_target_share",
        "rec_ewm_wopr",
    ]
]
Def_Data = Def_Data[
    [
        "game_id",
        "gsis_id",
        "defteam",
        "def_epa",
        "def_air_epa",
        "def_yac_epa",
        "def_yards_after_catch",
        "def_short_target_epa",
        "def_medium_target_epa",
        "def_deep_target_epa",
        "def_air_yards_efficiency",
        "def_air_yards_per_attempt",
        "def_catch_rate",
        "def_short_catch_rate",
        "def_medium_catch_rate",
        "def_deep_catch_rate",
    ]
]
#%%
pbp_passing_frames = []
passing.loc[passing.air_yards <= 7, "pass_length"] = 0
passing.loc[
    (passing.air_yards > 7) & (passing.air_yards < 15), "pass_length"
] = 1
passing.loc[(passing.air_yards >= 15), "pass_length"] = 2
passing["pass_length"] = passing.pass_length.astype(float)
for i in range(1, 12):
    print(i)
    passing_frame = passing.groupby(
        [
            "game_id",
            "game_date",
            "week",
            "season",
            f"offense_player_{i}_id",
            f"offense_player_{i}_position",
            "play_id",
            "posteam",
        ]
    ).mean()
    passing_frame = passing_frame.reset_index().rename(
        {
            f"offense_player_{i}_id": "gsis_id",
            f"offense_player_{i}_position": "position",
        },
        axis=1,
    )
    passing_frame = passing_frame.merge(
        OL_Data, on=["gsis_id", "game_id", "posteam"], how="left"
    )
    passing_frame = passing_frame.merge(
        passingData, on=["gsis_id", "game_id", "posteam"], how="left"
    )
    passing_frame = passing_frame.merge(
        receivingData, on=["gsis_id", "game_id", "posteam"], how="left"
    )

    pbp_passing_frames.append(passing_frame)
    passing_frame = passing.groupby(
        [
            "game_id",
            "game_date",
            "week",
            "season",
            f"defense_player_{i}_id",
            f"defense_player_{i}_position",
            "play_id",
            "defteam",
        ]
    ).mean()
    passing_frame = passing_frame.reset_index().rename(
        {
            f"defense_player_{i}_id": "gsis_id",
            f"defense_player_{i}_position": "position",
        },
        axis=1,
    )
    passing_frame = passing_frame.merge(
        Def_Data, on=["gsis_id", "game_id", "defteam"], how="left"
    )
    passing_frame.reset_index(drop=True, inplace=True)
    pbp_passing_frames.append(passing_frame)
#%%
df = pd.concat(pbp_passing_frames)
df["distance_to_sticks"] = df.air_yards - df.ydstogo
df["score_differential"] = df.posteam_score - df.defteam_score
df = df.groupby(["game_id", "play_id"]).mean()[
    [
        "OL_epa",
        "OL_short_target_epa",
        "OL_medium_target_epa",
        "OL_deep_target_epa",
        "qb_ewm_epa",
        "qb_ewm_air_epa",
        "qb_ewm_short_pass_epa",
        "qb_ewm_medium_pass_epa",
        "qb_ewm_deep_pass_epa",
        "qb_ewm_air_yards_efficiency",
        "qb_ewm_air_yards_per_attempt",
        "qb_ewm_completion_percentage",
        "qb_ewm_short_completion_percentage",
        "qb_ewm_medium_completion_percentage",
        "qb_ewm_deep_completion_percentage",
        "qb_ewm_int_rate",
        "qb_ewm_passer_rating",
        "rec_ewm_epa",
        "rec_ewm_yac_epa",
        "rec_ewm_yards_after_catch",
        "rec_ewm_short_target_epa",
        "rec_ewm_medium_target_epa",
        "rec_ewm_deep_target_epa",
        "rec_ewm_air_yards_efficiency",
        "rec_ewm_catch_rate",
        "rec_ewm_short_catch_rate",
        "rec_ewm_medium_catch_rate",
        "rec_ewm_deep_catch_rate",
        "rec_ewm_target_share",
        "rec_ewm_wopr",
        "def_epa",
        "def_air_epa",
        "def_yac_epa",
        "def_yards_after_catch",
        "def_short_target_epa",
        "def_medium_target_epa",
        "def_deep_target_epa",
        "def_catch_rate",
        "def_short_catch_rate",
        "def_medium_catch_rate",
        "def_deep_catch_rate",
        "def_air_yards_efficiency",
        "def_air_yards_per_attempt",
        "down",
        "ydstogo",
        "yardline_100",
        "score_differential",
        "half_seconds_remaining",
        "distance_to_sticks",
        "air_yards",
        "yards_after_catch",
        "number_of_pass_rushers",
        "complete_pass",
        "interception",
        "pass_length",
        "qb_hit",
        "qb_scramble"
    ]
]
#%%
# df.dropna(inplace=True)
df.loc[df.interception == 1, "complete_pass"] = 2
#%% Pass Result Models (Short, Medium, Deep)
model_features = [
    "OL_epa",
    "OL_short_target_epa",
    "qb_ewm_epa",
    "qb_ewm_short_pass_epa",
    "qb_ewm_air_yards_efficiency",
    "qb_ewm_completion_percentage",
    "qb_ewm_short_completion_percentage",
    "qb_ewm_int_rate",
    "qb_ewm_passer_rating",
    "rec_ewm_epa",
    "rec_ewm_short_target_epa",
    "rec_ewm_air_yards_efficiency",
    "rec_ewm_catch_rate",
    "rec_ewm_short_catch_rate",
    "def_epa",
    "def_short_target_epa",
    "def_catch_rate",
    "def_short_catch_rate",
    "def_air_yards_efficiency",
    "down",
    "ydstogo",
    "yardline_100",
    "distance_to_sticks",
    "air_yards",
    "number_of_pass_rushers",
]
short_subset = df[df.pass_length == 0]
short_subset.dropna(inplace=True)
X_train, X_test, y_train, y_test = train_test_split(
    short_subset[model_features],
    short_subset.complete_pass,
    test_size=0.25,
    shuffle=True,
)
model = NN().fit(X_train, y_train)
pickle.dump(model, open("NN_shortPassModel.pkl", "wb"))
#%% Pass Result Models (Short, Medium, Deep)
model_features = [
    "OL_epa",
    "OL_medium_target_epa",
    "qb_ewm_epa",
    "qb_ewm_medium_pass_epa",
    "qb_ewm_air_yards_efficiency",
    "qb_ewm_completion_percentage",
    "qb_ewm_medium_completion_percentage",
    "qb_ewm_int_rate",
    "qb_ewm_passer_rating",
    "rec_ewm_epa",
    "rec_ewm_medium_target_epa",
    "rec_ewm_air_yards_efficiency",
    "rec_ewm_catch_rate",
    "rec_ewm_medium_catch_rate",
    "def_epa",
    "def_medium_target_epa",
    "def_catch_rate",
    "def_medium_catch_rate",
    "def_air_yards_efficiency",
    "down",
    "ydstogo",
    "yardline_100",
    "distance_to_sticks",
    "air_yards",
    "number_of_pass_rushers",
]
medium_subset = df[df.pass_length == 1]
medium_subset.dropna(inplace=True)
X_train, X_test, y_train, y_test = train_test_split(
    medium_subset[model_features],
    medium_subset.complete_pass,
    test_size=0.25,
    shuffle=True,
)
model = NN().fit(X_train, y_train)
pickle.dump(model, open("NN_mediumPassModel.pkl", "wb"))
#%% Deep complete Pass Model
model_features = [
    "OL_epa",
    "OL_deep_target_epa",
    "qb_ewm_epa",
    "qb_ewm_deep_pass_epa",
    "qb_ewm_air_yards_efficiency",
    "qb_ewm_completion_percentage",
    "qb_ewm_deep_completion_percentage",
    "qb_ewm_int_rate",
    "qb_ewm_passer_rating",
    "rec_ewm_epa",
    "rec_ewm_deep_target_epa",
    "rec_ewm_air_yards_efficiency",
    "rec_ewm_catch_rate",
    "rec_ewm_deep_catch_rate",
    "def_epa",
    "def_deep_target_epa",
    "def_catch_rate",
    "def_deep_catch_rate",
    "def_air_yards_efficiency",
    "down",
    "ydstogo",
    "yardline_100",
    "distance_to_sticks",
    "air_yards",
    "number_of_pass_rushers",
]
deep_subset = df[df.pass_length == 2]
deep_subset.dropna(inplace=True)
X_train, X_test, y_train, y_test = train_test_split(
    deep_subset[model_features],
    deep_subset.complete_pass,
    test_size=0.25,
    shuffle=True,
)
model = NN().fit(X_train, y_train)
pickle.dump(model, open("NN_deepPassModel.pkl", "wb"))
#%% Pass Result QB Hit Models (Short, Medium, Deep)
model_features = [
    "OL_epa",
    "OL_short_target_epa",
    "qb_ewm_epa",
    "qb_ewm_short_pass_epa",
    "qb_ewm_air_yards_efficiency",
    "qb_ewm_completion_percentage",
    "qb_ewm_short_completion_percentage",
    "qb_ewm_int_rate",
    "qb_ewm_passer_rating",
    "rec_ewm_epa",
    "rec_ewm_short_target_epa",
    "rec_ewm_air_yards_efficiency",
    "rec_ewm_catch_rate",
    "rec_ewm_short_catch_rate",
    "def_epa",
    "def_short_target_epa",
    "def_catch_rate",
    "def_short_catch_rate",
    "def_air_yards_efficiency",
    "down",
    "ydstogo",
    "yardline_100",
    "distance_to_sticks",
    "air_yards",
    "number_of_pass_rushers",
    "qb_scramble"
]
short_subset = df[(df.pass_length == 0)&(df.qb_hit==1)]
short_subset.dropna(inplace=True)
X_train, X_test, y_train, y_test = train_test_split(
    short_subset[model_features],
    short_subset.complete_pass,
    test_size=0.25,
    shuffle=True,
)
model = NN().fit(X_train, y_train)
pickle.dump(model, open("NN_QBHitShortPassModel.pkl", "wb"))
#%% Pass Result Models (Short, Medium, Deep)
model_features = [
    "OL_epa",
    "OL_medium_target_epa",
    "qb_ewm_epa",
    "qb_ewm_medium_pass_epa",
    "qb_ewm_air_yards_efficiency",
    "qb_ewm_completion_percentage",
    "qb_ewm_medium_completion_percentage",
    "qb_ewm_int_rate",
    "qb_ewm_passer_rating",
    "rec_ewm_epa",
    "rec_ewm_medium_target_epa",
    "rec_ewm_air_yards_efficiency",
    "rec_ewm_catch_rate",
    "rec_ewm_medium_catch_rate",
    "def_epa",
    "def_medium_target_epa",
    "def_catch_rate",
    "def_medium_catch_rate",
    "def_air_yards_efficiency",
    "down",
    "ydstogo",
    "yardline_100",
    "distance_to_sticks",
    "air_yards",
    "number_of_pass_rushers",
    "qb_scramble"
]
medium_subset = df[(df.pass_length == 1)&(df.qb_hit==1)]
medium_subset.dropna(inplace=True)
X_train, X_test, y_train, y_test = train_test_split(
    medium_subset[model_features],
    medium_subset.complete_pass,
    test_size=0.25,
    shuffle=True,
)
model = NN().fit(X_train, y_train)
pickle.dump(model, open("NN_QBHitMediumPassModel.pkl", "wb"))
#%% Deep complete Pass Model
model_features = [
    "OL_epa",
    "OL_deep_target_epa",
    "qb_ewm_epa",
    "qb_ewm_deep_pass_epa",
    "qb_ewm_air_yards_efficiency",
    "qb_ewm_completion_percentage",
    "qb_ewm_deep_completion_percentage",
    "qb_ewm_int_rate",
    "qb_ewm_passer_rating",
    "rec_ewm_epa",
    "rec_ewm_deep_target_epa",
    "rec_ewm_air_yards_efficiency",
    "rec_ewm_catch_rate",
    "rec_ewm_deep_catch_rate",
    "def_epa",
    "def_deep_target_epa",
    "def_catch_rate",
    "def_deep_catch_rate",
    "def_air_yards_efficiency",
    "down",
    "ydstogo",
    "yardline_100",
    "distance_to_sticks",
    "air_yards",
    "number_of_pass_rushers",
    "qb_scramble"
]
deep_subset = df[(df.pass_length == 2)&(df.qb_hit==1)]
deep_subset.dropna(inplace=True)
X_train, X_test, y_train, y_test = train_test_split(
    deep_subset[model_features],
    deep_subset.complete_pass,
    test_size=0.25,
    shuffle=True,
)
model = NN().fit(X_train, y_train)
pickle.dump(model, open("NN_QBHitDeepPassModel.pkl", "wb"))
#%% YAC Model
model_features = [
    "rec_ewm_epa",
    "rec_ewm_yac_epa",
    "rec_ewm_yards_after_catch",
    "def_epa",
    "def_yac_epa",
    "def_yards_after_catch",
    "down",
    "ydstogo",
    "yardline_100",
    "distance_to_sticks",
    "air_yards",
    "number_of_pass_rushers",
]
yac_subset = df[df.yards_after_catch.isna() == False].dropna()
X_train, X_test, y_train, y_test = train_test_split(
    yac_subset[model_features],
    yac_subset.yards_after_catch,
    test_size=0.25,
    shuffle=True,
)
model = NN().fit(X_train, y_train)
pickle.dump(model, open("NN_YACModel.pkl", "wb"))
#%% Air Yards Model
model_features = [
    "qb_ewm_air_epa",
    "qb_ewm_air_yards_per_attempt",
    "qb_ewm_air_yards_efficiency",
    "def_air_epa",
    "def_air_yards_per_attempt",
    "def_air_yards_efficiency",
    "down",
    "ydstogo",
    "yardline_100",
    "score_differential",
    "half_seconds_remaining",
    "number_of_pass_rushers",
]
air_yards_subset = df[df.air_yards.isna() == False].dropna()
X_train, X_test, y_train, y_test = train_test_split(
    air_yards_subset[model_features],
    air_yards_subset.air_yards,
    test_size=0.25,
    shuffle=True,
)
model = NN().fit(X_train, y_train)
pickle.dump(model, open("NN_AirYardsModel.pkl", "wb"))
#%% Catch Fumble Model
temp = temp[temp.play_type == "pass"]
model_features = [
    "down",
    "ydstogo",
    "yardline_100",
    "score_differential",
    "half_seconds_remaining",
]
catch_fumble_subset = temp[model_features + ["complete_pass", "fumble_lost"]][
    temp.complete_pass == 1
].dropna()
model = NN().fit(
    catch_fumble_subset[model_features], catch_fumble_subset.fumble_lost
)
pickle.dump(model, open("NN_CatchFumbleModel.pkl", "wb"))
#%% Catch Fumble Return Yards Model
model_features = [
    "down",
    "ydstogo",
    "yardline_100",
    "score_differential",
    "half_seconds_remaining",
]
catch_fumble_return_yards_subset = temp[
    model_features + ["complete_pass", "fumble_lost", "fumble_recovery_1_yards"]
][(temp.complete_pass == 1) & (temp.fumble_lost == 1)].dropna()
model = NN().fit(
    catch_fumble_return_yards_subset[model_features],
    catch_fumble_return_yards_subset.fumble_recovery_1_yards,
)
pickle.dump(model, open("NN_CatchFumbleReturnYardsModel.pkl", "wb"))
#%% Interception Return Yards Model
model_features = [
    "down",
    "ydstogo",
    "yardline_100",
    "score_differential",
    "half_seconds_remaining",
]
int_return_yards_subset = temp[
    model_features + ["interception", "return_yards"]
][(temp.interception == 1)].dropna()
model = NN().fit(
    int_return_yards_subset[model_features],
    int_return_yards_subset.return_yards,
)
pickle.dump(model, open("NN_InterceptionReturnYardsModel.pkl", "wb"))
#%% Elapsed Time Complete Pass Model
model_features = [
    "air_yards",
    "yards_after_catch",
    "out_of_bounds",
    "clock_running",
]
et_complete_pass_subset = temp[
    model_features + ["fumble_lost", "complete_pass", "elapsed_time"]
][(temp.complete_pass==1)&
  (temp.fumble_lost==0)&
  (temp.elapsed_time>0)].dropna()
X_train, X_test, y_train, y_test = train_test_split(
    et_complete_pass_subset[model_features],
    et_complete_pass_subset.elapsed_time,
    test_size=0.01,
    shuffle=True,
)
model = NN().fit(X_train, y_train)
pickle.dump(model, open("NN_CompletePassElapsedTime.pkl", "wb"))
model_features = [
    "air_yards",
]
et_incomplete_pass_subset = temp[
    model_features + ["fumble_lost", "interception","sack", "complete_pass", "elapsed_time"]
][(temp.complete_pass==0)&
  (temp.fumble_lost==0)&
  (temp.interception==0)&
  (temp.sack==0)&
  (temp.elapsed_time>0)].dropna()
X_train, X_test, y_train, y_test = train_test_split(
    et_incomplete_pass_subset[model_features],
    et_incomplete_pass_subset.elapsed_time,
    test_size=0.01,
    shuffle=True,
)
model = NN().fit(X_train, y_train)
pickle.dump(model, open("NN_IncompletePassElapsedTime.pkl", "wb"))
#%% Elapsed Time Complete Pass Model 4th Quarter
model_features = [
    "air_yards",
    "yards_after_catch",
    "out_of_bounds",
    "clock_running",
]
et_complete_pass_subset = temp[
    model_features + ["fumble_lost", "complete_pass", "elapsed_time"]
][(temp.complete_pass==1)&
  (temp.fumble_lost==0)&
  (temp.elapsed_time>0)&
  (temp.qtr==4)].dropna()
X_train, X_test, y_train, y_test = train_test_split(
    et_complete_pass_subset[model_features],
    et_complete_pass_subset.elapsed_time,
    test_size=0.01,
    shuffle=True,
)
model = NN().fit(X_train, y_train)
pickle.dump(model, open("NN_4thQtrCompletePassElapsedTime.pkl", "wb"))
model_features = [
    "air_yards",
]
et_incomplete_pass_subset = temp[
    model_features + ["fumble_lost", "interception","sack", "complete_pass", "elapsed_time"]
][(temp.complete_pass==0)&
  (temp.fumble_lost==0)&
  (temp.interception==0)&
  (temp.sack==0)&
  (temp.elapsed_time>0)&
  (temp.qtr==4)].dropna()
X_train, X_test, y_train, y_test = train_test_split(
    et_incomplete_pass_subset[model_features],
    et_incomplete_pass_subset.elapsed_time,
    test_size=0.01,
    shuffle=True,
)
model = NN().fit(X_train, y_train)
pickle.dump(model, open("NN_4thQtrIncompletePassElapsedTime.pkl", "wb"))
#%% Elapsed Time Pass Intercepted Mode
model_features = [
    "down",
    "ydstogo",
    "yardline_100",
    "half_seconds_remaining",
    "return_yards",
]
et_pass_int_subset = temp[
    model_features + ["fumble_lost", "interception", "elapsed_time"]
][(temp.fumble_lost == 0) & (temp.interception == 1)].dropna()
X_train, X_test, y_train, y_test = train_test_split(
    et_pass_int_subset[model_features],
    et_pass_int_subset.elapsed_time,
    test_size=0.25,
    shuffle=True,
)
model = NN().fit(X_train, y_train)
pickle.dump(model, open("NN_PassIntElapsedTime.pkl", "wb"))
#%% Elapsed Time Catch Fumble Model
model_features = [
    "down",
    "ydstogo",
    "yardline_100",
    "half_seconds_remaining",
    "fumble_recovery_1_yards",
    "air_yards",
    "yards_after_catch",
]
et_catch_fumble_subset = temp[
    model_features + ["fumble_lost", "complete_pass", "elapsed_time"]
][(temp.fumble_lost == 1) & (temp.complete_pass == 1)].dropna()
X_train, X_test, y_train, y_test = train_test_split(
    et_catch_fumble_subset[model_features],
    et_catch_fumble_subset.elapsed_time,
    test_size=0.25,
    shuffle=True,
)
model = NN().fit(X_train, y_train)
pickle.dump(model, open("NN_CatchFumbleElapsedTime.pkl", "wb"))


#%% Elapsed Time Catch Fumble Model
model_features = [
    "down",
    "ydstogo",
    "yardline_100",
    "half_seconds_remaining",
    "air_yards",
    "yards_after_catch",
]
rec_ob_subset = temp[
    model_features + ["complete_pass", "elapsed_time",'out_of_bounds']
][(temp.complete_pass == 1)].dropna()
X_train, X_test, y_train, y_test = train_test_split(
    rec_ob_subset[model_features],
    rec_ob_subset.elapsed_time,
    test_size=0.25,
    shuffle=True,
)
model = NN().fit(X_train, y_train)
pickle.dump(model, open("NN_RecOB.pkl", "wb"))

