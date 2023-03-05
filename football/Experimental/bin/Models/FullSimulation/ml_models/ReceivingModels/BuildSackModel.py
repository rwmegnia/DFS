#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 10:55:57 202

@author: robertmegnia
"""
import pandas as pd
import nfl_data_py as nfl
import os
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier as NN
from sklearn.dummy import DummyClassifier as DC
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
# %% Filter Plays to Pass Plays
temp["time"] = temp.time.apply(
    lambda x: datetime.strptime(x, "%M:%S") if x is not np.nan else x
)
temp["end_time"] = temp.time.shift(-1)
temp["elapsed_time"] = (temp.time - temp.end_time).apply(
    lambda x: x.total_seconds()
)
pbp = temp[temp.n_offense.isna() == False]
pbp = pbp[pbp.play_type == "pass"]
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

passing = pbp[(pbp.play_type == "pass")]
passing = passing[(passing.n_offense == 11) & (passing.n_defense == 11)]
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
OffData = pd.read_csv("./OffSacRateData.csv")
OffData.loc[OffData.position == "QB", "qb_sack_rate"] = OffData.loc[
    OffData.position == "QB", "sack_allowed_rate"
]
DefData = pd.read_csv("./DefSacRateData.csv")
#%%
OffData = OffData[
    ["gsis_id", "posteam", "game_id", "sack_allowed_rate", "qb_sack_rate"]
]

DefData = DefData[["gsis_id", "defteam", "game_id", "sack_rate"]]
#%%
pbp_passing_frames = []
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
        OffData, on=["gsis_id", "game_id", "posteam"], how="left"
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
        DefData, on=["gsis_id", "game_id", "defteam"], how="left"
    )
    passing_frame.reset_index(drop=True, inplace=True)
    pbp_passing_frames.append(passing_frame)
#%%
df = pd.concat(pbp_passing_frames)
df["score_differential"] = df.posteam_score - df.defteam_score
df = df.groupby(["game_id", "play_id"]).mean()[
    [
        "sack_allowed_rate",
        "qb_sack_rate",
        "sack_rate",
        "down",
        "ydstogo",
        "yardline_100",
        "score_differential",
        "half_seconds_remaining",
        "number_of_pass_rushers",
        "yards_gained",
        "elapsed_time",
        "fumble",
        "fumble_lost",
        "fumble_recovery_1_yards",
        "return_touchdown",
        "return_yards",
        "sack",
    ]
]
#%% Sack Model
model_features = [
    "sack_allowed_rate",
    "qb_sack_rate",
    "sack_rate",
    "down",
    "ydstogo",
    "yardline_100",
    "number_of_pass_rushers",
    "score_differential",
    "half_seconds_remaining",
]
sacked_subset = df[model_features + ["sack"]].dropna()
X_train, X_test, y_train, y_test = train_test_split(
    sacked_subset[model_features],
    sacked_subset.sack,
    test_size=0.25,
    shuffle=True,
)
model = NN().fit(X_train, y_train)
pickle.dump(model, open("NN_sackModel.pkl", "wb"))

#%% Sack Lost Yards Model
model_features = [
    "sack_allowed_rate",
    "qb_sack_rate",
    "sack_rate",
    "down",
    "ydstogo",
    "yardline_100",
    "number_of_pass_rushers",
    "score_differential",
    "half_seconds_remaining",
]
sack_yards_subset = df[df.sack == 1][model_features + ["yards_gained"]].dropna()
X_train, X_test, y_train, y_test = train_test_split(
    sack_yards_subset[model_features],
    sack_yards_subset.yards_gained,
    test_size=0.25,
    shuffle=True,
)
model = NN().fit(X_train, y_train)
pickle.dump(model, open("NN_sackYardsModel.pkl", "wb"))
#%% Sack Elapsed Time Model
model_features = [
    "down",
    "ydstogo",
    "yardline_100",
    "score_differential",
    "half_seconds_remaining",
    "yards_gained",
]
sack_elapsed_time_subset = df[df.sack == 1][
    model_features + ["elapsed_time"]
].dropna()
X_train, X_test, y_train, y_test = train_test_split(
    sack_elapsed_time_subset[model_features],
    sack_elapsed_time_subset.elapsed_time,
    test_size=0.25,
    shuffle=True,
)
model = NN().fit(X_train, y_train)
pickle.dump(model, open("NN_sackElapsedTimeModel.pkl", "wb"))
#%% Strip Sack Model
model_features = [
    "sack_allowed_rate",
    "qb_sack_rate",
    "sack_rate",
    "down",
    "ydstogo",
    "yardline_100",
    "number_of_pass_rushers",
    "score_differential",
    "half_seconds_remaining",
]
strip_sack_subset = df[df.sack == 1][model_features + ["fumble_lost"]].dropna()
X_train, X_test, y_train, y_test = train_test_split(
    strip_sack_subset[model_features],
    strip_sack_subset.fumble_lost,
    test_size=0.25,
    shuffle=True,
)
model = NN().fit(X_train, y_train)
pickle.dump(model, open("NN_stripSackModel.pkl", "wb"))
#%% Strip Sack Return Yards Model
model_features = ["down"]
strip_sack_return_yards_subset = df[(df.sack == 1) & (df.fumble_lost == 1)][
    model_features + ["return_yards"]
].dropna()
model = DC(strategy="stratified").fit(
    strip_sack_return_yards_subset[["down"]],
    strip_sack_return_yards_subset.return_yards,
)
pickle.dump(model, open("NN_stripSackReturnYardsModel.pkl", "wb"))

#%% Strip Sack Return Touchdown Model
model_features = ["down"]
strip_sack_return_td_subset = df[(df.sack == 1) & (df.fumble_lost == 1)][
    model_features + ["return_touchdown"]
].dropna()
model = DC(strategy="stratified").fit(
    strip_sack_return_td_subset[["down"]],
    strip_sack_return_td_subset.return_touchdown,
)
pickle.dump(model, open("NN_stripSackReturnTDModel.pkl", "wb"))
#%% Strip Sack Elapsed Time Model
model_features = ["return_yards", "return_touchdown", "yards_gained"]
strip_sack_et_subset = df[(df.sack == 1) & (df.fumble_lost == 1)][
    model_features + ["elapsed_time"]
].dropna()
X_train, X_test, y_train, y_test = train_test_split(
    strip_sack_et_subset[model_features],
    strip_sack_et_subset.elapsed_time,
    test_size=0.25,
    shuffle=True,
)
model = NN().fit(X_train, y_train)
pickle.dump(model, open("NN_stripSackElapsedTimeModel.pkl", "wb"))

#%%
OffData = pd.read_csv("./OffHitRateData.csv")
OffData.loc[OffData.position == "QB", "qb_hit_rate"] = OffData.loc[
    OffData.position == "QB", "hit_allowed_rate"
]
DefData = pd.read_csv("./DefHitRateData.csv")
#%%
OffData = OffData[
    ["gsis_id", "posteam", "game_id", "hit_allowed_rate", "qb_hit_rate"]
]

DefData = DefData[["gsis_id", "defteam", "game_id", "hit_rate"]]
#%%
pbp_passing_frames = []
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
        OffData, on=["gsis_id", "game_id", "posteam"], how="left"
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
        DefData, on=["gsis_id", "game_id", "defteam"], how="left"
    )
    passing_frame.reset_index(drop=True, inplace=True)
    pbp_passing_frames.append(passing_frame)
#%%
df = pd.concat(pbp_passing_frames)
df["score_differential"] = df.posteam_score - df.defteam_score
df = df[df.sack==0].groupby(["game_id", "play_id"]).mean()[
    [
        "hit_allowed_rate",
        "qb_hit_rate",
        "hit_rate",
        "down",
        "ydstogo",
        "yardline_100",
        "score_differential",
        "half_seconds_remaining",
        "number_of_pass_rushers",
        "yards_gained",
        "elapsed_time",
        "qb_hit",
    ]
]
#%% QB Hit Model
model_features = [
    "hit_allowed_rate",
    "qb_hit_rate",
    "hit_rate",
    "down",
    "ydstogo",
    "yardline_100",
    "number_of_pass_rushers",
    "score_differential",
    "half_seconds_remaining",
]
hit_subset = df[model_features + ["qb_hit"]].dropna()
X_train, X_test, y_train, y_test = train_test_split(
    hit_subset[model_features],
    hit_subset.qb_hit,
    test_size=0.25,
    shuffle=True,
)
model = NN().fit(X_train, y_train)
pickle.dump(model, open("NN_QBHitModel.pkl", "wb"))
