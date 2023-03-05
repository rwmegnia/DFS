#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 13:34:29 2023

@author: robertmegnia
"""

import pandas as pd
import numpy as np

roster = pd.read_csv("2016_2022_Rosters.csv")
roster.rename({"club_code": "team"}, axis=1, inplace=True)
roster.team.replace({"OAK": "LV", "SD": "LAC"}, inplace=True)
#%% Get QB Data
qbs = roster[roster.position == "QB"]

# Read in QB Modeling Data
qb_data = pd.read_csv("../ml_models/ReceivingModels/QB_Passing_Model_Data.csv")
qb_rush_data = pd.read_csv(
    "../ml_models/RushingModels/Rushing_Data/RushingData.csv"
)
qb_rush_data = qb_rush_data[qb_rush_data.position == "QB"]
qb_sack_data = pd.read_csv("../ml_models/ReceivingModels/OffSacRateData.csv")
qb_sack_data = qb_sack_data[qb_sack_data.position == "QB"]
qb_team_data = pd.read_csv(
    "../ml_models/ReceivingModels/QB_Team_Passing_Model_Data.csv"
)
qb_team_rush_data = pd.read_csv(
    "../ml_models/RushingModels/Rushing_Data/TeamRushingData.csv"
)
qb_team_sack_data = pd.read_csv(
    "../ml_models/ReceivingModels/TeamOffSacRateData.csv"
)

# Rename team column
qb_data.rename(
    {"posteam": "team", "passer_player_id": "gsis_id"}, axis=1, inplace=True
)
qb_rush_data.rename({"posteam": "team"}, axis=1, inplace=True)
qb_sack_data.rename({"posteam": "team"}, axis=1, inplace=True)
qb_team_data.rename({"posteam": "team"}, axis=1, inplace=True)
qb_team_rush_data.rename({"posteam": "team"}, axis=1, inplace=True)
qb_team_sack_data.rename({"posteam": "team"}, axis=1, inplace=True)
# Drop unneeded columns
qb_data.drop(
    [
        "game_id",
    ],
    axis=1,
    inplace=True,
)
qb_rush_data.drop(
    [
        "game_id",
        "position",
        "game_date",
        "team_rush_epa",
        "team_ypc",
        "team_open_field_ypc",
    ],
    axis=1,
    inplace=True,
)
qb_sack_data.drop(
    ["game_id", "game_date", "team_sack_allowed_rate", "position"],
    axis=1,
    errors="ignore",
    inplace=True,
)
qb_team_data.drop(["game_id"], axis=1, inplace=True)
qb_team_rush_data.drop(["game_id", "game_date"], axis=1, inplace=True)
qb_team_sack_data.drop(["game_id", "game_date"], axis=1, inplace=True)
qbs = qbs.merge(qb_data, on=["week", "season", "team", "gsis_id"], how="left")
qbs = qbs.merge(
    qb_rush_data, on=["week", "season", "team", "gsis_id"], how="left"
)
qbs = qbs.merge(
    qb_sack_data, on=["week", "season", "team", "gsis_id"], how="left"
)
qbs = qbs.merge(qb_team_data, on=["week", "season", "team"], how="left")
qbs = qbs.merge(
    qb_team_rush_data, on=["week", "season", "team", "position"], how="left"
)
qbs = qbs.merge(qb_team_sack_data, on=["week", "season", "team"], how="left")
for stat in qb_data.drop(["gsis_id", "team", "week", "season"], axis=1).columns:
    qbs.loc[qbs[stat].isna() == True, stat] = qbs.loc[
        qbs[stat].isna() == True, f"team_{stat}"
    ]

for stat in qb_rush_data.drop(
    ["gsis_id", "team", "week", "season", "rush_share"], axis=1
).columns:
    qbs.loc[qbs[stat].isna() == True, stat] = qbs.loc[
        qbs[stat].isna() == True, f"team_{stat}"
    ]

qbs.loc[qbs.sack_allowed_rate.isna() == True, "sack_allowed_rate"] = qbs.loc[
    qbs.sack_allowed_rate.isna() == True, "team_sack_allowed_rate"
]
qbs.rename({"sack_allowed_rate": "qb_sack_rate"}, axis=1, inplace=True)
qbs.drop([c for c in qbs.columns if "team_" in c], axis=1, inplace=True)

#%% Read in RB Modeling Data
rbs = roster[roster.position.isin(["FB", "RB"])]

rush_data = pd.read_csv(
    "../ml_models/RushingModels/Rushing_Data/RushingData.csv"
)
rec_data = pd.read_csv("../ml_models/ReceivingModels/Receiving_Model_Data.csv")
team_rush_data = pd.read_csv(
    "../ml_models/RushingModels/Rushing_Data/TeamRushingData.csv"
)
team_rec_data = pd.read_csv(
    "../ml_models/ReceivingModels/Team_Receiving_Model_Data.csv"
)
rush_data = rush_data[rush_data.position == "RB"]
rush_data.rename({"posteam": "team"}, axis=1, inplace=True)
rec_data.rename(
    {"posteam": "team", "receiver_player_id": "gsis_id"}, axis=1, inplace=True
)
team_rush_data.rename({"posteam": "team"}, axis=1, inplace=True)
team_rec_data.rename({"posteam": "team"}, axis=1, inplace=True)
rush_data.drop(
    [
        "game_id",
        "game_date",
        "position",
        "team_rush_epa",
        "team_ypc",
        "team_open_field_ypc",
    ],
    axis=1,
    inplace=True,
)

rec_data.drop(
    [
        "game_id",
    ],
    axis=1,
    inplace=True,
)
team_rush_data.drop(["game_id", "game_date"], axis=1, inplace=True)
team_rec_data.drop(
    [
        "game_id",
    ],
    axis=1,
    inplace=True,
)
rbs = rbs.merge(rush_data, on=["week", "season", "team", "gsis_id"], how="left")
rbs = rbs.merge(
    team_rush_data, on=["team", "week", "season", "position"], how="left"
)
rbs = rbs.merge(rec_data, on=["team", "week", "season", "gsis_id"], how="left")
rbs = rbs.merge(team_rec_data, on=["team", "week", "season"], how="left")

for stat in rush_data.drop(
    ["week", "season", "team", "gsis_id", "rush_share"], axis=1
).columns:
    rbs.loc[rbs[stat].isna() == True, stat] = rbs.loc[
        rbs[stat].isna() == True, f"team_{stat}"
    ]

for stat in rec_data.drop(
    ["week", "season", "team", "gsis_id"], axis=1
).columns:
    if ("share" in stat) | ("wopr" in stat):
        continue
    rbs.loc[rbs[stat].isna() == True, stat] = rbs.loc[
        rbs[stat].isna() == True, f"team_{stat}"
    ]
rbs.drop([c for c in rbs.columns if "team_" in c], axis=1, inplace=True)
#%% Receiver Data
recs = roster[roster.position.isin(["WR", "TE"])]
rec_data = pd.read_csv("../ml_models/ReceivingModels/Receiving_Model_Data.csv")
rec_rush_data = pd.read_csv(
    "../ml_models/RushingModels/Rushing_Data/RushingData.csv"
)
team_rec_data = pd.read_csv(
    "../ml_models/ReceivingModels/Team_Receiving_Model_Data.csv"
)
team_rec_rush_data = pd.read_csv(
    "../ml_models/RushingModels/Rushing_Data/TeamRushingData.csv"
)
rec_data.rename(
    {"receiver_player_id": "gsis_id", "posteam": "team"}, axis=1, inplace=True
)

rec_data.drop(
    [
        "game_id",
    ],
    axis=1,
    inplace=True,
)
rec_rush_data.rename(
    {
        "posteam": "team",
    },
    axis=1,
    inplace=True,
)
rec_rush_data.drop(
    [
        "game_id",
        "game_date",
        "team_rush_epa",
        "team_ypc",
        "team_open_field_ypc",
    ],
    axis=1,
    inplace=True,
)
team_rec_data.rename({"posteam": "team"}, axis=1, inplace=True)
team_rec_data.drop(
    [
        "game_id",
    ],
    axis=1,
    inplace=True,
)
team_rec_rush_data.rename({"posteam": "team"}, axis=1, inplace=True)
team_rec_rush_data.drop(["game_id", "game_date"], axis=1, inplace=True)
recs = recs.merge(
    rec_data, on=["team", "week", "season", "gsis_id"], how="left"
)
recs = recs.merge(
    rec_rush_data,
    on=["team", "week", "season", "position", "gsis_id"],
    how="left",
)
recs = recs.merge(team_rec_data, on=["team", "week", "season"], how="left")
recs = recs.merge(
    team_rec_rush_data, on=["team", "week", "season", "position"], how="left"
)
for stat in rec_data.drop(
    ["week", "season", "team", "gsis_id"], axis=1
).columns:
    if ("share" in stat) | ("wopr" in stat):
        continue
    recs.loc[recs[stat].isna() == True, stat] = recs.loc[
        recs[stat].isna() == True, f"team_{stat}"
    ]

for stat in rec_rush_data.drop(
    ["week", "season", "team", "gsis_id", "position", "rush_share"], axis=1
).columns:
    if ("share" in stat) | ("wopr" in stat):
        continue
    recs.loc[recs[stat].isna() == True, stat] = recs.loc[
        recs[stat].isna() == True, f"team_{stat}"
    ]

recs.drop([c for c in recs.columns if "team_" in c], axis=1, inplace=True)
#%% OL Data

ols = roster[roster.position == "OL"]
ol_rush_data = pd.read_csv(
    "../ml_models/RushingModels/OL_Data/OL_RushingData.csv"
)
ol_pass_data = pd.read_csv(
    "../ml_models/ReceivingModels/OL_Data/OL_PassingData.csv"
)
ol_sack_data = pd.read_csv("../ml_models/ReceivingModels/OffSacRateData.csv")
ol_sack_data = ol_sack_data[ol_sack_data.position == "OL"]
team_ol_rush_data = pd.read_csv(
    "../ml_models/RushingModels/OL_Data/OL_TeamRushingData.csv"
)
team_ol_pass_data = pd.read_csv(
    "../ml_models/ReceivingModels/OL_Data/OL_TeamPassingData.csv"
)
team_ol_sack_data = pd.read_csv(
    "../ml_models/ReceivingModels/TeamOffSacRateData.csv"
)
ol_rush_data.rename({"posteam": "team"}, axis=1, inplace=True)
ol_pass_data.rename({"posteam": "team"}, axis=1, inplace=True)
ol_sack_data.rename({"posteam": "team"}, axis=1, inplace=True)
team_ol_rush_data.rename({"posteam": "team"}, axis=1, inplace=True)
team_ol_pass_data.rename({"posteam": "team"}, axis=1, inplace=True)
team_ol_sack_data.rename({"posteam": "team"}, axis=1, inplace=True)
ol_pass_data.drop(
    [
        "game_id",
        "position",
        "game_date",
        "team_OL_epa",
        "team_OL_short_target_epa",
        "team_OL_medium_target_epa",
        "team_OL_deep_target_epa",
    ],
    axis=1,
    errors="ignore",
    inplace=True,
)
ol_rush_data.drop(
    [
        "game_id",
        "position",
        "game_date",
        "ypc",
        "rush_epa",
        "open_field_ypc",
        "second_level_rushing_yards",
        "team_line_epa",
        "team_adjusted_line_yards",
    ],
    axis=1,
    errors="ignore",
    inplace=True,
)
ol_sack_data.drop(
    ["game_id", "game_date", "position", "team_sack_allowed_rate"],
    axis=1,
    errors="ignore",
    inplace=True,
)
team_ol_rush_data.drop(
    ["game_id", "game_date"], axis=1, errors="ignore", inplace=True
)
team_ol_pass_data.drop(
    ["game_id", "game_date"], axis=1, errors="ignore", inplace=True
)
team_ol_sack_data.drop(
    ["game_id", "game_date"], axis=1, errors="ignore", inplace=True
)
ols = ols.merge(
    ol_rush_data, on=["week", "season", "team", "gsis_id"], how="left"
)
ols = ols.merge(
    ol_pass_data, on=["week", "season", "team", "gsis_id"], how="left"
)
ols = ols.merge(
    ol_sack_data, on=["week", "season", "team", "gsis_id"], how="left"
)
ols = ols.merge(
    team_ol_rush_data, on=["week", "season", "team", "position"], how="left"
)
ols = ols.merge(team_ol_pass_data, on=["week", "season", "team"], how="left")
ols = ols.merge(team_ol_sack_data, on=["week", "season", "team"], how="left")
for stat in [
    "line_epa",
    "adjusted_line_yards",
    "OL_epa",
    "OL_short_target_epa",
    "OL_medium_target_epa",
    "OL_deep_target_epa",
    "sack_allowed_rate",
]:
    ols.loc[ols[stat].isna() == True, stat] = ols.loc[
        ols[stat].isna() == True, f"team_{stat}"
    ]

ols.drop(
    [
        "team_line_epa",
        "team_adjusted_line_yards",
        "team_OL_epa",
        "team_OL_short_target_epa",
        "team_OL_medium_target_epa",
        "team_OL_deep_target_epa",
        "team_sack_allowed_rate",
    ],
    axis=1,
    errors="ignore",
    inplace=True,
)
#%% Defensive Linemen
dls = roster[roster.position == "DL"]
dls.team.replace({"OAK": "LV", "SD": "LAC"}, inplace=True)
dl_rush_data = pd.read_csv(
    "../ml_models/RushingModels/DL_Data/DL_RushingStats.csv"
)
dl_pass_data = pd.read_csv("../ml_models/ReceivingModels/DefensivePassData.csv")
dl_pass_data = dl_pass_data[dl_pass_data.position == "DL"]
dl_sack_data = pd.read_csv("../ml_models/ReceivingModels/DefSacRateData.csv")
dl_sack_data = dl_sack_data[dl_sack_data.position == "DL"]
team_dl_rush_data = pd.read_csv(
    "../ml_models/RushingModels/DL_Data/DL_TeamRushingStats.csv"
)
team_dl_pass_data = pd.read_csv(
    "../ml_models/ReceivingModels/TeamDefensivePassData.csv"
)
team_dl_pass_data.drop([
    'short_target',
    'short_rec',
    'medium_target',
    'medium_rec',
    'deep_target',
    'deep_rec',
    'targets',
    'rec',
    'receiving_yards',
    'air_yards',
    'completed_air_yards',
    'posteam_pass_attempts',
    'posteam_air_yards',],axis=1,inplace=True)
team_dl_pass_data = team_dl_pass_data[team_dl_pass_data.position == "DL"]
team_dl_sack_data = pd.read_csv(
    "../ml_models/ReceivingModels/TeamDefSacRateData.csv"
)
dl_rush_data.rename({"defteam": "team"}, axis=1, inplace=True)
dl_pass_data.rename({"defteam": "team"}, axis=1, inplace=True)
dl_sack_data.rename({"defteam": "team"}, axis=1, inplace=True)
team_dl_rush_data.rename({"defteam": "team"}, axis=1, inplace=True)
team_dl_pass_data.rename({"defteam": "team"}, axis=1, inplace=True)
team_dl_sack_data.rename({"defteam": "team"}, axis=1, inplace=True)
dl_pass_data.drop(
    [
        "game_id",
        "position",
        "game_date",
        "team_def_epa",
        "team_def_yac_epa",
        "team_def_yards_after_catch",
        "team_def_short_target_epa",
        "team_def_medium_target_epa",
        "team_def_deep_target_epa",
        "team_def_air_yards_efficiency",
        "team_def_catch_rate",
        "team_def_short_catch_rate",
        "team_def_medium_catch_rate",
        "team_def_deep_catch_rate",
    ],
    axis=1,
    errors="ignore",
    inplace=True,
)
dl_rush_data.drop(
    [
        "game_id",
        "position",
        "game_date",
        "team_def_line_epa",
        "team_def_adjusted_line_yards",
    ],
    axis=1,
    errors="ignore",
    inplace=True,
)
dl_sack_data.drop(
    [
        "game_id",
        "game_date",
        "position",
        "team_sack_rate",
        "pass_attempt",
        "play_id",
    ],
    axis=1,
    errors="ignore",
    inplace=True,
)
team_dl_rush_data.drop(
    ["game_id", "game_date","position"], axis=1, errors="ignore", inplace=True
)
team_dl_pass_data.drop(
    ["game_id", "game_date", "position"], axis=1, errors="ignore", inplace=True
)
team_dl_sack_data.drop(
    ["game_id", "game_date"], axis=1, errors="ignore", inplace=True
)
dls = dls.merge(
    dl_rush_data, on=["week", "season", "team", "gsis_id"], how="left"
)
dls = dls.merge(
    dl_pass_data, on=["week", "season", "team", "gsis_id"], how="left"
)
dls = dls.merge(
    dl_sack_data, on=["week", "season", "team", "gsis_id"], how="left"
)
dls = dls.merge(team_dl_rush_data, on=["week", "season", "team"], how="left")
dls = dls.merge(team_dl_pass_data, on=["week", "season", "team"], how="left")
dls = dls.merge(team_dl_sack_data, on=["week", "season", "team"], how="left")
for stat in [
    "def_line_epa",
    "def_adjusted_line_yards",
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
    "sack_rate",
]:
    dls.loc[dls[stat].isna() == True, stat] = dls.loc[
        dls[stat].isna() == True, f"team_{stat}"
    ]
dls = dls[dls.team_def_epa.isna() == False]
dls.drop(
    [
        "team_def_epa",
        "team_def_line_epa",
        "team_def_adjusted_line_yards",
        "team_def_air_epa",
        "team_def_air_yards_per_attempt",
        "team_def_yac_epa",
        "team_def_yards_after_catch",
        "team_def_short_target_epa",
        "team_def_medium_target_epa",
        "team_def_deep_target_epa",
        "team_def_air_yards_efficiency",
        "team_def_catch_rate",
        "team_def_short_catch_rate",
        "team_def_medium_catch_rate",
        "team_def_deep_catch_rate",
        "team_sack_rate",
    ],
    axis=1,
    errors="ignore",
    inplace=True,
)
dls.drop_duplicates(inplace=True)
dls = dls.groupby(["week", "season", "gsis_id"], as_index=False).first()
#%%
lbs = roster[roster.position == "LB"]
lbs.team.replace({"OAK": "LV", "SD": "LAC"}, inplace=True)
lb_rush_data = pd.read_csv(
    "../ml_models/RushingModels/LB_Data/LB_RushingStats.csv"
)
lb_pass_data = pd.read_csv("../ml_models/ReceivingModels/DefensivePassData.csv")
lb_pass_data = lb_pass_data[lb_pass_data.position == "LB"]
lb_sack_data = pd.read_csv("../ml_models/ReceivingModels/DefSacRateData.csv")
lb_sack_data = lb_sack_data[lb_sack_data.position == "LB"]
team_lb_rush_data = pd.read_csv(
    "../ml_models/RushingModels/LB_Data/LB_TeamRushingStats.csv"
)
team_lb_rush_data.drop([
    'gsis_id',
    'position',
    'game_id',
    'def_second_level_rushing_yards',
    "game_date"
    ],axis=1,inplace=True)
team_lb_rush_data
team_lb_pass_data = pd.read_csv(
    "../ml_models/ReceivingModels/TeamDefensivePassData.csv"
)
team_lb_pass_data.drop([
    'short_target',
    'short_rec',
    'medium_target',
    'medium_rec',
    'deep_target',
    'deep_rec',
    'targets',
    'rec',
    'receiving_yards',
    'air_yards',
    'completed_air_yards',
    'posteam_pass_attempts',
    'posteam_air_yards',],axis=1,inplace=True)
team_lb_pass_data = team_lb_pass_data[team_lb_pass_data.position == "LB"]
team_lb_sack_data = pd.read_csv(
    "../ml_models/ReceivingModels/TeamDefSacRateData.csv"
)
lb_rush_data.rename({"defteam": "team"}, axis=1, inplace=True)
lb_pass_data.rename({"defteam": "team"}, axis=1, inplace=True)
lb_sack_data.rename({"defteam": "team"}, axis=1, inplace=True)
team_lb_rush_data.rename({"defteam": "team"}, axis=1, inplace=True)
team_lb_pass_data.rename({"defteam": "team"}, axis=1, inplace=True)
team_lb_sack_data.rename({"defteam": "team"}, axis=1, inplace=True)
lb_pass_data.drop(
    [
        "game_id",
        "position",
        "game_date",
        "team_def_epa",
        "team_def_yac_epa",
        "team_def_yards_after_catch",
        "team_def_short_target_epa",
        "team_def_medium_target_epa",
        "team_def_deep_target_epa",
        "team_def_air_yards_efficiency",
        "team_def_catch_rate",
        "team_def_short_catch_rate",
        "team_def_medium_catch_rate",
        "team_def_deep_catch_rate",
    ],
    axis=1,
    errors="ignore",
    inplace=True,
)
lb_rush_data.drop(
    [
        "game_id",
        "position",
        "game_date",
        "team_def_second_level_rushing_yards",
    ],
    axis=1,
    errors="ignore",
    inplace=True,
)
lb_sack_data.drop(
    [
        "game_id",
        "game_date",
        "position",
        "team_sack_rate",
        "pass_attempt",
        "play_id",
    ],
    axis=1,
    errors="ignore",
    inplace=True,
)

team_lb_pass_data.drop(
    ["game_id", "game_date", "position"], axis=1, errors="ignore", inplace=True
)
team_lb_sack_data.drop(
    ["game_id", "game_date"], axis=1, errors="ignore", inplace=True
)
lbs = lbs.merge(
    lb_rush_data, on=["week", "season", "team", "gsis_id"], how="left"
)
lbs = lbs.merge(
    lb_pass_data, on=["week", "season", "team", "gsis_id"], how="left"
)
lbs = lbs.merge(
    lb_sack_data, on=["week", "season", "team", "gsis_id"], how="left"
)
lbs = lbs.merge(team_lb_rush_data, on=["week", "season", "team"], how="left")
lbs = lbs.merge(team_lb_pass_data, on=["week", "season", "team"], how="left")
lbs = lbs.merge(team_lb_sack_data, on=["week", "season", "team"], how="left")
for stat in [
    "def_second_level_rushing_yards",
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
    "sack_rate",
]:
    lbs.loc[lbs[stat].isna() == True, stat] = lbs.loc[
        lbs[stat].isna() == True, f"team_{stat}"
    ]
lbs = lbs[lbs.team_def_epa.isna() == False]
lbs.drop(
    [
        "team_def_epa",
        "team_def_line_epa",
        "team_def_adjusted_line_yards",
        "team_def_air_epa",
        "team_def_air_yards_per_attempt",
        "team_def_yac_epa",
        "team_def_yards_after_catch",
        "team_def_short_target_epa",
        "team_def_medium_target_epa",
        "team_def_deep_target_epa",
        "team_def_air_yards_efficiency",
        "team_def_catch_rate",
        "team_def_short_catch_rate",
        "team_def_medium_catch_rate",
        "team_def_deep_catch_rate",
        "team_sack_rate",
        "team_def_second_level_rushing_yards"
    ],
    axis=1,
    errors="ignore",
    inplace=True,
)
lbs.drop_duplicates(inplace=True)
lbs = lbs.groupby(["week", "season", "gsis_id"], as_index=False).first()
#%%
dbs = roster[roster.position == "DB"]
dbs.team.replace({"OAK": "LV", "SD": "LAC"}, inplace=True)
db_rush_data = pd.read_csv(
    "../ml_models/RushingModels/DB_Data/DB_RushingStats.csv"
)
db_pass_data = pd.read_csv("../ml_models/ReceivingModels/DefensivePassData.csv")
db_pass_data = db_pass_data[db_pass_data.position == "DB"]

team_db_rush_data = pd.read_csv(
    "../ml_models/RushingModels/DB_Data/DB_TeamRushingStats.csv"
)
team_db_pass_data = pd.read_csv(
    "../ml_models/ReceivingModels/TeamDefensivePassData.csv"
)
team_db_pass_data = team_db_pass_data[team_db_pass_data.position == "DB"]

db_rush_data.rename({"defteam": "team"}, axis=1, inplace=True)
db_pass_data.rename({"defteam": "team"}, axis=1, inplace=True)
team_db_rush_data.rename({"defteam": "team"}, axis=1, inplace=True)
team_db_pass_data.rename({"defteam": "team"}, axis=1, inplace=True)
db_pass_data.drop(
    [
        "game_id",
        "position",
        "game_date",
        "team_def_epa",
        "team_def_yac_epa",
        "team_def_yards_after_catch",
        "team_def_short_target_epa",
        "team_def_medium_target_epa",
        "team_def_deep_target_epa",
        "team_def_air_yards_efficiency",
        "team_def_catch_rate",
        "team_def_short_catch_rate",
        "team_def_medium_catch_rate",
        "team_def_deep_catch_rate",
    ],
    axis=1,
    errors="ignore",
    inplace=True,
)
db_rush_data.drop(
    [
        "game_id",
        "position",
        "game_date",
        "team_def_open_field_rushing_yards",
    ],
    axis=1,
    errors="ignore",
    inplace=True,
)
team_db_rush_data.drop(
    ['game_id',
     'gsis_id',
     'game_date',
     'position',
     'def_open_field_rushing_yards'],axis=1,inplace=True)
team_db_pass_data.drop(
    ["game_id", "game_date", "position"], axis=1, errors="ignore", inplace=True
)
team_db_pass_data.drop([
    'short_target',
    'short_rec',
    'medium_target',
    'medium_rec',
    'deep_target',
    'deep_rec',
    'targets',
    'rec',
    'receiving_yards',
    'air_yards',
    'completed_air_yards',
    'posteam_pass_attempts',
    'posteam_air_yards',],axis=1,inplace=True)
dbs = dbs.merge(
    db_rush_data, on=["week", "season", "team", "gsis_id"], how="left"
)
dbs = dbs.merge(
    db_pass_data, on=["week", "season", "team", "gsis_id"], how="left"
)

dbs = dbs.merge(team_db_rush_data, on=["week", "season", "team"], how="left")
dbs = dbs.merge(team_db_pass_data, on=["week", "season", "team"], how="left")
for stat in [
    "def_open_field_rushing_yards",
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
]:
    dbs.loc[dbs[stat].isna() == True, stat] = dbs.loc[
        dbs[stat].isna() == True, f"team_{stat}"
    ]
dbs = dbs[dbs.team_def_epa.isna() == False]
dbs.drop(
    [
        "team_def_epa",
        "team_def_open_field_rushing_yards",
        "team_def_air_epa",
        "team_def_air_yards_per_attempt",
        "team_def_yac_epa",
        "team_def_yards_after_catch",
        "team_def_short_target_epa",
        "team_def_medium_target_epa",
        "team_def_deep_target_epa",
        "team_def_air_yards_efficiency",
        "team_def_catch_rate",
        "team_def_short_catch_rate",
        "team_def_medium_catch_rate",
        "team_def_deep_catch_rate",
    ],
    axis=1,
    errors="ignore",
    inplace=True,
)
dbs.drop_duplicates(inplace=True)
dbs = dbs.groupby(["week", "season", "gsis_id"], as_index=False).first()
#%%
new_roster = pd.concat([qbs, rbs, recs, ols, dls, lbs, dbs,roster[roster.position=='K']])
new_roster.to_csv("RosterModelData.csv", index=False)
