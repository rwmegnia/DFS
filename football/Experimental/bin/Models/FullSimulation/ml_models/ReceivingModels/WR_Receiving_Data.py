#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 14:20:36 2023

@author: robertmegnia
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 11:32:23 2023

@author: robertmegnia
"""

import pandas as pd
import numpy as np

# Read in play by play data and filter
# to pass players where a QB threw the ball

df = pd.read_csv(
    "/Volumes/XDrive/DFS/football/Experimental/bin/Models/FullSimulation/pbp_data/2016_2022_pbp_data.csv"
)
df["posteam_pass_attempts"] = df.groupby(
    ["posteam", "game_id"]
).pass_attempt.transform(np.sum)
df["posteam_air_yards"] = df.groupby(
    ["posteam", "game_id"]
).air_yards.transform(np.sum)
# Create column for completed air yards
df.loc[df.complete_pass == 1, "completed_air_yards"] = df.loc[
    df.complete_pass == 1
].air_yards

# Adjust pass_length column
df.loc[df.air_yards < 7.5, "pass_length"] = "short"
df.loc[(df.air_yards >= 7.5) & (df.air_yards < 15), "pass_length"] = "medium"
df.loc[df.air_yards >= 15, "pass_length"] = "deep"

# Create EPA columns
df.loc[df.pass_length == "short", "short_target_epa"] = df.loc[
    df.pass_length == "short"
].epa
df.loc[df.pass_length == "short", "short_target"] = 1
df.loc[(df.pass_length == "short") & (df.complete_pass == 1), "short_rec"] = 1
df["posteam_short_pass_attempts"] = df.groupby(
    ["posteam", "game_id"]
).short_target.transform(np.sum)

df.loc[df.pass_length == "medium", "medium_target_epa"] = df.loc[
    df.pass_length == "medium"
].epa
df.loc[df.pass_length == "medium", "medium_target"] = 1
df.loc[(df.pass_length == "medium") & (df.complete_pass == 1), "medium_rec"] = 1
df["posteam_medium_pass_attempts"] = df.groupby(
    ["posteam", "game_id"]
).medium_target.transform(np.sum)

df.loc[df.pass_length == "deep", "deep_target_epa"] = df.loc[
    df.pass_length == "deep"
].epa
df.loc[df.pass_length == "deep", "deep_target"] = 1
df.loc[(df.pass_length == "deep") & (df.complete_pass == 1), "deep_rec"] = 1
df["posteam_deep_pass_attempts"] = df.groupby(
    ["posteam", "game_id"]
).deep_target.transform(np.sum)


df = df[
    (df.passer_position == "QB")
    & (df.receiver_position.isin(["RB", "WR", "TE", "FB"]))
]
df = df[df.play_type == "pass"]


df.rename(
    {"pass_attempt": "targets", "complete_pass": "rec"}, axis=1, inplace=True
)
#%%
# Individual recs
recs = df.groupby(
    ["game_id", "receiver_player_id", "posteam", "week", "season"]
).agg(
    {
        "air_yards": np.sum,
        "completed_air_yards": np.sum,
        "posteam_pass_attempts": "first",
        "posteam_short_pass_attempts": "first",
        "posteam_medium_pass_attempts": "first",
        "posteam_deep_pass_attempts": "first",
        "posteam_air_yards": "first",
        "epa": np.sum,
        "yac_epa": np.sum,
        "yards_after_catch": np.sum,
        "short_target_epa": np.sum,
        "medium_target_epa": np.sum,
        "deep_target_epa": np.sum,
        "short_target": np.sum,
        "short_rec": np.sum,
        "medium_target": np.sum,
        "medium_rec": np.sum,
        "deep_target": np.sum,
        "deep_rec": np.sum,
        "targets": np.sum,
        "rec": np.sum,
        "receiving_yards": np.sum,
        "week": "first",
        "season": "first",
    }
)
recs["air_yards_efficiency"] = recs.completed_air_yards / recs.air_yards
recs["catch_rate"] = recs.rec / recs.targets
recs["short_catch_rate"] = recs.short_rec / recs.short_target
recs["medium_catch_rate"] = recs.medium_rec / recs.medium_target
recs["deep_catch_rate"] = recs.deep_rec / recs.deep_target
recs["epa"] = recs.epa / recs.targets
recs["short_target_epa"] = recs.short_target_epa / recs.short_target
recs["medium_target_epa"] = recs.medium_target_epa / recs.medium_target
recs["deep_target_epa"] = recs.deep_target_epa / recs.deep_target
recs["target_share"] = recs.targets / recs.posteam_pass_attempts
recs["short_target_share"] = (
    recs.short_target / recs.posteam_short_pass_attempts
)
recs["medium_target_share"] = (
    recs.medium_target / recs.posteam_medium_pass_attempts
)
recs["deep_target_share"] = recs.deep_target / recs.posteam_deep_pass_attempts
recs["air_yards_share"] = recs.air_yards / recs.posteam_air_yards
recs["wopr"] = recs.target_share * 1.5 + recs.air_yards_share * 0.7
REC_STATS = [
    "epa",
    "short_target_epa",
    "medium_target_epa",
    "deep_target_epa",
    "yac_epa",
    "yards_after_catch",
    "air_yards_efficiency",
    "catch_rate",
    "short_catch_rate",
    "short_target_share",
    "medium_catch_rate",
    "medium_target_share",
    "deep_catch_rate",
    "deep_target_share",
    "target_share",
    "wopr",
]
recs[REC_STATS] = recs.groupby("receiver_player_id").apply(
    lambda x: x.ewm(min_periods=1, span=8).mean().shift()[REC_STATS]
)
for stat in REC_STATS:
    recs.rename({stat: f"rec_ewm_{stat}"}, axis=1, inplace=True)
recs = recs[["rec_ewm_" + stat for stat in REC_STATS]]

recs.to_csv("./Receiving_Model_Data.csv")
#%%
team_recs = df.groupby(["game_id", "posteam", "week", "season"]).agg(
    {
        "air_yards": np.sum,
        "completed_air_yards": np.sum,
        "posteam_pass_attempts": "first",
        "posteam_short_pass_attempts": "first",
        "posteam_medium_pass_attempts": "first",
        "posteam_deep_pass_attempts": "first",
        "posteam_air_yards": "first",
        "epa": np.sum,
        "yac_epa": np.sum,
        "yards_after_catch": np.sum,
        "short_target_epa": np.sum,
        "medium_target_epa": np.sum,
        "deep_target_epa": np.sum,
        "short_target": np.sum,
        "short_rec": np.sum,
        "medium_target": np.sum,
        "medium_rec": np.sum,
        "deep_target": np.sum,
        "deep_rec": np.sum,
        "targets": np.sum,
        "rec": np.sum,
        "receiving_yards": np.sum,
        "week": "first",
        "season": "first",
    }
)
team_recs["air_yards_efficiency"] = (
    team_recs.completed_air_yards / team_recs.air_yards
)
team_recs["catch_rate"] = team_recs.rec / team_recs.targets
team_recs["short_catch_rate"] = team_recs.short_rec / team_recs.short_target
team_recs["medium_catch_rate"] = team_recs.medium_rec / team_recs.medium_target
team_recs["deep_catch_rate"] = team_recs.deep_rec / team_recs.deep_target
team_recs["epa"] = team_recs.epa / team_recs.targets
team_recs["short_target_epa"] = (
    team_recs.short_target_epa / team_recs.short_target
)
team_recs["medium_target_epa"] = (
    team_recs.medium_target_epa / team_recs.medium_target
)
team_recs["deep_target_epa"] = team_recs.deep_target_epa / team_recs.deep_target
REC_STATS = [
    "epa",
    "short_target_epa",
    "medium_target_epa",
    "deep_target_epa",
    "yac_epa",
    "yards_after_catch",
    "air_yards_efficiency",
    "catch_rate",
    "short_catch_rate",
    "medium_catch_rate",
    "deep_catch_rate",
]
team_recs[REC_STATS] = team_recs.groupby("posteam").apply(
    lambda x: x.ewm(min_periods=1, span=8).mean().shift()[REC_STATS]
)
for stat in REC_STATS:
    team_recs.rename({stat: f"team_rec_ewm_{stat}"}, axis=1, inplace=True)
team_recs = team_recs[["team_rec_ewm_" + stat for stat in REC_STATS]]
team_recs.to_csv("./Team_Receiving_Model_Data.csv")
#%%Redzone Data
df = pd.read_csv(
    "/Volumes/XDrive/DFS/football/Experimental/bin/Models/FullSimulation/pbp_data/2016_2022_pbp_data.csv"
)
df = df[df.yardline_100 <= 20]
df["posteam_pass_attempts"] = df.groupby(
    ["posteam", "game_id"]
).pass_attempt.transform(np.sum)
df["posteam_air_yards"] = df.groupby(
    ["posteam", "game_id"]
).air_yards.transform(np.sum)
# Create column for completed air yards
df.loc[df.complete_pass == 1, "completed_air_yards"] = df.loc[
    df.complete_pass == 1
].air_yards

# Adjust pass_length column
df.loc[df.air_yards < 7.5, "pass_length"] = "short"
df.loc[(df.air_yards >= 7.5) & (df.air_yards < 15), "pass_length"] = "medium"
df.loc[df.air_yards >= 15, "pass_length"] = "deep"

# Create EPA columns
df.loc[df.pass_length == "short", "short_target_epa"] = df.loc[
    df.pass_length == "short"
].epa
df.loc[df.pass_length == "short", "short_target"] = 1
df.loc[(df.pass_length == "short") & (df.complete_pass == 1), "short_rec"] = 1
df["posteam_short_pass_attempts"] = df.groupby(
    ["posteam", "game_id"]
).short_target.transform(np.sum)

df.loc[df.pass_length == "medium", "medium_target_epa"] = df.loc[
    df.pass_length == "medium"
].epa
df.loc[df.pass_length == "medium", "medium_target"] = 1
df.loc[(df.pass_length == "medium") & (df.complete_pass == 1), "medium_rec"] = 1
df["posteam_medium_pass_attempts"] = df.groupby(
    ["posteam", "game_id"]
).medium_target.transform(np.sum)

df.loc[df.pass_length == "deep", "deep_target_epa"] = df.loc[
    df.pass_length == "deep"
].epa
df.loc[df.pass_length == "deep", "deep_target"] = 1
df.loc[(df.pass_length == "deep") & (df.complete_pass == 1), "deep_rec"] = 1
df["posteam_deep_pass_attempts"] = df.groupby(
    ["posteam", "game_id"]
).deep_target.transform(np.sum)


df = df[
    (df.passer_position == "QB")
    & (df.receiver_position.isin(["RB", "WR", "TE", "FB"]))
]
df = df[df.play_type == "pass"]


df.rename(
    {"pass_attempt": "targets", "complete_pass": "rec"}, axis=1, inplace=True
)
# Individual recs
recs = df.groupby(
    ["game_id", "receiver_player_id", "posteam", "week", "season"]
).agg(
    {
        "air_yards": np.sum,
        "completed_air_yards": np.sum,
        "posteam_pass_attempts": "first",
        "posteam_short_pass_attempts": "first",
        "posteam_medium_pass_attempts": "first",
        "posteam_deep_pass_attempts": "first",
        "posteam_air_yards": "first",
        "epa": np.sum,
        "yac_epa": np.sum,
        "yards_after_catch": np.sum,
        "short_target_epa": np.sum,
        "medium_target_epa": np.sum,
        "deep_target_epa": np.sum,
        "short_target": np.sum,
        "short_rec": np.sum,
        "medium_target": np.sum,
        "medium_rec": np.sum,
        "deep_target": np.sum,
        "deep_rec": np.sum,
        "targets": np.sum,
        "rec": np.sum,
        "receiving_yards": np.sum,
        "week": "first",
        "season": "first",
    }
)
recs["air_yards_efficiency"] = recs.completed_air_yards / recs.air_yards
recs["catch_rate"] = recs.rec / recs.targets
recs["short_catch_rate"] = recs.short_rec / recs.short_target
recs["medium_catch_rate"] = recs.medium_rec / recs.medium_target
recs["deep_catch_rate"] = recs.deep_rec / recs.deep_target
recs["epa"] = recs.epa / recs.targets
recs["short_target_epa"] = recs.short_target_epa / recs.short_target
recs["medium_target_epa"] = recs.medium_target_epa / recs.medium_target
recs["deep_target_epa"] = recs.deep_target_epa / recs.deep_target
recs["target_share"] = recs.targets / recs.posteam_pass_attempts
recs["short_target_share"] = (
    recs.short_target / recs.posteam_short_pass_attempts
)
recs["medium_target_share"] = (
    recs.medium_target / recs.posteam_medium_pass_attempts
)
recs["deep_target_share"] = recs.deep_target / recs.posteam_deep_pass_attempts
recs["air_yards_share"] = recs.air_yards / recs.posteam_air_yards
recs["wopr"] = recs.target_share * 1.5 + recs.air_yards_share * 0.7
REC_STATS = [
    "epa",
    "short_target_epa",
    "medium_target_epa",
    "deep_target_epa",
    "yac_epa",
    "yards_after_catch",
    "air_yards_efficiency",
    "catch_rate",
    "short_catch_rate",
    "short_target_share",
    "medium_catch_rate",
    "medium_target_share",
    "deep_catch_rate",
    "deep_target_share",
    "target_share",
    "wopr",
]
recs[REC_STATS] = recs.groupby("receiver_player_id").apply(
    lambda x: x.ewm(min_periods=1, span=8).mean().shift()[REC_STATS]
)
for stat in REC_STATS:
    recs.rename({stat: f"rec_redzone_ewm_{stat}"}, axis=1, inplace=True)
recs = recs[["rec_redzone_ewm_" + stat for stat in REC_STATS]]
recs.to_csv("./Redzone_Receiving_Model_Data.csv")
#%%
team_recs = df.groupby(["game_id", "posteam", "week", "season"]).agg(
    {
        "air_yards": np.sum,
        "completed_air_yards": np.sum,
        "posteam_pass_attempts": "first",
        "posteam_short_pass_attempts": "first",
        "posteam_medium_pass_attempts": "first",
        "posteam_deep_pass_attempts": "first",
        "posteam_air_yards": "first",
        "epa": np.sum,
        "yac_epa": np.sum,
        "yards_after_catch": np.sum,
        "short_target_epa": np.sum,
        "medium_target_epa": np.sum,
        "deep_target_epa": np.sum,
        "short_target": np.sum,
        "short_rec": np.sum,
        "medium_target": np.sum,
        "medium_rec": np.sum,
        "deep_target": np.sum,
        "deep_rec": np.sum,
        "targets": np.sum,
        "rec": np.sum,
        "receiving_yards": np.sum,
        "week": "first",
        "season": "first",
    }
)
team_recs["air_yards_efficiency"] = (
    team_recs.completed_air_yards / team_recs.air_yards
)
team_recs["catch_rate"] = team_recs.rec / team_recs.targets
team_recs["short_catch_rate"] = team_recs.short_rec / team_recs.short_target
team_recs["medium_catch_rate"] = team_recs.medium_rec / team_recs.medium_target
team_recs["deep_catch_rate"] = team_recs.deep_rec / team_recs.deep_target
team_recs["epa"] = team_recs.epa / team_recs.targets
team_recs["short_target_epa"] = (
    team_recs.short_target_epa / team_recs.short_target
)
team_recs["medium_target_epa"] = (
    team_recs.medium_target_epa / team_recs.medium_target
)
team_recs["deep_target_epa"] = team_recs.deep_target_epa / team_recs.deep_target
REC_STATS = [
    "epa",
    "short_target_epa",
    "medium_target_epa",
    "deep_target_epa",
    "yac_epa",
    "yards_after_catch",
    "air_yards_efficiency",
    "catch_rate",
    "short_catch_rate",
    "medium_catch_rate",
    "deep_catch_rate",
]
team_recs[REC_STATS] = team_recs.groupby("posteam").apply(
    lambda x: x.ewm(min_periods=1, span=8).mean().shift()[REC_STATS]
)
for stat in REC_STATS:
    team_recs.rename(
        {stat: f"team_rec_redzone_ewm_{stat}"}, axis=1, inplace=True
    )
team_recs = team_recs[["team_rec_redzone_ewm_" + stat for stat in REC_STATS]]
team_recs.to_csv("./Team_Redzone_Receiving_Model_Data.csv")
