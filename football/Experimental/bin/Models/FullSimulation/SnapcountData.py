#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 23:40:50 2022

@author: robertmegnia

Create Database for PBP temp Model


"""

import pandas as pd
import nfl_data_py as nfl
import numpy as np
import os

depth_charts = nfl.import_depth_charts(range(2016, 2023))
depth_charts.drop("depth_position", axis=1, inplace=True)
depth_charts.drop_duplicates(inplace=True)
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
    temp = pd.read_csv("./pbp_data/2016_2022_pbp_data.csv")

temp = temp[temp.play_type.isna() == False]
temp = temp[temp.posteam.isna() == False]
temp = temp[
    ~temp.penalty_type.isin(
        [
            "Delay of Game",
            "False Start",
            "Neutral Zone Infraction",
            "Enroachment",
            "Defensive Delay of Game",
        ]
    )
]
temp = temp[temp.special_teams_play == 0]
posteam_snaps = (
    temp.groupby(["game_id", "posteam"], as_index=False)
    .size()
    .rename({"size": "posteam_snaps"}, axis=1)
)
defteam_snaps = (
    temp.groupby(["game_id", "defteam"], as_index=False)
    .size()
    .rename({"size": "defteam_snaps"}, axis=1)
)
temp = temp.merge(posteam_snaps, on=["game_id", "posteam"], how="left")
temp = temp.merge(defteam_snaps, on=["game_id", "defteam"], how="left")
temp.loc[temp.posteam == temp.home_team, "home_team_offense_snaps"] = temp.loc[
    temp.posteam == temp.home_team, "posteam_snaps"
]
temp.loc[temp.posteam == temp.away_team, "away_team_offense_snaps"] = temp.loc[
    temp.posteam == temp.away_team, "posteam_snaps"
]
temp.loc[temp.defteam == temp.home_team, "home_team_defense_snaps"] = temp.loc[
    temp.defteam == temp.home_team, "defteam_snaps"
]
temp.loc[temp.defteam == temp.away_team, "away_team_defense_snaps"] = temp.loc[
    temp.defteam == temp.away_team, "defteam_snaps"
]
temp = temp[temp.offense_players.isna() == False]
temp = temp[(temp.n_offense == 11) & (temp.n_defense == 11)]
#%%
# Get Player ID and Position of every player on the field
for i in range(1, 12):
    temp[f"offense_player_{i}_id"] = temp.offense_players.apply(
        lambda x: x.split(";")[i - 1]
    )
    temp = temp.merge(
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
    temp.drop_duplicates(inplace=True)
    temp[f"defense_player_{i}_id"] = temp.defense_players.apply(
        lambda x: x.split(";")[i - 1]
    )
    temp = temp.merge(
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
