#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 23:40:50 2022

@author: robertmegnia

Create Database for PBP Rushing Model


"""

import pandas as pd
import nfl_data_py as nfl
import numpy as np
import os


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
    temp.to_csv("../../pbp_data/2016_2022_pbp_data.csv")
else:
    temp = pd.read_csv("../../pbp_data/2016_2022_pbp_data.csv")

temp["posteam_rush_attempts"] = temp.groupby(
    ["posteam", "game_id"]
).rush_attempt.transform(np.sum)
# %% Filter Plays to relevant rushing plays
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

# %% Filter PBP Data to rush attempts
rushing = pbp[(pbp.rush_attempt == 1)]
rushing = rushing[(rushing.n_offense == 11) & (rushing.n_defense == 11)]
rushing = rushing[rushing.rusher_position.isin(["RB", "FB", "WR", "QB"])]

# Get Player ID and Position of every player on the field
for i in range(1, 12):
    print(i)
    rushing[f"offense_player_{i}_id"] = rushing.offense_players.apply(
        lambda x: x.split(";")[i - 1]
    )
    rushing = rushing.merge(
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
    rushing.drop_duplicates(inplace=True)
    rushing[f"defense_player_{i}_id"] = rushing.defense_players.apply(
        lambda x: x.split(";")[i - 1]
    )
    rushing = rushing.merge(
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
# %%Compute Some Customized Stats
# Open Field Yards
rushing.loc[(rushing.rushing_yards > 10), "open_field_rushing_yards"] = (
    rushing.loc[(rushing.rushing_yards > 10)].rushing_yards - 10
)
rushing.loc[(rushing.rushing_yards > 10), "adjusted_line_yards"] = 0
rushing.loc[(rushing.rushing_yards > 10), "line_epa"] = 0
rushing.open_field_rushing_yards.fillna(0, inplace=True)
# Second Level Yards
rushing.loc[
    (rushing.rushing_yards <= 10) & (rushing.rushing_yards >= 5),
    "second_level_rushing_yards",
] = (
    rushing.loc[
        (rushing.rushing_yards <= 10) & (rushing.rushing_yards >= 5)
    ].rushing_yards
    - 5
)
rushing.loc[
    (rushing.rushing_yards <= 10) & (rushing.rushing_yards >= 5),
    "adjusted_line_yards",
] = (
    rushing.loc[
        (rushing.rushing_yards <= 10) & (rushing.rushing_yards >= 5)
    ].rushing_yards
    * 0.5
)
rushing.loc[
    (rushing.rushing_yards <= 10) & (rushing.rushing_yards >= 5), "line_epa"
] = (
    rushing.loc[
        (rushing.rushing_yards <= 10) & (rushing.rushing_yards >= 5)
    ].epa
    * 0.5
)
rushing.second_level_rushing_yards.fillna(0, inplace=True)
# First/Negative Level Yards
rushing.loc[
    (rushing.rushing_yards < 5) & (rushing.rushing_yards > 0),
    "adjusted_line_yards",
] = rushing.loc[
    (rushing.rushing_yards < 5) & (rushing.rushing_yards > 0)
].rushing_yards
rushing.loc[
    (rushing.rushing_yards < 5) & (rushing.rushing_yards > 0), "line_epa"
] = rushing.loc[(rushing.rushing_yards < 5) & (rushing.rushing_yards > 0)].epa

rushing.loc[(rushing.rushing_yards <= 0), "adjusted_line_yards"] = (
    rushing.loc[(rushing.rushing_yards <= 0)].rushing_yards * 1.2
)
rushing.loc[(rushing.rushing_yards <= 0), "line_epa"] = (
    rushing.loc[(rushing.rushing_yards <= 0)].epa * 1.2
)

rushing["adjusted_line_yards"] = rushing.adjusted_line_yards * (
    1 + rushing.vegas_wpa
)
# %%
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
            "rusher_player_id",
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
    pbp_rushing_frames.append(rushing_frame)

pbp_rushing = pd.concat(pbp_rushing_frames)
pbp_rushing.loc[(pbp_rushing.gsis_id!=pbp_rushing.rusher_player_id),'rush_attempt']=0
pbp_rushing.position.replace(
    {
        "G": "OL",
        "C": "OL",
        "T": "OL",
        "DE": "DL",
        "DT": "DL",
        "NT": "DL",
        "FS": "DB",
        "SS": "DB",
        "CB": "DB",
        "MLB": "LB",
        "ILB": "LB",
        "OLB": "LB",
    },
    inplace=True,
)
# %%
# Get Relevant Offensive Rushing Data
#
RUSHING_STATS = [
    "rush_epa",
    "ypc",
    "open_field_ypc",
    "line_epa",
    "adjusted_line_yards",
    "rush_share",
]
rushing_stats_offense_position = (
    pbp_rushing[pbp_rushing.rusher_position.isin(["RB", "QB", "WR", "FB"])]
    .groupby(
        [
            "game_id",
            "week",
            "season",
            "posteam",
            "position",
            "gsis_id",
            "game_date",
        ]
    )
    .agg(
        {
            "epa": np.mean,
            "rushing_yards": np.mean,
            "rush_attempt": np.sum,
            "open_field_rushing_yards": np.mean,
            "second_level_rushing_yards": np.mean,
            "line_epa": np.mean,
            "adjusted_line_yards": np.mean,
            "posteam_rush_attempts": "first",
        }
    )
    .reset_index()
)
rushing_stats_offense_position["rush_share"] = (
    rushing_stats_offense_position.rush_attempt
    / rushing_stats_offense_position.posteam_rush_attempts
)
rushing_stats_offense_position.rename(
    {
        "epa": "rush_epa",
        "rushing_yards": "ypc",
        "open_field_rushing_yards": "open_field_ypc",
    },
    axis=1,
    inplace=True,
)

rushing_stats_offense_position[
    RUSHING_STATS
] = rushing_stats_offense_position.groupby(
    ["posteam", "gsis_id", "position"]
).apply(
    lambda x: x.ewm(span=8, min_periods=1).mean()[RUSHING_STATS].shift(1)
)

rushing_stats_position = rushing_stats_offense_position[
    [
        "week",
        "season",
        "game_id",
        "posteam",
        "gsis_id",
        "position",
        "game_date",
        "rush_epa",
        "ypc",
        "open_field_ypc",
        "rush_share",
    ]
]
rushing_stats_position = rushing_stats_position[
    rushing_stats_position.position.isin(["QB", "RB", "WR", "FB"])
]
ol_stats_position = rushing_stats_offense_position[
    [
        "week",
        "season",
        "game_id",
        "posteam",
        "gsis_id",
        "position",
        "game_date",
        "line_epa",
        "adjusted_line_yards",
    ]
]
ol_stats_position = rushing_stats_offense_position[
    rushing_stats_offense_position.position.isin(["OL", "FB", "RB"])
]

#%%
# Get Team RB Stats
rb_rushing_offense_team = (
    pbp_rushing[pbp_rushing.rusher_position.isin(["RB", "WR", "QB", "FB"])]
    .groupby(["game_id", "week", "position", "season", "posteam", "game_date"])
    .mean()[
        [
            "epa",
            "rushing_yards",
            "open_field_rushing_yards",
        ]
    ]
    .reset_index()
)

rb_rushing_offense_team.rename(
    {
        "epa": "team_rush_epa",
        "rushing_yards": "team_ypc",
        "open_field_rushing_yards": "team_open_field_ypc",
    },
    axis=1,
    inplace=True,
)

rb_rushing_offense_team[
    [
        "team_rush_epa",
        "team_ypc",
        "team_open_field_ypc",
    ]
] = rb_rushing_offense_team.groupby(["posteam", "position"]).apply(
    lambda x: x.ewm(span=8, min_periods=1)
    .mean()[
        [
            "team_rush_epa",
            "team_ypc",
            "team_open_field_ypc",
        ]
    ]
    .shift(1)
)

rb_rushing_offense_team = rb_rushing_offense_team[
    [
        "team_rush_epa",
        "team_ypc",
        "team_open_field_ypc",
        "week",
        "season",
        "game_id",
        "position",
        "posteam",
        "game_date",
    ]
]

## Merge Inividual Stats frame with team frame
rushing_stats_position = rushing_stats_position.merge(
    rb_rushing_offense_team,
    on=["week", "season", "game_id", "posteam", "position", "game_date"],
    how="left",
)
# Fill Individual NaNs with team data
for stat in ["rush_epa", "ypc", "open_field_ypc"]:
    rushing_stats_position.loc[
        rushing_stats_position[stat].isna() == True, stat
    ] = rushing_stats_position.loc[
        rushing_stats_position[stat].isna() == True, f"team_{stat}"
    ]
rushing_stats_position.to_csv("./Rushing_Data/RushingData.csv", index=False)
rb_rushing_offense_team.to_csv(
    "./Rushing_Data/TeamRushingData.csv", index=False
)
#%%
# Get TeamOffensiveLine Stats
ol_rushing_offense_team = (
    pbp_rushing[
        (pbp_rushing.rusher_position.isin(["RB", "QB", "WR", "FB"]))
        & (pbp_rushing.position.isin(["OL", "FB", "RB"]))
    ]
    .groupby(["game_id", "week", "season", "position", "posteam", "game_date"])
    .mean()[
        [
            "line_epa",
            "adjusted_line_yards",
        ]
    ]
    .reset_index()
)

ol_rushing_offense_team.rename(
    {
        "line_epa": "team_line_epa",
        "adjusted_line_yards": "team_adjusted_line_yards",
    },
    axis=1,
    inplace=True,
)
ol_rushing_offense_team[
    [
        "team_line_epa",
        "team_adjusted_line_yards",
    ]
] = ol_rushing_offense_team.groupby(["posteam"]).apply(
    lambda x: x.ewm(span=8, min_periods=4)
    .mean()[
        [
            "team_line_epa",
            "team_adjusted_line_yards",
        ]
    ]
    .shift(1)
)

ol_rushing_offense_team = ol_rushing_offense_team[
    [
        "team_line_epa",
        "team_adjusted_line_yards",
        "week",
        "season",
        "game_id",
        "position",
        "posteam",
        "game_date",
    ]
]
ol_stats_position = ol_stats_position.merge(
    ol_rushing_offense_team,
    on=["week", "season", "game_id", "position","posteam", "game_date"],
    how="left",
)
for stat in ["line_epa", "adjusted_line_yards"]:
    ol_stats_position.loc[
        ol_stats_position[stat].isna() == True, stat
    ] = ol_stats_position.loc[
        ol_stats_position[stat].isna() == True, f"team_{stat}"
    ]
ol_stats_position.to_csv("./OL_Data/OL_RushingData.csv", index=False)
ol_rushing_offense_team.to_csv("./OL_Data/OL_TeamRushingData.csv", index=False)
#%% Get Defense Stats
pbp_rushing.position.replace(
    {
        "FS": "DB",
        "SS": "DB",
        "CB": "DB",
        "DE": "DL",
        "DT": "DL",
        "NT": "DL",
        "ILB": "LB",
        "MLB": "LB",
        "OLB": "LB",
    },
    inplace=True,
)
rb_rushing_defense = (
    pbp_rushing[pbp_rushing.rusher_position == "RB"]
    .groupby(
        [
            "game_id",
            "week",
            "season",
            "defteam",
            "position",
            "gsis_id",
            "game_date",
        ]
    )
    .mean()[
        [
            "epa",
            "rushing_yards",
            "line_epa",
            "adjusted_line_yards",
            "open_field_rushing_yards",
            "second_level_rushing_yards",
        ]
    ]
    .reset_index()
)
rb_rushing_defense[
    [
        "epa",
        "rushing_yards",
        "line_epa",
        "adjusted_line_yards",
        "open_field_rushing_yards",
        "second_level_rushing_yards",
    ]
] = rb_rushing_defense.groupby(["defteam", "position", "gsis_id"]).apply(
    lambda x: x.ewm(span=8, min_periods=4)
    .mean()[
        [
            "epa",
            "rushing_yards",
            "line_epa",
            "adjusted_line_yards",
            "open_field_rushing_yards",
            "second_level_rushing_yards",
        ]
    ]
    .shift(1)
)
#%% Get Individual Defensive Line Stats
dl = rb_rushing_defense[rb_rushing_defense.position == "DL"][
    [
        "defteam",
        "position",
        "gsis_id",
        "game_id",
        "game_date",
        "week",
        "season",
        "line_epa",
        "adjusted_line_yards",
    ]
]
dl.rename(
    {
        "line_epa": "def_line_epa",
        "adjusted_line_yards": "def_adjusted_line_yards",
    },
    axis=1,
    inplace=True,
)
#%% Get Individual DB Data
db = rb_rushing_defense[rb_rushing_defense.position == "DB"]
db = db[
    [
        "game_id",
        "gsis_id",
        "week",
        "season",
        "defteam",
        "position",
        "game_date",
        "open_field_rushing_yards",
    ]
]
db.rename(
    {"open_field_rushing_yards": "def_open_field_rushing_yards"},
    axis=1,
    inplace=True,
)

#%%Get Individual LB Data
lb = rb_rushing_defense[rb_rushing_defense.position == "LB"]
lb = lb[
    [
        "game_id",
        "gsis_id",
        "week",
        "season",
        "defteam",
        "position",
        "game_date",
        "second_level_rushing_yards",
    ]
]
lb.rename(
    {"second_level_rushing_yards": "def_second_level_rushing_yards"},
    axis=1,
    inplace=True,
)
#%% Get Team Defensive Data
rb_rushing_defense_team = (
    pbp_rushing[pbp_rushing.rusher_position == "RB"]
    .groupby(["game_id", "week", "season", "defteam", "position", "game_date"])
    .mean()[
        [
            "epa",
            "rushing_yards",
            "line_epa",
            "adjusted_line_yards",
            "open_field_rushing_yards",
            "second_level_rushing_yards",
        ]
    ]
    .reset_index()
)
rb_rushing_defense_team[
    [
        "epa",
        "rushing_yards",
        "line_epa",
        "adjusted_line_yards",
        "open_field_rushing_yards",
        "second_level_rushing_yards",
    ]
] = rb_rushing_defense_team.groupby(["defteam", "position"]).apply(
    lambda x: x.ewm(span=8, min_periods=4)
    .mean()[
        [
            "epa",
            "rushing_yards",
            "line_epa",
            "adjusted_line_yards",
            "open_field_rushing_yards",
            "second_level_rushing_yards",
        ]
    ]
    .shift(1)
)
#%% Get Team DL Stats
dl_team = rb_rushing_defense_team[rb_rushing_defense_team.position == "DL"]
dl_team = dl_team[
    [
        "game_id",
        "week",
        "season",
        "defteam",
        "position",
        "game_date",
        "line_epa",
        "adjusted_line_yards",
    ]
]
dl_team.rename(
    {
        "line_epa": "team_def_line_epa",
        "adjusted_line_yards": "team_def_adjusted_line_yards",
    },
    axis=1,
    inplace=True,
)

#%% Get Team DB Stats
db_team = rb_rushing_defense_team[rb_rushing_defense_team.position == "DB"]
db_team = db_team[
    [
        "game_id",
        "week",
        "season",
        "defteam",
        "position",
        "game_date",
        "open_field_rushing_yards",
    ]
]
db_team.rename(
    {
        "open_field_rushing_yards": "team_def_open_field_rushing_yards",
    },
    axis=1,
    inplace=True,
)
#%% Get Team LB Stats
lb_team = rb_rushing_defense_team[rb_rushing_defense_team.position == "LB"]
lb_team = lb_team[
    [
        "game_id",
        "week",
        "season",
        "defteam",
        "position",
        "game_date",
        "second_level_rushing_yards",
    ]
]
lb_team.rename(
    {"second_level_rushing_yards": "team_def_second_level_rushing_yards"},
    axis=1,
    inplace=True,
)
#%% Merge Individual Defense Stats with Team Defense Stats
dl = dl.merge(
    dl_team,
    on=["game_id", "week", "season", "defteam", "position", "game_date"],
    how="left",
)
for stat in ["def_line_epa", "def_adjusted_line_yards"]:
    dl.loc[dl[stat].isna() == True, stat] = dl.loc[
        dl[stat].isna() == True, f"team_{stat}"
    ]
db = db.merge(
    db_team,
    on=["game_id", "week", "season", "defteam", "position", "game_date"],
    how="left",
)
db.loc[
    db["def_open_field_rushing_yards"].isna() == True,
    "def_open_field_rushing_yards",
] = db.loc[
    db["def_open_field_rushing_yards"].isna() == True,
    "team_def_open_field_rushing_yards",
]

lb = lb.merge(
    lb_team,
    on=["game_id", "week", "season", "defteam", "position", "game_date"],
    how="left",
)
lb.loc[
    lb["def_second_level_rushing_yards"].isna() == True,
    "def_second_level_rushing_yards",
] = lb.loc[
    lb["def_second_level_rushing_yards"].isna() == True,
    "team_def_second_level_rushing_yards",
]
dl_team.to_csv('./DL_Data/DL_TeamRushingStats.csv',index=False)
lb.to_csv("./LB_Data/LB_TeamRushingStats.csv", index=False)
db.to_csv("./DB_Data/DB_TeamRushingStats.csv", index=False)

dl.to_csv("./DL_Data/DL_RushingStats.csv", index=False)
lb.to_csv("./LB_Data/LB_RushingStats.csv", index=False)
db.to_csv("./DB_Data/DB_RushingStats.csv", index=False)
