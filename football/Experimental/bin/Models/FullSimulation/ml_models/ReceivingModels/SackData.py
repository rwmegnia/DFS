#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 12:30:42 2023

@author: robertmegnia
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 23:40:50 2022

@author: robertmegnia

Create Database for PBP passing Model


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
# %% Filter Plays to relevant passing plays
pbp = temp[temp.n_offense.isna() == False]
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
passing = pbp[(pbp.play_type=='pass')&
              (pbp.offense_players.isna()==False)&
              (pbp.defense_players.isna()==False)&
              (pbp.n_offense==11)&
              (pbp.n_defense==11)]

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
pbp_passing_frames = []
for i in range(1, 12):
    print(i)
    passing_frame = passing.groupby(
        [
            "game_id",
            "game_date",
            "week",
            "season",
            "play_id",
            f"offense_player_{i}_id",
            f"offense_player_{i}_position",
            "posteam",
        ]
    ).agg({'sack':np.sum,
           'pass_attempt':np.sum,
           'sack_player_id':'first',
           'qb_hit':np.sum})
    passing_frame = passing_frame.reset_index().rename(
        {
            f"offense_player_{i}_id": "gsis_id",
            f"offense_player_{i}_position": "position",
        },
        axis=1,
    )
    pbp_passing_frames.append(passing_frame)
    passing_frame = passing.groupby(
        [
            "game_id",
            "game_date",
            "week",
            "season",
            "play_id",
            f"defense_player_{i}_id",
            f"defense_player_{i}_position",
            "defteam",
        ]
    ).agg({'sack':np.sum,
           'pass_attempt':np.sum,
           'sack_player_id':'first',
           'qb_hit':np.sum})
    passing_frame = passing_frame.reset_index().rename(
        {
            f"defense_player_{i}_id": "gsis_id",
            f"defense_player_{i}_position": "position",
        },
        axis=1,
    )
    pbp_passing_frames.append(passing_frame)

pbp_passing = pd.concat(pbp_passing_frames)
pbp_passing.position.replace(
    {
        "G": "OL",
        "C": "OL",
        "T": "OL",
        "FB": "OL",
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
#%%Get Sack Rates for Offensive Players on Field
sack_rate_offense = (
    pbp_passing[pbp_passing.position.isin(['OL','QB'])]
    .groupby(["game_id", "week","position","gsis_id", "season", "posteam", "game_date"])
    .mean()[
        [
            "sack"
        ]
    ].reset_index().rename({'sack':'sack_allowed_rate'},axis=1)
)

sack_rate_offense['sack_allowed_rate']=sack_rate_offense.groupby(["gsis_id"]).apply(
    lambda x: x.ewm(span=8, min_periods=1)
    .mean()[
        [
            "sack_allowed_rate"
        ]
        ]
    .shift(1)
)
#%%
sack_rate_offense_team = (
    pbp_passing[pbp_passing.position.isin(['OL','QB'])]
    .groupby(["game_id", "week", "season", "posteam", "game_date"])
    .mean()[
        [
            "sack"
        ]
    ]
    .reset_index().rename({'sack':'team_sack_allowed_rate'},axis=1)
)


sack_rate_offense_team[
        [
            "team_sack_allowed_rate"
        ]
] = sack_rate_offense_team.groupby(["posteam"]).apply(
    lambda x: x.ewm(span=8, min_periods=1)
    .mean()[
        [
            "team_sack_allowed_rate"
        ]
        ]
    .shift(1)
)
    
sack_rate_offense=sack_rate_offense.merge(
        sack_rate_offense_team,on=['week','season','game_id','posteam','game_date'],how='left'
        )
sack_rate_offense.loc[sack_rate_offense.sack_allowed_rate.isna()==True,'sack_allowed_rate']=sack_rate_offense.loc[sack_rate_offense.sack_allowed_rate.isna()==True,f'team_sack_allowed_rate']
sack_rate_offense.to_csv('./OffSacRateData.csv',index=False)
sack_rate_offense_team.to_csv('TeamOffSacRateData.csv',index=False)

#%%Get QB Hit Rates for Offensive Players on Field
hit_rate_offense = (
    pbp_passing[pbp_passing.position.isin(['OL','QB'])]
    .groupby(["game_id", "week","position","gsis_id", "season", "posteam", "game_date"])
    .mean()[
        [
            "qb_hit"
        ]
    ].reset_index().rename({'qb_hit':'hit_allowed_rate'},axis=1)
)

hit_rate_offense['hit_allowed_rate']=hit_rate_offense.groupby(["gsis_id"]).apply(
    lambda x: x.ewm(span=8, min_periods=1)
    .mean()[
        [
            "hit_allowed_rate"
        ]
        ]
    .shift(1)
)
#%%
hit_rate_offense_team = (
    pbp_passing[pbp_passing.position.isin(['OL','QB'])]
    .groupby(["game_id", "week", "season", "posteam", "game_date"])
    .mean()[
        [
            "qb_hit"
        ]
    ]
    .reset_index().rename({'qb_hit':'team_hit_allowed_rate'},axis=1)
)


hit_rate_offense_team[
        [
            "team_hit_allowed_rate"
        ]
] = hit_rate_offense_team.groupby(["posteam"]).apply(
    lambda x: x.ewm(span=8, min_periods=1)
    .mean()[
        [
            "team_hit_allowed_rate"
        ]
        ]
    .shift(1)
)
    
hit_rate_offense=hit_rate_offense.merge(
        hit_rate_offense_team,on=['week','season','game_id','posteam','game_date'],how='left'
        )
hit_rate_offense.loc[hit_rate_offense.hit_allowed_rate.isna()==True,'hit_allowed_rate']=hit_rate_offense.loc[hit_rate_offense.hit_allowed_rate.isna()==True,f'team_hit_allowed_rate']
hit_rate_offense.to_csv('./OffHitRateData.csv',index=False)
hit_rate_offense_team.to_csv('TeamOffHitRateData.csv',index=False)
#%%
pbp_passing.position.replace({
    'FS':'DB',
    'SS':'DB',
    'CB':'DB',
    'DE':'DL',
    'DT':'DL',
    'NT':'DL',
    'ILB':'LB',
    'MLB':'LB',
    'OLB':'LB'},inplace=True)
#%%
sack_rate_defense = (
    pbp_passing[pbp_passing.position.isin(['DL','LB'])]
    .groupby(["game_id", "week", "season", "defteam", "position", "gsis_id", "game_date"])
    .mean('sack')
    .reset_index().rename({'sack':'sack_rate'},axis=1)
)
sack_rate_defense[
    [
         'sack_rate'
    ]
] = sack_rate_defense.groupby(["defteam", "position","gsis_id"]).apply(
    lambda x: x.ewm(span=8, min_periods=1)
    .mean()[
        [
            "sack_rate"
        ]
    ]
    .shift(1)
)
#%%
sack_rate_defense_team = (
    pbp_passing[pbp_passing.position.isin(['DL','LB'])]
    .groupby(["game_id", "week", "season", "defteam", "game_date"])
    .mean()[
        [
            "sack"
        ]
    ]
    .reset_index().rename({'sack':'team_sack_rate'},axis=1)
)


sack_rate_defense_team[
        [
            "team_sack_rate"
        ]
] = sack_rate_defense_team.groupby(["defteam"]).apply(
    lambda x: x.ewm(span=8, min_periods=1)
    .mean()[
        [
            "team_sack_rate"
        ]
        ]
    .shift(1)
)
    
sack_rate_defense=sack_rate_defense.merge(
        sack_rate_defense_team,on=['week','season','game_id','defteam','game_date'],how='left'
        )
sack_rate_defense.loc[sack_rate_defense.sack_rate.isna()==True,'sack_rate']=sack_rate_defense.loc[sack_rate_defense.sack_rate.isna()==True,'team_sack_rate']
sack_rate_defense.to_csv('./DefSacRateData.csv',index=False)
sack_rate_defense_team.to_csv('TeamDefSacRateData.csv',index=False)
#%%Defense Hit Rate
hit_rate_defense = (
    pbp_passing[pbp_passing.position.isin(['DL','LB'])]
    .groupby(["game_id", "week", "season", "defteam", "position", "gsis_id", "game_date"])
    .mean('qb_hit')
    .reset_index().rename({'qb_hit':'hit_rate'},axis=1)
)
hit_rate_defense[
    [
         'hit_rate'
    ]
] = hit_rate_defense.groupby(["defteam", "position","gsis_id"]).apply(
    lambda x: x.ewm(span=8, min_periods=1)
    .mean()[
        [
            "hit_rate"
        ]
    ]
    .shift(1)
)
#%%
hit_rate_defense_team = (
    pbp_passing[pbp_passing.position.isin(['DL','LB'])]
    .groupby(["game_id", "week", "season", "defteam", "game_date"])
    .mean()[
        [
            "qb_hit"
        ]
    ]
    .reset_index().rename({'qb_hit':'team_hit_rate'},axis=1)
)


hit_rate_defense_team[
        [
            "team_hit_rate"
        ]
] = hit_rate_defense_team.groupby(["defteam"]).apply(
    lambda x: x.ewm(span=8, min_periods=1)
    .mean()[
        [
            "team_hit_rate"
        ]
        ]
    .shift(1)
)
    
hit_rate_defense=hit_rate_defense.merge(
        hit_rate_defense_team,on=['week','season','game_id','defteam','game_date'],how='left'
        )
hit_rate_defense.loc[hit_rate_defense.hit_rate.isna()==True,'hit_rate']=hit_rate_defense.loc[hit_rate_defense.hit_rate.isna()==True,'team_hit_rate']
hit_rate_defense.to_csv('./DefHitRateData.csv',index=False)
hit_rate_defense_team.to_csv('TeamDefHitRateData.csv',index=False)