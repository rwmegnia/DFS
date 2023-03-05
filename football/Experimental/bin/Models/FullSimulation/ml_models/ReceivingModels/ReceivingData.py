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
# depth_charts = nfl.import_depth_charts(range(2016, 2023))
# depth_charts=fixDepthCharts(depth_charts)
# depth_charts.drop("depth_position", axis=1, inplace=True)
# depth_charts.drop_duplicates(inplace=True)
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
    temp.to_csv("../../pbp_data/2016_2022_pbp_data.csv",index=False)
else:
    temp = pd.read_csv("../../pbp_data/2016_2022_pbp_data.csv")
temp['posteam_pass_attempts']=temp.groupby(['posteam','game_id']).pass_attempt.transform(np.sum)
temp['posteam_air_yards']=temp.groupby(['posteam','game_id']).air_yards.transform(np.sum)
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

#Filter PBP Data to pass attempts
passing = pbp[(pbp.play_type == 'pass')]
passing.loc[passing.complete_pass==1,'completed_air_yards']=passing.loc[passing.complete_pass==1].air_yards
# Adjust pass_length column
passing.loc[passing.air_yards<7.5,'pass_length']='short'
passing.loc[(passing.air_yards>=7.5)&(passing.air_yards<15),'pass_length']='medium'
passing.loc[passing.air_yards>=15,'pass_length']='deep'

# Create EPA columns
passing.loc[passing.pass_length=='short','short_target_epa']=passing.loc[passing.pass_length=='short'].epa
passing.loc[passing.pass_length=='short','short_target']=1
passing.loc[(passing.pass_length=='short')&(passing.complete_pass==1),'short_rec']=1
passing.loc[passing.pass_length=='medium','medium_target_epa']=passing.loc[passing.pass_length=='medium'].epa
passing.loc[passing.pass_length=='medium','medium_target']=1
passing.loc[(passing.pass_length=='medium')&(passing.complete_pass==1),'medium_rec']=1

passing.loc[passing.pass_length=='deep','deep_target_epa']=passing.loc[passing.pass_length=='deep'].epa
passing.loc[passing.pass_length=='deep','deep_target']=1
passing.loc[(passing.pass_length=='deep')&(passing.complete_pass==1),'deep_rec']=1
passing.rename({'pass_attempt':'targets',
           'complete_pass':'rec'},axis=1,inplace=True)                                                                  
passing = passing[(passing.n_offense == 11) & (passing.n_defense == 11)]

# %%Get Player ID and Position of every player on the field
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
            "receiver_position",
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
            "receiver_position",
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
#%%
# Get OL EPA Stats on non-scramble pass plays
ol_passing_offense = (
    pbp_passing[(pbp_passing.receiver_position.isin(['RB','WR','TE','FB']))&(pbp_passing.position=='OL')&(pbp_passing.qb_scramble==0)]
    .groupby(["game_id", "week","position","gsis_id", "season", "posteam", "game_date"])
    .mean()[
        [
            "epa",
            'short_target_epa',
            'medium_target_epa',
            'deep_target_epa'
        ]
    ]
    .reset_index()
)

ol_passing_offense[
    [
        "epa",
        'short_target_epa',
        'medium_target_epa',
        'deep_target_epa'
    ]
] = ol_passing_offense.groupby(["gsis_id"]).apply(
    lambda x: x.ewm(span=5, min_periods=1)
    .mean()[
        [
            "epa",
            "short_target_epa",
            "medium_target_epa",
            "deep_target_epa",
        ]
        ]
    .shift(1)
)
ol_passing_offense.rename({
        'epa':'OL_epa',
        'short_target_epa':'OL_short_target_epa',
        'medium_target_epa':'OL_medium_target_epa',
        'deep_target_epa':'OL_deep_target_epa'},
    axis=1,inplace=True)
#%%
# Get Team OL Passing Offense Stats
ol_passing_offense_team = (
    pbp_passing[(pbp_passing.receiver_position.isin(['RB','WR','TE']))&(pbp_passing.position.isin(['OL']))&(pbp_passing.qb_scramble==0)]
    .groupby(["game_id", "week", "season", "posteam", "game_date"])
    .mean()[
        [
            "epa",
            "short_target_epa",
            "medium_target_epa",
            "deep_target_epa",
        ]
    ]
    .reset_index()
)

ol_passing_offense_team.rename(
    {'epa':'team_OL_epa',
     'short_target_epa':'team_OL_short_target_epa',
     'medium_target_epa':'team_OL_medium_target_epa',
     'deep_target_epa':'team_OL_deep_target_epa'},
    axis=1,
    inplace=True)
ol_passing_offense_team[
        [
            "team_OL_epa",
            "team_OL_short_target_epa",
            "team_OL_medium_target_epa",
            "team_OL_deep_target_epa",
        ]
] = ol_passing_offense_team.groupby(["posteam"]).apply(
    lambda x: x.ewm(span=8, min_periods=4)
    .mean()[
        [
            "team_OL_epa",
            "team_OL_short_target_epa",
            "team_OL_medium_target_epa",
            "team_OL_deep_target_epa",
        ]
        ]
    .shift(1)
)

#%%
ol_passing_offense=ol_passing_offense.merge(
    ol_passing_offense_team,on=['week','season','game_id','posteam','game_date'],how='left'
    )
for stat in ['OL_epa','OL_short_target_epa','OL_medium_target_epa','OL_deep_target_epa']:
    ol_passing_offense.loc[ol_passing_offense[stat].isna()==True,stat]=ol_passing_offense.loc[ol_passing_offense[stat].isna()==True,f'team_{stat}']
ol_passing_offense.to_csv('./OL_Data/OL_PassingData.csv',index=False)
ol_passing_offense_team.to_csv('./OL_Data/OL_TeamPassingData.csv',index=False)


#%% Get Defense Stats
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
passing_defense = (
    pbp_passing[pbp_passing.receiver_position.isin(['RB','WR','TE'])]
    .groupby(["game_id", "week", "season", "defteam", "position", "gsis_id", "game_date"])
    .agg(
                    {'air_yards':np.sum,
                    'completed_air_yards':np.sum,
                    'posteam_pass_attempts':'first',
                    'posteam_air_yards':'first',
                    'epa':np.sum,
                    'air_epa':np.sum,
                    'yac_epa':np.sum,
                    'yards_after_catch':np.sum,
                    'short_target_epa':np.sum,
                    'medium_target_epa':np.sum,
                    'deep_target_epa':np.sum,
                    'short_target':np.sum,
                    'short_rec':np.sum,
                    'medium_target':np.sum,
                    'medium_rec':np.sum,
                    'deep_target':np.sum,
                    'deep_rec':np.sum,
                    'targets':np.sum,
                    'rec':np.sum,
                    'receiving_yards':np.sum,})
    .reset_index()
)
passing_defense['air_yards_efficiency']=passing_defense.completed_air_yards/passing_defense.air_yards
passing_defense['air_yards_per_attempt']=passing_defense.air_yards/passing_defense.targets
passing_defense['catch_rate']=passing_defense.rec/passing_defense.targets
passing_defense['short_catch_rate']=passing_defense.short_rec/passing_defense.short_target
passing_defense['medium_catch_rate']=passing_defense.medium_rec/passing_defense.medium_target
passing_defense['deep_catch_rate']=passing_defense.deep_rec/passing_defense.deep_target
passing_defense['epa']=passing_defense.epa/passing_defense.targets
passing_defense['air_epa']=passing_defense.air_epa/passing_defense.targets
passing_defense['short_target_epa']=passing_defense.short_target_epa/passing_defense.short_target
passing_defense['medium_target_epa']=passing_defense.medium_target_epa/passing_defense.medium_target
passing_defense['deep_target_epa']=passing_defense.deep_target_epa/passing_defense.deep_target
#%% Passing Defense
passing_defense[
    [
        "epa",
        "air_epa",
        "short_target_epa",
        "medium_target_epa",
        "deep_target_epa",
        "yac_epa",
        "yards_after_catch",
        "catch_rate",
        "short_catch_rate",
        "medium_catch_rate",
        "deep_catch_rate",
        "air_yards_efficiency",
        "air_yards_per_attempt"
    ]
] = passing_defense.groupby(["defteam", "position","gsis_id"]).apply(
    lambda x: x.ewm(span=5, min_periods=1)
    .mean()[
        [
            "epa",
            "air_epa",
            "short_target_epa",
            "medium_target_epa",
            "deep_target_epa",
            "yac_epa",
            "yards_after_catch",
            "catch_rate",
            "short_catch_rate",
            "medium_catch_rate",
            "deep_catch_rate",
            "air_yards_efficiency",
            "air_yards_per_attempt"
        ]
    ]
    .shift(1)
)
passing_defense.rename(
    {'epa':'def_epa',
     'air_epa':'def_air_epa',
    'short_target_epa':'def_short_target_epa',
    'medium_target_epa':'def_medium_target_epa',
    'deep_target_epa':'def_deep_target_epa',
    'yac_epa':'def_yac_epa',
    'yards_after_catch':'def_yards_after_catch',
    "catch_rate":'def_catch_rate',
    'short_catch_rate':'def_short_catch_rate',
    'medium_catch_rate':'def_medium_catch_rate',
    'deep_catch_rate':'def_deep_catch_rate',
    'air_yards_efficiency':'def_air_yards_efficiency',
    'air_yards_per_attempt':'def_air_yards_per_attempt'
    },
    axis=1,inplace=True)

#%% Get Team Defensive Data
passing_defense_team = (
    pbp_passing[pbp_passing.receiver_position.isin(['RB','WR','TE'])]
    .groupby(["game_id", "week", "season", "defteam", "position", "game_date"])
    .agg(
                    {'air_yards':np.sum,
                    'completed_air_yards':np.sum,
                    'posteam_pass_attempts':'first',
                    'posteam_air_yards':'first',
                    'epa':np.sum,
                    'air_epa':np.sum,
                    'yac_epa':np.sum,
                    'yards_after_catch':np.sum,
                    'short_target_epa':np.sum,
                    'medium_target_epa':np.sum,
                    'deep_target_epa':np.sum,
                    'short_target':np.sum,
                    'short_rec':np.sum,
                    'medium_target':np.sum,
                    'medium_rec':np.sum,
                    'deep_target':np.sum,
                    'deep_rec':np.sum,
                    'targets':np.sum,
                    'rec':np.sum,
                    'receiving_yards':np.sum,})
)
passing_defense_team['air_yards_efficiency']=passing_defense_team.completed_air_yards/passing_defense_team.air_yards
passing_defense_team['air_yards_per_attempt']=passing_defense_team.air_yards/passing_defense_team.targets

passing_defense_team['catch_rate']=passing_defense_team.rec/passing_defense_team.targets
passing_defense_team['short_catch_rate']=passing_defense_team.short_rec/passing_defense_team.short_target
passing_defense_team['medium_catch_rate']=passing_defense_team.medium_rec/passing_defense_team.medium_target
passing_defense_team['deep_catch_rate']=passing_defense_team.deep_rec/passing_defense_team.deep_target
passing_defense_team['epa']=passing_defense_team.epa/passing_defense_team.targets
passing_defense_team['air_epa']=passing_defense_team.air_epa/passing_defense_team.targets
passing_defense_team['short_target_epa']=passing_defense_team.short_target_epa/passing_defense_team.short_target
passing_defense_team['medium_target_epa']=passing_defense_team.medium_target_epa/passing_defense_team.medium_target
passing_defense_team['deep_target_epa']=passing_defense_team.deep_target_epa/passing_defense_team.deep_target
passing_defense_team[
    [
        "epa",
        'air_epa',
        "short_target_epa",
        "medium_target_epa",
        "deep_target_epa",
        "yac_epa",
        "yards_after_catch",
        "catch_rate",
        "short_catch_rate",
        "medium_catch_rate",
        "deep_catch_rate",
        "air_yards_efficiency",
        "air_yards_per_attempt"
    ]
] = passing_defense_team.groupby(["defteam", "position"]).apply(
    lambda x: x.ewm(span=8, min_periods=1)
    .mean()[
        [
        "epa",
        "air_epa",
        "short_target_epa",
        "medium_target_epa",
        "deep_target_epa",
        "yac_epa",
        "catch_rate",
        "yards_after_catch",
        "short_catch_rate",
        "medium_catch_rate",
        "deep_catch_rate",
        "air_yards_efficiency",
        "air_yards_per_attempt"
        ]
    ]
    .shift(1)
)
passing_defense_team.rename({
                    'epa':'team_def_epa',
                    "air_epa":'team_def_air_epa',
                    'short_target_epa':'team_def_short_target_epa',
                    'medium_target_epa':'team_def_medium_target_epa',
                    'deep_target_epa':'team_def_deep_target_epa',
                    'yac_epa':'team_def_yac_epa',
                    'yards_after_catch':'team_def_yards_after_catch',
                    "catch_rate":'team_def_catch_rate',
                    'short_catch_rate':'team_def_short_catch_rate',
                    'medium_catch_rate':'team_def_medium_catch_rate',
                    'deep_catch_rate':'team_def_deep_catch_rate',
                    'air_yards_efficiency':'team_def_air_yards_efficiency',
                    'air_yards_per_attempt':'team_def_air_yards_per_attempt'
                    },
    axis=1,inplace=True)
passing_defense_team.to_csv('TeamDefensivePassData.csv')
#%% Merge Individual Defense Stats with Team Defense Stats
DEF_STATS=[
 'def_epa',
 'def_air_epa',
 'def_yac_epa',
 'def_yards_after_catch',
 'def_short_target_epa',
 'def_medium_target_epa',
 'def_deep_target_epa',
 'def_air_yards_efficiency',
 'def_air_yards_per_attempt',
 'def_catch_rate',
 'def_short_catch_rate',
 'def_medium_catch_rate',
 'def_deep_catch_rate']
passing_defense=passing_defense.merge(passing_defense_team,on=['week','season','game_id','defteam','game_date'],how='left')
for stat in DEF_STATS:
    passing_defense.loc[passing_defense[stat].isna()==True,stat]=passing_defense.loc[passing_defense[stat].isna()==True,f'team_{stat}']
passing_defense[['game_id',
 'week',
 'season',
 'defteam',
 'position',
 'gsis_id',
 'game_date',
 'def_epa',
 'def_air_epa',
 'def_yac_epa',
 'def_yards_after_catch',
 'def_short_target_epa',
 'def_medium_target_epa',
 'def_deep_target_epa',
 'def_air_yards_efficiency',
 'def_air_yards_per_attempt',
 'def_catch_rate',
 'def_short_catch_rate',
 'def_medium_catch_rate',
 'def_deep_catch_rate',
 'team_def_epa',
 'team_def_yac_epa',
 'team_def_yards_after_catch',
 'team_def_short_target_epa',
 'team_def_medium_target_epa',
 'team_def_deep_target_epa',
 'team_def_air_yards_efficiency',
 'team_def_catch_rate',
 'team_def_short_catch_rate',
 'team_def_medium_catch_rate',
 'team_def_deep_catch_rate']].to_csv('DefensivePassData.csv',index=False)