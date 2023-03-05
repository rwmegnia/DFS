#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 03:55:39 2022

@author: robertmegnia
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
import requests
import os
from os.path import exists

basedir = os.path.dirname(os.path.abspath(__file__))
etcdir = f"{basedir}/../../etc"
datadir = f"{basedir}/../../data"
projdir = f"{datadir}/Projections"
from config.Stochastic_model_config import *
from ModelFunctions import *
from getDKPts import *

os.chdir(f"{basedir}/../")


def getPMM(df, game_date):
    pos_frames = []
    methods = ["BR", "EN", "RF", "GB"]
    df["ML"] = df[methods].mean(axis=1)
    for pos in ["Forward", "Defenseman", "Goalie"]:
        if pos == "Defenseman":
            pos_frame = df[df.position_type == pos]
            methods = [
                "BR",
                "EN",
                "RF",
                "GB",
                "Stochastic",
                "TD_Proj",
                "TD_Proj2",
                "ML",
            ]
            pos_frame["Proj"] = pos_frame[methods].mean(axis=1)
        elif pos == "Goalie":
            pos_frame = df[df.position_type == pos]
            methods = ["BR", "EN", "NN", "RF", "GB", "Tweedie", "Stochastic", "ML"]
            pos_frame["Proj"] = pos_frame[methods].mean(axis=1)
        else:
            methods = [
                "BR",
                "EN",
                "RF",
                "GB",
                "Stochastic",
                "TD_Proj",
                "TD_Proj2",
                "ML",
            ]
            pos_frame = df[df.position_type == pos]
            pos_frame["Proj"] = pos_frame[methods].mean(axis=1)
        members = pd.concat([pos_frame[m] for m in methods])
        members.sort_values(ascending=False, inplace=True)
        members.name = "PMM"
        members = members.to_frame()
        pos_frame.sort_values(by="Proj", ascending=False, inplace=True)
        members = members[len(methods) - 1 :: len(methods)]
        pos_frame["PMM"] = members.values
        pos_frames.append(pos_frame)
    df = pd.concat(pos_frames)
    df.reset_index(drop=True, inplace=True)
    methods = ["BR", "EN", "RF", "GB"]
    df["ML"] = df[["PMM"] + methods].mean(axis=1)
    df.drop("Proj", axis=1, inplace=True)
    return df


#%%
# Build Stochastic Projections Database
datadir = f"{basedir}/../../data"
skater_db = pd.read_csv(f"{datadir}/game_logs/SkaterStatsDatabase.csv")
skater_db["game_date_string"] = skater_db.game_date
skater_db.game_date = pd.to_datetime(skater_db.game_date)
skater_db.sort_values(by="game_date", inplace=True)
#
goalie_db = pd.read_csv(f"{datadir}/game_logs/GoalieStatsDatabase.csv")
goalie_db["game_date_string"] = goalie_db.game_date
goalie_db.game_date = pd.to_datetime(goalie_db.game_date)
goalie_db.sort_values(by="game_date", inplace=True)
#
team_db = pd.read_csv(f"{datadir}/game_logs/TeamStatsDatabase.csv")
team_db["game_date_string"] = team_db.game_date
team_db.game_date = pd.to_datetime(team_db.game_date)
team_db.sort_values(by="game_date", inplace=True)
proj_frames = []
for game_date in skater_db.game_date_string.unique():
    print(game_date)
    season = 2021
    if game_date < "2022-02-01":
        continue
    if exists(f"{projdir}/{season}/{game_date}_Projections.csv"):
        continue
    print(game_date)
    skater_stats_df = skater_db[skater_db.game_date < game_date]
    skater_stats_df.sort_values(by="game_date", inplace=True)
    goalie_stats_df = goalie_db[goalie_db.game_date < game_date]
    goalie_stats_df.sort_values(by="game_date", inplace=True)
    team_stats_df = team_db[team_db.game_date < game_date]
    team_stats_df.sort_values(by="game_date", inplace=True)
    skater_games_df = skater_db[skater_db.game_date == game_date]
    goalie_games_df = goalie_db[goalie_db.game_date == game_date]
    team_games_df = team_db[team_db.game_date == game_date]
    season = skater_games_df.season.unique()[0]
    # Stochastic Projections
    skater_proj_df = pd.concat(
        [
            skater_games_df,
            pd.concat(
                [
                    a
                    for a in skater_games_df.apply(
                        lambda x: StochasticPrediction(
                            x.player_id,
                            x.position,
                            x.line,
                            skater_stats_df,
                            x.opp,
                            skater_games_df[skater_games_df.player_id == x.player_id],
                            "Skater",
                        ),
                        axis=1,
                    )
                ]
            ).set_index(skater_games_df.index),
        ],
        axis=1,
    )
    goalie_proj_df = pd.concat(
        [
            goalie_games_df,
            pd.concat(
                [
                    a
                    for a in goalie_games_df.apply(
                        lambda x: StochasticPrediction(
                            x.player_id,
                            x.position,
                            1,
                            goalie_stats_df,
                            x.opp,
                            goalie_games_df[goalie_games_df.player_id == x.player_id],
                            "Goalie",
                        ),
                        axis=1,
                    )
                ]
            ).set_index(goalie_games_df.index),
        ],
        axis=1,
    )
    # ML Projections
    offense_ml_proj_df1 = MLPrediction(
        skater_stats_df[skater_stats_df.line == 1],
        goalie_stats_df,
        skater_games_df[skater_games_df.line == 1],
        goalie_games_df,
        "Forward",
        line=1,
    )
    offense_ml_proj_df2 = MLPrediction(
        skater_stats_df[skater_stats_df.line == 2],
        goalie_stats_df,
        skater_games_df[skater_games_df.line == 2],
        goalie_games_df,
        "Forward",
        line=2,
    )
    offense_ml_proj_df3 = MLPrediction(
        skater_stats_df[skater_stats_df.line == 3],
        goalie_stats_df,
        skater_games_df[skater_games_df.line == 3],
        goalie_games_df,
        "Forward",
        line=3,
    )
    offense_ml_proj_df4 = MLPrediction(
        skater_stats_df[skater_stats_df.line == 4],
        goalie_stats_df,
        skater_games_df[skater_games_df.line == 4],
        goalie_games_df,
        "Forward",
        line=4,
    )
    try:
        offense_ml_proj_df = pd.concat(
            [
                offense_ml_proj_df1,
                offense_ml_proj_df2,
                offense_ml_proj_df3,
                offense_ml_proj_df4,
            ]
        )
    except ValueError:
        continue
    defense_ml_proj_df1 = MLPrediction(
        skater_stats_df[skater_stats_df.line == 1],
        goalie_stats_df,
        skater_games_df[skater_games_df.line == 1],
        goalie_games_df,
        "Defenseman",
        line=1,
    )
    defense_ml_proj_df2 = MLPrediction(
        skater_stats_df[skater_stats_df.line == 2],
        goalie_stats_df,
        skater_games_df[skater_games_df.line == 2],
        goalie_games_df,
        "Defenseman",
        line=2,
    )
    defense_ml_proj_df3 = MLPrediction(
        skater_stats_df[skater_stats_df.line == 3],
        goalie_stats_df,
        skater_games_df[skater_games_df.line == 3],
        goalie_games_df,
        "Defenseman",
        line=3,
    )
    defense_ml_proj_df = pd.concat(
        [defense_ml_proj_df1, defense_ml_proj_df2, defense_ml_proj_df3]
    )
    goalie_ml_proj_df = MLPrediction(
        skater_stats_df,
        goalie_stats_df,
        skater_games_df,
        goalie_games_df,
        "Goalie",
        line=1,
    )
    # Top Down Projections
    team_proj_df = TeamStatsPredictions(team_games_df, team_stats_df)
    if (len(skater_proj_df) == 0) | (offense_ml_proj_df is None):
        continue
    ml_proj_df = pd.concat([offense_ml_proj_df, defense_ml_proj_df, goalie_ml_proj_df])
    ml_proj_df = ml_proj_df.merge(
        team_proj_df[
            [
                "team",
                "proj_goals",
                "proj_assists",
                "proj_shots",
                "proj_blocked",
                "proj_DKPts",
            ]
        ],
        on="team",
        how="left",
    )
    ml_proj_df = TopDownPrediction(ml_proj_df)
    stochastic_proj_df = pd.concat([skater_proj_df, goalie_proj_df])
    proj_frame = ml_proj_df.merge(
        stochastic_proj_df,
        on=[
            "full_name",
            "player_id",
            "position",
            "position_type",
            "team",
            "game_location",
            "opp",
            "game_date",
            "DKPts",
        ],
        how="left",
    )
    proj_frame = getPMM(proj_frame, game_date)

    proj_frame["Projection"] = proj_frame[
        ["Stochastic", "ML", "PMM", "TD_Proj", "TD_Proj2"]
    ].mean(axis=1)
    proj_frame.to_csv(f"{projdir}/2021/{game_date}_Projections.csv", index=False)
