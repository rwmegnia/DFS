#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 20:56:13 2022

@author: robertmegnia
"""
import pandas as pd
import requests
import os
from config.MLModel_config import *

basedir = os.path.dirname(os.path.abspath(__file__))
etcdir = f"{basedir}/../../etc"
datadir = f"{basedir}/../../data"
projdir = f"{datadir}/Projections"
os.chdir("../")
from Models.ModelDict import *


def rolling_average(df, window=1):
    return df.rolling(min_periods=1, window=window).mean().shift(1)


def MLPrediction(
    skater_stats_df,
    goalie_stats_df,
    skater_games_df,
    goalie_games_df,
    skater_db,
    goalie_db,
    stat_type,
):
    if stat_type != "Goalie":
        goalie_db["depth_team"] = goalie_db.groupby(
            ["opp", "game_date"]
        ).timeOnIce.rank(ascending=False, method="first")
        goalie_db = goalie_db[goalie_db.depth_team == 1]
        goalie_db["team"] = goalie_db["opp"]
        goalie_db = goalie_db[NonFeatures + OpposingGoalieColumns]
        goalie_db.rename(
            dict([(c, f"goalie_{c}") for c in OpposingGoalieColumns]),
            axis=1,
            inplace=True,
        )
        goalie_db_Features = [f"goalie_{c}" for c in OpposingGoalieColumns]
        #
        DK_stats = DK_Stats["Skater"]
        weekly_proj_df = skater_games_df
        stats_df = skater_stats_df
        opp_stats_df = goalie_stats_df
    else:
        goalie_db["depth_team"] = goalie_db.groupby(
            ["team", "game_date"]
        ).timeOnIce.rank(ascending=False, method="first")
        DK_stats = DK_Stats["Goalie"]
        weekly_proj_df = goalie_games_df.groupby("full_name", as_index=False).first()
        weekly_proj_df["depth_team"] = 1
        stats_df = goalie_stats_df
        opp_stats_df = skater_stats_df

    weekly_proj_df.drop_duplicates(inplace=True)
    # Sort stats database chronologically and filter down to active players for the week
    stats_df.sort_values(by="game_date", inplace=True)
    stats_df = stats_df[stats_df.player_id.isin(weekly_proj_df.player_id)]

    features = (
        stats_df.groupby(["player_id"])[DK_stats]
        .apply(lambda x: rolling_average(x))
        .drop(NonFeatures, axis=1, errors="ignore")
    )
    features[["player_id", "team"]] = stats_df[["player_id", "team"]]
    # Reinsert gsis_id, get last of rolling averages, insert known features
    if len(features) == 0:
        return
    features = features.groupby(["player_id"], as_index=False).last()
    if stat_type in ["Offense", "Defense"]:
        opp_features = opp_stats_df.groupby("player_id")[goalie_db_Features].apply(
            lambda x: rolling_average(x)
        )
        opp_features[["team", "game_date"]] = opp_stats_df[["team", "game_date"]]
        opp_features = opp_features.groupby("team", as_index=False).last()
    else:
        opp_features = opp_stats_df.groupby(["opp", "game_date"]).sum()[
            OpposingTeamColumns
        ]
        opp_features["shooting_percentage"] = opp_features.goals / opp_features.shots
        opp_features["give_take_ratio"] = (
            opp_features.giveaways / opp_features.takeaways
        )
        opp_features = (
            opp_features.groupby("opp")
            .apply(lambda x: rolling_average(x))
            .add_prefix("opp_")
        )
        opp_features = opp_features.groupby("opp").last()
    # Merge Offense and Defense features
    features.set_index("player_id", inplace=True)
    features = features.join(
        weekly_proj_df[["player_id", "opp"]].set_index("player_id")
    ).reset_index()
    if stat_type in ["Offense", "Defense"]:
        features = features.merge(opp_features, on=["team"], how="left")
    else:
        features = features.merge(opp_features, on=["opp"], how="left")

    features.rename({"DKPts": "avg_DKPts"}, axis=1, inplace=True)
    ProjFrame = weekly_proj_df[NonFeatures]
    ProjFrame.set_index("player_id", inplace=True)
    ProjFrame = ProjFrame.join(
        features.set_index("player_id").drop(NonFeatures, axis=1, errors="ignore")
    )
    ProjFrame.dropna(inplace=True)
    ProjFrame.drop_duplicates(inplace=True)
    if len(ProjFrame) == 0:
        return ProjFrame
    ProjFrame.loc[ProjFrame.position.isin(["L", "C", "R"]), "position_type"] = "Offense"
    ProjFrame.loc[ProjFrame.position == "D", "position_type"] = "Defense"
    ProjFrame.loc[ProjFrame.position == "G", "position_type"] = "Goalie"
    for position in ProjFrame.position_type.unique():
        models = ModelDict[position]
        for method, model in models.items():
            ProjFrame.loc[ProjFrame.position_type == position, method] = model.predict(
                ProjFrame.loc[
                    ProjFrame.position_type == position,
                    features.drop(NonFeatures, axis=1, errors="ignore").columns,
                ]
            )
            ProjFrame.loc[
                (ProjFrame.position_type == position) & (ProjFrame[method] < 0), method
            ] = 0
    ProjFrame["ML"] = ProjFrame[models.keys()].mean(axis=1)
    ProjFrame["DKPts"] = weekly_proj_df.set_index("player_id").DKPts
    ProjFrame.reset_index(inplace=True)
    return ProjFrame


#%%
# Build ML Projections Database
datadir = f"{basedir}/../../data"
for stat_type in ["Goalie"]:
    skater_db = pd.read_csv(f"{datadir}/game_logs/SkaterStatsDatabase.csv")
    if stat_type == "Offense":
        skater_db = skater_db[skater_db.position.isin(["L", "C", "R"])]
    elif stat_type == "Defense":
        skater_db = skater_db[skater_db.position == "D"]
    goalie_db = pd.read_csv(f"{datadir}/game_logs/GoalieStatsDatabase.csv")
    skater_db["game_date_string"] = skater_db.game_date
    goalie_db["game_date_string"] = goalie_db.game_date
    skater_db.game_date = pd.to_datetime(skater_db.game_date)
    skater_db.sort_values(by="game_date", inplace=True)
    goalie_db.game_date = pd.to_datetime(goalie_db.game_date)
    goalie_db.sort_values(by="game_date", inplace=True)
    if stat_type in ["Offense", "Defense"]:
        goalie_db["depth_team"] = goalie_db.groupby(
            ["opp", "game_date"]
        ).timeOnIce.rank(ascending=False, method="first")
        goalie_db = goalie_db[goalie_db.depth_team == 1]
        goalie_db["team"] = goalie_db["opp"]
        goalie_db = goalie_db[NonFeatures + OpposingGoalieColumns]
        goalie_db.rename(
            dict([(c, f"goalie_{c}") for c in OpposingGoalieColumns]),
            axis=1,
            inplace=True,
        )
        goalie_db_Features = [f"goalie_{c}" for c in OpposingGoalieColumns]
    else:
        goalie_db["depth_team"] = goalie_db.groupby(
            ["team", "game_date"]
        ).timeOnIce.rank(ascending=False, method="first")
    for game_date in skater_db.game_date_string.unique():
        print(game_date)
        proj_frames = []
        skater_stats_df = skater_db[skater_db.game_date < game_date]
        if len(skater_stats_df) == 0:
            continue
        goalie_stats_df = goalie_db[goalie_db.game_date < game_date]
        skater_stats_df.sort_values(by="game_date", inplace=True)
        goalie_stats_df.sort_values(by="game_date", inplace=True)
        skater_games_df = skater_db[skater_db.game_date == game_date]
        goalie_games_df = goalie_db[goalie_db.game_date == game_date]
        season = skater_games_df.season.unique()[0]
        ml_proj_df = MLPrediction(
            skater_stats_df,
            goalie_stats_df,
            skater_games_df,
            goalie_games_df,
            stat_type,
        )
        if ml_proj_df is None:
            continue
        proj_frames.append(ml_proj_df)
        if len(proj_frames) == 0:
            continue
        projdir = f"{datadir}/Projections"
        ml_proj_df.to_csv(
            f"{projdir}/{int(season)}/ML/{game_date}_{stat_type}MLProjections.csv",
            index=False,
        )
