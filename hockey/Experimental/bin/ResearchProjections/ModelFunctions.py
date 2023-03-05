#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 22:04:16 2022

@author: robertmegnia
"""

import numpy as np
import pandas as pd
import os

basedir = os.path.dirname(os.path.abspath(__file__))
etcdir = f"{basedir}/../../etc"
datadir = f"{basedir}/../../data"
projdir = f"{datadir}/Projections"
#%%
from config.Stochastic_model_config import *
from config.MLModel_config import *
from getDKPts import *

os.chdir(f"{basedir}/../")
from Models.ModelDict import *

ProjFileColumns = [
    "full_name",
    "player_id",
    "position",
    "position_type",
    "team",
    "game_location",
    "opp",
    "game_date",
    "DKPts",
    "season",
    "game_date_string",
    "Floor",
    "Ceiling",
    "Stochastic",
    "BR",
    "EN",
    "NN",
    "RF",
    "GB",
    "Tweedie",
    "ML",
]


def rolling_average(df, window=8):
    return df.rolling(min_periods=1, window=window).mean()


def StochasticPrediction(
    player, position, Salary, stats_df, opp, weekly_proj_df, stat_type
):
    if stat_type == "Goalie":
        STATS = DK_Goalie_Stats
        stats_df["line"] = 1
    else:
        STATS = DK_Skater_Stats
    player_weight = 0.65
    opp_weight = 0.35
    stats_df.sort_values(by="game_date", inplace=True)
    name = stats_df[stats_df.player_id == player].full_name.unique()
    n_games = 15
    feature_frame = pd.DataFrame({})
    opp_feature_frame = pd.DataFrame({})
    player_df = stats_df[stats_df.player_id == player][-n_games:]
    opp_df = stats_df[
        (stats_df.opp == opp)
        & (stats_df.position == position)
        & ((stats_df.Salary >= Salary - 500) & (stats_df.Salary <= Salary + 500))
    ][-n_games:]
    if (len(player_df) == 0) | ((len(opp_df) == 0)):
        return pd.DataFrame(
            {"Floor": [np.nan], "Ceiling": [np.nan], "Stochastic": [np.nan]}
        )
    for stat in STATS:
        mean = player_df[stat].mean()
        std = player_df[stat].std()
        stats = np.random.normal(loc=mean, scale=std, size=10000)
        feature_frame[stat] = stats
        #
        opp_mean = opp_df[stat].mean()
        opp_std = opp_df[stat].std()
        opp_stats = np.random.normal(loc=opp_mean, scale=opp_std, size=10000)
        opp_feature_frame[stat] = opp_stats
    feature_frame.fillna(0, inplace=True)
    opp_feature_frame.fillna(0, inplace=True)
    feature_frame = feature_frame.mask(feature_frame.lt(0), 0)
    opp_feature_frame = opp_feature_frame.mask(opp_feature_frame.lt(0), 0)
    ProjectionsFrame = pd.DataFrame({})
    if stat_type == "Goalie":
        Stochastic = getGoalieDKPts(feature_frame)
        opp_Stochastic = getGoalieDKPts(opp_feature_frame)
    else:
        Stochastic = getSkaterDKPts(feature_frame)
        opp_Stochastic = getSkaterDKPts(opp_feature_frame)
    ProjectionsFrame["Stochastic"] = (Stochastic * player_weight) + (
        opp_Stochastic * opp_weight
    )
    feature_frame = pd.concat([feature_frame, opp_feature_frame]).reset_index(drop=True)
    Floor = round(ProjectionsFrame.Stochastic.quantile(0.15), 1)
    if Floor < 0:
        Floor = 0
    if stat_type == "Skater":
        Ceiling = round(ProjectionsFrame.Stochastic.quantile(0.85), 1)
        Stochastic = round(ProjectionsFrame.Stochastic.quantile(0.4), 1)
        ShotProb = len(feature_frame[feature_frame.shots >= 5]) / 20000
        BlockProb = len(feature_frame[feature_frame.blocked >= 3]) / 20000
        print(name, Stochastic)
        ProjFrame = pd.DataFrame(
            {
                "Floor": [Floor],
                "Ceiling": [Ceiling],
                "Stochastic": [Stochastic],
                "ShotProb": [ShotProb],
                "BlockProb": [BlockProb],
            }
        )
    else:
        Ceiling = round(ProjectionsFrame.Stochastic.quantile(0.85), 1)
        Stochastic = round(ProjectionsFrame.Stochastic.mean(), 1)
        print(name, Stochastic)
        ProjFrame = pd.DataFrame(
            {"Floor": [Floor], "Ceiling": [Ceiling], "Stochastic": [Stochastic]}
        )
    return ProjFrame


def MLPrediction(
    skater_stats_df, goalie_stats_df, skater_games_df, goalie_games_df, stat_type, line
):
    print(stat_type)
    if stat_type in ["Forward", "Defenseman"]:
        weekly_proj_df = skater_games_df[skater_games_df.position_type == stat_type]
        stats_df = skater_stats_df[skater_stats_df.position_type == stat_type]
        # Goalie Stats
        goalie_stats_df["line"] = goalie_stats_df.groupby(
            ["opp", "game_date"]
        ).timeOnIce.rank(ascending=False, method="first")
        goalie_stats_df = goalie_stats_df[goalie_stats_df.line == 1]
        goalie_stats_df["team"] = goalie_stats_df["opp"]
        goalie_stats_df = goalie_stats_df[NonFeatures + OpposingGoalieColumns]
        goalie_stats_df.rename(
            dict([(c, f"goalie_{c}") for c in OpposingGoalieColumns]),
            axis=1,
            inplace=True,
        )
        goalie_db_Features = [f"goalie_{c}" for c in OpposingGoalieColumns]
        # Opponent Stats
        opp_stats_df = stats_df.groupby(["opp", "game_date"]).sum()[OpposingTeamColumns]
        opp_stats_df = (
            opp_stats_df.groupby("opp")
            .apply(lambda x: rolling_average(x))
            .add_prefix("opp_")
        )
        if "line" in NonFeatures:
            NonFeatures.remove("line")
    else:
        weekly_proj_df = goalie_games_df.groupby("full_name", as_index=False).first()
        weekly_proj_df["line"] = 1
        NonFeatures.append("line")
        stats_df = goalie_stats_df
        opp_stats_df = skater_stats_df

    weekly_proj_df.drop_duplicates(inplace=True)
    # Sort stats database chronologically and filter down to active players for the week
    stats_df.sort_values(by="game_date", inplace=True)
    stats_df = stats_df[stats_df.player_id.isin(weekly_proj_df.player_id)]

    features = (
        stats_df.groupby(["player_id"])
        .apply(lambda x: rolling_average(x))
        .drop(NonFeatures, axis=1, errors="ignore")
    )
    features[["player_id", "team"]] = stats_df[["player_id", "team"]]
    # Reinsert gsis_id, get last of rolling averages, insert known features
    if len(features) == 0:
        return
    features = features.groupby(["player_id"], as_index=False).last()
    if stat_type in ["Forward", "Defenseman"]:
        goalie_features = goalie_stats_df.groupby("player_id")[
            goalie_db_Features
        ].apply(lambda x: rolling_average(x))
        goalie_features[["team", "game_date"]] = goalie_stats_df[["team", "game_date"]]
        goalie_features = goalie_features.groupby("team", as_index=False).last()
        opp_features = opp_stats_df.groupby("opp").last()
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
    if stat_type in ["Forward", "Defenseman"]:
        features = features.merge(goalie_features, on=["team"], how="left")
        features = features.merge(opp_features, on=["opp"], how="left")
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
    for position in ProjFrame.position_type.unique():
        models = ModelDict[f"{position}{line}"]
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
    # Predict Player Shares
    for position in ProjFrame.position_type.unique():
        if position == "Goalie":
            break
        for stat in ["goals", "assists", "shots", "blocked", "DKPts"]:
            model = SharesModelDict[f"{position}{line}_{stat}"]
            ProjFrame.loc[
                ProjFrame.position_type == position, f"proj_{stat}_share"
            ] = model.predict(
                ProjFrame.loc[
                    ProjFrame.position_type == position,
                    features.drop(NonFeatures, axis=1, errors="ignore").columns,
                ]
            )
            ProjFrame.loc[
                (ProjFrame.position_type == position) & (ProjFrame[stat] < 0), stat
            ] = 0
    ProjFrame["ML"] = ProjFrame[models.keys()].mean(axis=1)
    ProjFrame["DKPts"] = weekly_proj_df.set_index("player_id").DKPts
    ProjFrame.reset_index(inplace=True)
    return ProjFrame


#
def TeamStatsPredictions(team_games_df, team_stats_df):
    features = (
        team_stats_df.groupby(["team", "opp"])
        .apply(lambda x: rolling_average(x))
        .drop(NonFeatures, axis=1, errors="ignore")
    )
    team_stats_df = team_stats_df[team_stats_df.index.isin(features.index)]
    features[["team", "opp"]] = team_stats_df[["team", "opp"]]
    # Reinsert gsis_id, get last of rolling averages, insert known features
    if len(features) == 0:
        return
    features = features.groupby(["team"], as_index=False).last()
    opp_features = (
        team_stats_df.groupby(["opp", "game_date"])
        .sum()
        .drop(TeamNonFeatures, axis=1, errors="ignore")
    )
    opp_features = (
        opp_features.groupby("opp")
        .apply(lambda x: rolling_average(x))
        .add_prefix("opp_")
    )
    opp_features = opp_features.groupby("opp").last()
    # Merge Offense and Defense features
    features = features.merge(opp_features, on=["opp"], how="left")

    ProjFrame = team_games_df[TeamNonFeatures]
    ProjFrame.set_index("team", inplace=True)
    ProjFrame = ProjFrame.join(
        features.set_index("team").drop(TeamNonFeatures, axis=1, errors="ignore")
    )
    ProjFrame.dropna(inplace=True)
    ProjFrame.drop_duplicates(inplace=True)
    if len(ProjFrame) == 0:
        return ProjFrame
    for stat in ["goals", "assists", "shots", "blocked", "DKPts"]:
        model = TeamModelDict[f"RF{stat}"]
        ProjFrame[f"proj_{stat}"] = model.predict(
            ProjFrame[features.drop(TeamNonFeatures, axis=1, errors="ignore").columns]
        )
        ProjFrame.loc[(ProjFrame[f"proj_{stat}"] < 0), f"proj_{stat}"] = 0
    ProjFrame[stat] = team_games_df.drop_duplicates().set_index("team")[stat]
    ProjFrame.reset_index(inplace=True)
    return ProjFrame


def TopDownPrediction(df):
    df["TD_Proj"] = (
        (df.proj_shots_share * df.proj_shots * 1.5)
        + (df.proj_blocked_share * df.proj_blocked * 1.3)
        + (df.proj_goals_share * df.proj_goals * 8.5)
        + (df.proj_assists_share * df.proj_assists * 5)
    )
    df["TD_Proj2"] = df.proj_DKPts_share * df.proj_DKPts
    return df
