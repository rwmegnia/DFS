#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 22:04:16 2022

@author: robertmegnia
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 08:59:54 2021

@author: robertmegnia
"""
import numpy as np
import pandas as pd
import os

basedir = os.path.dirname(os.path.abspath(__file__))
etcdir = f"{basedir}/../../etc"
datadir = f"{basedir}/../../data"
projdir = f"{datadir}/Projections"
from config.MLModel_config import *
from config.StochasticModel_config import *
from getDKPts import getDKPts

os.chdir("../")
from Models.ModelDict import *

os.chdir(f"{basedir}/../")

#%%
def rolling_average(df, window=10):
    return df.rolling(min_periods=1, window=window).mean()


def MLPrediction(stats_df, game_proj_df):

    # Filter player down to active players for the week
    opp_stats_df = stats_df[stats_df.opp.isin(game_proj_df.opp)]
    stats_df = stats_df[stats_df.player_id.isin(game_proj_df.player_id)]
    # Create prediction features frame by taking 15 game running average of player stats
    features = (
        stats_df.groupby(["player_id"])
        .apply(lambda x: rolling_average(x))
        .drop(NonFeatures + KnownFeatures, axis=1, errors="ignore")
    )

    # Reassign player_id,team to features frame
    features[["player_id", "team_abbreviation"]] = stats_df[
        ["player_id", "team_abbreviation"]
    ]
    if len(features) == 0:
        return

    # Get last row of features frame which is the latest running average for that player
    features = features.groupby(["player_id"], as_index=False).last()

    opp_features = (
        opp_stats_df.groupby(["opp", "game_date"])
        .agg(DefenseFeatures)
    )
    opp_features = (
        opp_features.groupby("opp")
        .apply(lambda x: rolling_average(x))
        .add_suffix("_allowed")
    )
    opp_features = opp_features.groupby("opp").last()

    # Merge Offense and Defense features
    features.set_index("player_id", inplace=True)
    features = features.join(
        game_proj_df[["player_id", "opp",'salary']].set_index("player_id")
    ).reset_index()
    features = features.merge(opp_features, on=["opp"], how="left")

    # Setup projection frame
    game_proj_df.rename({"team": "team_abbreviation"}, axis=1, inplace=True)
    ProjFrame = game_proj_df
    ProjFrame.set_index("player_id", inplace=True)
    ProjFrame = ProjFrame.join(
        features.set_index("player_id").drop(NonFeatures+KnownFeatures, axis=1, errors="ignore")
    )
    ProjFrame.dropna(inplace=True)
    if len(ProjFrame) == 0:
        return ProjFrame

    # Make projections
    for starter in ["starter", "bench"]:
        if starter == "starter":
            start = True
        else:
            start = False
        models = ModelDict[starter]
        for method, model in models.items():
            ProjFrame.loc[ProjFrame.starter == start, method] = model.predict(
                ProjFrame.loc[
                    ProjFrame.starter == start,
                    features.drop(NonFeatures, axis=1, errors="ignore").columns,
                ]
            )
            ProjFrame.loc[
                (ProjFrame.starter == start) & (ProjFrame[method] < 0), method
            ] = 0
        ProjFrame.loc[ProjFrame.starter == start,"ML"] = ProjFrame.loc[ProjFrame.starter==start][models.keys()].mean(axis=1)
    # Predict Player Shares
    for starter in ['starter','bench']:
        if starter=='starter':
            start=True
        else:
            start=False
        for stat in ["pts","fg3m","ast","reb","stl","blk","tov","dkpts"]:
            methods=['RF','GB']
            for method in methods:
                model = SharesModelDict[method][f"{starter}_{stat}"]
                ProjFrame.loc[
                    ProjFrame.starter==start, f"proj_pct_{stat}_{method}"
                ] = model.predict(
                    ProjFrame.loc[
                        ProjFrame.starter == start,
                        features.drop(NonFeatures, axis=1, errors="ignore").columns,
                    ]
                )
                ProjFrame.loc[
                    (ProjFrame.starter==start)
                    & (ProjFrame[f"proj_pct_{stat}_{method}"] < 0),
                    stat,
                ] = 0
            ProjFrame.loc[ProjFrame.starter==start,f"proj_pct_{stat}"] = ProjFrame.loc[ProjFrame.starter==start][[f"proj_pct_{stat}_{m}" for m in methods]].mean(axis=1)
        ProjFrame.reset_index(inplace=True)
    return ProjFrame


#
def TeamStatsPredictions(team_games_df, team_stats_df):
    # Get 15 game running average of team statistics
    features = (
        team_stats_df.groupby(["team_abbreviation", "opp"])
        .apply(lambda x: rolling_average(x))
        .drop(TeamNonFeatures, axis=1, errors="ignore")
    )
    features[["team_abbreviation", "opp"]] = team_stats_df[["team_abbreviation", "opp"]]
    if len(features) == 0:
        return

    # Get last row of team running average statistics
    features = features.groupby(["team_abbreviation"], as_index=False).last()

    # Get 15 game running average of oppponent statistics and latest average
    opp_features = (
        team_stats_df.groupby(["opp", "game_date"])
        .sum()
        .drop(TeamNonFeatures+TeamKnownFeatures, axis=1, errors="ignore")
    )
    opp_features = (
        opp_features.groupby("opp")
        .apply(lambda x: rolling_average(x))
        .add_suffix("_allowed")
    )
    opp_features = opp_features.groupby("opp").last()

    # Merge Offense and Defense features
    features = features.merge(opp_features, on=["opp"], how="left")

    # Setup projection
    ProjFrame = team_games_df.set_index('team_abbreviation')
    ProjFrame = ProjFrame.join(
        features.set_index("team_abbreviation").drop(TeamNonFeatures+TeamKnownFeatures, axis=1, errors="ignore")
    )
    ProjFrame.dropna(inplace=True)
    ProjFrame.drop_duplicates(inplace=True)
    if len(ProjFrame) == 0:
        return ProjFrame
    for stat in ["pts",'fg3m','ast','stl','blk','reb','to', "dkpts"]:
        methods=['RF','GB']
        for method in methods:
            model = TeamModelDict[method][f"{method}{stat}"]
            ProjFrame[f"{method}_proj_{stat}"] = model.predict(
                ProjFrame[features.drop(TeamNonFeatures, axis=1, errors="ignore").columns]
            )
            ProjFrame.loc[(ProjFrame[f"{method}_proj_{stat}"] < 0), f"{method}_proj_{stat}"] = 0
        ProjFrame[f'proj_{stat}']=ProjFrame[[f'{m}_proj_{stat}' for m in methods]].mean(axis=1)
    return ProjFrame


def StochasticPrediction(
    player, position, starter, salary, moneyline, stats_df, opp, game_proj_df
):
    if moneyline < 0:
        favorite = True
    else:
        favorite = False
    name = stats_df[stats_df.player_id == player].player_name.unique()
    n_games = 10
    feature_frame = pd.DataFrame({})
    opp_feature_frame = pd.DataFrame({})
    player_df = stats_df[stats_df.player_id == player][-n_games:]
    opp_df = stats_df[
        (stats_df.opp == opp)
        & (stats_df.position == position)
        & (stats_df.started == starter)
        & ((stats_df.salary >= salary - 500) & (stats_df.salary <= salary + 500))
    ][-n_games:]
    if len(player_df) < n_games:
        return pd.DataFrame(
            {
                "Floor": [np.nan],
                "Ceiling": [np.nan],
                "Stochastic": [np.nan],
                "DDProb": [np.nan],
                "TDProb": [np.nan],
            }
        )
    for stat in DK_Stats:
        mean = player_df[stat].mean()
        std = player_df[stat].std()
        stats = np.random.normal(loc=mean, scale=std, size=10000)
        feature_frame[stat] = stats
        #
        opp_mean = opp_df[stat].mean()
        opp_std = opp_df[stat].std()
        opp_stats = np.random.normal(loc=opp_mean, scale=opp_std, size=1000)
        opp_feature_frame[stat] = opp_stats
    feature_frame.fillna(0, inplace=True)
    opp_feature_frame.fillna(0, inplace=True)
    feature_frame = feature_frame.mask(feature_frame.lt(0), 0)
    opp_feature_frame = opp_feature_frame.mask(opp_feature_frame.lt(0), 0)
    ProjectionsFrame = pd.DataFrame({})
    Stochastic = getDKPts(feature_frame)
    opp_Stochastic = getDKPts(opp_feature_frame)
    if Stochastic.DKPts.mean() > opp_Stochastic.DKPts.mean():
        player_edge = True
    else:
        player_edge = False
    if (player_edge == True) & (favorite == True):
        player_weight = 1 - (1 / ((1 - (100 / (np.abs(moneyline) * -1)))))
    elif (player_edge == True) & (favorite == False):
        player_weight = 1 / ((1 - (100 / (np.abs(moneyline) * -1))))
    elif (player_edge == False) & (favorite == True):
        player_weight = 1 / ((1 - (100 / (np.abs(moneyline) * -1))))
    else:
        player_weight = 1 - (1 / ((1 - (100 / (np.abs(moneyline) * -1)))))
    opp_weight = 1 - player_weight
    if salary >= 9000:
        player_weight = 1
        opp_weight = 0
        feature_frame = pd.concat([feature_frame, feature_frame]).reset_index(drop=True)
    else:
        feature_frame = pd.concat([feature_frame, opp_feature_frame]).reset_index(
            drop=True
        )
    ProjectionsFrame["Stochastic"] = (Stochastic.DKPts * player_weight) + (
        opp_Stochastic.DKPts * opp_weight
    )
    Floor = round(ProjectionsFrame.Stochastic.quantile(0.15), 1)
    if Floor < 0:
        Floor = 0
    Ceiling = round(ProjectionsFrame.Stochastic.quantile(0.85), 1)
    Stochastic = round(ProjectionsFrame.Stochastic.quantile(0.55), 1)
    DDProb = len(feature_frame[feature_frame.doubleDouble == 1]) / 20000
    TDProb = len(feature_frame[feature_frame.tripleDouble == 1]) / 20000
    print(name, Stochastic)
    ProjFrame = pd.DataFrame(
        {
            "Floor": [Floor],
            "Ceiling": [Ceiling],
            "Stochastic": [Stochastic],
            "DDProb": [DDProb],
            "TDProb": [TDProb],
        }
    )
    return ProjFrame


def TopDownPrediction(df):
    df["proj_pts"] = df[["proj_pts", "proj_team_score"]].mean(axis=1)
    df["TD_Proj"] = (
        (df.proj_pct_pts * df.proj_pts)
        + (df.proj_pct_fg3m * df.proj_fg3m * 0.5)
        + (df.proj_pct_ast * df.proj_ast * 1.5)
        + (df.proj_pct_reb * df.proj_reb * 1.25)
        + (df.proj_pct_stl * df.proj_stl * 2)
        + (df.proj_pct_blk * df.proj_blk * 2)
        - (df.proj_pct_tov * df.proj_to * 0.5)
    )
    df["TD_Proj2"] = df.proj_pct_dkpts * df.proj_dkpts
    return df

