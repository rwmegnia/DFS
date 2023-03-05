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
import scipy
import pandas as pd
import os
from MLB_API_TOOLS import *

basedir = os.path.dirname(os.path.abspath(__file__))
etcdir = f"{basedir}/../../etc"
datadir = f"{basedir}/../../data"
projdir = f"{datadir}/Projections"
from config.MLModel_config import *
from config.StochasticModel_config import *

os.chdir("../")
from Models.ModelDict import *

os.chdir(f"{basedir}/../")

#%%
def rolling_average(df, window):
    return df.rolling(min_periods=1, window=window).mean()


def MLBatterPrediction(
    batter_stats_df,
    pitcher_stats_df,
    batter_games_df,
    bat_hand,
    pitch_hand,
):
    N_games=20
    # Determine player stats/opponent stats to be used based on position_type
    position_type = "Batter"
    NonFeatures = BatterNonFeatures
    game_proj_df = batter_games_df
    stats_df = batter_stats_df
    # pitcher Stats
    pitcher_stats_df["line"] = 1
    pitcher_stats_df.drop("line", axis=1, inplace=True)
    pitcher_db_Features = [f"pitcher_{c}" for c in OpposingPitcherColumns]
    # Opponent Stats
    opp_stats_df = stats_df.groupby(["opp", "game_date"]).sum()[
        OpposingTeamColumns
    ]
    opp_stats_df = (
        opp_stats_df.groupby(
            "opp",
        )
        .apply(lambda x: rolling_average(x,N_games))
        .add_prefix("opp_")
    )
    opp_stats_df.reset_index(inplace=True)
    game_proj_df.drop_duplicates(inplace=True)
    # Filter player down to active players for the week
    stats_df = stats_df[stats_df.player_id.isin(game_proj_df.player_id)]

    # Create prediction features frame by taking 15 game running average of player stats
    features = (
        stats_df.groupby(["player_id"])
        .apply(lambda x: rolling_average(x,N_games))
        .drop(NonFeatures, axis=1, errors="ignore")
    )

    # Reassign player_id,team to features frame
    features[["player_id", "team", "Salary"]] = stats_df[
        ["player_id", "team", "Salary"]
    ]
    if len(features) == 0:
        return

    # Get last row of features frame which is the latest running average for that player
    features = features.groupby(["player_id"], as_index=False).last()
    features.drop("Salary", axis=1, inplace=True)
    features = features.merge(
        game_proj_df[["player_id", "Salary"]], on="player_id", how="left"
    )
    # Establish opponent/pitcher features depending on position type being predicted
    features.fillna(0, inplace=True)
    pitcher_features = pitcher_stats_df.groupby("player_id", as_index=False)[
        [
            f"pitcher_{c}"
            for c in OpposingPitcherColumns
            if f"pitcher_{c}" in pitcher_stats_df.columns
        ]
    ].apply(lambda x: rolling_average(x, window=7))
    pitcher_features[["player_id"]] = pitcher_stats_df[["player_id"]]
    pitcher_features = pitcher_features.groupby(
        "player_id", as_index=False
    ).last()
    pitcher_features.rename(
        {"player_id": "opp_pitcher_id"}, axis=1, inplace=True
    )
    opp_features = opp_stats_df.groupby("opp").last()
    # Merge Offense and Defense features
    features.set_index("player_id", inplace=True)
    features = features.join(
        game_proj_df[["player_id", "opp"]].set_index("player_id")
    ).reset_index()
    features = features.merge(opp_features, on=["opp"], how="left")
    # Setup projection frame
    ProjFrame = game_proj_df[
        [c for c in NonFeatures if c in game_proj_df.columns]
    ]
    ProjFrame = ProjFrame.merge(
        pitcher_features, on=["opp_pitcher_id"], how="left"
    )
    ProjFrame.set_index("player_id", inplace=True)
    ProjFrame = ProjFrame.join(
        features.set_index("player_id").drop(
            NonFeatures, axis=1, errors="ignore"
        )
    )
    ProjFrame.dropna(inplace=True)
    ProjFrame.drop_duplicates(inplace=True)
    if len(ProjFrame) == 0:
        return ProjFrame

    # Make projections
    models = ModelDict[f"{position_type}_{bat_hand}vs{pitch_hand}"]
    for method, model in models.items():
        ProjFrame[method] = model.predict(ProjFrame[batter_features])
        ProjFrame.loc[
            ProjFrame[method] < 0,
            method,
        ] = 0
    ProjFrame["ML"] = ProjFrame[models.keys()].mean(axis=1)
    ProjFrame.reset_index(inplace=True)
    return ProjFrame


def MLPitcherPrediction(
    batter_stats_df,
    pitcher_stats_df,
    pitcher_games_df,
    pitch_hand,
):
    N_games=7
    position_type = "Pitcher"
    NonFeatures = PitcherNonFeatures
    game_proj_df = pitcher_games_df.groupby("full_name", as_index=False).first()
    stats_df = pitcher_stats_df
    opp_stats_df = batter_stats_df
    game_proj_df.drop_duplicates(inplace=True)
    # Filter player down to active players for the week
    stats_df = stats_df[stats_df.player_id.isin(game_proj_df.player_id)]
    stats_df.replace("-.--", np.nan, inplace=True)
    # Create prediction features frame by taking 15 game running average of player stats
    features = (
        stats_df.groupby(["player_id"])
        .apply(lambda x: rolling_average(x, window=N_games))
        .drop(NonFeatures, axis=1, errors="ignore")
    )

    # Reassign player_id,team to features frame
    features[["player_id", "team", "Salary"]] = stats_df[
        ["player_id", "team", "Salary"]
    ]
    if len(features) == 0:
        return

    # Get last row of features frame which is the latest running average for that player
    features = features.groupby(["player_id"]).last().reset_index()
    features.drop("Salary", axis=1, inplace=True)
    features = features.merge(
        game_proj_df[["player_id", "Salary"]], on="player_id", how="left"
    )
    opp_features = opp_stats_df.groupby(["opp", "game_date"]).sum()[
        OpposingTeamColumns
    ]
    opp_features = (
        opp_features.groupby("opp")
        .apply(lambda x: rolling_average(x,N_games))
        .add_prefix("opp_")
    )
    opp_features = opp_features.groupby("opp").last().reset_index()
    # Merge Offense and Defense features
    features.set_index("player_id", inplace=True)
    features = features.join(
        game_proj_df[["player_id", "opp"]].set_index("player_id")
    ).reset_index()
    features = features.merge(opp_features, on=["opp"], how="left")

    # Setup projection frame
    ProjFrame = game_proj_df[
        [c for c in NonFeatures if c in game_proj_df.columns]
    ]
    ProjFrame.set_index("player_id", inplace=True)
    ProjFrame = ProjFrame.join(
        features.set_index("player_id").drop(
            NonFeatures, axis=1, errors="ignore"
        )
    )
    ProjFrame.dropna(inplace=True)
    ProjFrame.drop_duplicates(inplace=True)
    if len(ProjFrame) == 0:
        return ProjFrame

    # Make projections
    models = ModelDict[f"{position_type}_{pitch_hand}"]
    for method, model in models.items():
        ProjFrame[method] = model.predict(ProjFrame[PitcherFeaturesOrder])
        ProjFrame.loc[
            ProjFrame[method] < 0,
            method,
        ] = 0
    ProjFrame["ML"] = ProjFrame[models.keys()].mean(axis=1)
    ProjFrame.reset_index(inplace=True)
    return ProjFrame


def BattersStochastic(players, stats_df):
    # Make sure switch hitters have bat hand opposite that of opposing pitcher hand
    N_games=30
    players.loc[
        (players.handedness == "S") & (players.opp_pitcher_hand == "R"),
        "handedness",
    ] = "L"
    players.loc[
        (players.handedness == "S") & (players.opp_pitcher_hand == "L"),
        "handedness",
    ] = "R"
    players["splits"] = "vs_" + players.opp_pitcher_hand + "HP"
    players.ownership_proj.fillna(0, inplace=True)
    # Filter Players to Batters and set index to ID and splits to join with stats frame
    batters = players[~players.position.isin(['P','RP','SP'])] 
    batters = batters[batters.RotoPosition.isna() == False]
    batters.set_index(["player_id", "splits"], inplace=True)
    stats_df.set_index(["player_id", "splits"], inplace=True)
    batters = batters.join(stats_df[DK_Batter_Stats], lsuffix="_a")

    # Get each batters last 40 games
    batters = batters.groupby("ID").tail(N_games+5)

    # Scale batter stats frame to z scores to get opponent influence
    # Get batter stats from last 40 games
    scaled_batters = (
        stats_df[stats_df.game_date > "2022-01-1"]
        .groupby(["player_id", "splits"])
        .tail(N_games+5)
    )
    scaled_batters.sort_index(inplace=True)

    # Get batters average/standard dev stats over last 20 games
    scaled_averages = (
        scaled_batters.groupby(["player_id", "splits"], as_index=False)
        .rolling(window=N_games, min_periods=5)
        .mean()[DK_Batter_Stats]
        .groupby(["player_id", "splits"])
        .tail(N_games)
    )

    scaled_stds = (
        scaled_batters.groupby(["player_id", "splits"], as_index=False)
        .rolling(window=N_games, min_periods=5)
        .std()[DK_Batter_Stats]
        .groupby(["player_id", "splits"])
        .tail(N_games)
    )

    scaled_batters = scaled_batters.groupby(["player_id", "splits"]).tail(N_games)

    # Get batters Z scores over last 20 games
    scaled_batters[DK_Batter_Stats] = (
        (scaled_batters[DK_Batter_Stats] - scaled_averages[DK_Batter_Stats])
        / scaled_stds[DK_Batter_Stats]
    ).values
    #scaled_batters.fillna(0,inplace=True)
    # Find out what opposing pitcher allowed to previous batters over last
    # 7 starts
    opp_pitcher_stats = (
        scaled_batters.groupby(["opp_pitcher_id", "game_date"])
        .mean()
        .groupby(["opp_pitcher_id"])
        .tail(7)[DK_Batter_Stats]
        .reset_index()
    )
    
    # Take averages of previous data frame
    opp_pitcher_stats = (
        opp_pitcher_stats[
            opp_pitcher_stats.opp_pitcher_id.isin(players.opp_pitcher_id)
        ]
        .groupby("opp_pitcher_id")
        .mean()
    )
    opp_pitcher_stats.fillna(0,inplace=True)
    #%%
    averages = batters.groupby("ID").mean()
    averages.opp_pitcher_id = averages.opp_pitcher_id.astype(int)
    averages = averages.reset_index().set_index("opp_pitcher_id")
    stds = batters.groupby(["ID","opp_pitcher_id"]).std()
    stds = stds.reset_index().set_index('opp_pitcher_id')
    
    # Convert average Z Scores to quantiles
    #shape=(averages[DK_Batter_Stats]/stds[DK_Batter_Stats])**2
    #shape=opp_pitcher_stats.join(shape,lsuffix='_shape')
    # quantiles = 1 - scipy.stats.gamma.sf(shape[[c for c in shape.columns if '_shape' not in c]],
    #                                      shape[[c for c in shape.columns if '_shape' in c]])
    quantiles = 1 - scipy.stats.norm.sf(opp_pitcher_stats)
    quantiles = pd.DataFrame(quantiles, columns=DK_Batter_Stats).set_index(
        opp_pitcher_stats.index
    )
    quantiles = averages.join(quantiles[DK_Batter_Stats], lsuffix="_quant")[
        DK_Batter_Stats
    ]
    quantiles.fillna(0.5, inplace=True)
    averages.sort_index(inplace=True)
    stds.sort_index(inplace=True)
    quantiles.sort_index(inplace=True)
    # averages.fillna(0,inplace=True)
    sims = np.random.normal(
        averages[DK_Batter_Stats],
        stds[DK_Batter_Stats],
        size=(10000, len(averages), len(DK_Batter_Stats)),
    )
    # sims = np.random.gamma(
    #     (averages[DK_Batter_Stats]/stds[DK_Batter_Stats])**2,
    #     (stds[DK_Batter_Stats]**2)/averages[DK_Batter_Stats],
    #     size=(10000, len(averages), len(DK_Batter_Stats)),
    # )
    # sims[sims < 0] = 0
    #
    batters = players[~players.position.isin(['P','RP','SP'])] 
    batters = batters[batters.RotoPosition.isna() == False]

    # Get Floor Stats
    low = pd.DataFrame(
        np.quantile(sims, 0.1, axis=0), columns=DK_Batter_Stats
    ).set_index(averages.ID)
    low.rename({"DKPts": "Floor1"}, axis=1, inplace=True)
    low["Floor"] = getDKPts(low, "batters")
    low["Floor"] = low[["Floor", "Floor1"]].mean(axis=1)

    # Get Ceiling Stats
    high = pd.DataFrame(
        np.quantile(sims, 0.9, axis=0), columns=DK_Batter_Stats
    ).set_index(averages.ID)
    high["Ceiling"] = getDKPts(high, "batters")
    high["Ceiling"] = high[["Ceiling", "DKPts"]].mean(axis=1)

    # Get Stochastic Stats/Projections
    median = pd.concat(
        [
            pd.DataFrame(
                np.diag(
                    pd.DataFrame(
                        sims[:, i, :], columns=DK_Batter_Stats
                    ).quantile(quantiles.values[i, :])
                ).reshape(1, -1),
                columns=DK_Batter_Stats,
            )
            for i in range(0, len(batters))
        ]
    ).set_index(averages.ID)
    median.rename({"DKPts": "Stochastic1"}, axis=1, inplace=True)
    median["Stochastic"] = getDKPts(median, "batters")
    median["Stochastic"] = median[["Stochastic", "Stochastic1"]].mean(axis=1)
    batters.set_index("ID", inplace=True)
    batters = batters.join(low["Floor"].round(1))
    batters = batters.join(high["Ceiling"].round(1))
    batters = batters.join(median["Stochastic"].round(1))
    batters.reset_index(inplace=True)
    return batters


def PitchersStochastic(players, stats_df):
    N_games=7
    # Filter Players to Pitchers and set index to PlayerID
    pitchers = players[players.position_type == "Pitcher"]
    pitchers = pitchers[pitchers.RG_projection.isna() == False]
    pitchers.set_index(["player_id"], inplace=True)
    stats_df.set_index(["player_id"], inplace=True)
    pitchers = pitchers.join(stats_df[DK_Pitcher_Stats], lsuffix="_a")

    # Get each Pitchers last 14 games
    pitchers = pitchers.groupby("ID").tail(14)

    # Scale pitcher stats frame to z scores to get opponent influence
    # Get pitcher stats form last 14 games
    scaled_pitchers = (
        stats_df[stats_df.game_date > "2021-01-1"]
        .groupby(["player_id"])
        .tail(N_games*2)
    )
    scaled_pitchers.sort_index(inplace=True)
    # Get batters average/standard dev stats over last 20 games
    scaled_averages = (
        scaled_pitchers.groupby(["player_id"], as_index=False)
        .rolling(window=N_games, min_periods=3)
        .mean()[DK_Pitcher_Stats]
        .groupby(["player_id"])
        .tail(N_games)
    )
    scaled_stds = (
        scaled_pitchers.groupby(["player_id"], as_index=False)
        .rolling(window=N_games, min_periods=3)
        .std()[DK_Pitcher_Stats]
        .groupby(["player_id"])
        .tail(N_games)
    )
    scaled_pitchers = scaled_pitchers.groupby(["player_id"]).tail(N_games)
    # Get pitchers Z scores over last 7 games
    scaled_pitchers[DK_Pitcher_Stats] = (
        (scaled_pitchers[DK_Pitcher_Stats] - scaled_averages[DK_Pitcher_Stats])
        / scaled_stds[DK_Pitcher_Stats]
    ).values
    opp_stats = (
        scaled_pitchers.groupby(["opp", "game_date"])
        .mean()
        .groupby(["opp"])
        .tail(N_games)[DK_Pitcher_Stats]
        .reset_index()
    )
    opp_stats = opp_stats[opp_stats.opp.isin(players.opp)].groupby("opp").mean()
    quantiles = 1 - scipy.stats.norm.sf(opp_stats)
    # quantiles = 1 - scipy.stats.gamma.sf(opp_stats)
    quantiles = pd.DataFrame(quantiles, columns=DK_Pitcher_Stats).set_index(
        opp_stats.index
    )
    averages = pitchers.groupby(["ID", "opp"]).mean()
    averages = averages.reset_index().set_index("opp")
    stds = pitchers.groupby(["ID","opp"]).std()
    stds = stds.reset_index().set_index("opp")
    quantiles = averages.join(quantiles[DK_Pitcher_Stats], lsuffix="_quant")[
        DK_Pitcher_Stats
    ]
    quantiles.fillna(0.5, inplace=True)
    averages.sort_index(inplace=True)
    stds.sort_index(inplace=True)
    sims = np.random.normal(
        averages[DK_Pitcher_Stats],
        stds[DK_Pitcher_Stats],
        size=(10000, len(averages), len(DK_Pitcher_Stats)),
    )
    # sims = np.random.normal(
    #     (averages[DK_Batter_Stats]/stds[DK_Batter_Stats])**2,
    #     (stds[DK_Batter_Stats]**2)/averages[DK_Batter_Stats],
    #     size=(10000, len(averages), len(DK_Batter_Stats)),
    # )
    sims[sims < 0] = 0
    #
    pitchers = players[players.position_type == "Pitcher"]
    pitchers = pitchers[pitchers.RG_projection.isna() == False]
    low = pd.DataFrame(
        np.quantile(sims, 0.1, axis=0), columns=DK_Pitcher_Stats
    ).set_index(averages.ID)
    low.rename({"DKPts": "Floor1"}, axis=1, inplace=True)
    low["Floor"] = getDKPts(low, "pitchers")
    low["Floor"] = low[["Floor", "Floor1"]].mean(axis=1)
    high = pd.DataFrame(
        np.quantile(sims, 0.9, axis=0), columns=DK_Pitcher_Stats
    ).set_index(averages.ID)
    high["Ceiling"] = getDKPts(high, "pitchers")
    high["Ceiling"] = high[["Ceiling", "DKPts"]].mean(axis=1)
    median = pd.concat(
        [
            pd.DataFrame(
                np.diag(
                    pd.DataFrame(
                        sims[:, i, :], columns=DK_Pitcher_Stats
                    ).quantile(quantiles.values[i, :])
                ).reshape(1, -1),
                columns=DK_Pitcher_Stats,
            )
            for i in range(0, len(pitchers))
        ]
    ).set_index(averages.ID)
    median.rename({"DKPts": "Stochastic1"}, axis=1, inplace=True)
    median["Stochastic"] = getDKPts(median, "pitchers")
    median["Stochastic"] = median[["Stochastic", "Stochastic1"]].mean(axis=1)
    pitchers.set_index("ID", inplace=True)
    pitchers = pitchers.join(low["Floor"].round(1))
    pitchers = pitchers.join(high["Ceiling"].round(1))
    pitchers = pitchers.join(median["Stochastic"].round(1))
    pitchers.reset_index(inplace=True)
    return pitchers
