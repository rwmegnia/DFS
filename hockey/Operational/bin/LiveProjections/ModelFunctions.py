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
from NHL_API_TOOLS import *

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
def rolling_average(df, window=15):
    return df.rolling(min_periods=1, window=window).mean()


def MLPrediction(
    skater_stats_df,
    goalie_stats_df,
    skater_games_df,
    goalie_games_df,
    position_type,
    line,
):
    print(position_type)
    # Determine player stats/opponent stats to be used based on position_type
    if position_type in ["Forward", "Defenseman"]:
        if "line" in NonFeatures:
            NonFeatures.remove("line")
        game_proj_df = skater_games_df
        stats_df = skater_stats_df
        # Goalie Stats
        goalie_stats_df["line"] = 1
        goalie_stats_df.drop("line", axis=1, inplace=True)
        goalie_db_Features = [f"goalie_{c}" for c in OpposingGoalieColumns]
        # Opponent Stats
        opp_stats_df = stats_df.groupby(["opp", "game_date"]).sum()[OpposingTeamColumns]
        opp_stats_df = (
            opp_stats_df.groupby("opp")
            .apply(lambda x: rolling_average(x))
            .add_prefix("opp_")
        )

    else:
        NonFeatures.append("line")
        game_proj_df = goalie_games_df.groupby("full_name", as_index=False).first()
        stats_df = goalie_stats_df
        opp_stats_df = skater_stats_df

    game_proj_df.drop_duplicates(inplace=True)
    # Filter player down to active players for the week
    stats_df = stats_df[stats_df.player_id.isin(game_proj_df.player_id)]

    # Create prediction features frame by taking 15 game running average of player stats
    features = (
        stats_df.groupby(["player_id"])
        .apply(lambda x: rolling_average(x))
        .drop(NonFeatures, axis=1, errors="ignore")
    )

    # Reassign player_id,team to features frame
    features[["player_id", "team"]] = stats_df[["player_id", "team"]]
    if len(features) == 0:
        return

    # Get last row of features frame which is the latest running average for that player
    features = features.groupby(["player_id"], as_index=False).last()

    # Establish opponent/goalie features depending on position type being predicted
    if position_type in ["Forward", "Defenseman"]:
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
        game_proj_df[["player_id", "opp"]].set_index("player_id")
    ).reset_index()
    if position_type in ["Forward", "Defenseman"]:
        features = features.merge(goalie_features, on=["team"], how="left")
        features = features.merge(opp_features, on=["opp"], how="left")
    else:
        features = features.merge(opp_features, on=["opp"], how="left")

    # Setup projection frame
    ProjFrame = game_proj_df[NonFeatures]
    ProjFrame.set_index("player_id", inplace=True)
    ProjFrame = ProjFrame.join(
        features.set_index("player_id").drop(NonFeatures, axis=1, errors="ignore")
    )
    ProjFrame.dropna(inplace=True)
    ProjFrame.drop_duplicates(inplace=True)
    if len(ProjFrame) == 0:
        return ProjFrame

    # Make projections
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
                (ProjFrame.position_type == position)
                & (ProjFrame[f"proj_{stat}_share"] < 0),
                stat,
            ] = 0
    ProjFrame["ML"] = ProjFrame[models.keys()].mean(axis=1)
    ProjFrame.reset_index(inplace=True)
    return ProjFrame


#
def TeamStatsPredictions(team_games_df, team_stats_df):
    # Get 15 game running average of team statistics
    features = (
        team_stats_df.groupby(["team", "opp"])
        .apply(lambda x: rolling_average(x))
        .drop(NonFeatures, axis=1, errors="ignore")
    )
    features[["team", "opp"]] = team_stats_df[["team", "opp"]]
    if len(features) == 0:
        return

    # Get last row of team running average statistics
    features = features.groupby(["team"], as_index=False).last()

    # Get 15 game running average of oppponent statistics and latest average
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
    if "proj_team_score" not in TeamNonFeatures:
        TeamNonFeatures.append("proj_team_score")

    # Setup projection
    ProjFrame = team_games_df[TeamNonFeatures]
    TeamNonFeatures.remove("proj_team_score")
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
    ProjFrame.reset_index(inplace=True)
    ProjFrame["proj_opp_score"] = ProjFrame.team.apply(
        lambda x: ProjFrame[ProjFrame.opp == x].proj_team_score.values[0]
    )
    ProjFrame.loc[
        ProjFrame.proj_team_score < ProjFrame.proj_opp_score, "opp_Winner"
    ] = 1
    ProjFrame.opp_Winner.fillna(0, inplace=True)
    ProjFrame["opp_goalie_saves"] = ProjFrame.proj_shots - ProjFrame.proj_goals
    ProjFrame["opp_goalie_proj"] = (
        (ProjFrame.opp_goalie_saves * 0.7)
        - (ProjFrame.proj_goals * 3.5)
        + (ProjFrame.opp_Winner * 6)
    )
    ProjFrame.loc[ProjFrame.opp_goalie_saves >= 35, "opp_goalie_proj"] += 3
    return ProjFrame


def StochasticPrediction(
    player, position, line, moneyline, stats_df, opp, game_proj_df, position_type
):
    if position_type == "Goalie":
        STATS = DK_Goalie_Stats
        stats_df["line"] = 1
    else:
        STATS = DK_Skater_Stats
    if moneyline > 0:
        favorite = True
    else:
        favorite = False
    name = stats_df[stats_df.player_id == player].full_name.unique()
    n_games = 15
    feature_frame = pd.DataFrame({})
    opp_feature_frame = pd.DataFrame({})
    player_df = stats_df[stats_df.player_id == player][-n_games:]
    opp_df = stats_df[
        (stats_df.opp == opp)
        & (stats_df.position == position)
        & (stats_df.line == line)
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
        opp_stats = np.random.normal(loc=opp_mean, scale=opp_std, size=1000)
        opp_feature_frame[stat] = opp_stats
    feature_frame.fillna(0, inplace=True)
    opp_feature_frame.fillna(0, inplace=True)
    feature_frame = feature_frame.mask(feature_frame.lt(0), 0)
    opp_feature_frame = opp_feature_frame.mask(opp_feature_frame.lt(0), 0)
    ProjectionsFrame = pd.DataFrame({})
    if position_type == "Goalie":
        Stochastic = getGoalieDKPts(feature_frame)
        opp_Stochastic = getGoalieDKPts(opp_feature_frame)
    else:
        Stochastic = getSkaterDKPts(feature_frame)
        opp_Stochastic = getSkaterDKPts(opp_feature_frame)
    if Stochastic.mean() > opp_Stochastic.mean():
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
    ProjectionsFrame["Stochastic"] = (Stochastic * player_weight) + (
        opp_Stochastic * opp_weight
    )
    Floor = round(ProjectionsFrame.Stochastic.quantile(0.15), 1)
    if Floor < 0:
        Floor = 0
    if position_type == "Skater":
        Ceiling = round(ProjectionsFrame.Stochastic.quantile(0.85), 1)
        Stochastic = round(ProjectionsFrame.Stochastic.mean(), 1)
        ShotProb = len(feature_frame[feature_frame.shots >= 5]) / 10000
        BlockProb = len(feature_frame[feature_frame.blocked >= 3]) / 10000
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


def TopDownPrediction(df):
    df["proj_goals"] = df[["proj_goals", "proj_team_score"]].mean(axis=1)
    df["TD_Proj"] = (
        (df.proj_shots_share * df.proj_shots * 1.5)
        + (df.proj_blocked_share * df.proj_blocked * 1.3)
        + (df.proj_goals_share * df.proj_goals * 8.5)
        + (df.proj_assists_share * df.proj_assists * 5)
    )
    df["TD_Proj"] += df.proj_DKPts_share * df.proj_DKPts
    df["TD_Proj"] /= 2
    return df
