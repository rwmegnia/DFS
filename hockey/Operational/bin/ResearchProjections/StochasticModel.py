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
import nflfastpy as nfl
from scipy.stats import norm
import requests
import os

basedir = os.path.dirname(os.path.abspath(__file__))
etcdir = f"{basedir}/../../etc"
datadir = f"{basedir}/../../data"
projdir = f"{datadir}/Projections"
#%%
from config.Stochastic_model_config import *
from getDKPts import *

os.chdir(f"{basedir}/../")


def rolling_average(df, window=8):
    return df.rolling(min_periods=1, window=window).mean()


def StochasticPrediction(player, stats_df, opp, weekly_proj_df, stat_type):
    if stat_type == "Goalie":
        STATS = DK_Goalie_Stats
    else:
        STATS = DK_Skater_Stats
    stats_df.sort_values(by="game_date", inplace=True)
    name = stats_df[stats_df.player_id == player].full_name.unique()
    n_games = 10
    feature_frame = pd.DataFrame({})
    player_df = stats_df[stats_df.player_id == player][-n_games:]
    if len(player_df) == 0:
        return pd.DataFrame(
            {"Floor": [np.nan], "Ceiling": [np.nan], "Stochastic": [np.nan]}
        )
    for stat in STATS:
        mean = player_df[stat].mean()
        std = player_df[stat].std()
        stats = np.random.normal(loc=mean, scale=std, size=10000)
        feature_frame[stat] = stats
    feature_frame.fillna(0, inplace=True)
    feature_frame = feature_frame.mask(feature_frame.lt(0), 0)
    ProjectionsFrame = pd.DataFrame({})
    if stat_type == "Goalie":
        ProjectionsFrame["Stochastic"] = getGoalieDKPts(feature_frame)
    else:
        ProjectionsFrame["Stochastic"] = getSkaterDKPts(feature_frame)
    Floor = round(ProjectionsFrame.Stochastic.quantile(0.15), 1)
    if Floor < 0:
        Floor = 0
    Ceiling = round(ProjectionsFrame.Stochastic.quantile(0.85), 1)
    Stochastic = round(ProjectionsFrame.Stochastic.mean(), 1)
    print(name, Stochastic)
    ProjFrame = pd.DataFrame(
        {"Floor": [Floor], "Ceiling": [Ceiling], "Stochastic": [Stochastic],}
    )
    return ProjFrame


#%%
# Build Stochastic Projections Database
datadir = f"{basedir}/../../data"
for stat_type in ["Skater"]:
    db = pd.read_csv(f"{datadir}/game_logs/{stat_type}StatsDatabase.csv")
    db["game_date_string"] = db.game_date
    db.game_date = pd.to_datetime(db.game_date)
    db.sort_values(by="game_date", inplace=True)
    for game_date in db.game_date_string.unique():
        print(game_date)
        proj_frames = []
        stats_df = db[db.game_date < game_date]
        stats_df.sort_values(by="game_date", inplace=True)
        games_df = db[db.game_date == game_date]
        season = games_df.season.unique()[0]
        proj_df = pd.concat(
            [
                games_df,
                pd.concat(
                    [
                        a
                        for a in games_df.apply(
                            lambda x: StochasticPrediction(
                                x.player_id,
                                stats_df,
                                x.opp,
                                games_df[games_df.player_id == x.player_id],
                                stat_type,
                            ),
                            axis=1,
                        )
                    ]
                ).set_index(games_df.index),
            ],
            axis=1,
        )
        proj_frames.append(proj_df)
        if len(proj_frames) == 0:
            continue
        projdir = f"{datadir}/Projections"
        proj_df.to_csv(
            f"{projdir}/{int(season)}/Stochastic/{game_date}_{stat_type}StochasticProjections.csv",
            index=False,
        )
