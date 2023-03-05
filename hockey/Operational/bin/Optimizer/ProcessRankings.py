#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 20:33:41 2022

@author: robertmegnia
"""
import pickle
import os
from datetime import datetime
import numpy as np
import pandas as pd
from pydfs_lineup_optimizer.tz import get_timezone
from pytz import timezone

basedir = os.path.dirname(os.path.abspath(__file__))
etcdir = f"{basedir}/../../etc"
datadir = f"{basedir}/../../data"


def processRankings(proj, game_date, powerplay=True):
    # Rank players by different metrics
    proj["MyRank"] = proj.Stochastic.rank(ascending=False, method="min")
    proj["RGRank"] = proj.RG_projection.rank(ascending=False, method="min")
    proj["ShotRank"] = proj.ShotProb.rank(ascending=False, method="min")
    proj["MLRank"] = proj.ML.rank(ascending=False, method="min")
    proj["PMMRank"] = proj.PMM.rank(ascending=False, method="min")
    proj["TDRank"] = proj.TD_Proj.rank(ascending=False, method="min")
    pos_frames = []
    for position_type in ["Forward", "Defenseman"]:
        pos_df = proj[proj.position_type == position_type]
        pos_df["ConsRank"] = pos_df[
            ["MyRank", "ShotRank", "RGRank", "PMMRank", "MLRank", "TDRank"]
        ].mean(axis=1)
        for method in ["RF", "GB"]:
            model = pickle.load(
                open(
                    f"{etcdir}/model_pickles/{position_type}_{method}_DKPts_Rank_model.pkl",
                    "rb",
                )
            )
            pos_df = pos_df[pos_df.ConsRank.isna() == False]
            pos_df[method] = model.predict(pos_df.ConsRank[:, np.newaxis])
        pos_df["ScaledProj"] = pos_df[["RF", "GB"]].mean(axis=1)
        pos_frames.append(pos_df)
    goalie_df = proj[proj.position == "G"]
    goalie_df["ConsRank"] = goalie_df[["MyRank", "RGRank", "PMMRank", "MLRank"]].mean(
        axis=1
    )
    pos_frames.append(goalie_df)
    proj = pd.concat(pos_frames)
    proj["Projection"] = proj[["Stochastic", "RG_projection", "PMM", "TD_Proj"]].mean(
        axis=1
    )
    proj.loc[proj.position == "G", "ScaledProj"] = proj.loc[
        proj.position == "G", "RG_projection"
    ]
    proj.game_location.replace("@", "away", inplace=True)
    proj.game_location.replace("VS", "home", inplace=True)
    proj.loc[proj.game_location == "away", "away_team"] = proj.loc[
        proj.game_location == "away", "team"
    ]
    proj.loc[proj.game_location == "away", "home_team"] = proj.loc[
        proj.game_location == "away", "opp"
    ]
    proj.loc[proj.game_location == "home", "home_team"] = proj.loc[
        proj.game_location == "home", "team"
    ]
    proj.loc[proj.game_location == "home", "away_team"] = proj.loc[
        proj.game_location == "home", "opp"
    ]
    try:
        proj["game_date"] = proj["game_date"].apply(
            lambda x: datetime.strptime(x, "%Y-%m-%d").replace(
                tzinfo=timezone(get_timezone())
            )
        )
    except ValueError:
        proj.game_date = pd.to_datetime(proj.game_date)
    proj.fillna(0, inplace=True)
    if powerplay == True:
        proj = proj[(proj.powerplay == True) | (proj.line <= 3)]
    return proj
