#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 22:04:16 2022

@author: robertmegnia


"""
import pandas as pd
import os
from getDKSalaries import getDKSalaries
from reformatNames import reformatNames
from datetime import datetime

from PMM import getPMM

basedir = os.path.dirname(os.path.abspath(__file__))
etcdir = f"{basedir}/../../etc"
datadir = f"{basedir}/../../data"
projdir = f"{datadir}/Projections"
from config.MLModel_config import *
from config.StochasticModel_config import *
from ScrapeStartingLineups import scrapeStartingLineups
from ScrapeBettingOdds import ScrapeBettingOdds
from ModelFunctions import *
#%%
def getMinutes(df):
    df["mins"] = df["min"].apply(lambda x: x.split(":")[0]).astype(float)
    df["seconds"] = (
        df["min"]
        .apply(lambda x: x.split(":")[1] if len(x.split(":")) > 1 else 0)
        .astype(float)
    )
    df["mins"] = df.mins + (df.seconds / 60)
    return df["mins"]
#%%
datadir = f"{basedir}/../../data"
season = 2021
game_date = datetime.now().strftime("%Y-%m-%d")
contestType = "Classic"
SWAP = False
# Get Player Pool for the Day, Starting lineups, and odds
start_df = scrapeStartingLineups()
odds_df = ScrapeBettingOdds()
start_df=start_df.merge(odds_df.drop('spread',axis=1),on=['team_abbreviation'])
# Get Draftkings Salaries for the day
salaries, slate = getDKSalaries(game_date, contestType)
start_df = start_df[start_df.team_abbreviation.isin(salaries.team)]
#%%
# Load in database to pull statistics from and filter out current and future dates
db = pd.read_csv(f"{datadir}/game_logs/PlayerStatsDatabase.csv")
db.player_name.replace("Nene", "Nene Nene", inplace=True)
player_IDs = db.groupby("player_id", as_index=False).last()[
    ["team_abbreviation", "player_name", "nickname", "player_id"]
]
player_IDs = reformatNames(player_IDs)
salaries = salaries.merge(
    player_IDs[["player_id", "player_name"]], on="player_name", how="left"
)
start_df = start_df.merge(
    salaries[["position", "player_id", "Salary","Game Info","ID", "RotoName", "player_name"]],
    on=["Salary", "RotoName"],
    how="left",
)
start_df = start_df[start_df.player_id.isna() == False]
start_df.player_id = start_df.player_id.astype(int)
db["game_date_string"] = db.game_date
db.game_date = pd.to_datetime(db.game_date)
db.sort_values(by="game_date", inplace=True)
stats_df = db[db.game_date < game_date]
stats_df.sort_values(by="game_date", inplace=True)
season = 2021
# Need to merge starting lineups with salary database and then with player_ids database.
#%%
# Execute Stochastic Model Projection
stochastic_proj_df = pd.concat(
    [
        start_df,
        pd.concat(
            [
                a
                for a in start_df.apply(
                    lambda x: StochasticPrediction(
                        x.player_id,
                        x.position,
                        x.starter,
                        x.Salary,
                        x.moneyline,
                        stats_df,
                        x.opp,
                        start_df[start_df.player_id == x.player_id],
                    ),
                    axis=1,
                )
            ]
        ).set_index(start_df.index),
    ],
    axis=1,
)
#%%
# # Read in player database
player_db = pd.read_csv(f"{datadir}/game_logs/PlayerStatsDatabase.csv")
player_db["game_date_string"] = player_db.game_date
player_db.game_date = pd.to_datetime(player_db.game_date)
player_db.sort_values(by="game_date", inplace=True)
player_stats_df = player_db[player_db.game_date < game_date]
start_df.rename({'Salary':'salary'},axis=1,inplace=True)
ml_proj_df = MLPrediction(player_stats_df, start_df.reset_index())
#%%
# Predict Top Down
team_db = pd.read_csv(f"{datadir}/game_logs/TeamStatsDatabase.csv")
team_db["game_date_string"] = team_db.game_date
team_db.game_date = pd.to_datetime(team_db.game_date)
team_db.sort_values(by="game_date", inplace=True)
team_stats_df = team_db[team_db.game_date < game_date]
start_df.rename({'team':'team_abbreviation'},axis=1,inplace=True)
team_games_df = (
    start_df.groupby("team_abbreviation")
    .first()[
        ["opp", "game_date", "proj_team_score",'total_line']
    ]
    .reset_index()
)
team_proj_df = TeamStatsPredictions(team_games_df, team_stats_df)
#
ml_proj_df = ml_proj_df.merge(
    team_proj_df.reset_index()[
        [
            "team_abbreviation",
            "proj_pts",
            "proj_fg3m",
            "proj_ast",
            "proj_reb",
            "proj_blk",
            "proj_stl",
            "proj_to",
            "proj_dkpts",
        ]
    ],
    on="team_abbreviation",
    how="left",
)
ml_proj_df = TopDownPrediction(ml_proj_df)
ml_proj_df = stochastic_proj_df.merge(
    ml_proj_df[["player_id", "BR", "EN","NN", "RF", "GB", "Tweedie", "TD_Proj","TD_Proj2"]],
    on="player_id",
    how="left",
)
final_proj_df = getPMM(ml_proj_df, game_date)

#%%
final_proj_df["Projection"] = final_proj_df[
    ["Stochastic", "ML", "RG_projection", "PMM", "TD_Proj"]
].mean(axis=1)
final_proj_df.loc[final_proj_df.RG_projection==0,'Projection']=0
final_proj_df["game_date"] = game_date
projdir = f"{datadir}/Projections/RealTime/{season}/{contestType}"
final_proj_df = final_proj_df[final_proj_df.RG_projection.isna() == False]
try:
    if SWAP==False:
        final_proj_df.to_csv(
            f"{projdir}/{game_date}/{slate}_Projections.csv", index=False,
        )
    else:
        final_proj_df.to_csv(
            f"{projdir}/{game_date}/{slate}_Projections_LATESWAP.csv", index=False,
        )
except FileNotFoundError:
    os.mkdir(f"{projdir}/{game_date}")
    final_proj_df.to_csv(
        f"{projdir}/{game_date}/{slate}_Projections.csv", index=False,
    )
