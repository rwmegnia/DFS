#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 22:04:16 2022

@author: robertmegnia


"""
import pandas as pd
import os
import sys
from NHL_API_TOOLS import *
from getDKSalaries import getDKSalaries
from getPlayerPool import getPlayerPool
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
# from ProcessSlateStats import processSlateStats

START_DF_COLS = [
    "RotoName",
    "RotoPosition",
    "RG_projection",
    "ownership_proj",
    "team",
    "line",
    "powerplay",
    "O/U",
    "spread",
    "moneyline",
    "proj_team_score",
]

SALARY_DF_COLS = ["full_name", "team", "position", "Roster Position", "Salary", "ID"]
#%%
datadir = f"{basedir}/../../data"
season = 2022
contestType = "Classic"

# Get Player Pool for the Day, Starting lineups, and odds
players = getPlayerPool(season)
start_df = scrapeStartingLineups()
odds_df = ScrapeBettingOdds()
start_df = start_df.merge(odds_df, on="team", how="left")
# Alert if players were not merged from starting lineups
print(
    " The following players were not merged from the roto grinders starting\
      lineups..."
)
missing_players = start_df[~start_df.RotoName.isin(players.RotoName)][
    ["RotoName", "RotoPosition", "team"]
]
print(missing_players)

# Filter players to those in  starting lineups
players = players[players.RotoName.isin(start_df.RotoName)]
players = players.merge(start_df[START_DF_COLS], on=["RotoName", "team"], how="left")
game_date = players.game_date.unique()[0]
print(game_date)

# Get Draftkings Salaries for the day
salaries = getDKSalaries(game_date, contestType)

# Merge player pool with salaries
players = players.drop("position", axis=1).merge(
    salaries[SALARY_DF_COLS], on=["full_name", "team"], how="left"
)

# Alert if players not merged from salaries file
missing_players = players[players.Salary.isna() == True][
    ["RotoName", "RotoPosition", "team"]
]
print(" The following players were not merged from the DraftKings Salaries File...")
print(missing_players)
players["season"] = season
#%%
# Perform Skaters and Goalie Stochastic Projections individually
stochastic_proj_frames = []
if contestType=='Showdown':
    CPT=players[players['Roster Position']=='CPT']
    players = players[players['Roster Position']=='FLEX']
for position_type in ["Goalie", "Skater"]:
    # Load in database to pull statistics from and filter out current and future dates
    # 
    db = pd.read_csv(f"{datadir}/game_logs/{position_type}StatsDatabase.csv")
    db["game_date_string"] = db.game_date
    db.game_date = pd.to_datetime(db.game_date)
    db.sort_values(by="game_date", inplace=True)
    stats_df = db[db.game_date < game_date]
    stats_df.sort_values(by="game_date", inplace=True)
    season = players.season.unique()[0]
    # Execute Stochastic Model Projection
    if position_type=='Skater':
        proj_df=skatersStochastic(players,stats_df)
    else:
        proj_df = pd.concat(
            [
                players,
                pd.concat(
                    [
                        a
                        for a in players.apply(
                            lambda x: StochasticPrediction(
                                x.player_id,
                                x.position,
                                x.Salary,
                                x.line,
                                x.moneyline,
                                stats_df,
                                x.opp,
                                players[players.player_id == x.player_id],
                                position_type,
                                contestType,
                            ),
                            axis=1,
                        )
                    ]
                ).set_index(players.index),
            ],
            axis=1,
        )
    if position_type == "Skater":
        proj_df = proj_df[proj_df.position != "G"]
    else:
        proj_df = proj_df[proj_df.position == "G"]
    stochastic_proj_frames.append(proj_df)
    if len(stochastic_proj_frames) == 0:
        continue
stochastic_proj_df = pd.concat(stochastic_proj_frames)
if contestType=='Showdown':
    CPT = CPT.merge(stochastic_proj_df[['player_id','Ceiling','Stochastic',
                                        'HDSC','ShotProb','BlockProb']],on='player_id',how='left')
    stochastic_proj_df=pd.concat([CPT,stochastic_proj_df])
#%%
# Make ML Predictions for Forwards, Defenseman, and Goalie
ml_proj_frames = []
# Read in goalie/skater databases
skater_db = pd.read_csv(f"{datadir}/game_logs/SkaterStatsDatabase.csv")
goalie_db = pd.read_csv(f"{datadir}/game_logs/GoalieStatsDatabase.csv")
skater_db["game_date_string"] = skater_db.game_date
goalie_db["game_date_string"] = goalie_db.game_date
skater_db.game_date = pd.to_datetime(skater_db.game_date)
goalie_db.game_date = pd.to_datetime(goalie_db.game_date)
skater_db.sort_values(by="game_date", inplace=True)
goalie_db.sort_values(by="game_date", inplace=True)
goalie_db["line"] = 1
# Make sure to predict goalies first since goalie database gets altered for skaters
for position_type in ["Goalie", "Forward", "Defenseman"]:
    # Determine number of lines based on position type
    if position_type == "Forward":
        linemax = 4
    elif position_type == "Defenseman":
        linemax = 3
    else:
        linemax = 1
    for line in range(1, linemax + 1):
        # Filter databases to position type and line
        if position_type in ["Forward", "Defenseman"]:
            position_type_db = skater_db[
                (skater_db.position_type == position_type) & (skater_db.line == line)
            ]
            skater_games_df = players[
                (players.position_type == position_type) & (players.line == line)
            ]
            goalie_games_df = players[players.position == "G"]

            goalie_db["team"] = goalie_db["opp"]
            try:
                goalie_db = goalie_db[NonFeatures + OpposingGoalieColumns]
                goalie_db.rename(
                    dict([(c, f"goalie_{c}") for c in OpposingGoalieColumns]),
                    axis=1,
                    inplace=True,
                )
            except KeyError:
                pass
        else:
            position_type_db = skater_db
            skater_games_df = players[players.position != "G"]
            goalie_games_df = players[players.position == "G"]
        skater_stats_df = position_type_db[position_type_db.game_date < game_date]
        if len(skater_stats_df) == 0:
            continue
        goalie_stats_df = goalie_db[goalie_db.game_date < game_date]
        ml_proj_df = MLPrediction(
            skater_stats_df,
            goalie_stats_df,
            skater_games_df,
            goalie_games_df,
            position_type,
            line,
        )
        if ml_proj_df is None:
            continue
        ml_proj_frames.append(ml_proj_df)
ml_proj_df = pd.concat(ml_proj_frames)
#%%
# Predict Top Down
team_db = pd.read_csv(f"{datadir}/game_logs/TeamStatsDatabase.csv")
team_db["game_date_string"] = team_db.game_date
team_db.game_date = pd.to_datetime(team_db.game_date)
team_db.sort_values(by="game_date", inplace=True)
team_stats_df = team_db[team_db.game_date < game_date]
team_games_df = (
    players.groupby("team")
    .first()[
        ["season", "game_location", "opp", "game_date", "game_id", "proj_team_score"]
    ]
    .reset_index()
)
team_proj_df = TeamStatsPredictions(team_games_df, team_stats_df)
#
ml_proj_df = ml_proj_df.merge(
    team_proj_df.reset_index()[
        [
            "team",
            "proj_goals",
            "proj_team_score",
            "proj_assists",
            "proj_shots",
            "proj_blocked",
            "proj_DKPts",
            "opp_goalie_proj",
        ]
    ],
    on="team",
    how="left",
)
ml_proj_df = TopDownPrediction(ml_proj_df)
ml_proj_df.loc[ml_proj_df.position == "G", "TD_Proj"] = ml_proj_df.loc[
    ml_proj_df.position == "G"
].apply(
    lambda x: ml_proj_df[ml_proj_df.opp == x.team]["opp_goalie_proj"].values[0], axis=1
)
ml_proj_df.drop("opp_HDSC", axis=1, inplace=True)
ml_proj_df = stochastic_proj_df.merge(
    ml_proj_df[
        ["player_id", "BR", "EN", "NN", "RF", "GB", "Tweedie", "TD_Proj", "team_HDSC"]
    ],
    on="player_id",
    how="left",
)
final_proj_df = getPMM(ml_proj_df, game_date)

#%%
final_proj_df["Projection"] = final_proj_df[
    ["Stochastic", "ML", "RG_projection", "PMM", "TD_Proj"]
].mean(axis=1)
final_proj_df["game_date"] = game_date
projdir = f"{datadir}/Projections/RealTime/{season}/{contestType}"
final_proj_df = final_proj_df[final_proj_df.RG_projection.isna() == False]
try:
    final_proj_df.to_csv(
        f"{projdir}/{game_date}/{game_date}_Projections.csv", index=False,
    )
except:
    os.mkdir(f"{projdir}/{game_date}")
    os.mkdir(f"{projdir}/{game_date}/Analysis")
    final_proj_df.to_csv(
        f"{projdir}/{game_date}/{game_date}_Projections.csv", index=False,
    )
