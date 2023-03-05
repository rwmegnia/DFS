#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 19:00:06 2022

@author: robertmegnia
"""

import pandas as pd
import os
from os.path import exists
from datetime import datetime
from MLB_API_TOOLS import *
from getDKSalaries import getDKSalaries
from getPlayerPool import getPlayerPool
from PMM import getPMM, getPMMScaled
from ProcessSlateStats import processSlateStats
from ProcessRankings import processRankings

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

START_DF_COLS = [
    "RotoName",
    "RotoPosition",
    "RG_projection",
    "ownership_proj",
    "team",
    "handedness",
    "opp_pitcher_hand",
    "order",
    "total_runs",
    "moneyline",
    "proj_runs",
    "game_time",
]
SALARY_DF_COLS = [
    "RotoName",
    "team",
    "position",
    "Roster Position",
    "Salary",
    "ID",
    "game_time",
]
#%%
datadir = f"{basedir}/../../data"
season = 2022
n = 0
ctypes=['Classic','Showdown']
for ctype in ctypes:
    print(n + 1, ctype)
    n += 1
contestType = input("Select Contest Type by number (1,2,3, etc...) ")
contestType = ctypes[int(contestType)-1]
game_date = datetime.now().strftime("%Y-%m-%d")
# Get Player Pool for the Day, Starting lineups, and odds
players = getPlayerPool(game_date, season)
players.reset_index(drop=True,inplace=True)
players.drop(players.loc[(players.RotoName=='joseramir')&(players.position_type=='Pitcher')].index,axis=0,inplace=True)
players.drop(players.loc[(players.RotoName=='joshsmith')&(players.position_type=='Pitcher')].index,axis=0,inplace=True)

start_df = scrapeStartingLineups()
odds_df = ScrapeBettingOdds()
odds_df["game_time"] = odds_df["game_time"].apply(lambda x: x.zfill(7))
# odds_df.loc[(odds_df.team=='ATL')|(odds_df.team=='STL'),'game_time']='07:20PM'
# if not exists(f'{datadir}/BettingOdds/{game_date}_BettingOdds.csv'):
#     odds_df = ScrapeBettingOdds()
#     odds_df['game_time'] = odds_df['game_time'].apply(lambda x: x.zfill(7))
# else:
#     odds_df = pd.read_csv(f'{datadir}/BettingOdds/{game_date}_BettingOdds.csv')
#     odds_df['game_time'] = odds_df['game_time'].apply(lambda x: x.zfill(7))

# Add padded zeros
start_df["game_time"] = start_df["game_time"].apply(lambda x: x.zfill(7))
start_df = start_df.merge(odds_df, on=["team", "game_time"], how="left")
# start_df.loc[(start_df.team=='NYM')|(start_df.team=='SD'),'game_time']='07:08PM'
starting_pitchers = start_df[start_df.RotoPosition == "P"]
starting_pitchers["position_type"] = "Pitcher"
#%%
# Shoheai Ohtani is a two-way player, make sure he has
# correct position type if he's a starting pitcher
if "shohohtan" in starting_pitchers.RotoName.to_list():
    players.loc[players.RotoName == "shohohtan", "position_type"] = "Pitcher"
    start_df.drop(start_df[(start_df.RotoName=="shohohtan")&(start_df.order!=9)].index,inplace=True)
starting_pitchers = starting_pitchers.merge(
    players[["RotoName", "position_type", "player_id", "team", "game_time"]],
    on=["RotoName", "position_type", "team", "game_time"],
    how="left",
).drop_duplicates()
starting_pitchers.drop("team", axis=1, inplace=True)
starting_pitchers.rename(
    {"opp": "team", "player_id": "opp_pitcher_id"}, axis=1, inplace=True
)
starting_pitchers = starting_pitchers[["opp_pitcher_id", "team", "game_time"]]
if game_date=='2022-08-29':
    starting_pitchers.loc[starting_pitchers.team=='STL','opp_pitcher_id']=502624
    chase_anderson=pd.DataFrame({'full_name':['chase anderson'],'player_id':[502624],'position':[10],'position_type':['Pitcher'],'team':['CIN'],'game_location':['home'],'opp':['STL'],'game_id':[663018],'game_time':['06:40PM'],'RotoName':['chasander'],'season':[2022]})
    players=pd.concat([players,chase_anderson])
#%%
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
players = players.merge(
    start_df[START_DF_COLS], on=["RotoName", "team", "game_time"], how="left"
)
game_date = players.game_date.unique()[0]
print(game_date)
#%%
# Get Draftkings Salaries for the day
# salaries, slate = getDKSalaries(game_date, contestType)
salaries = getDKSalaries(game_date, contestType)
starting_pitchers = starting_pitchers[
    starting_pitchers.team.isin(salaries.team)
]
#%%
players = players[players.team.isin(salaries.team)]
# salaries= salaries.merge(starting_pitchers, on =['team','game_time'])
# Merge player pool with salaries
players = players.drop("position", axis=1).merge(
    salaries[SALARY_DF_COLS], on=["RotoName", "team", "game_time"], how="left"
)

players = players.merge(starting_pitchers, on=["team", "game_time"], how="left")
players["season"] = season
players = players[players.ID.isna() == False]
players = players[players.RotoPosition.isna() == False]
players.drop_duplicates(inplace=True)
# Alert if players not merged from salaries file
missing_players = players[players.Salary.isna() 
                          == True][
    ["RotoName", "RotoPosition", "team"]
]

print(
    " The following players were not merged from the DraftKings Salaries File..."
)
print(missing_players)
#%%
# Execute Stochastic Model Projection

stochastic_proj_frames = []
# Execute Hitter Stochastic Projections
db = pd.read_csv(f"{datadir}/game_logs/batterstatsDatabase.csv")
db["game_date_string"] = db.game_date
db.game_date = pd.to_datetime(db.game_date)
db.sort_values(by="game_date", inplace=True)
stats_df = db[db.game_date < game_date]
stats_df.sort_values(by="game_date", inplace=True)
batters = BattersStochastic(players, stats_df)

# Execute PitcherStochastic Projections
db = pd.read_csv(f"{datadir}/game_logs/PitcherstatsDatabase.csv")
db = db[db.position.isin(["P", "SP", "RP"])]
stats_df = db[db.game_date < game_date]
stats_df.sort_values(by="game_date", inplace=True)
pitchers = PitchersStochastic(players, stats_df)

# Concatenate Hitter/Pitcher Stochastic Projections
stochastic_proj_df = pd.concat([batters, pitchers])

#%%
# Make ML Predictions
ml_proj_frames = []

# Read in pitcher/batter databases
batter_db = pd.read_csv(f"{datadir}/game_logs/batterstatsDatabase.csv")
pitcher_db = pd.read_csv(f"{datadir}/game_logs/pitcherStatsDatabase.csv")
batter_db["game_date_string"] = batter_db.game_date
pitcher_db["game_date_string"] = pitcher_db.game_date
batter_db.game_date = pd.to_datetime(batter_db.game_date)
pitcher_db.game_date = pd.to_datetime(pitcher_db.game_date)
batter_db.sort_values(by="game_date", inplace=True)
pitcher_db.sort_values(by="game_date", inplace=True)

# Make sure to predict pitchers first since pitcher database gets altered for batters
position_type_db = batter_db
batter_stats_df = position_type_db[position_type_db.game_date < game_date]
#%%
for pitch_hand in ["L", "R"]:
    pitcher_games_df = players[
        (players.position.isin(["RP", "SP"]))
        & (players.handedness == pitch_hand)
    ]

    pitcher_stats_df = pitcher_db[pitcher_db.game_date < game_date]
    ml_proj_df = MLPitcherPrediction(
        batter_stats_df,
        pitcher_stats_df,
        pitcher_games_df,
        pitch_hand,
    )
    if ml_proj_df is None:
        continue
    ml_proj_frames.append(ml_proj_df)
#%%
players.loc[
    (players.handedness == "S") & (players.opp_pitcher_hand == "L"),
    "handedness",
] = "R"
players.loc[
    (players.handedness == "S") & (players.opp_pitcher_hand == "R"),
    "handedness",
] = "L"
for bat_hand in ["L", "R"]:
    for pitch_hand in ["L", "R"]:
        position_type_db = batter_db[batter_db.splits == f"vs_{pitch_hand}HP"]
        batter_games_df = players[
            (players.handedness == bat_hand)
            & (players.opp_pitcher_hand == pitch_hand)
        ]
        pitcher_games_df = players[
            (players.position == "SP") & (players.handedness == pitch_hand)
        ]

        pitcher_db["team"] = pitcher_db["opp"]
        try:
            pitcher_db = pitcher_db[PitcherNonFeatures + OpposingPitcherColumns]
            pitcher_db.rename(
                dict([(c, f"pitcher_{c}") for c in OpposingPitcherColumns]),
                axis=1,
                inplace=True,
            )
        except KeyError:
            pass
        batter_stats_df = position_type_db[
            position_type_db.game_date < game_date
        ]
        if len(batter_stats_df) == 0:
            continue
        pitcher_stats_df = pitcher_db[pitcher_db.game_date < game_date]
        ml_proj_df = MLBatterPrediction(
            batter_stats_df,
            pitcher_stats_df,
            batter_games_df,
            bat_hand,
            pitch_hand,
        )
        ml_proj_frames.append(ml_proj_df)
#%%
ml_proj_df = pd.concat(ml_proj_frames)
ml_proj_df['AVG']=ml_proj_df.hits/ml_proj_df.atBats
ml_proj_df['OBP']=ml_proj_df[['hits','baseOnBalls']].sum(axis=1)/ml_proj_df.atBats
ml_proj_df["oppK_prcnt"] = ml_proj_df.strikeOuts / ml_proj_df.atBats
opp_Kprcnt = (
    ml_proj_df[ml_proj_df.position_type != "Pitcher"]
    .groupby("opp", as_index=False)
    .oppK_prcnt.mean()
)
ml_proj_df.drop("oppK_prcnt", axis=1, inplace=True)
ml_proj_df = ml_proj_df.merge(opp_Kprcnt, on="opp", how="left")
ml_proj_df["opp_pitcher_era"] = (
    9 * ml_proj_df.pitcher_earnedRuns
) / ml_proj_df.pitcher_inningsPitched
ml_proj_df["era"] = (9 * ml_proj_df.earnedRuns) / ml_proj_df.inningsPitched
ml_proj_df['WHIP']=ml_proj_df[['baseOnBalls','hits']].sum(axis=1)/ml_proj_df.inningsPitched
ml_proj_df = stochastic_proj_df.merge(
    ml_proj_df[
        [
            "player_id",
            "BR",
            "EN",
            "NN",
            "RF",
            "GB",
            "Tweedie",
            "exWOBA",
            "wOBA",
            "AVG",
            "OBP",
            "ISO",
            "opp_pitcher_era",
            "SB_perAtBat",
            "K_prcnt",
            "oppK_prcnt",
            "era",
            "WHIP",
        ]
    ],
    on="player_id",
    how="left",
)

final_proj_df = getPMM(ml_proj_df, game_date)
final_proj_df = final_proj_df.groupby("ID", as_index=False).first()
final_proj_df.loc[
    final_proj_df.game_location == "home", "home_team"
] = final_proj_df.loc[final_proj_df.game_location == "home", "team"]
final_proj_df.loc[
    final_proj_df.game_location == "home", "away_team"
] = final_proj_df.loc[final_proj_df.game_location == "home", "opp"]
final_proj_df.loc[
    final_proj_df.game_location == "away", "away_team"
] = final_proj_df.loc[final_proj_df.game_location == "away", "team"]
final_proj_df.loc[
    final_proj_df.game_location == "away", "home_team"
] = final_proj_df.loc[final_proj_df.game_location == "away", "opp"]

#%%
final_proj_df["Projection"] = final_proj_df[
    ["Stochastic", "ML", "PMM", "RG_projection"]
].mean(axis=1)
final_proj_df["game_date"] = game_date
projdir = f"{datadir}/Projections/RealTime/{season}/{contestType}"
final_proj_df = final_proj_df[final_proj_df.RG_projection.isna() == False]
final_proj_df.drop_duplicates(inplace=True)
final_proj_df = processRankings(final_proj_df, game_date, Scaled=False)
processed = processSlateStats(final_proj_df)
try:
    final_proj_df.to_csv(
        f"{projdir}/{game_date}/{game_date}_Projections.csv",
        index=False,
    )

except OSError:
    os.mkdir(f"{projdir}/{game_date}")
    final_proj_df.to_csv(
        f"{projdir}/{game_date}/{game_date}_Projections.csv",
        index=False,
    )

try:
    processed.to_csv(
        f"{projdir}/{game_date}/Analytics/{game_date}_ProjectionsAnalysis.csv",
        index=False,
    )
except OSError:
    os.mkdir(f"{projdir}/{game_date}/Analytics")
    processed.to_csv(
        f"{projdir}/{game_date}/Analytics/{game_date}_ProjectionsAnalysis.csv",
        index=False,
    )
