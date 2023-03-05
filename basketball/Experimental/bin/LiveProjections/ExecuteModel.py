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
def getMinutes(df):
    df["mins"] = df["min"].apply(lambda x: x.split(":")[0]).astype(float)
    df["seconds"] = (
        df["min"]
        .apply(lambda x: x.split(":")[1] if len(x.split(":")) > 1 else 0)
        .astype(float)
    )
    df["mins"] = df.mins + (df.seconds / 60)
    return df["mins"]
datadir = f"{basedir}/../../data"
season = 2022
game_date = datetime.now().strftime("%Y-%m-%d")
# game_date='2022-10-26'
contestType = "Classic"
SWAP = False
# Get Player Pool for the Day, Starting lineups, and odds
start_df = scrapeStartingLineups(game_date)
odds_df = ScrapeBettingOdds()
start_df=start_df.merge(odds_df,on=['team_abbreviation','game_date'])
start_df.drop(['spread_line','moneyline'],axis=1,inplace=True)
# Get Draftkings Salaries for the day
salaries= getDKSalaries(game_date, contestType)
if contestType=='Showdown':
    salaries=salaries[salaries['Roster Position']=='UTIL']
start_df = start_df[start_df.team_abbreviation.isin(salaries.team)]

# Load in database to pull statistics from and filter out current and future dates
db = pd.read_csv(f"{datadir}/game_logs/PlayerStatsDatabase.csv")
db.player_name.replace("Nene", "Nene Nene", inplace=True)
player_IDs = db.groupby("player_id", as_index=False).last()[
    ["team_abbreviation", "player_name", "nickname", "player_id"]
]
player_IDs = reformatNames(player_IDs)

salaries = salaries.merge(
    player_IDs[["player_id", "RotoName"]],on="RotoName", how="left"
)

if contestType=='Showdown':
    start_df = start_df.drop('Salary',axis=1).merge(
        salaries[["position", "player_id", "Salary","Game Info","ID", "RotoName", "player_name"]],
        on=["RotoName"],
        how="left",
    )
else:
    start_df = start_df.drop('Salary',axis=1).merge(
        salaries[["position", "player_id", "Salary","Game Info","ID", "RotoName", "player_name",'Roster Position']],
        on=["RotoName"],
        how="left",
    )
db["game_date_string"] = db.game_date
db.game_date = pd.to_datetime(db.game_date)
db.sort_values(by="game_date", inplace=True)
config=db[db.started==True].groupby(['game_id','team_abbreviation'],as_index=False).player_id.sum().reset_index(drop=True)
config.rename({'player_id':'config'},axis=1,inplace=True)
db=db.merge(config,on=['game_id','team_abbreviation'],how='left')
stats_df = db[db.game_date < game_date]

stats_df.sort_values(by="game_date", inplace=True)
stats_df['rotoname']=stats_df.player_name.apply(lambda x: x.split(' ')[0][0:4].lower()+x.split(' ')[1][0:5].lower())
season = 2022
config=start_df[start_df.starter==True].groupby('team_abbreviation').player_id.sum().reset_index()
config.rename({'player_id':'config'},axis=1,inplace=True)
start_df=start_df.merge(config,on='team_abbreviation',how='left')
start_df=MLMinutesPrediction(stats_df, start_df)
#%%
# Filter players to starters projected to play > 28 minutes
# start_df.drop(start_df[(start_df.starter==True)&(start_df.proj_mins<28)].index,inplace=True)
# Filter bench players projected to play under 15
# start_df.drop(start_df[(start_df.starter==False)&(start_df.proj_mins<15)].index,inplace=True)
start_df.reset_index(inplace=True)
stochastic_proj_df=StochasticPrediction(stats_df,start_df)
stochastic_proj_df.reset_index(inplace=True)
stochastic_proj_df=stochastic_proj_df.groupby(['player_id',
                                               'position',
                                               'RotoName', 
                                               'RotoPosition',
                                               'team_abbreviation',
                                               'opp',
                                               'player_name',
                                               'Game Info',
                                               'full_name',
                                               'game_date',
                                               ],as_index=False).mean()
#QC
stochastic_proj_df.loc[(stochastic_proj_df.RG_projection==0)&(stochastic_proj_df.Stochastic!=0),'Stochastic']=0
stochastic_proj_df.loc[(stochastic_proj_df.RG_projection==0)&(stochastic_proj_df.Stochastic!=0),'Ceiling']=0
stochastic_proj_df.loc[(stochastic_proj_df.RG_projection==0)&(stochastic_proj_df.Stochastic!=0),'Floor']=0

#%%
# # Read in player database
# player_db = pd.read_csv(f"{datadir}/game_logs/PlayerStatsDatabase.csv")
# player_db["game_date_string"] = player_db.game_date
# player_db.game_date = pd.to_datetime(player_db.game_date)
# player_db.sort_values(by="game_date", inplace=True)
player_stats_df = db[db.game_date < game_date]
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
#%%
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
ml_proj_df=ml_proj_df.groupby('player_id',as_index=False).mean()
ml_proj_df = stochastic_proj_df.merge(
    ml_proj_df[["player_id", "EN", "RF", "GB", "TD_Proj","TD_Proj2"]],
    on="player_id",
    how="left",
)
#%%
# # Quality Control
ml_proj_df.loc[(ml_proj_df.RG_projection==0),['Stochastic','TD_Proj',
                                              'TD_Proj2','EN',
                                              'RF','GB','Median']]=0
ml_proj_df.loc[(ml_proj_df.Stochastic<5)&
                  (ml_proj_df.RG_projection>5),'Stochastic']=(
                      ml_proj_df.loc[(ml_proj_df.Stochastic<5)&
                                        (ml_proj_df.RG_projection>5),'RG_projection'])
ml_proj_df.loc[(ml_proj_df.Median<5)&
                  (ml_proj_df.RG_projection>5),'Median']=(
                      ml_proj_df.loc[(ml_proj_df.Median<5)&
                                        (ml_proj_df.RG_projection>5),'RG_projection'])
for ml_method in ['EN','RF','GB','TD_Proj','TD_Proj2']:
    ml_proj_df.loc[(ml_proj_df[ml_method]==0)&
                    (ml_proj_df.RG_projection!=0),ml_method]=ml_proj_df.loc[(
                        ml_proj_df[ml_method]==0)&
                        (ml_proj_df.RG_projection!=0),'RG_projection']
    ml_proj_df.loc[(ml_proj_df[ml_method].isna()==True),ml_method]=ml_proj_df.loc[(
                        ml_proj_df[ml_method].isna()==True),'RG_projection']


final_proj_df=getPMM(ml_proj_df,game_date)
final_proj_df["Projection"] = final_proj_df[
    ["Stochastic", "ML", "RG_projection", "PMM", "TD_Proj","Median"]
].mean(axis=1)
final_proj_df.loc[final_proj_df.RG_projection==0,'Projection']=0
final_proj_df["game_date"] = game_date
projdir = f"{datadir}/Projections/RealTime/{season}/{contestType}"
final_proj_df = final_proj_df[final_proj_df.RG_projection.isna() == False]
final_proj_df.drop_duplicates(inplace=True)
final_proj_df=final_proj_df.merge(odds_df[['team_abbreviation','spread_line','moneyline']],on=['team_abbreviation'],how='left')
#%%
try:
    if SWAP==False:
        final_proj_df.to_csv(
            f"{projdir}/{game_date}/{game_date}_Projections.csv", index=False,
        )
    else:
        final_proj_df.to_csv(
            f"{projdir}/{game_date}/{game_date}_Projections_LATESWAP.csv", index=False,
        )
except OSError:
    os.mkdir(f"{projdir}/{game_date}")
    final_proj_df.to_csv(
        f"{projdir}/{game_date}/{game_date}_Projections.csv", index=False,
    )
