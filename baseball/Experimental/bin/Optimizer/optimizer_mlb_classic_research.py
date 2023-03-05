#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 16:05:01 2022

@author: robertmegnia
"""

# -*- coding: utf-8 -*-
import pandas as pd
from pydfs_lineup_optimizer import (
    Site,
    Sport,
    get_optimizer,
    PositionsStack,
    TeamStack,
)

# from pydfs_lineup_optimizer.stacks import GameStack
from pydfs_lineup_optimizer.fantasy_points_strategy import (
    RandomFantasyPointsStrategy,
)
from pydfs_lineup_optimizer.exposure_strategy import AfterEachExposureStrategy
import os
import numpy as np
import warnings
from datetime import datetime
from ProcessRankings import processRankings
from optimizerTools import *

warnings.simplefilter("ignore")
basedir = os.path.dirname(os.path.abspath(__file__))
game_date='2022-09-30'
datadir = f"{basedir}/../../data"
contestType='Classic'
salary_dir=f"/Volumes/XDrive/DFS/DraftKingsSalaryMegaDatabase2/MLB/{contestType}/{game_date}"
season = 2022
projdir = f"{basedir}/../../data/Projections/RealTime/{season}/{contestType}/{game_date}"


def mse(a, b):
    x = round(np.mean((a - b) ** 2), 2)
    return x


def mae(a, b):
    rms = round(np.mean(np.abs(a - b)), 2)
    return rms


#%%
### Read in Projections
proj = pd.read_csv(
    f"{datadir}/Projections/RealTime/{season}/{contestType}/{game_date}/{game_date}_Projections.csv"
)
proj.drop(proj.loc[(proj['Roster Position']=='CPT')&(proj.position_type=='Pitcher')].index,inplace=True)
slates = [
    file
    for file in os.listdir(
        f"{salary_dir}"
    )
    if ".csv" in file
]

n = 0
for file in slates:
    if len(file.split('_'))>2:
        print(n + 1, file)
        n += 1
if len(slates) == 1:
    slate = 1
else:
    slate = input("Select Slate by number (1,2,3, etc...) ")
slate = slates[int(slate) - 1]
slate_salaries=pd.read_csv(f'{salary_dir}/{slate}')
proj.drop('ID',axis=1,inplace=True)
proj=proj.merge(slate_salaries[['RotoName','team','position','Roster Position','ID']],
                on=['RotoName','team','position','Roster Position'],
                how='left')
proj=proj[proj.ID.isna()==False]
# proj = processRankings(proj, game_date, Scaled=False)
proj["game_date"] = pd.to_datetime(proj.game_date)
proj = proj[proj.DKPts.isna() == False]
proj.drop_duplicates(inplace=True)
proj=proj[proj.game_time.isin(slate_salaries.game_time)]

proj.loc[proj.Ceiling.isna() == True, "Ceiling"] = proj.loc[
    proj.Ceiling.isna() == True, "Projection"
]
proj.loc[proj.Floor.isna() == True, "Floor"] = proj.loc[
    proj.Floor.isna() == True, "Projection"
]
proj.loc[proj.Stochastic.isna() == True, "Stochastic"] = proj.loc[
    proj.Stochastic.isna() == True, "Projection"
]

proj.loc[proj.ML.isna() == True, "ML"] = proj.loc[
    proj.ML.isna() == True, "Projection"
]
# proj["Projection"] = proj[['Stochastic','RG_projection']].mean(axis=1)
# Create optimizer
if contestType == "Showdown":
    proj = proj[proj.position != "G"]
    proj.loc[proj["Roster Position"] == "CPT", "Projection"] *= 1.5
    proj.loc[proj["Roster Position"] == "CPT", "Stochastic"] *= 1.5
    proj.loc[proj["Roster Position"] == "CPT", "RG_projection"] *= 1.5
    proj.loc[proj["Roster Position"] == "CPT", "DKPts"] *= 1.5
    optimizer = get_optimizer(Site.DRAFTKINGS_CAPTAIN_MODE, Sport.BASEBALL)
else:
    optimizer = get_optimizer(Site.DRAFTKINGS, Sport.BASEBALL)
#%%
# proj.loc[proj.position_type=='Pitcher','Projection']=proj.loc[proj.position_type=='Pitcher','Floor']
# proj.loc[proj.position_type!='Pitcher','Projection']=proj.loc[proj.position_type!='Pitcher','Ceiling']

players = loadPlayers(proj, "ownership_proj", contestType)
optimizer.player_pool.load_players(players.to_list())
# optimizer.player_pool.lock_player("Shohei Ohtani")
# optimizer.player_pool.lock_player("Alex Manoah")
optimizer.set_min_salary_cap(49500)
# optimizer.add_stack(TeamStack(4))
# optimizer.add_stack(TeamStack(3))

# optimizer.add_stack(TeamStack(4,for_teams=['TEX'],for_positions=["C", "1B", "2B", "3B", "SS", "OF"]))
# optimizer.set_max_repeating_players(6)
# optimizer.set_projected_ownership(15,20)
# optimizer.player_pool.exclude_teams(['OAK'])
# player=optimizer.player_pool.get_player_by_name('Luis Garcia','RP')
# optimizer.player_pool.lock_player(player)
Include = ["Aaron Judge"
           ]
# Locks(Include, optimizer)
Exclude = ['Omar Navarez',"Jacob Degrom"]
# Exclusions(Exclude, optimizer)
optimizer.restrict_positions_for_opposing_team(
    ["SP","P","RP"], ["C", "1B", "2B", "3B", "SS", "OF"]
)
# optimizer.player_pool.exclude_teams(['BAL','PIT'])

# optimizer.add_stack(
#     TeamStack(
#         4,for_teams=['NYY'],for_positions=["C", "1B", "2B", "3B", "SS", "OF"]
#     )
# )
# optimizer.add_stack(
#     TeamStack(
#         5, for_teams=["ATL"], for_positions=["C", "1B", "2B", "3B", "SS", "OF"]
#     )
# )
# optimizer.set_max_repeating_players(5)
# optimizer.set_deviation(0,1)
LineupResults = []
AllLineupResults = []
cash_line=105.8
# for lineup in optimizer.optimize(20,exposure_strategy=AfterEachExposureStrategy):
for lineup in optimizer.optimize(10):
    print_lineups(lineup, proj)
    LineupResults.append(lineup.actual_fantasy_points_per_game_projection)
    AllLineupResults.append(lineup.actual_fantasy_points_per_game_projection)
slate = slate.split("_Projections.csv")[0]
optimizer.print_statistic()
# optimizer.export(f"{datadir}/ExportedLineups/ResearchLineups.csv")
# lineups = pd.read_csv(f"{datadir}/ExportedLineups/ResearchLineups.csv")
# batter_db = pd.read_csv(f"{datadir}/game_logs/batterstatsDatabase.csv")
# pitcher_db = pd.read_csv(f"{datadir}/game_logs/pitcherStatsDatabase.csv")
# lineups["maxScore"] = lineups.T.apply(
#     lambda x: getMaxLineupScore(x, proj, pitcher_db, batter_db,research=True)
# )
# lineups.sort_values(by="maxScore", ascending=False, inplace=True)
# print(lineups[lineups.maxScore == lineups.maxScore.max()].T)
print(np.mean(LineupResults))
print(np.max(LineupResults))
LineupResults=np.asarray(LineupResults)
successRate=len(LineupResults[LineupResults>cash_line])/len(LineupResults)*100
successRate=round(successRate,2)
print(f'Success rate = {successRate}%')
