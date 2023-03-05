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
from optimizerTools import (
    Locks,
    Exclusions,
    loadPlayers,
    print_lineups,
    getSlateSalaries,
    getTopLines,
)

warnings.simplefilter("ignore")
basedir = os.path.dirname(os.path.abspath(__file__))
etcdir = f"{basedir}/../../etc"
datadir = f"{basedir}/../../data"
game_date = "2023-02-23"
contestType = "Classic"
method='Projection'
season = 2022
salary_dir = (
    f"/Volumes/XDrive/DFS/DraftKingsSalaryMegaDatabase2/NHL/Classic/{game_date}"
)
proj = pd.read_csv(
    f"{datadir}/Projections/RealTime/{season}/Classic/{game_date}/{game_date}_Projections.csv"
)


def mse(a, b):
    x = round(np.mean((a - b) ** 2), 2)
    return x


def mae(a, b):
    mae = round(np.mean(np.abs(a - b)), 2)
    return mae


def rmse(a, b):
    rmse = np.sqrt((((a - b) ** 2).sum()) / len(a))
    return rmse


slate_salaries = getSlateSalaries(salary_dir)
proj.drop("ID", axis=1, inplace=True)
proj = proj.merge(
    slate_salaries[
        ["full_name", "team", "position", "Roster Position", "ID", "game_time"]
    ],
    on=["full_name", "team", "position"],
    how="left",
)
proj = proj[
    (proj.Projection.isna() == False)
    & (proj.Salary.isna() == False)
    & (proj.ID.isna() == False)
]
proj["game_date"] = proj.game_date.astype(str)
proj = proj[proj.line < 4]
proj = processRankings(proj, game_date, powerplay=False, Scaled=False)
proj.drop(
    proj[
        (proj.position == "D") & (proj.powerplay == False) & (proj.line > 2)
    ].index,
    inplace=True,
)
proj.loc[proj.position == "G", "Stochastic"] = proj.loc[
    proj.position == "G", "RG_projection"
]
proj["Projection"] = proj[["RG_projection", "Stochastic"]].mean(axis=1)
proj = getTopLines(proj)
# Offense
proj.loc[(proj.position_type == "Forward"), "Adj_ratio"] = (
    proj.loc[proj.position_type == "Forward", "ratio"]
    * proj.loc[proj.position_type == "Forward", method]
) / 3
# proj.loc[(proj.position_type=='Forward'),'Floor']=(
#     (proj.loc[proj.position_type=='Forward','ratio']*
#     proj.loc[proj.position_type=='Forward','Floor'])/3)
# proj.loc[(proj.position_type=='Forward'),'Ceiling']=(
#     (proj.loc[proj.position_type=='Forward','ratio']*
#     proj.loc[proj.position_type=='Forward','Ceiling'])/3)

# Defense
proj.loc[(proj.position_type == "Defenseman"), "Adj_ratio"] = (
    proj.loc[proj.position_type == "Defenseman", "ratio"]
    * proj.loc[proj.position_type == "Defenseman",method]
) / 2

# proj.loc[(proj.position_type=='Defenseman'),'Floor']=(
#     (proj.loc[proj.position_type=='Defenseman','ratio']*
#     proj.loc[proj.position_type=='Defenseman','Floor'])/2)

# proj.loc[(proj.position_type=='Defenseman'),'Ceiling']=(
#     (proj.loc[proj.position_type=='Defenseman','ratio']*
#     proj.loc[proj.position_type=='Defenseman','Ceiling'])/2)

# Goalie
proj.loc[proj.position_type == "Goalie", "Adj_ratio"] = proj.loc[
    proj.position_type == "Goalie", "ratio"
]
# proj.loc[proj.position_type=='Goalie','Floor']=proj.loc[proj.position_type=='Goalie','ratio']
# proj.loc[proj.position_type=='Goalie','Ceiling']=proj.loc[proj.position_type=='Goalie','ratio']
# proj=proj[(proj.ratio>1.0)]
# proj=proj[proj.moneyline<0]
proj = proj[proj.ratio.isna() == False]
players = loadPlayers(proj,'ownership_proj', contestType)
optimizer = get_optimizer(Site.DRAFTKINGS, Sport.HOCKEY)
optimizer.player_pool.load_players(players.to_list())
optimizer.add_stack(PositionsStack([["G"], ("LW", "C", "RW", "D")]))
optimizer.add_stack(PositionsStack(["C","D",("LW","RW"),("LW","RW","D")]))
optimizer.add_stack(PositionsStack([('C','LW','RW'),("C","LW","RW","D"),("C","LW","RW","D")]))
optimizer.set_min_salary_cap(49500)


Include = ["Zach Hyman"]
# Locks(Include,optimizer)
# Exclude = ["Leon Draisaitl"]
# Exclusions(Exclude,optimizer)
optimizer.set_spacing_for_positions(["C", "LW", "RW", "D"], 1)
optimizer.restrict_positions_for_opposing_team(["G"], ["LW", "RW", "D", "C"])
optimizer.set_players_with_same_position({'C':1})

LineupResults = []
AllLineupResults = []
for lineup in optimizer.optimize(n=40,randomness=True):
    print_lineups(lineup, proj)
    LineupResults.append(lineup.actual_fantasy_points_per_game_projection)
    AllLineupResults.append(lineup.actual_fantasy_points_per_game_projection)

print(np.mean(LineupResults))
print(np.max(LineupResults))
