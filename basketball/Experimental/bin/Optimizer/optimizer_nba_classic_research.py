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
    selectShowdownSlate,
)
from sklearn.metrics import r2_score

warnings.simplefilter("ignore")
basedir = os.path.dirname(os.path.abspath(__file__))
etcdir = f"{basedir}/../../etc"
datadir = f"{basedir}/../../data"
game_date = "2023-02-02"
contestType = "Classic"
season = 2022
method = "ownership_proj"
salary_dir = (
    f"/Volumes/XDrive/DFS/DraftKingsSalaryMegaDatabase2/NBA/Classic/{game_date}"
)


def mse(a, b):
    x = round(np.mean((a - b) ** 2), 2)
    return x


def mae(a, b):
    rms = round(np.mean(np.abs(a - b)), 2)
    return rms


proj = pd.read_csv(
    f"{datadir}/Projections/RealTime/{season}/Classic/{game_date}/{game_date}_Projections.csv"
)
proj = proj[proj.dkpts != 0]
slate_salaries = selectShowdownSlate(salary_dir)
proj.drop(["ID", "game_time", "Salary"], axis=1, errors="ignore", inplace=True)
proj = proj.merge(
    slate_salaries[
        [
            "RotoName",
            "team_abbreviation",
            "position",
            "Roster Position",
            "ID",
            "game_time",
            "Salary",
        ]
    ],
    on=["RotoName", "team_abbreviation", "position"],
    how="left",
)
# proj.drop(
#     proj[
#         (proj.starter == True)
#         & (proj.DG_proj_mins < 28)
#         & (proj.ownership_proj < 15)
#     ].index,
#     inplace=True,
# )
# proj.drop(
#     proj[
#         (proj.starter == False)
#         & (proj.DG_proj_mins < 15)
#         & (proj.ownership_proj < 15)
#     ].index,
#     inplace=True,
# )
proj["game_time"] = pd.to_datetime(proj.game_time)
proj = proj[proj.dkpts.isna() == False]
proj.drop_duplicates(inplace=True)
proj = proj[proj.ID.isna() == False]
#### Settings for research projectiosn files ends here
proj = proj[(proj.Projection.isna() == False) & (proj.Salary.isna() == False)]
proj["game_date"] = proj.game_date.astype(str)
proj["min_exposure"] = 0
proj["max_exposure"] = 1
proj = processRankings(proj, Scaled=False)
proj.loc[proj.RG_projection == 0, method] = 0
proj.loc[proj[method].isna() == True, method] = proj.loc[
    proj[method].isna() == True, "RG_projection"
]
proj.loc[proj["Floor"].isna() == True, "Floor"] = (
    proj.loc[proj["Floor"].isna() == True, "RG_projection"] * 0.65
)
proj.loc[proj["Ceiling"].isna() == True, "Ceiling"] = (
    proj.loc[proj["Ceiling"].isna() == True, "RG_projection"] * 1.35
)
proj.loc[proj.Floor.isna() == True, "Floor"] = proj.loc[
    proj.Floor.isna() == True, "RG_projection"
]
# proj=proj[proj.proj_mins>20]
proj["Floor"] = proj[["Floor", f"{method}"]].mean(axis=1)
proj["Ceiling"] = proj[["Ceiling", f"{method}"]].mean(axis=1)
proj = proj[proj.dkpts > 0]
proj["Projection"] = proj[["RG_projection", "ML", "DG_Proj"]].mean(axis=1)
# proj=proj[proj.starter==True]
# Create optimizer
if contestType == "Showdown":
    proj.loc[proj["Roster Position"] == "CPT", "Projection"] *= 1.5
    proj.loc[proj["Roster Position"] == "CPT", "DKPts"] *= 1.5
    optimizer = get_optimizer(Site.DRAFTKINGS_CAPTAIN_MODE, Sport.BASKETBALL)
else:
    optimizer = get_optimizer(Site.DRAFTKINGS, Sport.BASKETBALL)
proj["min_exposure"] = 0
proj["max_exposure"] = 1
# proj.loc[proj.player_name == "Naji Marshall", "min_exposure"] = 1.0
# proj.loc[proj.player_name=='Orlando Robinson','min_exposure']=0.5
# proj.loc[proj.player_name=='Kyle Lowry','min_exposure']=0.5
# proj.loc[proj.player_name == "Jose Alvarado", "min_exposure"] = 1.0
proj = proj[proj.Projection != 0]
proj.drop(
    proj[
        (proj.starter == True)
        & (proj.proj_mins < 28)
        & (proj.ownership_proj < 15)
    ].index,
    inplace=True,
)
proj.drop(
    proj[
        (proj.starter == False)
        & (proj.proj_mins < 15)
        & (proj.ownership_proj < 15)
    ].index,
    inplace=True,
)
players = loadPlayers(proj, method, contestType)
optimizer.player_pool.load_players(players.to_list())
# optimizer.set_projected_ownership(0.09,0.13)
# optimizer.set_players_from_one_team({'PHI': 2,
#                                       'IND':2})
Exclude = ["Gary Payton"]
# Exclusions(Exclude, optimizer)
Include = ["Tj Mcconnell", "Lebron James", "Luka Doncic"]
# Locks(Include,optimizer
# Locks(Include,optimizer)
# optimizer.player_pool.exclude('')
# optimizer.set_total_teams(max_teams=3)
optimizer.set_min_salary_cap(49700)
optimizer.set_projected_ownership(8, 12)
# optimizer.set_fantasy_points_strategy(RandomFantasyPointsStrategy(1.25))
# Execute Optimizer
LineupResults = []
AllLineupResults = []
for lineup in optimizer.optimize(n=100):
    print_lineups(lineup, proj)
    LineupResults.append(lineup.actual_fantasy_points_per_game_projection)
    AllLineupResults.append(lineup.actual_fantasy_points_per_game_projection)

print(np.mean(LineupResults))
print(np.max(LineupResults))
