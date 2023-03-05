#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 09:40:06 2022

@author: robertmegnia
"""
import pandas as pd
from pydfs_lineup_optimizer import Site, Sport, get_optimizer, PositionsStack, TeamStack
from pydfs_lineup_optimizer.fantasy_points_strategy import RandomFantasyPointsStrategy
import os
import warnings
from datetime import datetime
from ProcessRankings import processRankings
from optimizerTools import Locks, Exclusions, loadPlayers, selectShowdownSlate
import numpy as np

warnings.simplefilter("ignore")
basedir = os.path.dirname(os.path.abspath(__file__))
datadir = f"{basedir}/../../data"
game_date = datetime.now().strftime("%Y-%m-%d")
salary_dir=f"/Volumes/XDrive/DFS/DraftKingsSalaryMegaDatabase2/NBA/Showdown/{game_date}"
season = 2022
method='Projection'
### Read in Projections
cpt_proj = pd.read_csv(
    f"{datadir}/Projections/RealTime/{season}/Classic/{game_date}/{game_date}_Projections.csv"
)
flex_proj = pd.read_csv(
    f"{datadir}/Projections/RealTime/{season}/Classic/{game_date}/{game_date}_Projections.csv"
)
cpt_proj['Roster Position']='CPT'
flex_proj['Roster Position']='UTIL'
proj=pd.concat([cpt_proj,flex_proj])


slate_salaries=selectShowdownSlate(salary_dir)
proj.drop(['Salary','ID'],axis=1,
          inplace=True,
          errors='ignore')
proj=proj.merge(slate_salaries[['RotoName','team_abbreviation','Roster Position','ID','game_time','Salary']],
                on=['RotoName','team_abbreviation','Roster Position'],
                how='left')
proj=proj[proj.ID.isna()==False]
proj=proj[proj[method].isna()==False]


# Create optimizer
proj['Projection']=proj[['ML','RG_projection','DG_Proj']].mean(axis=1)
proj.loc[proj["Roster Position"] == "CPT", method] *= 1.5
proj['min_exposure']=0
proj['max_exposure']=1
optimizer = get_optimizer(Site.DRAFTKINGS_CAPTAIN_MODE, Sport.BASKETBALL)
proj['game_time']=pd.to_datetime(proj.game_time)
players = loadPlayers(proj, method, "Showdown")
optimizer.player_pool.load_players(players.to_list())


# player = optimizer.player_pool.get_player_by_name(
#     "Cale Makar", "FLEX"
# )  # using player name and position
# optimizer.player_pool.lock_player(player)
for lineup in optimizer.optimize(n=10):
    print(lineup)
optimizer.print_statistic()
# optimizer.export(f"{datadir}/ExportedLineups/{slate}_Lineups.csv")
