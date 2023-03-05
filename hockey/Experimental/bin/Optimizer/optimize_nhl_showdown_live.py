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
from optimizerTools import Locks, Exclusions, loadPlayers, getSlateSalaries, getTopLines
import numpy as np

warnings.simplefilter("ignore")
basedir = os.path.dirname(os.path.abspath(__file__))
datadir = f"{basedir}/../../data"
game_date = datetime.now().strftime("%Y-%m-%d")
# game_date='2022-10-26'
salary_dir=f"/Volumes/XDrive/DFS/DraftKingsSalaryMegaDatabase2/NHL/Showdown/{game_date}"
season = 2022
proj = pd.read_csv(
    f"{datadir}/Projections/RealTime/{season}/Classic/{game_date}/{game_date}_Projections.csv"
)
slate_salaries=getSlateSalaries(salary_dir)
proj.drop(['ID','Roster Position'],axis=1,inplace=True)
proj=proj.merge(slate_salaries[['full_name','team','position','Roster Position','ID','game_time']],
                on=['full_name','team','position'],
                how='left')
proj = proj[(proj.Projection.isna() == False) & 
            (proj.Salary.isna() == False) &
            (proj.ID.isna()==False)]
proj=proj[proj.position!='G']
proj = proj[(proj.Projection.isna() == False) & (proj.Salary.isna() == False)]
proj = processRankings(proj, game_date, powerplay=False, Scaled=False)


# Create optimizer
proj.loc[proj["Roster Position"] == "CPT", "Projection"] *= 1.5
optimizer = get_optimizer(Site.DRAFTKINGS_CAPTAIN_MODE, Sport.HOCKEY)
players = loadPlayers(proj, "Projection", "Showdown")
optimizer.player_pool.load_players(players.to_list())
# optimizer.set_min_salary_cap(49200)

# player = optimizer.player_pool.get_player_by_name(
#      "Cole Caufield", "CPT"
#  )  # using player name and position
# optimizer.player_pool.lock_player(player)
for lineup in optimizer.optimize(n=1000, randomness=True):
    print(lineup)
optimizer.print_statistic()
optimizer.export("/Users/robertmegnia/Desktop/NHL_Showdown_Lineups.csv")
