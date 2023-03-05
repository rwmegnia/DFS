# -*- coding: utf-8 -*-
import pandas as pd
from pydfs_lineup_optimizer import Site, Sport, get_optimizer, PositionsStack, TeamStack
from pydfs_lineup_optimizer.fantasy_points_strategy import RandomFantasyPointsStrategy
from pydfs_lineup_optimizer.exposure_strategy import AfterEachExposureStrategy

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
salary_dir=f"/Volumes/XDrive/DFS/DraftKingsSalaryMegaDatabase2/NHL/Classic/{game_date}"
season = 2022
### Read in Projections
proj = pd.read_csv(
    f"{datadir}/Projections/RealTime/{season}/Classic/{game_date}/{game_date}_Projections.csv"
)

slate_salaries=getSlateSalaries(salary_dir)
proj.drop('ID',axis=1,inplace=True)
proj=proj.merge(slate_salaries[['full_name','team','position','Roster Position','ID','game_time']],
                on=['full_name','team','position'],
                how='left')
proj = proj[(proj.Projection.isna() == False) & 
            (proj.Salary.isna() == False) &
            (proj.ID.isna()==False)]
proj["game_date"] = proj.game_date.astype(str)
proj=proj[proj.line<4]
proj = processRankings(proj, game_date, powerplay=False, Scaled=False)
proj.drop(proj[(proj.position=='D')&(proj.powerplay==False)&(proj.line>2)].index,inplace=True)
proj.loc[proj.game_location=='away','home_team']=proj.loc[proj.game_location=='away','opp']
proj.loc[proj.game_location=='away','away_team']=proj.loc[proj.game_location=='away','team']
proj.loc[proj.game_location=='home','home_team']=proj.loc[proj.game_location=='home','team']
proj.loc[proj.game_location=='home','away_team']=proj.loc[proj.game_location=='home','opp']
proj.loc[proj.position=='G','Stochastic']=proj.loc[proj.position=='G','RG_projection']
proj['game_date']=pd.to_datetime(proj.game_date)
proj['Projection']=proj[['Stochastic','RG_projection']].mean(axis=1)
proj=getTopLines(proj)
# Offense
proj.loc[(proj.position_type=='Forward'),'Adj_ratio']=(
    (proj.loc[proj.position_type=='Forward','ratio']* 
    proj.loc[proj.position_type=='Forward','Projection'])/3)
proj.loc[(proj.position_type=='Forward'),'Floor']=(
    (proj.loc[proj.position_type=='Forward','ratio']* 
    proj.loc[proj.position_type=='Forward','Floor'])/3)
proj.loc[(proj.position_type=='Forward'),'Ceiling']=(
    (proj.loc[proj.position_type=='Forward','ratio']* 
    proj.loc[proj.position_type=='Forward','Ceiling'])/3)

#Defense
proj.loc[(proj.position_type=='Defenseman'),'Adj_ratio']=(
    (proj.loc[proj.position_type=='Defenseman','ratio']* 
    proj.loc[proj.position_type=='Defenseman','Projection'])/2)

proj.loc[(proj.position_type=='Defenseman'),'Floor']=(
    (proj.loc[proj.position_type=='Defenseman','ratio']* 
    proj.loc[proj.position_type=='Defenseman','Floor'])/2)

proj.loc[(proj.position_type=='Defenseman'),'Ceiling']=(
    (proj.loc[proj.position_type=='Defenseman','ratio']* 
    proj.loc[proj.position_type=='Defenseman','Ceiling'])/2)

# Goalie
proj.loc[proj.position_type=='Goalie','Adj_ratio']=proj.loc[proj.position_type=='Goalie','ratio']
proj.loc[proj.position_type=='Goalie','Floor']=proj.loc[proj.position_type=='Goalie','ratio']
proj.loc[proj.position_type=='Goalie','Ceiling']=proj.loc[proj.position_type=='Goalie','ratio']

# proj=proj[proj.moneyline<0]
proj=proj[proj.ratio>1]
#%%
proj=proj[proj.ratio.isna()==False]
players = loadPlayers(proj, "Adj_ratio", "Classic")
# Create Opitmizer
optimizer = get_optimizer(Site.DRAFTKINGS, Sport.HOCKEY)
optimizer.player_pool.load_players(players.to_list())
optimizer.add_stack(PositionsStack([["G"], ("LW", "C", "RW", "D")]))
optimizer.add_stack(TeamStack(4))
# optimizer.add_stack(PositionsStack(["C","D",("LW","RW"),("LW","RW","D")]))
# optimizer.add_stack(PositionsStack([('LW','RW','C'),("C","LW","RW","D"),("C","LW","RW","D")]))
optimizer.set_min_salary_cap(49500)
optimizer.set_fantasy_points_strategy(RandomFantasyPointsStrategy(0.10))
Locks(['Auston Matthews'],optimizer)
optimizer.set_spacing_for_positions(["C","LW","RW","D"],2)
optimizer.restrict_positions_for_opposing_team(["G"], ["LW", "RW", "D", "C"])
# # optimizer.set_min_salary_cap(49500)
optimizer.set_players_with_same_position({'C':1})
# Execute Optimizer
for lineup in optimizer.optimize(n=80,randomness=True):
    lineup.actual_fantasy_points_per_game_projection=np.nan
    print(lineup)
optimizer.print_statistic()
optimizer.export(f"NHL_Lineups.csv")
df=pd.read_csv('NHL_Lineups.csv')
df.sample(20).to_csv('/Users/robertmegnia/Desktop/NHL_Lineups.csv',index=False)

