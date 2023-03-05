# -*- coding: utf-8 -*-
"""

Optimizer used for verification.

"""
import pandas as pd
import numpy as np
from pydfs_lineup_optimizer import (
    Site,
    Sport,
    get_optimizer,
    PositionsStack,
    TeamStack,
    CSVLineupExporter,
    AfterEachExposureStrategy,
)
from pydfs_lineup_optimizer.stacks import GameStack
from pydfs_lineup_optimizer.fantasy_points_strategy import *
from pydfs_lineup_optimizer.solvers.pulp_solver import PuLPSolver
from pydfs_lineup_optimizer.fantasy_points_strategy import (
    ProgressiveFantasyPointsStrategy,
)
from pydfs_lineup_optimizer import Player
from pydfs_lineup_optimizer.player import GameInfo, Player
from pydfs_lineup_optimizer.tz import get_timezone
from pytz import timezone
from OptimizerTools import *
import os
import warnings
from datetime import datetime
def mse(a, b):
    x = round(np.mean((a - b) ** 2), 2)
    return x


def mae(a, b):
    rms = round(np.mean(np.abs(a - b)), 2)
    return rms
warnings.simplefilter("ignore")
basedir = os.path.dirname(os.path.abspath(__file__))
etcdir = f"{basedir}/../../etc"
datadir = f"{basedir}/../../data"
# from getConsensusRank import getConsensusRanking

# Read in Projections data frame and create player pool
Slate = "Main"
season = 2022
week = 9
AllLineupResults = []
pf5=pd.DataFrame({},columns=['method','cashWon','AvgScore','MaxScore','week'])
for method in ['NicksAgg']:
#method = "wProjection"
    pre_bc_frames = []
    post_bc_frames = []
    team_frames=[]
    leverage_plays = []
    total_cash=0
    total_contests=0
    for season in range(2022,2023):
        for week in range(9,10):
            # week = np.random.choice(range(1, 18))
            # season = np.random.choice(range(2021, 2022))
            # season = 2021
            proj = pd.read_csv(
                f"{datadir}/Projections/{season}/WeeklyProjections/{season}_Week{week}_Projections_verified.csv"
            )
            proj.rename({'DKPts_x':'DKPts'},axis=1,inplace=True,errors='ignore')
            proj.drop(proj[(proj.position=='QB')&(proj.depth_team!=1)].index,inplace=True)
            if week==4:
                proj.drop(proj[proj.team.isin(['MIN','NO'])].index,inplace=True)
            elif week==5:
                proj.drop(proj[proj.team.isin(['NYG','GB'])].index,inplace=True)
            elif week==8:
                proj.drop(proj[proj.team.isin(['DEN','JAX'])].index,inplace=True)
            # proj=proj[proj.Ceiling>proj.UpsideScore]
            game_date=proj.game_date.unique()[0][0:10]
            contestResults=pd.read_csv(f'/Volumes/XDrive/DFS/football/Experimental/data/MillionaireMakerContestResults/2022/Week{week}_Millionaire_Results.csv')
            Standings=contestResults['Points'].to_frame()
            Prizes=contestResults[['Rank','Prize']]
            # Prizes['Rank']=Prizes.Rank.rank(method='min')
            proj.loc[proj.position == "DST", "full_name"] = proj.loc[
                proj.position == "DST", "full_name"
            ].apply(lambda x: x.split(" ")[0].upper())
            proj.rename({"game_id_y": "game_id"}, axis=1, inplace=True)
            proj["Ownership"] = proj.Ownership / 100
            proj.RosterPercent /= 100
            proj = proj[proj.Slate == Slate]
    
    
            # proj=proj[proj.FP_Proj.isna()==False]
            proj.game_location.replace("@", "away", inplace=True)
            proj.game_location.replace("VS", "home", inplace=True)
            proj.loc[proj.game_location == "away", "away_team"] = proj.loc[
                proj.game_location == "away", "team"
            ]
            proj.loc[proj.game_location == "away", "home_team"] = proj.loc[
                proj.game_location == "away", "opp"
            ]
            proj.loc[proj.game_location == "home", "home_team"] = proj.loc[
                proj.game_location == "home", "team"
            ]
            proj.loc[proj.game_location == "home", "away_team"] = proj.loc[
                proj.game_location == "home", "opp"
            ]
            proj["game_date"] = proj["game_time"].apply(
                lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").replace(
                    tzinfo=timezone(get_timezone())
                )
            )
            proj.loc[proj.Ceiling.isna() == True, "Ceiling"] = proj.loc[
                proj.Ceiling.isna() == True, method
            ]
            # proj = proj[proj.Projection >= 5]
            proj.fillna(0, inplace=True)
            proj['game_id']=proj.home_team+proj.away_team
            pass_teams = getPassingStackTeams(proj)
            rb_teams = getRunningBackTeams(proj)
            # Get Individual Game Information
            game_info = {}
            game_ids = (
                proj[proj.game_id != 0].groupby("game_id", as_index=False).first()
            )
            for row in game_ids.iterrows():
                row = row[1]
                game_info[row.game_id] = GameInfo(
                    row.home_team, row.away_team, row.game_date
                )
            # Create list of pydfs-lineup-optimizer Player objects
            # proj['Projection']=proj[['TopDown','RG_projection']].mean(axis=1)
            proj['Projection']=proj[['TopDown','Stochastic','ML','DC_proj']].mean(axis=1)
            # proj['Projection']=proj[['Projection','wProjection','FP_Proj2','FP_Proj2','RG_projection']].mean(axis=1)
            players = proj.apply(
                lambda x: Player(
                    player_id=x.ID,
                    first_name=x.full_name.split(" ")[0],
                    last_name=" ".join(x.full_name.split(" ")[1::]),
                    positions=[x.position],
                    team=x.team,
                    salary=x.salary,
                    fppg=x[method],
                    projected_ownership=x.AvgOwnership/100,
                    game_info=game_info[x.game_id],
                    min_deviation=x.min_deviation,
                    max_deviation=x.max_deviation,
                ),
                axis=1,
            )
            # Create optimizer
            optimizer = get_optimizer(Site.DRAFTKINGS, Sport.FOOTBALL)
            optimizer.player_pool.load_players(players.to_list())
            optimizer.set_min_salary_cap(49500)
            optimizer.set_players_with_same_position({"RB": 1})
            Exclude=['Noah Brown','Van Jefferson','Noah Fant','Jauan Jennings']
            # Exclusions(Exclude,optimizer)
            Include=['Justin Fields','Joe Mixon','NE ']
            Locks(Include,optimizer)
            # # optimizer.set_projected_ownership(12,20)
            optimizer.add_stack(PositionsStack(["QB"]))
            # optimizer.player_pool.exclude_teams(['DET','DAL'])
            # optimizer.set_fantasy_points_strategy(RandomFantasyPointsStrategy(0.5,1.2))
            # optimizer.add_stack(PositionsStack(["RB"],))
            # optimizer.add_stack(PositionsStack(["RB"],for_teams=rb_teams))
    
    
    
            optimizer.restrict_positions_for_opposing_team(
                ["DST"], ["QB", "RB", "WR", "TE"]
            )
    
            # optimizer.force_positions_for_opposing_team(("QB", "WR"))
            # Execute Optimizer
            LineupResults = []
            for lineup in optimizer.optimize(n=10):
                lineup_frame=print_lineups(lineup, proj)
                LineupResults.append(
                    lineup.actual_fantasy_points_per_game_projection
                )
                AllLineupResults.append(
                    lineup.actual_fantasy_points_per_game_projection
                )
                if contestResults is not False:
                    Standings=pd.concat([Standings,lineup_frame])
            if contestResults is not False:
                Standings.sort_values(by='Points',ascending=False,inplace=True)
                Standings=Standings[0:-10].reset_index(drop=True)
                Standings=pd.concat([Standings,Prizes.Prize.to_frame()],axis=1)
                CashWon=Standings[Standings.User=='rwmegnia'].Prize.sum()
                total_cash+=CashWon
                total_contests+=1
            else:
                CashWon=np.nan
            print(f'Average Points: {np.mean(LineupResults)}')
            print(f'Highest Score: {np.max(LineupResults)}')
            print(f'Prizes: {CashWon}$')
            print(np.mean(AllLineupResults))
            print(np.max(AllLineupResults))
            pf5=pf5.append(pd.DataFrame({'method':[method],
                                    'AvgScore':np.mean(LineupResults),
                                   'MaxScore':np.max(LineupResults),
                                   'cashWon':[CashWon],
                                   'week':[week]}))
            proj=proj[proj.RG_projection!=0]
    
        #     break
        # break
    
    # Single Stack
    # pf - no tuning
    # pf1 - qb/wr stack

    # pf2 - rb flex no stack 
    #                cashWon  AvgScore  MaxScore
    # method                                   
    # wProjection      1028   137.552     200.2
    # FP_Proj2          973   133.230     204.1
    # RG_projection     842   138.854     195.2
    # FP_Proj           756   136.478     206.3
    # TopDown           726   128.112     177.3
    # Projection        556   134.058     182.7
    # ML                430   121.674     181.1
    # Stochastic        188   121.652     175.2
    # DC_proj            98   112.926     149.9
    # pf3 - qb/wr stack rb flex, select teams




