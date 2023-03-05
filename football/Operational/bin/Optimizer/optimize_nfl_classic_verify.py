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

warnings.simplefilter("ignore")
basedir = os.path.dirname(os.path.abspath(__file__))
etcdir = f"{basedir}/../../etc"
datadir = f"{basedir}/../../data"
# from getConsensusRank import getConsensusRanking

# Read in Projections data frame and create player pool
Slate = "Main"
season = 2020
week = 1
AllLineupResults = []
method = "wProjection"
pre_bc_frames = []
post_bc_frames = []
team_frames=[]
fp = pd.read_csv(f"{datadir}/FantasyPros/2020_2021_FantasyPros_Projections.csv")
leverage_plays = []
total_cash=0
total_contests=0
for season in range(2021,2022):
    for week in range(1, 19):
        if (week == 18) & (season == 2020):
            continue
        # week = np.random.choice(range(1, 18))
        # season = np.random.choice(range(2021, 2022))
        # season = 2021
        proj = pd.read_csv(
            f"{datadir}/Projections/{season}/All/{season}_Week{week}_Projections.csv"
        )

        game_date=proj.game_date.unique()[0][0:10]
        try:
            files=os.listdir(f'/Volumes/XDrive/DFS/contestResults/NFL/{game_date}/')
            for file in files:
                if 'details' in file:
                    contestResults=pd.read_csv(f'/Volumes/XDrive/DFS/contestResults/NFL/{game_date}/{file}')
                    Standings=contestResults['Points'].to_frame()
                    Prizes=contestResults[['Rank','Prize']]
                    # Prizes['Rank']=Prizes.Rank.rank(method='min')
                    break
                else:
                    contestResults=False
        except Exception as error:
            print(error)
            contestResults=False
            pass
        proj.loc[proj.position == "DST", "full_name"] = proj.loc[
            proj.position == "DST", "full_name"
        ].apply(lambda x: x.split(" ")[0].upper())
        proj.rename({"game_id_y": "game_id"}, axis=1, inplace=True)
        proj["Ownership"] = proj.Ownership / 100
        proj.RosterPercent /= 100
        proj = proj[proj.Slate == Slate]
        proj = proj.merge(
            fp[["gsis_id", "FP_Proj", "season", "week"]],
            on=["gsis_id", "season", "week"],
            how="left",
        )


        proj=proj[proj.FP_Proj.isna()==False]
        proj["Projection"] = proj[["Projection", "FP_Proj"]].mean(axis=1)
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
        proj["game_date"] = proj["game_date"].apply(
            lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").replace(
                tzinfo=timezone(get_timezone())
            )
        )
        proj.loc[proj.Ceiling.isna() == True, "Ceiling"] = proj.loc[
            proj.Ceiling.isna() == True, "Projection"
        ]
        proj = proj[proj.Projection >= 5]
        proj.fillna(0, inplace=True)
        teams = (
                    proj[(proj.position.isin(['QB','WR','TE']))&(proj.depth_team<=2)].groupby(["game_id"])
                    .agg(
                        {
                            "Projection": np.sum,
                            "receiving_DKPts_share":np.sum,
                            "Ceiling": np.sum,
                            "proj_team_score": np.mean,
                            "team_DKPts":np.mean,
                            "team_receiving_DKPts":np.mean,
                            'team_receiving_DKPts_allowed':np.mean,
                            "total_line": np.mean,
                            "game_location": "first",
                            "opp": "first",
                            "Ownership": np.product,
                            "Leverage":np.sum,
                            'UpsideProb':np.sum,
                            'DKPts':np.sum
                        }
                    ))
        n_teams=5
        teams['score']=(teams.UpsideProb*teams.team_receiving_DKPts*teams.receiving_DKPts_share)
        pass_teams=teams.sort_values(by='score',ascending=False).head(5).index.to_list()
        #
        rb_teams1=proj[proj.position=='RB'].sort_values(by='Ownership',ascending=False).head(10).sort_values(by='spread_line').head(5).team.to_list()  
        rb_teams2=proj[proj.position=='RB'].sort_values(by='salary',ascending=False).head(5).team.to_list()      
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
        players = proj.apply(
            lambda x: Player(
                player_id=x.gsis_id,
                first_name=x.full_name.split(" ")[0],
                last_name=" ".join(x.full_name.split(" ")[1::]),
                positions=[x.position],
                team=x.team,
                salary=x.salary,
                fppg=x.Ownership*100,
                projected_ownership=x.Ownership,
                game_info=game_info[x.game_id],
            ),
            axis=1,
        )

        # Create optimizer
        optimizer = get_optimizer(Site.DRAFTKINGS, Sport.FOOTBALL)
        optimizer.player_pool.load_players(players.to_list())
        # optimizer.set_min_salary_cap(50000)
        optimizer.set_players_with_same_position({"RB": 1})
        # if (season ==2021)&(week==16):
        #     Include=['Joe Burrow']
        #     Locks(Include,optimizer)
        # Exclude=['Curtis Samuel']
        # Exclusions(Exclude,optimizer)

        # optimizer.set_projected_ownership(11,14)
        # optimizer.add_stack(PositionsStack(["QB",("WR",'TE')],for_teams=pass_teams))
        # optimizer.add_stack(PositionsStack(["RB"],for_teams=rb_teams2))
        # optimizer.add_stack(PositionsStack(["RB"],for_teams=rb_teams1))
        optimizer.set_deviation(0,0.02)

        optimizer.restrict_positions_for_opposing_team(
            ["DST"], ["QB", "RB", "WR", "TE"]
        )

        # optimizer.force_positions_for_opposing_team(("QB", "WR"))
        # Execute Optimizer
        LineupResults = []
        for lineup in optimizer.optimize(n=10,randomness=True):
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
    #     break
    # break

# Single Stack
# FP_Proj = 138.34 PPL and 162$/contest
# Projection = 134.69 PPL and 143$/contest
#
# Single Stack With Limited Teams
#
# Projection = 132 PPL and 137.7$/contest
# FP_Proj = 134.39PPL and 164$/contest
#
# Single Stack w/ WR FLEX
# FP_Proj = 138.34 PPL and 162$/contest
# Projection = 134.69 PPL and 143$/contest
#
# Single Stack With Limited Teams and WR FLEX
#
# Projection = 132.4 PPL and 169.76$/contest 
# FP_Proj = 135.47PPL and 184$/contest
#
# Single Stack With Limited Teams and WR FLEX and LowRisk RBs
#
# Projection = 133.3 PPL and 233.29$/contest 
# FP_Proj = 134.06PPL and 600.59 $/contest




