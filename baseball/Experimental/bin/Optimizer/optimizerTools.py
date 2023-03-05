#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 20:38:27 2022

@author: robertmegnia
"""
from pydfs_lineup_optimizer.player import GameInfo, Player
import numpy as np
import pandas as pd

def Locks(Players, optimizer):
    for p in Players:
        player = optimizer.player_pool.get_player_by_name(p)
        optimizer.player_pool.lock_player(player)


def Exclusions(Players, optimizer):
    for p in Players:
        try:
            player = optimizer.player_pool.get_player_by_name(p)
            if player is not None:
                optimizer.player_pool.remove_player(player)
        except:
            continue


def loadPlayers(proj, projection, contestType):
    if contestType == "Showdown":
        position = "Roster Position"
    else:
        position = "position"
    # Get Individual Game Information
    game_info = {}
    game_ids = proj.groupby("game_id", as_index=False).first()
    for row in game_ids.iterrows():
        row = row[1]
        game_info[row.game_id] = GameInfo(row.home_team, row.away_team, row.game_date)

        # Create list of pydfs-lineup-optimizer Player objects
    players = proj.apply(
        lambda x: Player(
            player_id=int(x.ID),
            first_name=x.full_name.split(" ")[0],
            last_name=" ".join(x.full_name.split(" ")[1::]),
            positions=x[position].split('/'),
            team=x.team,
            salary=x.Salary,
            fppg=x[projection],
            fppg_ceil=x.Ceiling,
            fppg_floor=x.Floor,
            projected_ownership=x.ownership_proj / 100,
            roster_order=x.order,
            game_info=game_info[x.game_id],
        ),
        axis=1,
    )
    return players


def print_lineups(lineup, proj):
    if "Roster Position" not in proj.columns:
        position = "position"
    elif "CPT" in proj["Roster Position"].unique():
        position = "Roster Position"
    else:
        position = "position"
    Total = 0
    Ownership = 0
    TotalSalary = 0
    for player in lineup.players:
        Name = player._player.full_name
        Position = player.positions[0]
        if Position == "DST":
            Name = Name.split(" ")[0]
        Team = "@".join(
            [player._player.game_info.away_team, player._player.game_info.home_team]
        )
        Points = round(player.fppg, 1)
        Id = player.id
        score = (
            proj.loc[(proj.ID==Id), "DKPts"]
            .values[0]
            .round(1)
        )
        Total += score
        Salary = player.salary
        Ownership += player.projected_ownership*10
        TotalSalary+=Salary
        print(
            "%20s %10s %10s %10s %10s %10s"
            % (Name, Team, Position, str(Points), str(score), str(Salary))
        )
    Proj = lineup.fantasy_points_projection
    lineup.actual_fantasy_points_per_game_projection = Total
    print(
        f"Lineup: \nProjected Points: {Proj}\nActual Points: {Total}\nSalary Used: {TotalSalary}\nOwnership: {Ownership}\n"
    )


def mergeSwapProjectedPoints(player, df,projection):

    try:
        proj = df[df.ID == int(player.id)][projection].values[0]
        player.fppg = proj
        print(proj)
        return player

    except:
        pass


def getMaxLineupScore(lineup,proj,pitcher_db,batter_db,contest='Classic',research=False):
    if contest=='Classic':
        lineup=lineup[['P','P.1','1B','2B','3B','SS','OF','OF.1','OF.2']]
        lineup.name='Player'
        lineup=lineup.to_frame()
        lineup['ID']=lineup.Player.apply(lambda x: int(x.split('(')[1].split(')')[0]))   
        lineup.reset_index(inplace=True)
        lineup.rename({'index':'pos'},axis=1,inplace=True)
        lineup=lineup.merge(proj,on='ID',how='left')
        sim_frame=pd.DataFrame({},columns=['P','P.1','C','1B','2B','3B','SS','OF','OF.1','OF.2'])
        sim_frame=simLineup(lineup,sim_frame,pitcher_db,batter_db,research)
        return sim_frame.sum(axis=1).max()
    else:
        lineup=lineup[['CPT','UTIL','UTIL.1','UTIL.2','UTIL.3','UTIL.4']]
        lineup.name='Player'
        lineup=lineup.to_frame()
        lineup['ID']=lineup.Player.apply(lambda x: int(x.split('(')[1].split(')')[0]))   
        lineup.reset_index(inplace=True)
        lineup.rename({'index':'pos'},axis=1,inplace=True)
        lineup=lineup.merge(proj,on='ID',how='left')
        sim_frame=pd.DataFrame({},columns=['CPT','UTIL','UTIL.1','UTIL.2','UTIL.3','UTIL.4'])
        sim_frame=simLineup(lineup,sim_frame,pitcher_db,batter_db,research)
        sim_frame['CPT']*=1.5
        return sim_frame.sum(axis=1).max()

def simLineup(lineup,sim_frame,pitcher_db,batter_db,research=False):
    for player in lineup.iterrows():
        player_id=player[1]['player_id']
        position=player[1]['pos']
        if player_id in pitcher_db.player_id.to_list():
            database=pitcher_db
        else:
            database=batter_db
        player_db=database[database.player_id==player_id][-20:]
        mean = player_db.DKPts.mean()
        std = player_db.DKPts.std()
        stats = np.random.normal(loc=mean, scale=std, size=10000)
        sim_frame[position]=stats
    return sim_frame
