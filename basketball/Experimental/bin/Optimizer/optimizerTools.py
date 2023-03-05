#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 20:38:27 2022

@author: robertmegnia
"""
from pydfs_lineup_optimizer.player import GameInfo, Player
import numpy as np
from datetime import datetime
import pandas as pd
import os

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
    game_ids = proj.groupby("Game Info", as_index=False).first()
    game_ids['away_team']=game_ids.apply(lambda x: x['Game Info'].split('@')[0],axis=1)
    game_ids['home_team']=game_ids.apply(lambda x: x['Game Info'].split('@')[1].split(' ')[0],axis=1)
    game_ids['game_date']=game_ids.apply(lambda x: ' '.join(x['Game Info'].split(' ')[1::]),axis=1)
    game_ids['game_date']=game_ids.game_date.apply(lambda x: datetime.strptime(x,'%m/%d/%Y %I:%M%p ET'))
    for row in game_ids.iterrows():
        row = row[1]
        game_info[row['Game Info']] = GameInfo(row.home_team, row.away_team, starts_at=row.game_time)

        # Create list of pydfs-lineup-optimizer Player objects
    players = proj.apply(
        lambda x: Player(
            player_id=int(x.ID),
            first_name=x.full_name.split(" ")[0],
            last_name=" ".join(x.full_name.split(" ")[1::]),
            positions=x[position].split('/'),
            team=x.team_abbreviation,
            salary=x.Salary,
            fppg=x[projection],
            min_exposure=x.min_exposure,
            max_exposure=x.max_exposure,
            # fppg_floor=x.Floor,
            # fppg_ceil=x.Ceiling,
            projected_ownership=x.ownership_proj/100,
            game_info=game_info[x['Game Info']],
            is_confirmed_starter=x.starter,
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
    for player in lineup.players:
        Name = player._player.full_name
        Position = player.positions
        Team = "@".join(
            [player._player.game_info.away_team, player._player.game_info.home_team]
        )
        Points = round(player.fppg, 1)
        Id = player.id
        score = (
            proj.loc[(proj.full_name == Name), "dkpts"]
            .values[0]
            .round(1)
        )
        Total += score
        Salary = player.salary
        Ownership += (player.projected_ownership*100)
        print(
            "%20s %10s %10s %10s %10s %10s"
            % (Name, Team, Position, str(Points), str(score), str(Salary))
        )
    Proj = lineup.fantasy_points_projection
    lineup.actual_fantasy_points_per_game_projection = Total
    print(
        f"Lineup: \nProjected Points: {Proj}\nActual Points: {Total}\nOwnership: {Ownership/8}\n"
    )


def mergeSwapProjectedPoints(player, df,method):

    try:
        proj = df[df.ID == int(player.id)][method].values[0]
        player.fppg = proj
        if player.id==25506871:
            player.min_exposure=0.10
            player.max_exposure=0.10
        else:
            player.min_exposure=0
            player.max_exposure=1
        print(proj)
        return player

    except:
        proj = 0
        player.fppg = proj
        print(proj)
        return player

def selectShowdownSlate(salary_dir):
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
    slate_salaries.rename({'team':'team_abbreviation'},axis=1,inplace=True)
    return slate_salaries