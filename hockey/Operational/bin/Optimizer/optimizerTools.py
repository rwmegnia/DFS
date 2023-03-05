#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 20:38:27 2022

@author: robertmegnia
"""
from pydfs_lineup_optimizer.player import GameInfo, Player


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
            positions=[x[position]],
            team=x.team,
            salary=x.Salary,
            fppg=x[projection],
            projected_ownership=x.ownership_proj / 10,
            roster_order=x.line,
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
            proj.loc[(proj.full_name == Name) & (proj[position] == Position), "DKPts"]
            .values[0]
            .round(1)
        )
        Total += score
        Salary = player.salary
        Ownership += player.projected_ownership
        print(
            "%20s %10s %10s %10s %10s %10s"
            % (Name, Team, Position, str(Points), str(score), str(Salary))
        )
    Proj = lineup.fantasy_points_projection
    lineup.actual_fantasy_points_per_game_projection = Total
    print(
        f"Lineup: \nProjected Points: {Proj}\nActual Points: {Total}\nOwnership: {Ownership}\n"
    )
