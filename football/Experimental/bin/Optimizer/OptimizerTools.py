#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 10:50:51 2022

@author: robertmegnia
"""
import pandas as pd
import os
import numpy as np
from pydfs_lineup_optimizer.tz import get_timezone
from pytz import timezone
from datetime import datetime

basedir = os.path.dirname(os.path.abspath(__file__))
etcdir = f"{basedir}/../../etc"
datadir = f"{basedir}/../../data"


def mse(a, b):
    x = round(np.mean((a - b) ** 2), 2)
    return x


def mae(a, b):
    rms = round(np.mean(np.abs(a - b)), 2)
    return rms


def selectLowRiskRBs(df):
    rbs = df[(df.position == "RB") & (df.Leverage < 1)]
    #
    high_usage = rbs.sort_values(by="rush_value", ascending=False).head(5)
    high_owned = rbs.sort_values(by="Ownership", ascending=False).head(5)
    rbs = rbs[rbs.index.isin(pd.concat([high_usage, high_owned]).index)]
    df.drop(
        df[(df.position == "RB") & (~df.index.isin(rbs.index))].index,
        inplace=True,
    )
    return df


def mergeSwapProjectedPoints(player, df, projection):

    try:
        proj = df[df.ID == int(player.id)][projection].values[0]
        player.fppg = proj
        print(proj)
        return player

    except:
        pass


def selectLowRiskWRs(df):
    rbs = df[(df.position == "RB") & (df.Leverage > 0)]
    #
    high_usage = (
        rbs.sort_values(by="Usage", ascending=False).iloc[0:10].sample(5)
    )
    high_owned = (
        rbs.sort_values(by="RosterPercent", ascending=False)
        .iloc[0:10]
        .sample(5)
    )
    rbs = rbs[rbs.index.isin(pd.concat([high_usage, high_owned]).index)]
    df.drop(
        df[(df.position == "RB") & (~df.index.isin(rbs.index))].index,
        inplace=True,
    )
    return df


def Locks(Players, optimizer):
    for p in Players:
        player = optimizer.player_pool.get_player_by_name(p)
        optimizer.player_pool.lock_player(player)


def Exclusions(Players, optimizer):
    for p in Players:
        try:
            player = optimizer.player_pool.get_player_by_name(p)
            optimizer.player_pool.remove_player(player)
        except:
            pass


def BiasCorrection(df, week, season):
    db = pd.read_csv(
        f"{datadir}/Projections/2020_2021_MyProj_vs_FantasyPros2.csv"
    )
    db["error"] = db.apply(lambda x: x.DKPts - x.Projection, axis=1)
    game_date = db[(db.season == season) & (db.week == week)].game_date.min()
    fp = db[(db.week == week) & (db.season == season)]
    db = db[db.game_date < game_date]
    pos_frames = []
    for pos in df.position.unique():
        print(pos)
        pos_frame = df[df.position == pos]
        bias = db[db.position == pos].groupby("ConsensusRank").mean().error
        bias.name = "bias"
        bias_df = bias.to_frame().reset_index()
        pos_frame1 = pos_frame[
            (~pos_frame.ConsensusRank.isin(bias_df.ConsensusRank))
            | (pos_frame.ConsensusRank >= 25)
        ]
        pos_frame2 = pos_frame[
            (pos_frame.ConsensusRank.isin(bias_df.ConsensusRank))
            & (pos_frame.ConsensusRank <= 25)
        ]
        pos_frame2["Projection"] = pos_frame2.apply(
            lambda x: x.Projection
            + bias_df[bias_df.ConsensusRank == x.ConsensusRank].bias.values[0]
            if x.ConsensusRank in bias_df.ConsensusRank
            else x.Projection,
            axis=1,
        )
        pos_frames.append(pos_frame1)
        pos_frames.append(pos_frame2)
    df = pd.concat(pos_frames)
    # df=df.merge(fp[['gsis_id','FP_Proj']],on='gsis_id',how='left')
    # df['Projection']=df[['Projection','FP_Proj']].mean(axis=1)
    return df


def print_lineups(lineup, proj):
    Total = 0
    Ownership = 0
    SalaryUsed = 0

    for player in lineup.players:
        Name = player._player.full_name
        Position = player.positions[0]
        if Position == "DST":
            Name = Name.split(" ")[0]
        Team = (
            player._player.game_info.away_team
            + "@"
            + player._player.game_info.home_team
        )
        Points = round(player.fppg, 1)
        Id = player.id
        score = proj.loc[proj.ID == Id, "DKPts"].values[0].round(1)
        Total += score
        Salary = player.salary
        SalaryUsed += Salary
        Ownership += player.projected_ownership * 100
        print(
            "%20s %10s %10s %10s %10s %10s"
            % (Name, Team, Position, str(Points), str(score), str(Salary))
        )
    Proj = lineup.fantasy_points_projection
    lineup.actual_fantasy_points_per_game_projection = Total
    print(
        f"Lineup: \nProjected Points: {Proj}\nActual Points: {Total}\nOwnership: {Ownership}\nSalary: {SalaryUsed}"
    )
    lineup_frame = pd.DataFrame({"Points": [Total], "User": ["rwmegnia"]})
    return lineup_frame


def filterProjections(proj, slate_salaries):
    # proj.loc[proj.injury_designation == "Questionable", "is_injured"] = True
    # proj.loc[proj.injury_designation != "Questionable", "is_injured"] = False
    # proj = proj[proj.FP_Proj2.isna() == False]
    proj.drop(["ID",'salary'], axis=1, inplace=True, errors="ignore")
    proj = proj.merge(
        slate_salaries[
            ["RotoName", "team", "position", "Roster Position", "ID",'salary']
        ],
        on=["RotoName", "team", "position", "Roster Position"],
        how="left",
    )
    proj = proj[proj.ID.isna() == False]
    # proj = selectLowRiskRBs(proj)
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
    proj["game_id"] = proj.home_team + proj.away_team
    proj["game_date"] = proj["game_date"].apply(
        lambda x: datetime.strptime(x, "%A %y-%m-%d %H:%M PM").replace(
            tzinfo=timezone(get_timezone())
        )
    )
    proj.loc[proj.Ceiling.isna() == True, "Ceiling"] = proj.loc[
        proj.Ceiling.isna() == True, "Projection"
    ]
    proj.fillna(0, inplace=True)
    proj.loc[proj.position == "DST", "full_name"] = proj.loc[
        proj.position == "DST", "full_name"
    ].apply(lambda x: x.split(" ")[0].upper())
    return proj


def getPassingStackTeams(proj):
    teams = (
        proj[(proj.position.isin(["QB", "WR"])) & (proj.depth_team <= 3)]
        .groupby(["team"])
        .agg(
            {
                "Projection": np.sum,
                "proj_receiving_DKPts_share": np.sum,
                "Ceiling": np.sum,
                "proj_team_score": np.mean,
                "proj_team_fpts": np.mean,
                "proj_team_receiving_fpts": np.mean,
                "total_line": np.mean,
                "game_location": "first",
                "opp": "first",
                "AvgOwnership": np.product,
                "Leverage": np.sum,
                "UpsideProb": np.sum,
            }
        )
    )
    n_teams = 5
    teams["score"] = (
        teams.UpsideProb
        * teams.proj_team_receiving_fpts
        * teams.proj_receiving_DKPts_share
    )
    teams = (
        teams.sort_values(by="score", ascending=False).head(10).index.to_list()
    )
    return teams


def getRunningBackTeams(proj):
    teams = (
        proj[proj.position == "RB"]
        .sort_values(by="AvgOwnership", ascending=False)
        .head(10)
        .sort_values(by="HVU", ascending=False)
        .head(5)
        .team.to_list()
    )
    return teams

def getSlateSalaries(salary_dir,game_date):
    slates = [
        file
        for file in os.listdir(f"{salary_dir}")
        if (".csv" in file) & (f"{game_date}_salaries.csv" not in file)
    ]

    n = 0
    for file in slates:
        if len(file.split("_")) > 2:
            print(n + 1, file)
            n += 1
    if len(slates) == 1:
        slate = 1
    else:
        slate = input("Select Slate by number (1,2,3, etc...) ")
    slate = slates[int(slate) - 1]
    slate_salaries = pd.read_csv(f"{salary_dir}/{slate}")
    slate_salaries.loc[
        slate_salaries.position == "DST", "RotoName"
    ] = slate_salaries.loc[slate_salaries.position == "DST", "team"]
    return slate_salaries
