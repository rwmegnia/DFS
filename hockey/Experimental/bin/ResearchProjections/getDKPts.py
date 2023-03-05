# -*- coding: utf-8 -*-
import requests
import pandas as pd


def getSkaterDKPts(df):
    """
    If shootout goal data can be found, add an additional 1.5 points
    """
    df["points"] = df.goals + df.assists
    df["sh_points"] = df.shortHandedGoals + df.shortHandedAssists
    df.loc[df.goals >= 3, "hatrick"] = 1
    df.hatrick.fillna(0, inplace=True)
    df.loc[df.shots >= 5, "five_shots"] = 1
    df.five_shots.fillna(0, inplace=True)
    df.loc[df.blocked >= 3, "three_blocks"] = 1
    df.three_blocks.fillna(0, inplace=True)
    df.loc[df.points >= 3, "three_points"] = 1
    df.three_points.fillna(0, inplace=True)
    df["DKPts"] = (
        (df.goals * 8.5)
        + (df.assists * 5)
        + (df.shots * 1.5)
        + (df.blocked * 1.3)
        + (df.sh_points * 2)
        + (df.hatrick * 3)
        + (df.five_shots * 3)
        + (df.three_blocks * 3)
        + (df.three_points * 3)
    )
    return df.DKPts


def getGoalieDKPts(df):
    """
Goalies

    """
    df["points"] = df.goals + df.assists
    df.loc[df.points >= 3, "three_points"] = 1
    df.three_points.fillna(0, inplace=True)
    df.loc[df.decision == "W", "decision"] = 6
    df.loc[df.decision == "OTL", "decision"] = 2
    df.loc[df.decision == "L", "decision"] = 0
    df.decision.fillna(0, inplace=True)
    df.loc[(df.goals_allowed == 0) & (df.timeOnIce >= 60), "shutout"] = 1
    df.shutout.fillna(0, inplace=True)
    df.loc[df.saves >= 35, "saves_35"] = 1
    df.saves_35.fillna(0, inplace=True)
    df["DKPts"] = (
        (df.decision)
        + (df.saves * 0.7)
        + (df.goals_allowed * -3.5)
        + (df.shutout * 4)
        + (df.three_points * 3)
        + (df.goals * 10)
        + (df.assists * 5)
    )
    return df.DKPts
