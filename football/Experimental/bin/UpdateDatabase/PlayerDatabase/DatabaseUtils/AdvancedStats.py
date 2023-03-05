#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 03:56:58 2022

@author: robertmegnia

Functions for Advanced Stats
"""


def getValuedTargets(rec_df, df):
    # Get targets behind opponent 20 yard line
    target_g20 = (
        df.loc[
            (df["pass_attempt"] == 1) & (df["gsis_id"].notnull()),
            ["gsis_id", "week", "yardline_100"],
        ]
        .assign(target_g20=lambda x: x.yardline_100 > 20)
        .groupby(["gsis_id", "week"])["target_g20"]
        .sum()
        .sort_values(ascending=False)
        .to_frame()
    )
    # Get targets from 20 yard line and in
    target_20 = (
        df.loc[
            (df["pass_attempt"] == 1) & (df["gsis_id"].notnull()),
            ["gsis_id", "week", "yardline_100"],
        ]
        .assign(
            target_20=lambda x: (x.yardline_100 <= 20) & (x.yardline_100 > 10)
        )
        .groupby(["gsis_id", "week"])["target_20"]
        .sum()
        .sort_values(ascending=False)
        .to_frame()
    )
    # Get targets from 10 yard line and in
    target_10 = (
        df.loc[
            (df["pass_attempt"] == 1) & (df["gsis_id"].notnull()),
            ["gsis_id", "week", "yardline_100"],
        ]
        .assign(
            target_10=lambda x: (x.yardline_100 <= 10) & (x.yardline_100 > 5)
        )
        .groupby(["gsis_id", "week"])["target_10"]
        .sum()
        .sort_values(ascending=False)
        .to_frame()
    )
    # Get targets from 5 yard line and in
    target_5 = (
        df.loc[
            (df["pass_attempt"] == 1) & (df["gsis_id"].notnull()),
            ["gsis_id", "week", "yardline_100"],
        ]
        .assign(target_5=lambda x: x.yardline_100 <= 5)
        .groupby(["gsis_id", "week"])["target_5"]
        .sum()
        .sort_values(ascending=False)
        .to_frame()
    )
    hvu = target_g20.join(target_20).join(target_10).join(target_5)
    # factors = 1.5, 2.,2.75
    hvu["target_value"] = hvu.apply(
        lambda x: x.target_g20
        + (x.target_20 * 1.5)
        + (x.target_10 * 2.0)
        + (x.target_5 * 2.75),
        axis=1,
    )
    rec_df = rec_df.merge(hvu["target_value"], on=["week", "gsis_id"])
    return rec_df


def getValuedRushes(rush_df, df):
    # Get rushs behind opponent 10 yard line
    rush_g10 = (
        df.loc[
            (df["rush_attempt"] == 1) & (df["gsis_id"].notnull()),
            ["gsis_id", "week", "yardline_100"],
        ]
        .assign(rush_g10=lambda x: x.yardline_100 > 10)
        .groupby(["gsis_id", "week"])["rush_g10"]
        .sum()
        .sort_values(ascending=False)
        .to_frame()
    )
    # Get rushs from 10 yard line and in
    rush_10 = (
        df.loc[
            (df["rush_attempt"] == 1) & (df["gsis_id"].notnull()),
            ["gsis_id", "week", "yardline_100"],
        ]
        .assign(rush_10=lambda x: (x.yardline_100 <= 10) & (x.yardline_100 > 5))
        .groupby(["gsis_id", "week"])["rush_10"]
        .sum()
        .sort_values(ascending=False)
        .to_frame()
    )
    # Get rushs from 5 yard line and in
    rush_5 = (
        df.loc[
            (df["rush_attempt"] == 1) & (df["gsis_id"].notnull()),
            ["gsis_id", "week", "yardline_100"],
        ]
        .assign(rush_5=lambda x: x.yardline_100 <= 5)
        .groupby(["gsis_id", "week"])["rush_5"]
        .sum()
        .sort_values(ascending=False)
        .to_frame()
    )
    hvu = rush_g10.join(rush_10).join(rush_5)
    # factors = 2.8, 4,5.5
    hvu["rush_value"] = hvu.apply(
        lambda x: x.rush_g10 + (x.rush_10 * 2) + (x.rush_5 * 4),
        axis=1,
    )
    rush_df = rush_df.merge(hvu["rush_value"], on=["week", "gsis_id"])
    return rush_df


def getRecAdvancedStats(rec_df, df):
    # Target Shares and Air Yards Share
    rec_df = getAirTargetShares(rec_df, df)
    rec_df = getRedZoneLooks(rec_df, df)
    rec_df = getWOPR(rec_df)
    return rec_df


def getAirTargetShares(rec_df, df):
    team_stats = (
        df.loc[(df["pass_attempt"] == 1) & (df["receiver_player_id"].notnull())]
        .groupby(["game_id", "posteam"], as_index=False)[
            ["air_yards", "pass_attempt"]
        ]
        .sum()
    )
    rec_df = rec_df.merge(
        team_stats, on=["game_id", "posteam"], suffixes=("_player", "_team")
    )
    rec_df["target_share"] = (
        rec_df["pass_attempt_player"] / rec_df["pass_attempt_team"]
    )
    rec_df["air_yards_share"] = (
        rec_df["air_yards_player"] / rec_df["air_yards_team"]
    )
    rec_df.drop(["pass_attempt_team", "air_yards_team"], axis=1, inplace=True)
    rec_df.rename(
        {"pass_attempt_player": "targets", "air_yards_player": "air_yards"},
        axis=1,
        inplace=True,
    )
    return rec_df


def getRushShare(pos_df, df):
    team_stats = (
        df.loc[(df["rush_attempt"] == 1) & (df["rusher_player_id"].notnull())]
        .groupby(["game_id", "posteam"], as_index=False)[["rush_attempt"]]
        .sum()
    )
    pos_df = pos_df.merge(
        team_stats, on=["game_id", "posteam"], suffixes=("_player", "_team")
    )
    pos_df["rush_share"] = (
        pos_df["rush_attempt_player"] / pos_df["rush_attempt_team"]
    )
    pos_df.drop(["rush_attempt_team"], axis=1, inplace=True)
    pos_df.rename({"rush_attempt_player": "rush_attempt"}, axis=1, inplace=True)
    return pos_df


def getRedZoneLooks(rec_df, df):
    rz = (
        df.loc[
            (df["pass_attempt"] == 1) & (df["gsis_id"].notnull()),
            ["gsis_id", "week", "air_yards", "yardline_100"],
        ]
        .assign(rec_redzone_looks=lambda x: x.yardline_100 <= 20)
        .groupby(["gsis_id", "week"], as_index=False)["rec_redzone_looks"]
        .sum()
        .sort_values(by="rec_redzone_looks", ascending=False)
    )
    rec_df = rec_df.merge(rz, on=["week", "gsis_id"])
    return rec_df


def getRushRedZoneLooks(rush_df, df):
    rz = (
        df.loc[
            (df["rush_attempt"] == 1) & (df["gsis_id"].notnull()),
            ["gsis_id", "week", "air_yards", "yardline_100"],
        ]
        .assign(rush_redzone_looks=lambda x: x.yardline_100 <= 10)
        .groupby(["gsis_id", "week"], as_index=False)["rush_redzone_looks"]
        .sum()
        .sort_values(by="rush_redzone_looks", ascending=False)
    )
    rush_df = rush_df.merge(rz, on=["week", "gsis_id"])
    return rush_df


def getWOPR(rec_df):
    # Calculated Weighted Opportunity Rating (WOPR)
    rec_df["wopr"] = (
        rec_df["target_share"] * 1.5 + rec_df["air_yards_share"] * 0.7
    )
    return rec_df


def getPasserRating(cmp, td, pass_int, yds, att):
    a = ((cmp / att) - 0.3) * 5
    b = ((yds / att) - 3) * 0.25
    c = (td / att) * 20
    d = 2.375 - ((pass_int / att) * 25)
    terms = {"a": a, "b": b, "c": c, "d": d}
    top = 0
    for term in terms.keys():
        if terms[term] > 2.375:
            terms[term] = 2.375
        elif terms[term] < 0:
            terms[term] = 0
        top += terms[term]
    return (top / 6) * 100
