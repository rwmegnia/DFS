#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 09:18:56 2022

@author: robertmegnia
"""


def getDKPts(df):
    df.loc[df.pts >= 10, "doublePoints"] = 1
    df.loc[df.ast >= 10, "doubleAssists"] = 1
    df.loc[df.stl >= 10, "doubleSteals"] = 1
    df.loc[df.reb >= 10, "doubleRebounds"] = 1
    df.loc[df.blk >= 10, "doubleBlocks"] = 1
    df["doubleStats"] = df[
        [
            "doublePoints",
            "doubleAssists",
            "doubleSteals",
            "doubleRebounds",
            "doubleBlocks",
        ]
    ].sum(axis=1)
    df.loc[df.doubleStats >= 2, "doubleDouble"] = 1
    df.loc[df.doubleStats > 2, "tripleDouble"] = 1
    df.fillna(0, inplace=True)
    df["DKPts"] = (
        df.pts
        + (df.fg3m * 0.5)
        + (df.reb * 1.25)
        + (df.ast * 1.5)
        + (df.stl * 2)
        + df.blk * 2
        - (df.to * 0.5)
        + (df.doubleDouble * 1.5)
        + (df.tripleDouble * 3)
    )
    return df


def getFDPts(df):
    df["FDPts"] = (
        df.PTS
        + (df.AST * 1.5)
        + (df.REB * 1.2)
        + (df.STL * 3)
        + (df.BLK * 3)
        - (df.TOV)
    )
    return df.FDPts
