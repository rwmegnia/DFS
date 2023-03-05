#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 09:18:56 2022

@author: robertmegnia
"""


def getDKPts(df):
    df.loc[df.PTS >= 10, "doublePoints"] = 1
    df.loc[df.AST >= 10, "doubleAssists"] = 1
    df.loc[df.STL >= 10, "doubleSteals"] = 1
    df.loc[df.REB >= 10, "doubleRebounds"] = 1
    df.loc[df.BLK >= 10, "doubleBlocks"] = 1
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
        df.PTS
        + (df.FG3M * 0.5)
        + (df.REB * 1.25)
        + (df.AST * 1.5)
        + (df.STL * 2)
        + df.BLK * 2
        - (df.TO * 0.5)
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
