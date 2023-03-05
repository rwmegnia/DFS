#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 11:14:46 2022

@author: robertmegnia
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 00:19:28 2022

@author: robertmegnia
"""
import pandas as pd


def getPMM(df, game_date):
    dfs = []
    methods = ["BR","EN", "NN", "RF", "GB","Tweedie"]
    df["ML"] = df[methods].mean(axis=1)
    methods = [ "BR",
                "EN",
                "NN",
                "RF",
                "GB",
                "Tweedie",
                "ML",
            ]
    df["Proj"] = df[methods].mean(axis=1)
    members = pd.concat([df[m] for m in methods])
    members.sort_values(ascending=False, inplace=True)
    members.name = "PMM"
    members = members.to_frame()
    df.sort_values(by="Proj", ascending=False, inplace=True)
    members = members[len(methods) - 1 :: len(methods)]
    df["PMM"] = members.values
    dfs.append(df)
    df = pd.concat(dfs)
    df.reset_index(drop=True, inplace=True)
    methods = ["BR","EN", "NN", "RF", "GB","Tweedie"]
    df["ML"] = df[["PMM"] + methods].mean(axis=1)
    df.drop("Proj", axis=1, inplace=True)
    return df

def getPMMScaled(df, game_date):
    dfs = []
    methods = [ "Stochastic",
               "RG_projection",
               "PMM",
               "ML",
               "BR",
               "EN",
               "NN",
               "RF",
               "GB",
               "Tweedie",
               "Projection",
            ]
    members = pd.concat([df[m] for m in methods])
    members.sort_values(ascending=False, inplace=True)
    members.name = "PMM"
    members = members.to_frame()
    df.sort_values(by="ScaledProj", ascending=False, inplace=True)
    members = members[len(methods) - 1 :: len(methods)]
    df["PMM"] = members.values
    dfs.append(df)
    df = pd.concat(dfs)
    df.reset_index(drop=True, inplace=True)
    return df
