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
    methods = ["EN", "RF", "GB","Stochastic","RG_projection","DG_Proj","TD_Proj2","Median"]
    ml_methods=["EN","RF","GB"]
    df["ML"] = df[ml_methods].mean(axis=1)
    methods = [
                "EN",
                "RF",
                "GB",
                "ML",
                "Stochastic",
                "RG_projection",
                "DG_Proj",
                "TD_Proj2",
                "Median",
            ]
    df["Proj"] = df[methods].mean(axis=1)
    members = pd.concat([df[m] for m in methods])
    members.sort_values(ascending=False, inplace=True)
    members.name = "PMM"
    members = members.to_frame()
    df.sort_values(by="Proj", ascending=False, inplace=True)
    members = members[len(methods) - 1 :: len(methods)]
    df["PMM"] = members.values
    df.reset_index(drop=True, inplace=True)
    df.drop("Proj", axis=1, inplace=True)
    return df
