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
    pos_frames = []
    methods = ["BR", "EN", "NN", "RF", "GB", "Tweedie"]
    df["ML"] = df[methods].mean(axis=1)
    for pos in ["Forward", "Defenseman", "Goalie"]:
        if pos == "Defenseman":
            pos_frame = df[df.position_type == pos]
            methods = [
                "BR",
                "EN",
                "NN",
                "RF",
                "GB",
                "Tweedie",
                "Stochastic",
                "TD_Proj",
                "TD_Proj2",
                "ML",
            ]
            pos_frame["Proj"] = pos_frame[methods].mean(axis=1)
        elif pos == "Goalie":
            pos_frame = df[df.position_type == pos]
            methods = ["BR", "EN", "NN", "RF", "GB", "Tweedie", "Stochastic", "ML"]
            pos_frame["Proj"] = pos_frame[methods].mean(axis=1)
        else:
            methods = [
                "BR",
                "EN",
                "NN",
                "RF",
                "GB",
                "Tweedie",
                "Stochastic",
                "TD_Proj",
                "TD_Proj2",
                "ML",
            ]
            pos_frame = df[df.position_type == pos]
            pos_frame["Proj"] = pos_frame[methods].mean(axis=1)
        members = pd.concat([pos_frame[m] for m in methods])
        members.sort_values(ascending=False, inplace=True)
        members.name = "PMM"
        members = members.to_frame()
        pos_frame.sort_values(by="Proj", ascending=False, inplace=True)
        members = members[len(methods) - 1 :: len(methods)]
        pos_frame["PMM"] = members.values
        pos_frames.append(pos_frame)
    df = pd.concat(pos_frames)
    df.reset_index(drop=True, inplace=True)
    methods = ["BR", "EN", "NN", "RF", "GB", "Tweedie"]
    df["ML"] = df[["PMM"] + methods].mean(axis=1)
    df.drop("Proj", axis=1, inplace=True)
    return df
