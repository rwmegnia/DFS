#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 11:52:32 2022

@author: robertmegnia
"""
import unidecode


def reformatNames(df):
    df["first_name"] = df.player_name.apply(lambda x: x.split(" ")[0])
    # df["first_name"] = df.nickname
    df["last_name"] = df.player_name.apply(lambda x: " ".join(x.split(" ")[1::]))

    # Remove non-alpha numeric characters from first/last names.
    df["first_name"] = df.first_name.apply(
        lambda x: "".join(c for c in x if c.isalnum())
    )
    df["last_name"] = df.last_name.apply(lambda x: "".join(c for c in x if c.isalnum()))

    # Recreate full_name to fit format "Firstname Lastname" with no accents
    df["player_name"] = df.apply(lambda x: x.first_name + " " + x.last_name, axis=1)
    df["player_name"] = df.player_name.apply(lambda x: x.lower())
    df.drop(["first_name", "last_name"], axis=1, inplace=True)
    df["player_name"] = df.player_name.apply(
        lambda x: x.split(" ")[0][0].upper()
        + x.split(" ")[0][1::]
        + " "
        + x.split(" ")[-1][0].upper()
        + x.split(" ")[-1][1::]
    )
    df["player_name"] = df.player_name.apply(lambda x: unidecode.unidecode(x))

    # Create Column to match with RotoGrinders
    df["RotoName"] = df.player_name.apply(
        lambda x: x.lower().split(" ")[0][0:4] + x.lower().split(" ")[1][0:5]
    )
    return df
