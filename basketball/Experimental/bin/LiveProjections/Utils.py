#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 16:29:35 2022

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

def reformatFantasyPros(df):
    df.rename({'Name':'player_name',
               'Position':'position',
               'Team':'team_abbreviation',
               'Proj Pts.':'FP_Proj'},axis=1,inplace=True)
    df=reformatNames(df)
    return df

def reformatDailyGrind(df):
   df.rename({
       'Player':'player_name',
       'Pos':'position',
       'Team':'team_abbreviation',
       'Proj Mins':'DG_proj_mins',
       'Draftkings Proj':'DG_Proj',
       'Draftkings Own %':'DG_ownership_proj',
       'USG':'proj_usg'},axis=1,inplace=True)
   df['DG_ownership_proj']=df.DG_ownership_proj.apply(lambda x: float(x.split('%')[0]))
   df['proj_usg']/=100
   df=reformatNames(df)
   return df
    
    
    
    
    