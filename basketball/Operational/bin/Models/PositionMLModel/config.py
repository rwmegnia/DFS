#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 05:43:13 2022

@author: robertmegnia

Configuration file for Position Machine Learning Model
"""
import numpy as np

# Columns that can't/shouldn't be used in ML models
NonFeatures = [
    "game_id",
    "team_id",
    "team_abbreviation",
    "team_city",
    "player_id",
    "player_name",
    "nickname",
    "start_position",
    "comment",
    "game_date",
    "home_team_id",
    "visitor_team_id",
    "game_location",
    "opp",
    "position",
    "roster position",
    "started",
    "doublepoints",
    "doubleassists",
    "doublesteals",
    "doublerebounds",
    "doubleblocks",
    "doublestats",
    "doubledouble",
    "tripledouble",
    "season",
    "game_date_string",
]

DefenseFeatures = {'fgm':np.sum,
                   'fga':np.sum,
                   'fg_pct':np.mean,
                   'fg3m':np.sum,
                   'fg3a':np.sum,
                   'fg3_pct':np.mean,
                   'reb':np.sum,
                   'ast':np.sum,
                   'stl':np.sum,
                   'blk':np.sum,
                   'to':np.sum,
                   'pts':np.sum,
                   'plus_minus':np.sum,
                   'dkpts':np.sum,}
KnownFeatures = ["salary"]
