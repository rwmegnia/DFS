#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 22:38:16 2021

@author: robertmegnia
"""
import numpy as np

TEAM_STATS={
    'air_epa':np.mean,
    'air_yards':np.sum,
    'comp_air_epa':np.mean,
    'comp_yac_epa':np.mean,
    'complete_pass':np.sum,
    'epa':np.mean,
    'fumble':np.sum,
    'fumble_lost':np.sum,
    'interception':np.sum,
    'pass_attempt':np.sum,
    'passing_yards':np.sum,
    'pass_touchdown':np.sum,
    'qb_epa':np.mean,
    'qb_hit':np.sum,
    'sack':np.sum,
    'yards_after_catch':np.sum,
    'big_pass_play':np.sum,
    'rush_attempt':np.sum,
    'rush_touchdown':np.sum,
    'rushing_yards':np.sum,
    'fumble':np.sum,
    'fumble_lost':np.sum,
    'tackled_for_loss':np.sum,
    'big_run':np.sum,
    }

QB_STATS=[
    'full_name',
    'pass_cmp',
    'pass_att',
    'pass_yds',
    'pass_yds_per_att',
    'air_yards',
    'pass_td',
    'pass_int',
    'DKPts',
    'passer_rating']

RB_STATS=[
    'full_name',
    'rush_att',
    'rush_share',
    'rush_yds',
    'rush_td',
    'targets',
    'rec',
    'rec_yds',
    'rec_td',
    'redzone_looks',
    'PPO',
    'DKPts',
    ]

REC_STATS=[
    'full_name',
    'targets',
    'rec',
    'rec_yds',
    'yards_after_catch',
    'air_yards',
    'wopr',
    'rec_td',
    'redzone_looks',
    'PPO',
    'DKPts',
    ]