#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 00:12:06 2022

@author: robertmegnia
"""
# Stats to use for DKPts forecast

DK_stats=[
    'pass_yards',
    'pass_td',
    'rush_yards',
    'rush_td',
    'rec',
    'rec_yards',
    'rec_td',
    'fumbles_lost',
    'int',
    'DKPts',]


DK_DST_stats=[
    'fumble_recoveries',
    'interception',
    'sack',
    'blocks',
    'safety',
    'return_touchdown',
    'points_allowed',
    'DKPts']


UpsideScores={
    'QB':25,
    'RB':20,
    'WR':20,
    'TE':15,
    'DST':10}