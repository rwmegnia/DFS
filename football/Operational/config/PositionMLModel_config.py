#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 05:43:13 2022

@author: robertmegnia

Configuration file for Position Machine Learning Model
"""

# DK Stats Columns
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
    'pass_yards_allowed',
    'pass_td_allowed',
    'rush_yards_allowed',
    'rush_td_allowed',
    'rec_allowed',
    'rec_yards_allowed',
    'rec_td_allowed',
    'fumbles_lost_allowed',
    'int_allowed',
    'pass_yards_share',
    'pass_td_share',
    'rush_yards_share',
    'rush_td_share',
    'rec_share',
    'rec_yards_share',
    'rec_td_share',
    'fumbles_lost_share',
    'int_share']

#DK_
# Columns that can't/shouldn't be used in ML models
NonFeatures=['full_name',
             'gsis_id',
             'opp',
             'week',
             'position',
             'season',
             'game_date',
             'start_time',
             'team',
             'game_location',
             'game_id',
             'game_day',
             'Slate']

# Columns that we know the values of going into a week
KnownFeatures=['total_line',
               'proj_team_score',
               'spread_line',
               'opp_Rank',
               'Adj_opp_Rank',]

# Columns that correspond to team shares won't be used for in opponent features
TeamShareFeatures=['DKPts_share_pos',
                   'DKPts_share_skill_pos',
                   'DKPts_share',
                   'pass_yards_share',
                   'pass_td_share',
                   'rush_yards_share',
                   'rush_td_share',
                   'rec_yards_share',
                   'rec_td_share',
                   'rec_share',
                   'fumbles_lost_share',
                   'int_share',
                   'passing_DKPts_share',
                   'rushing_DKPts_share',
                   'receiving_DKPts_share']

# Reduce low bias by excluding too many low-ranked players
pos_ranks={'QB':29,
           'RB':47,
           'WR':70,
           'TE':27,
           'DST':23}