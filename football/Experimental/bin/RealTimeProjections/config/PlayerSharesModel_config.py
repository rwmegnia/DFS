#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 21:04:30 2022

@author: robertmegnia
"""

#
# Rush Share features
rush_DKPts_share_features=[
    'rush_redzone_looks',
    'rush_yards',
    'rushing_DKPts',
    'rush_att',
    'rush_share',
    'rushing_DKPts_share',
    'depth_team']

# Rush Share features
receiving_DKPts_share_features=[
    'rec_redzone_looks',
    'rec_yards',
    'receiving_DKPts',
    'targets',
    'target_share',
    'rec_share',
    'receiving_DKPts_share',
    'depth_team']


# Columns that can't/shouldn't be used in ML models
PSM_NonFeatures=['full_name',
             'gsis_id',
             'opp',
             'week',
             'position',
             'season',
             'game_date',
             'team',
             'game_location',
             'game_day',
             'Slate',]

# Columns that we know the values of going into a week
PSM_KnownFeatures=['total_line',
               'proj_team_score',
               'spread_line',
               'opp_Rank',
               'Adj_opp_Rank',
               'depth_team']

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
                   'int_share']
UpsideScores={
    'QB':25,
    'RB':20,
    'WR':20,
    'TE':15,
    'DST':10}

PlayerShareModel_features_dict={
    'receiving_DKPts_share':receiving_DKPts_share_features,
    'rushing_DKPts_share':rush_DKPts_share_features}
