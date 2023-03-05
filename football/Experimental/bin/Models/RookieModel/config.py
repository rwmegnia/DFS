#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 05:43:13 2022

@author: robertmegnia

Configuration file for Position Machine Learning Model
"""



#DK_
# Columns that can't/shouldn't be used in ML models
NonFeatures=['full_name',
             'gsis_id',
             'opp',
             'week',
             'position',
             'season',
             'rookie_year',
             'game_date',
             'start_time',
             'team',
             'game_location',
             'game_id',
             'game_day',
             'Slate',]

# Rookie Model Features

QBRookieFeatures=['team_DKPts',
                'team_DKPts_allowed',
                'team_passing_DKPts',
                'team_passing_DKPts_allowed']


RBRookieFeatures=['team_DKPts',
                  'team_DKPts_allowed',
                  'team_rushing_DKPts',
                  'team_rushing_DKPts_allowed',
                  'team_receiving_DKPts',
                  'team_receiving_DKPts_allowed']

RecRookieFeatures=['team_DKPts',
                  'team_DKPts_allowed',
                  'team_passing_DKPts',
                  'team_passing_DKPts_allowed',
                  'team_receiving_DKPts',
                  'team_receiving_DKPts_allowed']

RookieFeatures={
    'QB':QBRookieFeatures,
    'RB':RBRookieFeatures,
    'WR':RecRookieFeatures,
    'TE':RecRookieFeatures}
# Columns that we know the values of going into a week
KnownFeatures=['total_line',
               'proj_team_score',
               'spread_line',
               'opp_Rank',
               'Adj_opp_Rank',
               'depth_team',
               'salary']


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

#
# Rush Share features
rush_DKPts_share_features=[
    'rush_redzone_looks',
    'rush_yards',
    'rushing_DKPts',
    'rush_att',
    'rush_share',
    'avg_rushing_DKPts_share']

# Rush Share features
receiving_DKPts_share_features=[
    'rec_redzone_looks',
    'rec_yards',
    'receiving_DKPts',
    'targets',
    'target_share',
    'rec_share',
    'avg_receiving_DKPts_share']







