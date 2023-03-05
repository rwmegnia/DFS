#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 05:43:13 2022

@author: robertmegnia

Configuration file for Ownership Machine Learning Model
"""
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
             'Slate',]

# Columns that we know the values of going into a week
KnownFeatures=['total_line',
                'proj_team_score',
                'opp_Rank',
                'Projection',
                'Value',]
# KnownFeatures=['total_line',
#                 'proj_team_score',
#                 'spread_line',
#                 'opp_Rank',
#                 'Adj_opp_Rank',
#                 'Projection',
#                 'Value',
#                 'salary',
#                 'UpsideProb']








