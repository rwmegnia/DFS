#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 21:04:30 2022

@author: robertmegnia
"""

# Features to predict total fantasy points from a team


team_fpts_features=[
    'ImpliedPoints',
    'TotalPoints',
    'spread_line',
    'team_fpts',
    'team_fpts_allowed']

# 
team_pass_attempt_features=[
                            'ImpliedPoints',
                            'TotalPoints',
                            'spread_line',
                            'team_fpts',
                            'pass_attempt',
                            'pass_attempt_allowed',
                            'rush_attempt',
                            'rush_attempt_allowed']

# 
team_rush_attempt_features=[
                            'ImpliedPoints',
                            'TotalPoints',
                            'spread_line',
                            'team_fpts',
                            'pass_attempt',
                            'pass_attempt_allowed',
                            'rush_attempt',
                            'rush_attempt_allowed',
                            'epa']

# Features to predict passing fantasy points
team_passing_DKPts_features=[
    'ImpliedPoints',
    'TotalPoints',
    'spread_line',
    'team_fpts',
    'pass_attempt',
    'rush_attempt',
    'passing_fpts',
    'passing_fpts_allowed',]

# Features to predict rushing fantasy points
team_rushing_DKPts_features=[
    'ImpliedPoints',
    'TotalPoints',
    'spread_line',
    'team_fpts',
    'pass_attempt',
    'rush_attempt',
    'passing_fpts',
    'passing_fpts_allowed',
    'rushing_fpts',
    'rushing_fpts_allowed',]

# Features to predict rushing fantasy points
team_receiving_DKPts_features=[
    'ImpliedPoints',
    'TotalPoints',
    'spread_line',
    'team_fpts',
    'pass_attempt',
    'rush_attempt',
    'passing_fpts',
    'passing_fpts_allowed',
    'rushing_fpts',
    'rushing_fpts_allowed',
    'receiving_fpts',
    'receiving_fpts_allowed']


# Columns that we know the values of going into a week
TSM_KnownFeatures=[
               'ImpliedPoints',
               'spread_line',
               'TotalPoints',]


TSM_NonFeatures=['team',
             'home_team',
             'away_team',
             'home_score',
             'away_score',
             'proj_away_score',
             'proj_home_score',
             'week',
             'season',
             'game_date',
             'game_id',
             'Location',
             'opp',
             'QB1_full_name',
             'RB1_full_name',
             'RB2_full_name',
             'RB3_full_name',
             'WR1_full_name',
             'WR2_full_name',
             'WR3_full_name',
             'WR4_full_name',
             'WR5_full_name',
             'TE1_full_name',
             'TE2_full_name',
             'TE3_full_name',
             'QB1_opp',
             'RB1_opp',
             'RB2_opp',
             'RB3_opp',
             'WR1_opp',
             'WR2_opp',
             'WR3_opp',
             'WR4_opp',
             'WR5_opp',
             'TE1_opp',
             'TE2_opp',
             'TE3_opp',]

TeamStatsModel_features_dict={
    'team_fpts':team_fpts_features,
    'pass_attempt':team_pass_attempt_features,
    'rush_attempt':team_rush_attempt_features,
    'passing_fpts':team_passing_DKPts_features,
    'rushing_fpts':team_rushing_DKPts_features,
    'receiving_fpts':team_receiving_DKPts_features}
