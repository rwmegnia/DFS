#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 19:56:40 2022

@author: robertmegnia
"""

# Determine if Onside Kick is necessary
KickoffOption_model_features=['game_seconds_remaining',
                              'score_differential',
                              'defteam_timeouts_remaining']

# KickoffFrom Model - Accounts for potential penalty on previous play

KickoffFrom_model_features=['qtr']

##
KickoffResult_model_features=['qtr']
# Kickoff Result Possibilities
#
# return
# touchback
# out_of_bounds

## Kickoff Distance Model
#
KickoffResult_model_features=['kickoff_from']

## Kickoff Return Model
#
KickoffReturn_model_features=['returned_from']
#
#
KickoffReturnTD_elapsed_time_features=['return_yards','randomizer']

KickoffReturn_elapsed_time_features=['return_yards','randomizer']

KickoffReturnFumble_elapsed_time_features=['return_yards','fumble_yards','randomizer']

TwoPointDecision_model_features=['score_differential','game_seconds_remaining']

XPFrom_model_features=['qtr']
XP_model_features=['kick_from']
XPBlockedReturn_model_features=['qtr']
### Coaching Models
coach_run_pass_model_features=['qtr',
                               'down',
                               'ydstogo',
                               'yardline',
                               'half_seconds_remaining',
                               'game_seconds_remaining',
                               'score_differential']

