#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 05:43:13 2022

@author: robertmegnia

Configuration file for Position Machine Learning Model
"""


# Columns that can't/shouldn't be used in ML models
BatterNonFeatures = ['gamesPlayed',
 'flyOuts',
 'groundOuts',
 'groundIntoDoublePlay',
 'groundIntoTriplePlay',
 'leftOnBase',
 'sacBunts',
 'sacFlies',
 'catchersInterference',
 'pickoffs',
 'atBatsPerHomeRun',
 'full_name',
 'player_id',
 'position',
 'position_type',
 'team',
 'game_location',
 'opp',
 'order',
 'note',
 'season',
 'game_date',
 'Roster Position',
 'bats',
 'launchSpeed',
 'launchAngle',
 'distance',
 'bat_result',
 'stolenBasePercentage']

PitcherNonFeatures = ['gamesPlayed',
 'flyOuts',
 'groundOuts',
 'sacBunts',
 'sacFlies',
 'catchersInterference',
 'pickoffs',
 'stolenBasePercentage',
 'full_name',
 'player_id',
 'position',
 'position_type',
 'team',
 'game_location',
 'opp',
 'order',
 'note',
 'season',
 'game_date',
 'Roster Position',
 'game_id']


OpposingPitcherColumns = [
 'runs',
 'doubles',
 'triples',
 'homeRuns',
 'strikeOuts',
 'baseOnBalls',
 'intentionalWalks',
 'hits',
 'hitByPitch',
 'atBats',
 'caughtStealing',
 'stolenBases',
 'numberOfPitches',
 'inningsPitched',
 'earnedRuns',
 'battersFaced',
 'pitchesThrown',
 'balls',
 'strikes',
 'hitBatsmen',
 'DKPts',
 'Salary',]

OpposingTeamColumns = [
    "hits",
    "doubles",
    "triples",
    "homeRuns",
    "DKPts",
    "runs",
    "stolenBases",
    "rbi",
]
