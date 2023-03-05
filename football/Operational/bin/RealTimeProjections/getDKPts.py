#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 00:23:50 2022

@author: robertmegnia
"""

def getDKPts(df,stat_type):
    if stat_type=='Offense':
        df.loc[df.pass_yards>=300,'pass_yards']=df.loc[df.pass_yards>=300,'pass_yards']+30
        df.loc[df.rush_yards>=100,'rush_yards']=df.loc[df.rush_yards>=100,'rush_yards']+30
        df.loc[df.rec_yards>=100,'rec_yards']=df.loc[df.rec_yards>=100,'rec_yards']+30
        return (df.pass_yards*0.04)+(df.rush_yards*.1)+(df.pass_td*4)+(df.rush_td*6)+df.rec+(df.rec_yards*.1)+(df.rec_td*6)-(df.int)-(df.fumbles_lost)
    else:
        points=(df.fumble_recoveries*2)+(df.interception*2)+df.sack+(df.blocks*6)+(df.safety*2)+(df.return_touchdown*6)
        points_allowed=df.points_allowed.values[0]
        if points_allowed==0:
            return points+10
        elif (points_allowed>0)&(points_allowed<=6):
            return points+7
        elif (points_allowed>6)&(points_allowed<=13):
            return points+4
        elif (points_allowed>13)&(points_allowed<=20):
            return points+1
        elif (points_allowed>20)&(points_allowed<=27):
            return points
        elif (points_allowed>27)&(points_allowed<=34):
            return points-1
        else:
            return points-4
        

