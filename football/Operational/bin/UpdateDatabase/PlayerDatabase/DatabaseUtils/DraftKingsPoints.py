#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 03:46:08 2022

@author: robertmegnia

Functions for computing NFL Draftkings Points
"""
def getDKPts_passing(pass_yds,pass_td,pass_int,fumbles_lost,two_point_conv):
            if pass_yds>=300:
                pass_yds+=75
            return (pass_yds*0.04)+(pass_td*4)-(pass_int)-(fumbles_lost)+(two_point_conv*2)

def getDKPts_rushing(rush_yds,rush_td,fumbles_lost,two_point_conv):
            if rush_yds>=100:
                rush_yds+=30
            return (rush_yds*0.1)+(rush_td*6)-(fumbles_lost)+(two_point_conv*2)
        
def getDKPts_receiving(rec_yds,rec,rec_td,fumbles_lost,two_point_conv):
            if rec_yds>=100:
                rec_yds+=30
            return (rec_yds*.1)+rec+(rec_td*6)-fumbles_lost+(two_point_conv*2)
        
def getDKPts(pass_yds,pass_td,rush_yds,rush_td,rec,rec_yds,rec_td,fumbles_lost,pass_int,two_point_conv):
            if pass_yds>=300:
                pass_yds+=75
            if rush_yds>=100:
                rush_yds+=30
            if rec_yds>=100:
                rec_yds+=30
            return (pass_yds*0.04)+(rush_yds*.1)+(pass_td*4)+(rush_td*6)+rec+(rec_yds*.1)+(rec_td*6)-(pass_int)-(fumbles_lost)+(two_point_conv*2)
    
def getDKPtsDST(fumbles,ints,sacks,blocks,safety,return_touchdown,points_allowed):
        points=(fumbles*2)+(ints*2)+sacks+(blocks*2)+(safety*2)+(return_touchdown*6)
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