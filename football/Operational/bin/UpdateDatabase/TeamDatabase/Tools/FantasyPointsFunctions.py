#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 01:46:28 2021

@author: robertmegnia
"""

def get_team_passing_fpts(pass_yards,pass_touchdown,td=4):
    '''
    

    Parameters
    ----------
    td : int
        touchdown points awarded - default 4
    pass_yards : float
    pass_touchdown : int

    Returns
    -------
    passing_fpts - float

    '''
    
    return (pass_yards*.04)+(pass_touchdown*td)
    
def get_team_rushing_fpts(rush_yards,rush_touchdown):
    '''
    Parameters
    ----------
    rush_yards : float
    rush_touchdown : int

    Returns
    -------
    rushing_fpts - float

    '''
    return (rush_yards*.1)+(rush_touchdown*6)

def get_team_receiving_fpts(rec_yards,rec,rec_td,ppr=1):
    '''
    Parameters
    ----------
    ppr: float
    (0.5,1,0)
    rec_yards : float
    rec_td : int

    Returns
    -------
    receiving_fpts - float

    '''
    return (rec*ppr)+(rec_yards*.1)+(rec_td*6)