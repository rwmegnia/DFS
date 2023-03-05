#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 03:55:11 2022

@author: robertmegnia
"""

def getEPAPerPlay(pos_df,df): 
        ## Get Team Offesive EPA
        off_plays=df.groupby(['week','posteam']).posteam.value_counts()
        off_plays.reset_index(level=1,drop=True,inplace=True)
        off_epa=df.groupby(['week','posteam']).sum().epa
        off_epa_per_play=off_epa/off_plays
        off_epa_per_play.name='offense_epa'
        off_epa_per_play=off_epa_per_play.to_frame()
        pos_df.set_index(['week','posteam'],inplace=True)
        pos_df=pos_df.merge(off_epa_per_play,on=['week','posteam'])
        pos_df.reset_index(inplace=True)
        ## Get Team Pass Offensive EPA
        pass_df=df[df['pass']==1]
        off_pass_plays=pass_df.groupby(['week','posteam']).posteam.value_counts()
        off_pass_plays.reset_index(level=1,drop=True,inplace=True)
        off_pass_epa=pass_df.groupby(['week','posteam']).sum().epa
        off_pass_epa_per_play=off_pass_epa/off_pass_plays
        off_pass_epa_per_play.name='offense_pass_epa'
        off_pass_epa_per_play=off_pass_epa_per_play.to_frame()
        pos_df.set_index(['week','posteam'],inplace=True)
        pos_df=pos_df.merge(off_pass_epa_per_play,on=['week','posteam'])
        pos_df.reset_index(inplace=True)
        ## Get Team Rush Offensive EPA
        rush_df=df[df['rush']==1]
        off_rush_plays=rush_df.groupby(['week','posteam']).posteam.value_counts()
        off_rush_plays.reset_index(level=1,drop=True,inplace=True)
        off_rush_epa=rush_df.groupby(['week','posteam']).sum().epa
        off_rush_epa_per_play=off_rush_epa/off_rush_plays
        off_rush_epa_per_play.name='offense_rush_epa'
        off_rush_epa_per_play=off_rush_epa_per_play.to_frame()
        pos_df.set_index(['week','posteam'],inplace=True)
        pos_df=pos_df.merge(off_rush_epa_per_play,on=['week','posteam'])
        pos_df.reset_index(inplace=True)
        ## Get Team Defensive EPA
        def_plays=df.groupby(['week','defteam']).defteam.value_counts()
        def_plays.reset_index(level=1,drop=True,inplace=True)
        def_epa=df.groupby(['week','defteam']).sum().epa
        def_epa_per_play=def_epa/def_plays
        def_epa_per_play.name='defense_epa'
        def_epa_per_play=def_epa_per_play.to_frame()
        def_epa_per_play.reset_index(inplace=True)
        def_epa_per_play.rename({'defteam':'posteam'},axis=1,inplace=True)
        pos_df.set_index(['week','posteam'],inplace=True)
        pos_df=pos_df.merge(def_epa_per_play,on=['week','posteam'])
        ## Get Team Pass Defensive EPA
        def_pass_plays=pass_df.groupby(['week','defteam']).defteam.value_counts()
        def_pass_plays.reset_index(level=1,drop=True,inplace=True)
        def_pass_epa=pass_df.groupby(['week','defteam']).sum().epa
        def_pass_epa_per_play=def_pass_epa/def_pass_plays
        def_pass_epa_per_play.name='defense_pass_epa'
        def_pass_epa_per_play=def_pass_epa_per_play.to_frame()
        def_pass_epa_per_play.reset_index(inplace=True)
        def_pass_epa_per_play.rename({'defteam':'posteam'},axis=1,inplace=True)
        pos_df.set_index(['week','posteam'],inplace=True)
        pos_df=pos_df.merge(def_pass_epa_per_play,on=['week','posteam'])
        ## Get Team Rush Defensive EPA
        def_rush_plays=rush_df.groupby(['week','defteam']).defteam.value_counts()
        def_rush_plays.reset_index(level=1,drop=True,inplace=True)
        def_rush_epa=rush_df.groupby(['week','defteam']).sum().epa
        def_rush_epa_per_play=def_rush_epa/def_rush_plays
        def_rush_epa_per_play.name='defense_rush_epa'
        def_rush_epa_per_play=def_rush_epa_per_play.to_frame()
        def_rush_epa_per_play.reset_index(inplace=True)
        def_rush_epa_per_play.rename({'defteam':'posteam'},axis=1,inplace=True)
        pos_df.set_index(['week','posteam'],inplace=True)
        pos_df=pos_df.merge(def_rush_epa_per_play,on=['week','posteam'])
        return pos_df