#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 03:59:08 2022

@author: robertmegnia
"""

def getSharedStats(pos_df):
        # Get Points Share per position
        pos_tp=pos_df.groupby(['team','season','week','position']).sum().DKPts
        pos_tp.name='team_DKPts_pos'
        pos_tp=pos_tp.to_frame()
        pos_df=pos_df.merge(pos_tp,on=['position','team','season','week'],how='left')
        pos_df['DKPts_share_pos']=pos_df.DKPts/pos_df.team_DKPts_pos
        pos_df.loc[pos_df.DKPts_share_pos>=1,'DKPts_share_pos']=1
        
        # Get Points Share per skill position (RB,WR,TE)
        skill_pos_tp=pos_df[pos_df.position.isin(['RB','WR','TE'])].groupby(['team','season','week']).sum().DKPts
        skill_pos_tp.name='team_DKPts_skill_pos'
        skill_pos_tp=skill_pos_tp.to_frame()
        pos_df=pos_df.merge(skill_pos_tp,on=['team','season','week'],how='left')
        pos_df.loc[pos_df.position.isin(['RB','WR','TE']),'DKPts_share_skill_pos']=pos_df.loc[pos_df.position.isin(['RB','WR','TE']),'DKPts']/pos_df.loc[pos_df.position.isin(['RB','WR','TE']),'team_DKPts_skill_pos']
        pos_df.loc[pos_df.DKPts_share_skill_pos>=1,'DKPts_share_skill_pos']=1
        
        # Get Total Team Points Share
        tp=pos_df.groupby(['team','season','week']).sum().DKPts
        tp.name='team_DKPts'
        tp=tp.to_frame()
        pos_df=pos_df.merge(tp,on=['team','season','week'],how='left')
        pos_df['DKPts_share']=pos_df.DKPts/pos_df.team_DKPts
        pos_df.loc[pos_df.DKPts_share>=1,'DKPts_share']=1
        
        # Get Shared Statistics
        for shared_stat in ['pass_yards','pass_td','rush_yards','rush_td','rec_yards','rec_td','rec','fumbles_lost','int','passing_DKPts','rushing_DKPts','receiving_DKPts']:
            ss=pos_df.groupby(['team','season','week']).sum()[shared_stat]
            ss.name=f'team_{shared_stat}'
            ss=ss.to_frame()
            pos_df=pos_df.merge(ss,on=['team','season','week'],how='left')
            pos_df[f'{shared_stat}_share']=pos_df[shared_stat]/pos_df[f'team_{shared_stat}']
            pos_df.loc[pos_df[f'{shared_stat}_share']>=1,f'{shared_stat}_share']=1
        return pos_df