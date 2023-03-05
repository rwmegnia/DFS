#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 04:00:40 2022

@author: robertmegnia
"""
import pandas as pd
def getOppRanks(df,stat_type):
        frames=[]
        for season in df.season.unique():
            # If working after 2020 need to loop through 18 weeks not 17
            if season>2020:
                end_week=18
            else:
                end_week=17
            season_df=df.loc[df.season==season]
            # Create Frame for Opponent Rankings
            opp_frames=[]
            if stat_type=='Offense':
                for pos in ['QB','RB','WR','TE','FB']:
                    for week in range(1,end_week+1):
                        opp_frame=pd.DataFrame({'opp':season_df.team.unique()})
                        opp_frame['week']=week
                        opp_frame['position']=pos
                        opp_frame['season']=season
                        opp_frames.append(opp_frame)
                opp_frame=pd.concat(opp_frames)
            else:
                for pos in ['DST']:
                    for week in range(1,end_week+1):
                        opp_frame=pd.DataFrame({'opp':season_df.team.unique()})
                        opp_frame['week']=week
                        opp_frame['position']=pos
                        opp_frame['season']=season
                        opp_frames.append(opp_frame)
                opp_frame=pd.concat(opp_frames)
            weekly_sum=season_df.groupby(['week','opp','position'],as_index=False).DKPts.sum()
            opp_frame=opp_frame.merge(weekly_sum,on=['week','opp','position'],how='left')
            opp_frame.DKPts.fillna(0,inplace=True)
            opp_frame.sort_values(by=['opp','week'],inplace=True)
            running_sum=opp_frame.groupby(['opp','position']).apply(lambda x: x.DKPts.rolling(window=16,min_periods=1).sum().shift(1)).reset_index()  
            running_sum.rename({'DKPts':'running_DKPts'},axis=1,inplace=True)
            running_sum.set_index('level_2',inplace=True)
            opp_frame['running_DKPts']=running_sum['running_DKPts']
            opp_frame['opp_Rank']=opp_frame.groupby(['position','week']).running_DKPts.rank(method='min')
            opp_frame.drop('DKPts',axis=1,inplace=True)
            opp_frame=opp_frame.merge(season_df,on=['opp','week','season','position'],how='left')
            frames.append(opp_frame)
        df=pd.concat(frames)
        frames=[]
        df.season=df.season.astype(int)
        for season in range(df.season.min()+1,df.season.max()+1):
            season_df=df.loc[df.season==season]
            week17=df.loc[(df.season==season-1)&(df.week==17)].groupby(['opp','position']).opp_Rank.mean()
            week17.name='opp_Rank'
            week17=week17.to_frame()
            week1=df.loc[(df.season==season)&(df.week==1)].merge(week17,on=['opp','position'])
            week1.drop('opp_Rank_x',axis=1,inplace=True)
            week1.rename({'opp_Rank_y':'opp_Rank'},axis=1,inplace=True)
            season_df=pd.concat([week1,season_df.drop(season_df[season_df.week==1].index)])
            frames.append(season_df)
        df=pd.concat(frames)
        return df.drop('running_DKPts',axis=1)

    
def getAdjOppRanks(df,stat_type):
        if stat_type=='Offense':
            DKPtsColumn='team_DKPts_pos'
            positions=['QB','RB','WR','TE','FB']

        else:
            DKPtsColumn='DKPts'
            positions=['DST']

        df.sort_values(by=['team','position','game_date'],inplace=True)
        df.reset_index(drop=True,inplace=True)
        df['DKPts_std']=df.groupby(['team'])[DKPtsColumn].apply(lambda x: x.rolling(min_periods=8,window=16,win_type='triang').std())
        df['DKPts_avg']=df.groupby(['team'])[DKPtsColumn].apply(lambda x: x.rolling(min_periods=8,window=16,win_type='triang').mean())
        df=df[df.DKPts_avg!=0]
        df['zscore']=df.apply(lambda x: (x[DKPtsColumn]-x.DKPts_avg)/x.DKPts_std,axis=1)
        frames=[]
        for season in sorted(df.season.unique()):
            print(season)
            if season>2020:
                end_week=18
            else:
                end_week=17
            season_df=df.loc[df.season==season]
            opp_frames=[]
            for pos in positions: 
                for week in range(1,end_week+1):
                    opp_frame=pd.DataFrame({'opp':season_df.team.unique()})
                    opp_frame['week']=week
                    opp_frame['position']=pos
                    opp_frame['season']=season
                    opp_frames.append(opp_frame)
            opp_frame=pd.concat(opp_frames)
            weekly_sum=season_df.groupby(['week','opp','position'],as_index=False).zscore.mean()
            opp_frame=opp_frame.merge(weekly_sum,on=['week','opp','position'],how='left')
            opp_frame.zscore.fillna(0,inplace=True)
            opp_frame.sort_values(by=['opp','week'],inplace=True)
            running_sum=opp_frame.groupby(['opp','position']).apply(lambda x: x.zscore.rolling(window=16,min_periods=1).sum().shift(1)).reset_index()  
            running_sum.rename({'zscore':'running_zscore'},axis=1,inplace=True)
            running_sum.set_index('level_2',inplace=True)
            opp_frame['running_zscore']=running_sum['running_zscore']
            opp_frame['Adj_opp_Rank']=opp_frame.groupby(['position','week']).running_zscore.rank(method='min')
            opp_frame.drop('zscore',axis=1,inplace=True)
            opp_frame=opp_frame.merge(season_df,on=['opp','week','season','position'],how='left')
            frames.append(opp_frame)
        df=pd.concat(frames)
        frames=[]
        df.season=df.season.astype(int)
        for season in range(df.season.min()+1,df.season.max()+1):
            season_df=df.loc[df.season==season]
            week17=df.loc[(df.season==season-1)&(df.week==17)].groupby(['opp','position']).Adj_opp_Rank.mean()
            week17.name='Adj_opp_Rank'
            week17=week17.to_frame()
            week1=df.loc[(df.season==season)&(df.week==1)].merge(week17,on=['opp','position'])
            week1.drop('Adj_opp_Rank_x',axis=1,inplace=True)
            week1.rename({'Adj_opp_Rank_y':'Adj_opp_Rank'},axis=1,inplace=True)
            season_df=pd.concat([week1,season_df.drop(season_df[season_df.week==1].index)])
            frames.append(season_df)
        df=pd.concat(frames)
        return df.drop(['DKPts_std','DKPts_avg','zscore','running_zscore'],axis=1)