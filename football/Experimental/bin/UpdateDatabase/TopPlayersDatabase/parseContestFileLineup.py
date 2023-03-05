# -*- coding: utf-8 -*-
import pandas as pd
def parseContestFileLineup(lineup_string,week,season):
    def_dict={'WAS Football Team ':'WAS',
              'Saints ':'NO',
              'Chargers ':'LAC',
              'Ravens ':'BAL',
              'Jets ':'NYJ',
              'Patriots ':'NE',
              'Dolphins ':'MIA',
              'Colts ':'IND',
              'Lions ':'DET',
              'Cardinals ':'ARI',
              'Buccaneers ':'TB',
              'Giants ':'NYG',
              'Bears ':'CHI',
              'Chiefs ':'KC',
              'KC':'KC',
              'Packers ':'GB',
              'Titans ':'TEN',
              'Eagles ':'PHI',
              'Steelers ':'PIT',
              'Rams ':'LA',
              'Panthers ':'CAR',
              '49ers ':'SF',
              'Browns ':'CLE',
              'Falcons ':'ATL',
              'Seahawks ':'SEA',
              'Bengals ':'CIN',
              'Vikings ':'MIN',
              'Raiders ':'OAK',
              'Jaguars ':'JAX',
              'Texans ':'HOU',
              'Cowboys ':'DAL',
              'Bills ':'BUF',
              'Broncos ':'DEN'}
    try:
        QB=lineup_string.split('QB ')[1].split(' RB')[0]
        RB1=lineup_string.split(' RB ')[1]
        RB2=lineup_string.split(' RB ')[2].split(' WR' )[0]
        WR1=lineup_string.split(' WR ')[1]
        WR2=lineup_string.split(' WR ')[2]
        WR3=lineup_string.split(' WR ')[3].split(' FLEX')[0]
        FLEX=lineup_string.split(' FLEX ')[1].split(' TE ')[0]
        TE=lineup_string.split(' TE ')[1].split(' DST')[0]
        DST=lineup_string.split(' DST ')[1]
        DST=def_dict[DST]
    except:
        QB=lineup_string.split('QB ')[1].split(' RB')[0]
        RB1=lineup_string.split(' RB ')[1]
        RB2=lineup_string.split(' RB ')[2].split(' WR' )[0]
        WR1=lineup_string.split(' WR ')[1].split(' FLEX')[0]        
        WR2=lineup_string.split(' WR ')[2].split(' TE')[0]
        WR3=lineup_string.split(' WR ')[3].split(' FLEX')[0]
        FLEX=lineup_string.split(' FLEX ')[1].split(' WR ')[0]
        TE=lineup_string.split(' TE ')[1].split(' DST')[0]
        DST=' '.join(lineup_string.split(' DST ')[1].split(' ')[0:4])
        DST=def_dict[DST]
    lineup=pd.DataFrame({'full_name':[QB,RB1,RB2,WR1,WR2,WR3,FLEX,TE,DST],
                         'roster_position':['QB','RB','RB','WR','WR','WR','FLEX','TE','DST']})
    lineup['week']=week
    lineup['season']=season
    lineup.loc[lineup.roster_position!='DST','full_name']=lineup.loc[lineup.roster_position!='DST'].full_name.apply(lambda x: x.split(' ')[0]+' '+x.split(' ')[1])
    return lineup

def parseShowdownLineup(row):
    '''
    Take a lineup string from draftkings contest results file and transform it into a row
    that can be placed into a data frame

    Parameters
    ----------
    lineup_string : TYPE
        DESCRIPTION.
    week : TYPE
        DESCRIPTION.
    season : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    lineup_string=row.Lineup
    lineup_id=row.lineup_id
    Duplicates=row.Duplicates
    Duplicated=row.Duplicated
    Week=row.week
    Day=row.Day
    print(lineup_string)
    # Parse each player from each lineup
    # try:
    FLEX1=' '.join(lineup_string.split('FLEX ')[1].split(' ')[0:2])
    FLEX2=' '.join(lineup_string.split('FLEX ')[2].split(' ')[0:2])
    FLEX3=' '.join(lineup_string.split('FLEX ')[3].split(' ')[0:2])
    FLEX4=' '.join(lineup_string.split('FLEX ')[4].split(' ')[0:2])
    FLEX5=' '.join(lineup_string.split('FLEX ')[5].split(' ')[0:2])
    CPT=' '.join(lineup_string.split('CPT')[1].split(' ')[1:3])
    lineup=pd.DataFrame({'full_name':[FLEX1,FLEX2,FLEX3,FLEX4,FLEX5,CPT],
                         'roster_position':['FLEX1','FLEX2','FLEX3','FLEX4','FLEX5','CPT']})
    lineup['lineup_id']=lineup_id
    lineup['Duplicates']=Duplicates
    lineup['Duplicated']=Duplicated
    lineup['Day']=Day
    lineup['Week']=Week
    return lineup

def reformatNames(df):
        dst=df[df.roster_position=='DST']
        df=df[df.roster_position!='DST']
        # Create first/last name columns for df
        df['first_name']=df.full_name.apply(lambda x: x.split(' ')[0])
        df['last_name']=df.full_name.apply(lambda x: ' '.join(x.split(' ')[1::]))

        # Remove suffix from last name but keep prefix
        df['last_name']=df.last_name.apply(lambda x: x if x in ['St. Brown','Vander Laan'] else x.split(' ')[0])

        # Remove non-alpha numeric characters from first names.
        df['first_name']=df.first_name.apply(lambda x: ''.join(c for c in x if c.isalnum()))
        df['last_name']=df.last_name.apply(lambda x: ''.join(c for c in x if c.isalnum()))
        # Recreate full_name
        df['full_name']=df.apply(lambda x: x.first_name+' '+x.last_name,axis=1)
        df['full_name']=df.full_name.apply(lambda x: x.lower())
        df.drop(['first_name','last_name'],axis=1,inplace=True)
        df.full_name=df.full_name.apply(lambda x: x.split(' ')[0][0].upper()+x.split(' ')[0][1::]+' '+x.split(' ')[-1][0].upper()+x.split(' ')[-1][1::])
        df=pd.concat([df,dst])
        return df