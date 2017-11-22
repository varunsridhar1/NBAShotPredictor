# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 22:52:43 2017

@author: Varun
"""

import pandas as pd
pd.options.mode.chained_assignment = None

df1 = pd.read_csv('shot_logs_fgpct_dbpm.csv')
df2 = pd.read_csv('days_since_shotlogs.csv')

df1['DAYS_SINCE_START'] = df2['DAYS_SINCE_START']
df1.to_csv('FINAL.csv')