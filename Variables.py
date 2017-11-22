import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

daysSince={0:'OCT', 3:'NOV', 33:'DEC', 64:'JAN', 95:'FEB', 123:'MAR', 154:'APR', 184:'MAY', 215:'JUN'}

def numDays(month, day):
    if(month == 'OCT'):
        return (int(day)-28)
    elif(month == 'NOV'):
        return (3+int(day))
    elif(month == 'DEC'):
        return (33+int(day))
    elif(month == 'JAN'):
        return (64+int(day))
    elif(month == 'FEB'):
        return (95+int(day))
    elif(month == 'MAR'):
        return (123+int(day))
    elif(month == 'APR'):
        return (154+int(day))
    elif(month == 'MAY'):
        return (184+int(day))
    elif(month == 'JUN'):
        return (215+int(day))

samples = pd.read_csv('updated_shotlogs.csv',usecols = ['MATCHUP'],dtype={'MATCHUP': str})
matchups = samples.values
daysSince = []
lim = matchups.shape[0]

for i in range (0, lim):
    month = (str(matchups[i])[2:5])
    day = (str(matchups[i])[6:8])
    daysSince.append(numDays(month, day))
    
ds = pd.DataFrame(daysSince, columns=['DAYS_SINCE_START'])
ds.to_csv('days_since_shotlogs.csv')

