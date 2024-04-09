#%%
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as pt
from matplotlib import style
style.use('ggplot')

import os , sys
current_dir = os.path.dirname(os.path.abspath(__file__))
path = '/home/geo/Downloads/geo'
sys.path.append(path)

from notification_bot.loguru_notification import loguru_notf
from financial_bot.volatility_smile.sabr import sabr_model
from financial_bot.options.options import black_scholes

file_dir = f'{current_dir}/volatility-smile.csv'

class data_visualization:
    
    def plot_volatility(self,df):
        pt.figure(figsize=[60,150])
    
        pt.subplot(7,2,1)
        pt.plot(df.loc[0])
        pt.xlabel('delta')
        pt.ylabel('iv')

        pt.subplot(7,2,2)
        pt.plot(df.loc[1])
        pt.xlabel('delta')
        pt.ylabel('iv')

        pt.subplot(7,2,3)
        pt.plot(df.loc[2])
        pt.xlabel('delta')
        pt.ylabel('iv')

        pt.subplot(7,2,4)
        pt.plot(df.loc[3])
        pt.xlabel('delta')
        pt.ylabel('iv')
        
        pt.subplot(7,2,5)
        pt.plot(df.loc[4])
        pt.xlabel('delta')
        pt.ylabel('iv')

        pt.subplot(7,2,6)
        pt.plot(df.loc[5])
        pt.xlabel('delta')
        pt.ylabel('iv')
        
        pt.subplot(7,2,7)
        pt.plot(df.loc[6])
        pt.xlabel('delta')
        pt.ylabel('iv')
        
        pt.subplot(7,2,8)
        pt.plot(df.loc[7])
        pt.xlabel('delta')
        pt.ylabel('iv')

        pt.subplot(7,2,9)
        pt.plot(df.loc[8])
        pt.xlabel('delta')
        pt.ylabel('iv')
        
        pt.subplot(7,2,10)
        pt.plot(df.loc[9])
        pt.xlabel('delta')
        pt.ylabel('iv')

        pt.subplot(7,2,11)
        pt.plot(df.loc[10])
        pt.xlabel('delta')
        pt.ylabel('iv')

        pt.subplot(7,2,12)
        pt.plot(df.loc[11])
        pt.xlabel('delta')
        pt.ylabel('iv')
        
        pt.show()

@dataclass
class data_interpolation:
    x_axis : float
    y_axis : float
    kind = 'linear'

    def interpolation_1d(self,range):
        itp_init = interp1d(self.x_axis,self.y_axis,kind=self.kind)
        x = range
        y = itp_init(x)
        return y
    
def case_for_sbar(df):
    model = sabr_model()
    # data_visual = data_visualization()
    beta = 0.5

    df['params'] = df.apply(lambda x : model.optimization(
                np.array([x['0.1'],x['0.25'],x['0.5'],x['0.75'],x['0.9']])/100,
                x['forward'],
                [x['0.1 strike'],x['0.25 strike'],x['0.5 strike'],x['0.75 strike'],x['0.9 strike']],
                x['time'],beta),axis=1)
        
    df['volatility'] = (
        df.apply(lambda x : [ model.calculate_volatility(
            x['forward'],
            x[item], # strike price.
            x['time'],
            beta,x['params'][0],x['params'][1],x['params'][2]
            ) for item in range(1,5+1)],axis=1)
        .apply(lambda x : np.array(x)))

    df['volatility_full_range'] = df['volatility'].apply(lambda x : 
                            data_interpolation(delta,x).interpolation_1d(range_init))
    return df

def volatility_smile(df):
    volatility = []
    for _ in range(len(df)):
        volatility.append(df['volatility'][_]/100)
    df = pd.DataFrame(volatility)
    df.columns = ['.1','.25','.5','.75','.9']
    return df

if __name__ == '__main__':

    spotrate = 7.38/100
    strike_price = 7.4

    delta = np.array([0.1,0.25,0.5,0.75,0.9])
    range_init = np.arange(0.1,0.9,0.01)
    
    df = pd.read_csv(file_dir)
    df = case_for_sbar(df)
    df = volatility_smile(df)
    print(df)
    # data_visualization().plot_volatility(df)