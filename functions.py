import pandas as pd
import numpy as np
import datetime
import yfinance as yf

# gets data
def get_data(tickers, exchange='',day_period=4000):
    """ 
    tickers as list
    exchange as string example '.TO' for TSX
    needs yfinance, datetime, pandas
    returns last 420 trading days
    """
    
    # parse ticker list
    tickerstring=""
    for t in tickers:
        tickerstring+=t+exchange+" "
    
    # get start and end date
    ct = datetime.datetime.today()
    end = ct.strftime("%Y-%m-%d")
    start=ct-datetime.timedelta(days=day_period)
    start=start.strftime("%Y-%m-%d")
    
    # request data from yfinance
    data = yf.download(tickers=tickerstring, start=start, end=end,interval="1d")  
    data = pd.DataFrame(data)
    # clean up data using other function
    data=df_cleanup(data)
    return data


# function to clean up yfinance output for pandas
# takes a pandas dataframe 
# removes columns we don't need and renames columns
def df_cleanup(df):
    df=df.reset_index()
    # list to store columns we want to keep
    col_list=[]
    # list to store new names of columns
    col_newname=[]
    # create list of columns to loop through
    cols=df.columns.to_list()
    for c in cols:
        if c[0]=='Date':
            col_list.append(c)
            col_newname.append(c[0])
        if c[0]=='Adj Close':
            col_list.append(c)
            col_newname.append(c[1])
        if c[0]=='Volume':
            col_list.append(c)
            col_newname.append(c[1]+'_volume')
    # create datafram with only desired columns
    dfclean=df[col_list]
    # change the names of the columns
    dfclean.columns=col_newname
    # make change date column to datetime object
    dfclean['Date']=pd.to_datetime(dfclean['Date'], format='%Y-%m-%d')
    # return the clean dataframe
    return dfclean  

# adds various technical indicators to dataframe
def add_technicals(df):
    df=df.copy()
    # first pass through all of the columns
    for c in df.columns:
        
        ### get technicals on price data ###
        if '_' not in c and 'Date' not in c:
            
            ### name columns that we want to add ###
            # columns ending with _drop are for calculations and will be dropped
            sma50=c+'_'+'sma50'
            sma200=c+'_'+'sma200'
            sma50_prop=c+'_sma50_prop' # price/sma50
            sma200_prop=c+'_sma200_prop' # price/sma200
            sma50_200_prop=c+'_sma50_200_prop' # sma50/sma200
            gain=c+'_gain'
            
            rsipos=c+'_rsipos_drop'
            rsineg=c+'_rsineg_drop'
            rollpos=c+'_rollpos_drop'
            rollneg=c+'_rollneg_drop'
            rsi=c+'_rsi' # 14 day relative strength index
            
            high52=c+'_52high'
            low52=c+'_52low'
            high52_prop=c+'_52high_prop'
            
            ### add columns indicators ###
            
            # add simple moving averages
            df[sma50]=df[c].rolling(50).mean() 
            df[sma200]=df[c].rolling(200).mean()
            
            # add gain/loss column
            df[gain]=(df[c]-df[c].shift())/df[c].shift()
            
            # calculate RSI using gain/loss column
            df[rsipos]=df[gain].apply(rsi_pos)
            df[rsineg]=df[gain].apply(rsi_neg)
            df[rollpos]=df[rsipos].rolling(14).mean()
            df[rollneg]=abs(df[rsineg].rolling(14).mean())
            df[rsi]=(100-(100/(1+(df[rollpos]/df[rollneg]))))/100
            
            
            # add 52 week high and lows
            df[high52]=df[c].rolling(200).max() 
            df[low52]=df[c].rolling(200).min()
            
            # add price proportional to sma and 52 week high and lows
            df[high52_prop]=(df[c]-df[low52])/(df[high52]-df[low52])
            df[sma50_prop]=df[c]/df[sma50]
            df[sma200_prop]=df[c]/df[sma200]
            df[sma50_200_prop]=df[sma50]/df[sma200]
                 
        ### get volume techincals ###
        elif c[-6:]=='volume':
            vol=c+'_avg'
            df[vol]=df[c].rolling(200).mean() 
    
    # second pass to use new columns to create additional features
    for c in df.columns:
        if c[-6:]=='volume':
            # define names
            measure=c+'_measure'
            avg=c+'_avg'
            # add column to calculate days volume compared to average
            df[measure]=df[c]/df[avg]
            
    
    
    # for loop drop columns no longer needed
    
    return df

# support functions for add_technicals function
def rsi_pos(x):
    if x==np.nan:
        return x
    if x>0:
        return x
    else:
        return 0

def rsi_neg(x):
    if x==np.nan:
        return x
    if x<0:
        return x
    else:
        return 0

# function to calculate strength relative to market benchmarks passed
# must have more than one benchmark or it won't work for some reason...
def relative_strength(df,benchmarks=['SPY','DJT']):
    # get start and end date
    df=df.copy()
    start=df['Date'].iloc[0]
    end=df['Date'].iloc[-1]+datetime.timedelta(days=1)
    # download index values for timeframe
    ind_data = yf.download(tickers=benchmarks, start=start, end=end,interval="1d") 
    ind_data=df_cleanup(ind_data) # clean result
    # merge with main df
    df=df.merge(ind_data,on='Date')
    
    
    for b in benchmarks:
        for c in df.columns:
            if ('_' not in c 
                and 'Date' not in c 
                and c not in benchmarks):
                
                # new column names
                rp=c+'_'+b+'_rp'
                sma_rp=c+'_'+b+'sma_rp'
                rs=c+'_'+b+'_rs'
                
                # add mansfield relative performance indicator - will be called rs
                df[rp]=df[c]/df[b]*100
                df[sma_rp]=df[rp].rolling(200).mean()
                df[rs]=df[rp]/df[sma_rp]
    
    return df

# set target - returns 0 or 1 based on whether the stock went up 10% in next 10 days
def set_target(df,n=10):
    x=df.copy()
    for c in x.columns:
        if '_' not in c and 'Date' not in c:
            # name columns to add
            target=c+'_target'
            target_gain=c+'_target_gains'
            # add columns
            x[target_gain]=(x[c].shift(-n)-x[c])/x[c]
            x[target]=x[target_gain].apply(binary_target)
    return x

#supporting function for set_target
def binary_target(x):
    if x:
        if x>0.05:
            return 1
        else:
            return 0
    else:
        return 0

# get only specified features
# need to add a list of features and descriptions
def get_features(df,features=[]):
    x=df.copy()
    cols=[]
    stocks=[]
    for c in x.columns:
        if ('_' not in c 
            and 'Date' not in c
           and c!='SPY'
           and c!='DJI'
           and c!='DJT'):
            stocks.append(c)

    for f in features:
        for c in x.columns:
            if c[-len(f):]==f:
                cols.append(c)  
    return x[cols],stocks


# gets a random sample of tickers and returns tuple containing list of train and test tickers
def get_samples(file='TSX-Tickers.csv',exchange='.TO',n=50):
    df=pd.read_csv(file)
    total=df.sample(n)
    n_train=int(n*.7)
    n_test=n-n_train
    train=total.head(n_train)
    test=total.tail(n_test)
    
    return train['Symbol'].tolist(),test['Symbol'].tolist()

# this function takes a dataset, removes unwanted featues and generates train set
# takes a while to run, there may be a better way to do it
def generate_trainset(df,features,n_period=30):
    x=df.copy()
    x,stocks=get_features(x,features)
    
    # for each stock in df
    y_values=[]
    x_values=[]
    for s in stocks:
        tempcol=[]
        # get technical columns related to stock
        for c in x.columns:
            if c[:len(s)]==s:
                tempcol.append(c)
        # put data in temporary df and get dropna values
        # more work may be required here to make sure this is working correctly
        tempdf=x[tempcol]
        tempdf=tempdf.dropna()
        print(s)
        # get 30 days of technicals and match with target
        for i in range(n_period,tempdf.shape[0]):
            # create fresh list to be populated below
            x_list=[]
            # add y_value
            y_values.append(tempdf.iloc[i-1,12])
            # get slice of features
            x_slice=tempdf.iloc[(i-n_period):i,:12]
            # loop through each of 30 days and append list of technicals
            for r in range(x_slice.shape[0]):
                x_list.append(x_slice.iloc[r].to_list())
            # append the 30 day by 12 technical idicator to be matched with y_value
            x_values.append(x_list)    

    return np.array(y_values).reshape(-1,1),np.array(x_values)





# utility to export features for a stock to csv to play around in excel and validate results
def get_stock_csv(df,stock,features,path=r"C:\Users\Alex\Desktop\Stock-Prediction-LSTM\\"):
    stockdf=df.copy()
    cols=[]
    for c in stockdf.columns:
        if stock in c:
            cols.append(c)
    
    res=stockdf[cols]
    res.to_csv(path+stock+'.csv')