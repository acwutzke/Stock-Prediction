import pandas as pd
import numpy as np
import datetime
import yfinance as yf
from datetime import timedelta

#####################################################################################################
#####################################################################################################

# simple moving average normalized by the current stock price at the current date
def SMA(df, days): 
  for c in df.columns:
    if '_' not in c and 'Date' not in c:
      sma = pd.Series(df[c]/df[c].rolling(days).mean(), name = c + '_sma_' + str(days))
      df = df.join(sma) 
  return df

# force index
# gain over specified number of days multiplied by volume multiplier
# if volume is above average, gain increase and vice versa
def FI(df,days):
  for c in df.columns:
    if c[-6:]=='volume':
      today=df[c[:(len(c)-7)]]
      past=df[c[:(len(c)-7)]].shift(days)
      avgvol=df[c].rolling(days).sum()/days/df[c].rolling(days*2).mean()

      fi=pd.Series((today-past)/past*avgvol,name = c[:(len(c)-7)] + '_fi_' + str(days))

      df = df.join(fi)
  return df

# Relative strength index
def RSI(df,days):
  for c in df.columns:
    if '_' not in c and 'Date' not in c:
      # add daily gains
      gain=c+'_gain'
      df[gain]=(df[c]-df[c].shift())/df[c].shift(1)

      # add rsi
      rsipos=c+'_rsipos_drop'
      rsineg=c+'_rsineg_drop'
      rollpos=c+'_rollpos_drop'
      rollneg=c+'_rollneg_drop'
      rsi=c+'_rsi_'+str(days)
      df[rsipos]=df[gain].apply(rsi_pos)
      df[rsineg]=df[gain].apply(rsi_neg)
      df[rollpos]=df[rsipos].rolling(days).mean()
      df[rollneg]=abs(df[rsineg].rolling(days).mean())
      df[rsi]=(100-(100/(1+(df[rollpos]/df[rollneg]))))/100
  
  # remove calculation columns
  for c in df.columns:
    if c[-4:]=='drop':
      df=df.drop(c, axis=1)

  return df

# Supporting fuctions for the calculation of RSI above
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

# MACD
# 
def MACD(df, day1, day2):
  for c in df.columns:
    if '_' not in c and 'Date' not in c:
      sma1=df[c].rolling(day1).mean()
      sma2=df[c].rolling(day2).mean()
      currentprice=df[c]
      macd = pd.Series((sma1-sma2)/currentprice, name = c + '_macd_' + str(day1)+'_'+str(day2))
      df = df.join(macd)
  return df

# get change in MACD over given time period
def MACDiff(df,days):
  for c in df.columns:
    if 'macd' in c and 'diff' not in c:
      today=df[c]
      past=df[c].shift(days)
      macdiff = pd.Series(today-past,name = c + '_diff'+str(days))
      df = df.join(macdiff)
  return df

# set a target of 0 or 1 depending on whether the stock 
# went up by more that 5% in the next n days
def set_target(df,n=10):
    for c in df.columns:
        if '_' not in c and 'Date' not in c:
            # name columns to add
            target=c+'_y'
            target_gain=c+'_target_gains'
            # add columns
            df[target_gain]=(df[c].shift(-n)-df[c])/df[c]
            df[target]=df[target_gain].apply(binary_target)
    return df

def binary_target(x):
    if x:
        if x>0.05:
            return 1
        else:
            return 0
    else:
        return 0

# add index price compares to the index moving average
# index price > index moving average is bullish
# index should be chosen based on the group of stocks 
def index(df,index='SPY',exchange='',days=30):
  start=df['Date'].iloc[0]
  end=df['Date'].iloc[-1]+timedelta(days=1)
  data=yf.download(tickers=index, start=start, end=end,interval="1d")  
  data=pd.DataFrame(data)
  data=data.reset_index()[['Date','Adj Close']]
  sma_calc=data['Adj Close']/data['Adj Close'].rolling(days).mean()
  for c in df.columns:
    if '_' not in c and 'Date' not in c:
      sma=pd.Series(sma_calc, name=c+'_index_'+index+str(days))
      df=df.join(sma)
  return df

#####################################################################################################
#####################################################################################################

def generate_xgb_set(df,features=[]):
  stocks=[]
  for c in df.columns:
    if '_' not in c and c!='Date':
      stocks.append(c)
  
  first=0
  for s in stocks:
    if len(s)<3:
      continue
    
    
    cols=[]
    colfeat=['Date']
    for c in df.columns:
      if s == c[:len(s)]:
        cols.append(c)
    for f in features:
      for col in cols:
        if f in col:
          colfeat.append(col)
    
    tempcol=df[colfeat].columns
    newcol=[]
    for t in tempcol:
      newcol.append(t[-(len(t)-t.find('_')):])


    tempdf=df[colfeat]
    tempdf.columns=newcol
    tempdf['STOCK']=s

    if first==0:
      dftotal=tempdf
      first+=1
    else:

      dftotal=dftotal.append(tempdf)

  return dftotal

#####################################################################################################
#####################################################################################################

# back test model
def backtest_xgb(val_data,pred,confidence=0.2,n_positions=5, start_cash=100000):

  """
  Takes validation data and returns the performance for the period for plotting

  Input:

  val_data : numpy array with shape n * [date, ticker, 10 day gain/loss]
  pred : numpy array with shape n * [model prediction]
  confidence : number between 0 and 1 indicating how high model prediction must be to buy stock
  n_positions : max number of stock positions at any given time, diversification factor
  start_cash : hypothetical starting cash to buy stocks

  Output:

  dates : list of dates
  total_hist : total value (cash + fmv positions) of portfolio on each date
  cash_hist : cash on each date
  fmv_hist : fmv positions on each date
  holdings : holdings on each date


  """

  # organize validation data and predictions into single sorted dataframe
  pred_df=pd.DataFrame(pred)
  extra_df=pd.DataFrame(val_data)
  extra_df['prediction']=pred_df[0]
  df=extra_df.sort_values(by=0)

  df=df.reindex(columns=[0,2,1,'prediction'])
  df.columns=[0,1,2,'prediction']
  # return df




  # initialize variables
  cash=start_cash
  max_positions=n_positions
  d=None
  shape=df.shape

  pos={}
  fmv=0
  fmv_hist=[]
  cash_hist=[]
  total_hist=[]
  dates=[]
  holdings=[]
  x=0

  # start loop through all of the rows in the df
  for r in range(df.shape[0]):
    # if new date make new df 
    if df.iloc[r,0]!=d:
      d=df.iloc[r,0]
      temp=df.loc[(df[0] == d) & (df['prediction'] > confidence)]
      temp=temp.sort_values(by='prediction',ascending=False)
      


      # first remove positions sold after 10 days and add back cash
      # add one more day to each position that is not 10 days
      ls=list(pos.keys())
      for p in ls:
        if pos[p][2]==10:
          cash=cash+(pos[p][0]*(1+pos[p][1]))
          pos.pop(p)
        else:
          pos[p][2]+=1
        
      # add positions
      ls=list(pos.keys())
      for t in range(temp.shape[0]):
        n_pos=len(pos)
        if len(pos)<max_positions:
          if temp.iloc[t,1] not in ls:
            purch_price=cash/(max_positions-n_pos)
            pos[temp.iloc[t,1]]=[cash/(max_positions-n_pos),temp.iloc[t,2],0]
            cash=cash-purch_price

      
      # record fmv and cash to history
      fmv=0
      for p in pos:
        fmv=fmv+pos[p][0]

      fmv_hist.append(fmv)

      cash_hist.append(cash)

      total_hist.append(fmv+cash)
      dates.append(d)

      holdings.append(list(pos.keys()))

      


      # testing code
      # if str(d)[:10]=='2018-12-03':
      #   return d, pos, cash+fmv
      # else:
      #   pass
        
  return dates, total_hist, fmv_hist, cash_hist, holdings


#####################################################################################################
#####################################################################################################

# gets data
def get_data(tickers, exchange='.TO',start='2015-01-01',end='2017-12-31'):
    """ 
    tickers as list
    exchange as string, for example '.TO' for TSX
    needs yfinance, datetime, pandas
    returns last 420 trading days
    """
    
    # parse ticker list
    tickerstring=""
    for t in tickers:
        tickerstring+=t+exchange+" "
    
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

# gets list of tickers to be fed into get data
def get_samples(file='TSX-Tickers.csv',n='all',col_name='SYM'):
    if n!='all':
        df=pd.read_csv(file)
        df=df.sample(n)
        df=df.dropna()
        return df[col_name].tolist()

    df=pd.read_csv(file)
    df=df.dropna()
    return df[col_name].tolist()



