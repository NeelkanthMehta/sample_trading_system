"""
Udemy | Mayank Rasu | Algorithmic Trading and Quantitative Analysis
====================================================================

Section 08: Building an automated Trading System on a shoestring budget

A trading system which can extract data periodically, perform analysis, execute trading strategy and place/ close
orders in an automated manner.

701240952
9244
"""

# Importing libraries
import copy
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fxcmpy
import statsmodels.api as sm
from stocktrends import Renko

token = "80717b31497561cce32e4a4ae76797605c4f1733"
con = fxcmpy.fxcmpy(access_token=token, log_level="error", server='demo')
pairs = ['EUR/USD', 'GBP/USD', 'USD/CHF', 'AUD/USD', 'USD/CAD']

pos_size = 10


def MACD(df, a=12, b=26, c=9):
    """Calculates MACD"""
    MAF = df['Close'].ewm(span=a, min_periods=a).mean()
    MAS = df['Close'].ewm(span=b, min_periods=b).mean()
    MACD= MAF - MAS
    sig = MACD.ewm(span=c, min_periods=c).mean()
    return MACD.dropna(inplace=True), sig.dropna(inplace=True)


def ATR(df, n):
    """Computes True Range and Average True Range"""
    HML = abs(df['High'] - df['Low'])
    HMPC= abs(df['High'] - df['Close'].shift(1))
    LMPC= abs(df['Low']  - df['Close'].shift(1))
    TR  = pd.concat([HML, HMPC, LMPC], axis=1).max(axis=1, skipna=False)
    return TR.rolling(n).mean()


def slope(Ser, n):
    """Computes slope of n consecutive points on a plot"""
    slopes = [i*0 for i in range(n-1)]
    for i in range(n, len(Ser)+1):
        y = Ser[i-n:i]
        x = np.array(range(n))
        y_scl = (y - y.min())/ (y.max() - y.min())
        x_scl = (x - x.min())/ (x.max() - x.min())
        x_scl = sm.add_constant(x_scl)
        model = sm.OLS(y_scl, x_scl)
        result= model.fit()
        slopes.append(result.params[-1])
    slope_angl= (np.rad2deg(np.arctan(np.array(slopes))))
    return np.array(slope_angl)


def renko_DF(DF):
    """Convert ohlc into renko bricks"""
    df = DF.copy()
    df.reset_index(inplace=True)
    df = df.iloc[:,[0, 1, 2, 3, 4, 6]]
    df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    df2 = Renko(df)
    df2.brick_size = round(ATR(DF, 120), 4)
    renko_df = df2.get_ohlc_data()
    renko_df['bar_num'] = np.where(renko_df['uptrend']==True, 1, np.where(renko_df['uptrend']==False))
    for i in range(1, len(renko_df['bar_num'])):
        if renko_df['bar_num'][i]>0 and renko_df['bar_num'][i-1]>0:
            renko_df['bar_num'][i] += renko_df['bar_num'][i-1]
        elif renko_df['bar_num'][i] < 0 and renko_df['bar_num'][i-1] < 0:
            renko_df['bar_num'][i] += renko_df['bar_num'][i-1]
    renko_df.drop_duplicates(subset='date', keep='last', inplace=True)
    return renko_df


def renko_merge(DF):
    """Merging renko df with original ohlc"""
    df = copy.deepcopy(DF)
    renko = renko_DF(df)
    df['date'] = df.index
    merged_df = df.merge(renko.loc[:, ['date', 'bar_num']], how='outer', on='date')
    merged_df['bar_num'].fillna(method='ffill', inplace=True)
    merged_df['MACD'], merged_df['sig'] = MACD(merged_df, 12, 26, 9)
    merged_df['slope'] = slope(merged_df['MACD'], 5)
    merged_df['sigslope'] = slope(merged_df['sig'], 5)
    return merged_df


def trading_signal(merged_df, l_s):
    """Generates Signal"""
    signal = ''
    df = copy.deepcopy(merged_df)
    if l_s == "":
        if df['bar_num'].tolist()[-1] >= 2 and df['MACD'].tolist()[-1] > df['sig'].tolist()[-1] and df['slope'].tolist()[-1] > df['sigslope'].tolist()[-1]:
            signal == 'Buy'
        elif df['bar_num'].tolist()[-1] <= -2 and df['MACD'].tolist()[-1] < df['sig'].tolist()[-1] and df['slope'].tolist()[-1] < df['sigslope'].tolist()[-1]:
            signal == 'Sell'
    elif l_s == "Long":
        if df['bar_num'].tolist()[-1] <= -2 and df['MACD'].tolist()[-1] < df['sig'].tolist()[-1] and df['slope'].tolist()[-1] < df['sigslope'].tolist()[-1]:
            signal == 'Close_Sell'
        elif df['MACD'].tolist()[-1] < df['sig'].tolist()[-1] and df['slope'].tolist()[-1] < df['sigslope'].tolist()[-1]:
            signal == 'Close'
    elif l_s == 'Short':
        if df['bar_num'].tolist()[-1] >= 2 and df['MACD'].tolist()[-1] > df['sig'].tolist()[-1] and df['slope'].tolist()[-1] > df['sigslope'].tolist()[-1]:
            signal == 'Close_Buy'
        elif df['MACD'].tolist()[-1] > df['sig'].tolist()[-1] and df['slope'].tolist()[-1] > df['sigslope'].tolist()[-1]:
            signal == 'Close'
    return signal


def main():
    try:
        open_pos = con.get_open_positions()
        for currency in pairs:
            long_short = ""
            if len(open_pos) > 0:
                open_pos_cur = open_pos[open_pos['currency']==currency]
                if len(open_pos_cur)>0:
                    if open_pos_cur['isBuy'].tolist()[0] == True:
                        long_short = 'long'
                    elif open_pos_cur['isBuy'].tolist()[0]==False:
                        long_short = 'short'
            data = con.get_candles(currency, period='m5', number=250)
            ohlc = data.iloc[:, [0, 1, 2, 3, 8]]
            ohlc.columns = ['Open', 'Close', 'High', 'Low', 'Volume']
            signal = trading_signal(renko_merge(ohlc), long_short)
            if signal == "Buy":
                con.open_trade(symbol=currency, is_buy=True, is_in_pips=True, amount=pos_size, time_in_force='GTC', stop=-8, trailing_step=True, order_type='AtMarket')
                print('New long position initiated for %s' % currency)
            elif signal == "Sell":
                con.open_trade(symbol=currency, is_buy=False, is_in_pips=True, amount=pos_size, time_in_force='GTC', stop=-8, trailing_step=True, order_type='AtMarket')
                print('New short position initiated for %s' % currency)
            elif signal == "Close":
                con.close_all_for_symbol(symbol=currency)
                print('all position closed for %s' % currency)
            elif signal == 'Close_Buy':
                con.close_all_for_symbol(symbol=currency)
                print('Existing Short position closed for %s' % currency)
                con.open_trade(symbol=currency, is_buy=True, is_in_pips=True, amount=pos_size, time_in_force='GTC', stop=-8, trailing_step=True, order_type='AtMarket')
                print('New long position initiated for %s' % currency)
            elif signal == 'Close_Sell':
                con.close_all_for_symbol(symbol=currency)
                print('Existing long position closed for %s' % currency)
                con.open_trade(symbol=currency, is_buy=False, is_in_pips=True, amount=pos_size, time_in_force='GTC', stop=-8, trailing_step=True, order_type='AtMarket')
                print('New short position initiated for %s' % currency)
    except:
        print('error encountred...skipping this iteration')


# Continuous execution
starttime = time.time()
timeout   = starttime + 60 * 60 * 1
while time.time() <= timeout:
    try:
        print('passthrough at ', time.strftime('%Y-%n-%d %H:%M:%S', time.localtime(time.time())))
        main()
        time.sleep(300 - ((time.time()-starttime)% 300.0))
    except:
        print('\n\n keyboard exception received, exiting..')
        exit()

# Close all positions and exit
for currency in pairs:
    print('Closing all positions for ', currency)
    con.close_all_for_symbol(currency)
con.close()

# # Getting candlestick data
# con.get_candles(instrument=pair, period='m5', number=250)
# """periods: m1, m5, m15, m30, H1, H2, H3, H4, H5 and H8, D1, W1, M1"""
#
# # Streaming data
# '''For streaming data, we first need to subscribe to a currency pair'''
# con.subscribe_market_data(pair)
# con.get_last_price(pair)
# con.get_prices(pair)
# con.unsubscribe_market_data(pair)
#
# starttime = time.time()
# timeout   = starttime + 60*0.2
# while time.time() <= timeout:
#     try:
#         print(con.get_last_price(pair)[0])
#     except:
#         print("Couldn't fetch data for %s" % pair)
#         pass
# con.unsubscribe_market_data(pair)
#
# # Trading account data
# con.get_accounts().T
# con.get_open_positions().T
# con.get_open_positions_summary().T
# con.get_closed_positions()
# con.get_orders()
#
# # Orders
# con.create_market_buy_order(symbol=pair, amount=100)
# con.create_market_buy_order(symbol='USD/CAD', amount=10)
# con.create_market_sell_order(symbol='USD/CAD', amount=20)
# con.create_market_sell_order(symbol=pair, amount=10)
#
# order = con.open_trade(
#     symbol='USD/CAD',
#     is_buy=True,
#     rate=1.35,
#     is_in_pips=False,
#     amount=10,
#     time_in_force='GTC',
#     stop=1.28,
#     trailing_step=True,
#     order_type='AtMarket',
#     limit=1.45
# )
#
# order = con.open_trade(
#     symbol='USD/CAD',
#     is_buy=False,
#     rate=1.35,
#     is_in_pips=False,
#     amount=10,
#     time_in_force='GTC',
#     stop=1.28,
#     trailing_step=True,
#     order_type='AtMarket',
#     limit=9
# )
#
# # con.close_trade(trade_id=tradeId, amount=1000)
# con.close_all_for_symbol(pair)
#
# # Closing connections
# con.close()

