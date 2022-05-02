# note: this file needs an internet connection AND login credentials to
# robinhood! They connect to the internet in a sorta sketchy way, but
# at least your credentials won't be saved to this device, in case you're
# worried about that.

import robin_stocks as r
from getpass import getpass
import json
import numpy as np

# given a stick's ticker, get info about that stock.
# we also handle crypto tickers with this ticker--bitcoin acceptable here.
# we return data, including a boolean value at the front:
# 0 for stock, 1 for crypto.
def getData(ticker):
    cryptoBool = ((ticker[0]=='!') and (ticker[0]!='$'))
    # figure out if we have a crypto or a stock
    symbol = ticker[1:]

    if(not cryptoBool):
        # we gather its data as a stock
        history=r.robinhood.get_stock_historicals(symbol,interval='day',span='5year')
    else:
        # we gather its data as a crypto
        history=r.robinhood.get_crypto_historicals(symbol,interval='day',span='5year')

    # now, historical data is a list of dicts that looks like:
    # {'begins_at': '2022-04-13T14:00:00Z', 'open_price': '184.520000', 'close_price': '184.680000', 'high_price': '184.820000', 'low_price': '184.250000', 'volume': 497035, 'session': 'reg', 'interpolated': False, 'symbol': 'GLD'}

    # interesting note: crypto always says that the volume is zero. Weird???
    # maybe the neural network can leverage that information if we leave it in?

    # so just output the important stuff:
    data = list()
    prices = list()
    vols = list()
    sumPrice = 0.0
    sumVol = 0.0
    useRealVol = True # we set this to false if we have a crypto
    try:
        for moment in history:
            openPrice = float(moment['open_price'])
            closePrice = float(moment['close_price'])
            avPrice = (openPrice+closePrice) / 2.0
            vol = float(moment['volume'])
            prices.append(avPrice)
            vols.append(vol)
            sumPrice+=avPrice
            sumVol += vol

        aP = sumPrice/len(prices) # this is now the average price
        sP = np.std(prices)

        if (vol!=0):
            aV = sumVol/len(vols)
            sV = np.std(vols)
        else:
            useRealVol = False

        for i in range(len(prices)):
            p = prices[i]
            v = vols[i]
            scaledPrice = (p - aP) / sP
            if useRealVol:
                scaledVol = (v - aV) / sV
            else:
                scaledVol = 0.0
            datum = [scaledPrice, scaledVol]
            data.append(datum)
    except:
        print(ticker+" has issue!")
    return data

def getTickers():
    # here's the list of stocks that we want to take data for. This will
    # control the size of the dataset that we construct.
    stockTickersList = [
        "F", "SPY", "QQQ", "TQQQ", "AAPL", "EBAY", "FL", "X","MSFT",
        "AMZN","XLK","HPQ","DELL","HPE","INTU","MCHP","NVDA","GME","AMD","TWTR",
        "GOOG","T","BAC","NIO","AAL","BA","VALE","AMC","PLTR","PFE","SNAP","XRAY",
        "GFI","SES","TDS","AU","BMI","SI","SMCI",
        "ACC","QLD","ARKW","ARKK","ITB","XHE","KCE","VCR","XTN","INDS","FDIS",
        "IHI","XLY","BOSS","IYC","BLOK","CALF","PLUG","JNJ","SAVA","NFLX","IBM",
        "LMT","LULU","EXPR","IPOF","TRV","VOO","VTI","SPHD","USO","ICLN","TAN",
        "BND","MJ","SLV","VGT","LIT","VEA","TSLA","DIS","FB","LCID","GPRO",
        "BABA","HOOD","ACB","NOK","DAL","GOOGL","RIVN","COIN","KO","CGC","SPCE",
        "PYPL","UBER","RBLX","FCEL","GM","GE","SQ","IDEX","ZNGA","PSEC","DKNG",
        "ABNB","UAL","CRON","SIRI","NKLA","LUV","MRO","NKE","PTON","RIOT",
        "GLD","TSM","UVXY","VXX","SOXS","VYM","VWO","SQQQ","PSQ","TZA",
        "XOM","CVX","NVO","ABBV","AZN","SHEL","BMY","BACHY","PTR","CVS","COP",
        "PNGAY","BUD","EL","BP","BKNG","SBUX","EADSY","REGN","BDX","VRTX",
        "EOG","PSA","ICE","CL","AON","ITW","FIS","PXD","NTR","MET","MRNA","SHOP",
        "SRE","YMM","NUTX","FRGE","SWVL","RILY","MOMO","VAXX","IESC","CWS","GBGR",
        "VIOV","TCEHY"
    ]
    # we want to add that these are stocks:
    newStockList = list()
    for stock in stockTickersList:
        newStockList.append("$"+stock)

    # and the same for our cryptos:
    cryptoTickersList=[
        "BTC","SOL","SHIB","MATIC","LTC","ETH","ETC","DOGE","COMP","BSV","BCH",
        "XMR"
    ]
    # we denote that these are cryptos with "!"
    newCryptoList = list()
    for crypto in cryptoTickersList:
        newCryptoList.append("!"+crypto)

    # combine the two lists:
    tickersList = list()
    for stock in newStockList:
        tickersList.append(stock)
    for crypto in newCryptoList:
        tickersList.append(crypto)

    # make sure no duplicates:
    newList = list()
    for ticker1 in tickersList:
        foundSame = 0
        for ticker2 in tickersList:
            if(ticker1==ticker2):
                foundSame+=1
        if foundSame <= 1:
            newList.append(ticker1)
        else:
            print("Repeat found:", ticker1)
    # sick, let's return.
    print("Prepared", len(newList),"tickers")
    return newList

# just made this for organizing purposes, but this runs everything.
def mainFunction():
    # let's keep some secrecy on these inputs:
    u=getpass("Enter username:")
    p=getpass("Enter password:")
    login_info = r.robinhood.authentication.login(username=u,password=p,store_session=False)

    tickersList = getTickers()

    X = list()
    for ticker in tickersList:
        X.append(getData(ticker))

    # and output to a file:
    '''with open("dataset.txt",'w') as f:
        for x in X:
            line = ""
            for point in x:
                line+="{:.2f}".format(point) + " "
            f.write(line+"\n")'''
    with open("dataset.json", "w") as f:
        json.dump(X, f)

    r.robinhood.authentication.logout() # end our session--no funny business.

if __name__ == '__main__':
    mainFunction()
