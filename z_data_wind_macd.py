from WindPy import *
import numpy as np
w.start()

all_stocks = np.loadtxt("./data/metadata.tsv", dtype=str)

# stock_i = ["000002.SZ", "000001.SZ", "000004.SZ"]
stock_i = all_stocks[1:100]
start_date = "2016-03-01"
end_date = "2016-03-01"

print(",".join(stock_i))

start = 1
data = []

while start < len(all_stocks):
    end = start + 100
    if end > len(all_stocks):
        end = len(all_stocks)

    stock_i = all_stocks[start:end]
    print("start:", start, " end:", end)
    stock_i_macd = w.wsd(",".join(stock_i), "MACD", start_date, end_date,
             "MACD_L=26;MACD_S=12;MACD_N=9;MACD_IO=1;PriceAdj=F")
    print("get ", len(stock_i_macd.Data), " records")
    data_0 = stock_i_macd.Data[0]
    data.extend(data_0)
    start = end

print("total len:", len(data))

with open("./data/macd_test.txt", "w") as f:
    np.savetxt(f, data, fmt="%f", delimiter=',')
