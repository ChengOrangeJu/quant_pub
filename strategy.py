import numpy as np
import pandas as pd
import tensorflow as tf

def get_macd(filename):
    df = pd.read_csv(filename, encoding='utf-8')
    df = df[['s_info_windcode', 'trade_dt',  'vmacd_macd']]
    print(df.head(5))
    return df



if __name__ == "__main__":


    # stocks: 3500
    stocks = np.loadtxt("./data/metadata.tsv", dtype=str)
    print(stocks)

    # stock embeddings:
    # 3500 * 32
    stock_embeddings=np.loadtxt("./data/final_embeddings",delimiter=",")

    indicator_macd = get_macd("./data/indicators.csv")

    # W = tf.Variable(np.random.randn(node_in, node_out))
