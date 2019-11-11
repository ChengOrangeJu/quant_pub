import numpy as np
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Conv1D
from keras.layers import Concatenate
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers import Lambda
import keras.backend as K
import pandas as pd
import tensorflow as tf
import sys
from keras.initializers import Constant


def load_stock_data():
    stocks = np.loadtxt("./data/stock_index.tsv", dtype=str)
    embeddings = np.loadtxt("./data/final_embeddings",delimiter=",")
    macd = np.loadtxt("./data/macd.txt", dtype=float)

    revenue_df=pd.read_csv("./data/revenue.csv", encoding='utf-8')[['instrument', 'pct_change_m']]
    revenue_dict = revenue_df.set_index('instrument').T.to_dict('index')['pct_change_m']
    revenue = []
    for i in stocks:
        if i in revenue_dict.keys():
            revenue.append(revenue_dict[i])
        else:
            revenue.append(0)
    # print("revenue", revenue)
    # print("stocks loaded", type(stocks), len(stocks))
    # print("embeddings loaded", type(embeddings), len(embeddings), " shape:", embeddings.shape)
    return stocks, np.asarray([np.nan_to_num(macd)]), np.asarray([revenue]), embeddings


def custom_loss(y_true, y_pred):


    x = y_true
    y = y_pred * macd
    y = tf.Print(y, [y],'refactored indicator', summarize=4000)

    mx = K.mean(x)
    mx = K.print_tensor(mx,message="mx = ")
    my = K.mean(y)
    my = K.print_tensor(my,message="my = ")

    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den

    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return 1 - K.square(r)



stocks, macd, revenue, stock_embeddings = load_stock_data()
print("macd", type(macd), macd)

print("revenue", type(revenue), revenue)

model = Sequential()
stock_embedding_shape = stock_embeddings.shape

#embedding layer
embedding_layer = Embedding(stock_embedding_shape[0], # 3500
                            stock_embedding_shape[1], # 32
                            embeddings_initializer=Constant(stock_embeddings),
                            # weights=[stock_embeddings],
                            input_length=stock_embedding_shape[0], # 3500
                            trainable=False)
model.add(embedding_layer)

#conv1d_layer
conv1d_layer = Conv1D(1, 1, strides=1, padding='valid', data_format='channels_last', dilation_rate=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
model.add(conv1d_layer)

flatten_layer = Flatten()
model.add(flatten_layer)
model.add(Activation('softmax'))
# model.compile(loss=custom_loss, optimizer='adam')
model.compile(loss=custom_loss, optimizer=tf.train.AdamOptimizer())

stocks_indexed = np.asarray([list(range(0, 3500))])

# print("stocks_indexed", type(stocks_indexed), stocks_indexed)
# print("mean:", np.mean(revenue[0]))
print("macd", type(macd), macd)

# model.fit(stocks_indexed, revenue, epochs=1000)
model.fit(stocks_indexed, revenue, epochs=10)
print("weight_J", model.layers[1].get_weights())
model.summary()