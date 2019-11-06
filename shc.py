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
    print("revenue", revenue)
    print("stocks loaded", type(stocks), len(stocks))
    print("embeddings loaded", type(embeddings), len(embeddings), " shape:", embeddings.shape)
    return stocks, macd, revenue, embeddings


def custom_loss(y_true, y_pred):

    pass


stocks, macd, revenue, stock_embeddings = load_stock_data()

model = Sequential()
stock_embedding_shape = stock_embeddings.shape


#embedding layer
embedding_layer = Embedding(stock_embedding_shape[0], # 3500
                            stock_embedding_shape[1], # 32
                            weights=[stock_embeddings],
                            input_length=stock_embedding_shape[0], # 3500
                            trainable=False)
model.add(embedding_layer)

#conv1d_layer
conv1d_layer = Conv1D(1, 1, strides=1, padding='valid', data_format='channels_last', dilation_rate=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
model.add(conv1d_layer)

flatten_layer = Flatten()
model.add(flatten_layer)
model.add(Activation('softmax'))

model.compile(loss=custom_loss,)


model.fit(x_train, y_train, epochs=10)