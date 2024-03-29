import numpy as np
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Conv1D
from keras.layers import Concatenate
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers import Lambda
import keras.backend as K


model = Sequential()

embedding_matrix = np.zeros((3, 2))
# for word, i in word_index.items():
#     embedding_vector = embeddings_index.get(word)
#     if embedding_vector is not None:
#         # words not found in embedding index will be all-zeros.
#         embedding_matrix[i] = embedding_vector
embedding_matrix[0] = np.asarray([1,2],dtype='float32')
embedding_matrix[1] = np.asarray([3,-1],dtype='float32')
embedding_matrix[2] = np.asarray([6,5],dtype='float32')

embedding_layer = Embedding(3,
                            2,
                            weights=[embedding_matrix],
                            input_length=3,
                            trainable=False)

model.add(embedding_layer)

conv1d_layer = Conv1D(1, 1, strides=1, padding='valid', data_format='channels_last', dilation_rate=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

model.add(conv1d_layer)

flatten_layer = Flatten()
model.add(flatten_layer)

model.add(Activation('softmax'))