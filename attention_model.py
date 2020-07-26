import keras.backend as K
from keras.layers import Multiply
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def get_data_recurrent(n, time_steps, input_dim, attention_column=10,attention_index=4):
    x = np.random.standard_normal(size=(n, time_steps, input_dim))  # 标准正态分布随机特征值
    y = np.random.randint(low=0, high=2, size=(n, 1))  # 二分类，随机标签值
    x[:, attention_column, attention_index] = np.reshape(y[:],len(y))
    return x, y



def get_activations(model, inputs, print_shape_only=False, layer_name=None):
    """
    生成测试数据
    x为输入，y为目标
    x为随机数构成的二维矩阵，维度为（time_steps, input_dim），在某一时间步与维度上，x与y相等。
    n为样本数
    timestep为时间布数
    input_dim为样本维度
    attention_column，attention_index处x值与y相等
    """
    activations = []
    inp = model.input
    if layer_name is None:
        outputs = [layer.output for layer in model.layers]
    else:
        outputs = [layer.output for layer in model.layers if layer.name == layer_name]  # all layer outputs
    funcs = [K.function([inp] + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
    layer_outputs = [func([inputs, 1.])[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
    return activations


def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a1 = Permute((2, 1))(inputs)
    #时间步的注意力分配
    a1 = Dense(TIME_STEPS, activation='softmax')(a1)
    a1 = Lambda(lambda x: K.mean(x, axis=1))(a1)
    a1 = RepeatVector(input_dim)(a1)
    a1 = Permute((2, 1), name='attention_vec1')(a1)
    #维度的注意力分配
    a2 = Dense(input_dim, activation='softmax', name='attention_vec2')(inputs)
    a2 = Lambda(lambda x: K.mean(x, axis=1))(a2)
    a2 = RepeatVector(TIME_STEPS)(a2)
    a_probs = Multiply()([a1, a2])
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul



def lstm_model_with_attention():
    K.clear_session()  # 清除之前的模型，省得压满内存
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    attention_mul = attention_3d_block(inputs)
    lstm_units = 32
    attention_mul = LSTM(lstm_units, return_sequences=False)(attention_mul)
    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(input=[inputs], output=output)
    return model


INPUT_DIM = 9
TIME_STEPS = 20

if __name__ == '__main__':

    np.random.seed(1337)  # for reproducibility

    # if True, the attention vector is shared across the input_dimensions where the attention is applied.

    N = 300000
    inputs_1, outputs = get_data_recurrent(N, TIME_STEPS, INPUT_DIM)
    m = lstm_model_with_attention()


    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    m.summary()

    m.fit([inputs_1], outputs, epochs=1, batch_size=64, validation_split=0.1)

    #画出时间步上的注意力权重分配
    attention_vectors = []
    for i in range(300):
        testing_inputs_1, testing_outputs = get_data_recurrent(1, TIME_STEPS, INPUT_DIM)
        attention_vector = np.mean(get_activations(m,
                                                   testing_inputs_1,
                                                   print_shape_only=True,
                                                   layer_name='attention_vec1')[0], axis=2).squeeze()

        attention_vectors.append(attention_vector)

    attention_vector_final = np.mean(np.array(attention_vectors), axis=0)
    pd.DataFrame(attention_vector_final, columns=['attention (%)']).plot(kind='bar',
                                                                         title='Attention Mechanism as '
                                                                               'a function of input'
                                                                               ' timestep.')
    plt.show()

    #画出维度上的注意力权重分配
    attention_vectors2 = []
    for i in range(300):
        testing_inputs_2, testing_outputs2 = get_data_recurrent(1, TIME_STEPS, INPUT_DIM)
        attention_vector2 = np.mean(get_activations(m,
                                                   testing_inputs_2,
                                                   print_shape_only=True,
                                                   layer_name='attention_vec2')[0], axis=1).squeeze()

        attention_vectors2.append(attention_vector2)

    attention_vector_final2 = np.mean(np.array(attention_vectors2), axis=0)
    pd.DataFrame(attention_vector_final2, columns=['attention (%)']).plot(kind='bar',
                                                                         title='Attention Mechanism as '
                                                                               'a function of input'
                                                                               ' dimensions.')
    plt.show()

