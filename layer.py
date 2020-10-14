# https://github.com/zhougr1993/DeepInterestNetwork/blob/master/din/Dice.py
import tensorflow as tf
import tensorflow.keras.layers as nn

class attention(tf.keras.layers.Layer):
    def __init__(self, keys_dim, dim_layers):
        super(attention, self).__init__()
        self.keys_dim = keys_dim

        self.fc = tf.keras.Sequential()
        for dim_layer in dim_layers[:-1]:
            self.fc.add(nn.Dense(dim_layer, activation='sigmoid'))
        self.fc.add(nn.Dense(dim_layers[-1], activation=None))

    def call(self, queries, keys, keys_length):
        # 将 queries 复制多次，复制次数与序列长度相同
        # tf.shape(keys)[1] 即序列长度
        # 得到(batch, his_len, (item_dim+cate_dim))，his_len为序列长度
        queries = tf.tile(tf.expand_dims(queries, 1), [1, tf.shape(keys)[1], 1])
        # outer product ?
        # *运算等价于 tf.multiply，是元素级别的相乘，对应位置相乘。 而 tf.matmul 则是矩阵乘法
        # (batch, his_len, (item_dim+cate_dim) * 4)
        din_all = tf.concat([queries, keys, queries-keys, queries*keys], axis=-1)
        # (batch, his_len, 1) => (batch, 1, his_len)
        outputs = tf.transpose(self.fc(din_all), [0,2,1])

        # Mask, 返回shape为(batch, his_len)的布尔矩阵
        key_masks = tf.sequence_mask(keys_length, max(keys_length), dtype=tf.bool) 
        # 扩展一个维度，使得 mask 与 outputs 维度匹配：(batch, 1, his_len)
        key_masks = tf.expand_dims(key_masks, 1) 
        # 构造一个与outputs形状相同的由近似等于0的数值(-2 ** 32 + 1)构成的矩阵
        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1) 
        # 从outputs中取出key_masks为True对应的部分，而False对应的部分用极小的数(-2 ** 32 + 1) 代替
        # outputs 最终维度仍然是 (batch, 1, his_len)
        outputs = tf.where(key_masks, outputs, paddings)

        # Scale，除以嵌入维度开平方后的数值，有利于训练
        outputs = outputs / (self.keys_dim ** 0.5)

        # Activation
        outputs = tf.keras.activations.softmax(outputs, -1)

        # 加权求和
        # keys: (batch, his_len, item_dim+cate_dim)
        # (batch, 1, his_len) x (batch, his_len, item_dim+cate_dim)
        # 得到：(batch, 1, item_dim+cate_dim) =>  (batch, item_dim+cate_dim)
        outputs = tf.squeeze(tf.matmul(outputs, keys))

        return outputs

class dice(tf.keras.layers.Layer):
    def __init__(self, feat_dim):
        super(dice, self).__init__()
        self.feat_dim = feat_dim
        self.alphas= tf.Variable(tf.zeros([feat_dim]), dtype=tf.float32)
        self.beta  = tf.Variable(tf.zeros([feat_dim]), dtype=tf.float32)

        # center=False, scale=False 目的是仅对batch数据标准化，得到 z_norm
        # 而不进行额外的 γ*z_norm + β 的操作
        self.bn = tf.keras.layers.BatchNormalization(center=False, scale=False)

    def call(self, _x, axis=-1, epsilon=0.000000001):

        # reduction_axes = list(range(len(_x.get_shape())))
        # del reduction_axes[axis]
        # broadcast_shape = [1] * len(_x.get_shape())
        # broadcast_shape[axis] = self.feat_dim

        # mean = tf.reduce_mean(_x, axis=reduction_axes)
        # brodcast_mean = tf.reshape(mean, broadcast_shape)
        # std = tf.reduce_mean(tf.square(_x - brodcast_mean) + epsilon, axis=reduction_axes)
        # std = tf.sqrt(std)
        # brodcast_std = tf.reshape(std, broadcast_shape)

        # 根据论文中的描述：标准化后使用 sigmoid 函数得到 x_p
        x_normed = self.bn(_x)
        x_p = tf.keras.activations.sigmoid(self.beta * x_normed)
        # 根据论文公式计算激活函数的输出值
        return self.alphas * (1.0 - x_p) * _x + x_p * _x

# def parametric_relu(_x):
#     with tf.variable_scope(name_or_scope='', reuse=tf.AUTO_REUSE):
#         alphas = tf.get_variable('alpha', _x.get_shape()[-1],
#                                  initializer=tf.constant_initializer(0.0),
#                                  dtype=tf.float32)
#     pos = tf.nn.relu(_x)
#     neg = alphas * (_x - abs(_x)) * 0.5
#     return pos + neg

class Bilinear(tf.keras.layers.Layer):
    def __init__(self, units):
        super(Bilinear, self).__init__()
        # 带有 bias 的 Dense 层，相当于 GRU 论文中当前时刻t输入的x的权重 W 与 b
        self.linear_act = nn.Dense(units, activation=None, use_bias=True)
        # 不带 bias 的 Dense 层，相当于 GRU 论文中前一时刻t-1传递的隐含状态 h_(t-1) 的权重的 W
        self.linear_noact = nn.Dense(units, activation=None, use_bias=False)

    def call(self, a, b, gate_b=None):
        if gate_b is None:
            # 计算更新门与重置门的输出值
            return tf.keras.activations.sigmoid(self.linear_act(a) + self.linear_noact(b))
        else:
            # 计算候选隐含状态的输出值
            return tf.keras.activations.tanh(self.linear_act(a) + tf.math.multiply(gate_b, self.linear_noact(b)))

# GRU结构参考：https://www.cnblogs.com/mantch/p/11364343.html
class AUGRU(tf.keras.layers.Layer):
    def __init__(self, units):
        super(AUGRU, self).__init__()

        self.u_gate = Bilinear(units)
        self.r_gate = Bilinear(units)
        self.c_memo = Bilinear(units)

    def call(self, inputs, state, att_score):
        # 更新门
        u = self.u_gate(inputs, state)
        # 重置门
        r = self.r_gate(inputs, state)
        # 候选隐含状态
        c = self.c_memo(inputs, state, r)
        # 跟新门乘上注意力得分
        u_= att_score * u
        # 得到最终的隐含状态
        final = (1 - u_) * state + u_ * c

        return final
