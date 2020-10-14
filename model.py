import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as nn

from layer import attention, dice, AUGRU
from utils import sequence_mask

class Base(tf.keras.Model):
    def __init__(self, user_count, item_count, cate_count, cate_list,
                       user_dim, item_dim, cate_dim,
                       dim_layers):
        super(Base, self).__init__()
        # 商品嵌入维度
        self.item_dim = item_dim
        # 类别嵌入维度
        self.cate_dim = cate_dim

        # 用户、商品、类别嵌入向量
        self.user_emb = nn.Embedding(user_count, user_dim)
        self.item_emb = nn.Embedding(item_count, item_dim)
        self.cate_emb = nn.Embedding(cate_count, cate_dim)
        # 给商品一个偏置项
        self.item_bias= tf.Variable(tf.zeros([item_count]), trainable=True)
        # 商品列表
        self.cate_list = cate_list

        # 对历史序列向量接上BatchNormalization、Dense
        self.hist_bn = nn.BatchNormalization()
        self.hist_fc = nn.Dense(item_dim+cate_dim)

        # 构造模型的全连接层
        self.fc = tf.keras.Sequential()
        self.fc.add(nn.BatchNormalization())
        for dim_layer in dim_layers[:-1]:
            self.fc.add(nn.Dense(dim_layer, activation='sigmoid'))
        self.fc.add(nn.Dense(dim_layers[-1], activation=None))

    # 得到嵌入向量
    def get_emb(self, user, item, history):
        # 取出用户的嵌入向量
        user_emb = self.user_emb(user)
        # 取出商品的嵌入向量
        item_emb = self.item_emb(item)
        # 根据商品id得到商品类别，进而得到类别的嵌入向量
        item_cate_emb = self.cate_emb(tf.gather(self.cate_list, item))
        # 将商品向量与商品类别向量拼接, (batch, item_dim+cate_dim)
        item_join_emb = tf.concat([item_emb, item_cate_emb], -1)
        # 取出商品的偏置项
        item_bias= tf.gather(self.item_bias, item)

        # 历史商品序列的嵌入向量
        # TODO：history 维度是？
        hist_emb = self.item_emb(history)
        # 历史商品类别序列的嵌入向量
        hist_cate_emb = self.cate_emb(tf.gather(self.cate_list, history))
        # 将商品序列向量与商品类别序列向量拼接，(batch, his_len, item_dim+cate_dim)
        hist_join_emb = tf.concat([hist_emb, hist_cate_emb], -1)

        return user_emb, item_join_emb, item_bias, hist_join_emb


    def call(self, user, item, history, length):
        # 取得嵌入向量
        user_emb, item_join_emb, item_bias, hist_join_emb = self.get_emb(user, item, history)

        # length 中存放每个batch中用户序列长度
        # sequence_mask 是用来返回 len(length) 个 True, False 数组，True 对应的位置表示有效输入数据
        # 最终返回的是二维数组
        hist_mask = tf.sequence_mask(length, max(length), dtype=tf.float32)
        # 在最后一个轴扩展一个维度之后，用tf.tile函数在最后一个维度复制item_dim+cate_dim次
        # 目的是可以把长度为item_dim+cate_dim的嵌入向量完整取出来
        hist_mask = tf.tile(tf.expand_dims(hist_mask, -1), (1,1,self.item_dim+self.cate_dim))
        # 取出嵌入向量
        hist_join_emb = tf.math.multiply(hist_join_emb, hist_mask)
        # 执行 SUM　Pooling 操作，每个用户都得到长度为 item_dim+cate_dim 的向量
        hist_join_emb = tf.reduce_sum(hist_join_emb, 1)
        # 求和之后再除以序列的长度
        hist_join_emb = tf.math.divide(hist_join_emb, tf.cast(tf.tile(tf.expand_dims(length, -1),
                                                      [1,self.item_dim+self.cate_dim]), tf.float32))
        # 对嵌入向量执行 BatchNormalization 后，再接一个全连接层
        hist_hid_emb = self.hist_fc(self.hist_bn(hist_join_emb))
        # 对各种嵌入向量进行拼接
        join_emb = tf.concat([user_emb, item_join_emb, hist_hid_emb], -1)
        # 接上全连接层后，加上一个偏置项
        output = tf.squeeze(self.fc(join_emb)) + item_bias
        # 得到预测概率
        logit = tf.keras.activations.sigmoid(output)

        return output, logit


class DIN(Base):
    def __init__(self, user_count, item_count, cate_count, cate_list,
                       user_dim, item_dim, cate_dim,
                       dim_layers):
        super(DIN, self).__init__(user_count, item_count, cate_count, cate_list,
                                  user_dim, item_dim, cate_dim,
                                  dim_layers)
        # 传入两个参数：keys_dim, dim_layers
        self.hist_at = attention(item_dim+cate_dim, dim_layers)

        self.fc = tf.keras.Sequential()
        self.fc.add(nn.BatchNormalization())
        for dim_layer in dim_layers[:-1]:
            self.fc.add(nn.Dense(dim_layer, activation=None))
            # 添加自定义激活函数 dice 
            self.fc.add(dice(dim_layer))
        self.fc.add(nn.Dense(dim_layers[-1], activation=None))

    def call(self, user, item, history, length):
        user_emb, item_join_emb, item_bias, hist_join_emb = self.get_emb(user, item, history)

        # 传入三个参数：queries, keys, keys_length
        # 得到利用 attention 得分加权求和后shape为 (batch, item_dim+cate_dim) 的矩阵
        hist_attn_emb = self.hist_at(item_join_emb, hist_join_emb, length)
        # 接上全连接层
        hist_attn_emb = self.hist_fc(self.hist_bn(hist_attn_emb))

        # 对各种嵌入向量进行拼接
        join_emb = tf.concat([user_emb, item_join_emb, hist_attn_emb], -1)

        # 接上全连接层后，加上一个偏置项
        output = tf.squeeze(self.fc(join_emb)) + item_bias
        # 得到预测概率
        logit = tf.keras.activations.sigmoid(output)

        return output, logit

class DIEN(Base):
    def __init__(self, user_count, item_count, cate_count, cate_list,
                       user_dim, item_dim, cate_dim,
                       dim_layers):
        super(DIEN, self).__init__(user_count, item_count, cate_count, cate_list,
                                   user_dim, item_dim, cate_dim,
                                   dim_layers)

        self.hist_gru = nn.GRU(item_dim+cate_dim, return_sequences=True)
        self.hist_augru = AUGRU(item_dim+cate_dim)

    def call(self, user, item, history, length):
        user_emb, item_join_emb, item_bias, hist_join_emb = self.get_emb(user, item, history)

        # hist_join_emb: (batch, his_len, item_dim+cate_dim)
        # hist_gru_emb: (batch, his_len, item_dim+cate_dim)
        hist_gru_emb = self.hist_gru(hist_join_emb)
        # item_join_emb: (batch, item_dim+cate_dim)
        # hist_attn: (batch, 1, item_dim+cate_dim) x (batch, item_dim+cate_dim, his_len) 
        # hist_attn:  (batch, 1, his_len)
        hist_attn = tf.nn.softmax(tf.matmul(tf.expand_dims(item_join_emb, 1), hist_gru_emb, transpose_b=True))
        
        # hist_hid_emb: (batch, 1, item_dim+cate_dim)
        hist_hid_emb = tf.zeros_like(hist_gru_emb[:,0,:])
        for in_emb, in_att in zip(tf.transpose(hist_gru_emb, [1,0,2]), # (his_len, batch, item_dim+cate_dim)
                                  tf.transpose(hist_attn, [2,0,1])): # (his_len, batch, 1)
            # 调用 his_len 次 AUGRU 单元，
            # 每次给 AUGRU 单元传递三个参数：
            # 第一个参数是inputs，即GRU层每个时间步的输出值 (batch, item_dim+cate_dim)
            # 第二个参数是state，即上一个AUGRU单元传递出来的隐含状态 (batch, 1, item_dim+cate_dim)
            # 第三个参数是att_score，即与候选商品一起计算得到的注意力得分 (batch, 1)
            # 每次都得到 AUGRU 单元返回的隐含状态，重新赋值给 hist_hid_emb，进而可以传递下去
            hist_hid_emb = self.hist_augru(in_emb, hist_hid_emb, in_att)

        # 将各种 embedding 按最后一维拼接
        join_emb = tf.concat([user_emb, item_join_emb, hist_hid_emb], -1)
        # 连上全连接层后加上一个偏置项得到输出值
        output = tf.squeeze(self.fc(join_emb)) + item_bias
        # 得到预测概率
        logit = tf.keras.activations.sigmoid(output)

        return output, logit
