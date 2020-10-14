

## model.py

模型的定义，继承了tf.keras.Model，

```python
class Base(tf.keras.Model):
    # 初始化基本变量和工具，以便get_emb和get_call调用。
    def __init__(self, user_count, item_count, cate_count, cate_list,
                       user_dim, item_dim, cate_dim,
                       dim_layers):
        super(Base, self).__init__()
       
    # 得到嵌入向量
    def get_emb(self, user, item, history):
    
    # 生成模型结构
    def call(self, user, item, history, length):
```

第一次调用Base

```python
model = Base(user_count, item_count, cate_count, cate_list,
             args.user_dim, args.item_dim, args.cate_dim, args.dim_layers)
```

传入基本变量，执行\__init__函数。

第二次调用，传入数据，调用call函数，进行计算，得到模型的输出结果。

```python 
 output,_ = model(u,i,hist_i,sl)
```



## main.py

训练函数，核心代码

```python
def train_one_step(u,i,y,hist_i,sl):
    # 梯度带
    with tf.GradientTape() as tape:
        # 调用模型，得到模型的输出
        output,_ = model(u,i,hist_i,sl)
        # 对模型的结果计算和更新梯度
        loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=output,
                                                        labels=tf.cast(y, dtype=tf.float32)))
    # #计算梯度
    gradient = tape.gradient(loss, model.trainable_variables)  
    clip_gradient, _ = tf.clip_by_global_norm(gradient, 5.0)
    #  #更新梯度
    optimizer.apply_gradients(zip(clip_gradient, model.trainable_variables))

    loss_metric(loss)
```

