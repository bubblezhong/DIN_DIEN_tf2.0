# DIN_DIEN_tf2
Deep Interest Network for Click-Through Rate Prediction / Deep Interest Evolution Network for Click-Through Rate Prediction。

代码来源：[zxxwin/DIEN-tf2](https://github.com/zxxwin/DIEN-tf2)

包含了三个模型的底层复现。具体细节可以参考以下两篇博客：

[[CTR模型] DIN（Deep Interest Network）模型解读与Deepctr实现](https://blog.csdn.net/zhong_ddbb/article/details/108992936)

[[CTR模型] DIEN（Deep Interest Evolution Network）模型解读与Deepctr实现](https://blog.csdn.net/zhong_ddbb/article/details/109002729)



## Deep Interest Netw

ork for Click-Through Rate Prediction Deep Interest Evolution Network for Click-Through Rate Prediction

I reference [zhougr1993](https://github.com/zhougr1993/DeepInterestNetwork) and [mouna99](https://github.com/mouna99/dien) code and converte it to TensorFlow 2.0.
This code performs similarly to the paper on ml-20 and amazon datasets.
You can modify the `model` called in `main.py` and then utilize a model such as Base, DIN, DIEN.

Requirements

- python 3.6
- tensorflow 2.0

Run `python main.py`