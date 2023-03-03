# -*- encoding:utf-8 -*-
"""
@作者：Javen-Huang
@文件名：KD_T.py
@时间：2023/3/1  21:55
@文档说明:测试温度T对结果的影响
"""
import numpy as np
import matplotlib.pyplot as plt
logits = np.array([-5,2,7,9])
softmax_1 = np.exp(logits)/sum(np.exp(logits))
print(softmax_1)
plt.plot(softmax_1,label= 'softmax_1')
plt.legend()
plt.show()
plt.plot(softmax_1,label= "T=1")

T= 3
softmax_3 = np.exp(logits/T)/sum(np.exp(logits/T))
plt.plot(softmax_3,label= "T=3")
T= 5
softmax_5 = np.exp(logits/T)/sum(np.exp(logits/T))
plt.plot(softmax_5,label= "T=5")
T= 7
softmax_7 = np.exp(logits/T)/sum(np.exp(logits/T))
plt.plot(softmax_7,label= "T=7")

plt.xticks(np.arange(4),['Cat','Dog','Donkey','Horse'])
plt.legend()
plt.show()
