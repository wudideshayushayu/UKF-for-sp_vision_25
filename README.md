# UKF-for-sp_vision_25
Many thanks to Tongji University 适用于同济大学框架的无迹卡尔曼（UKF）
UKF C++类库使用说明
此为无迹卡尔曼滤波(UKF)的C++实现。
使用步骤：
初始化：使用初始状态(x0)、协方差(P0)及滤波参数构造UKF对象。
迭代：在循环中，先调用predict(Q, f)进行预测，再用update(z, R, h)结合测量值更新。
用户需自行定义非线性状态转移函数f和测量函数h。
通用卡尔曼，观测所有你想观测的
参数推荐值：a: 0.001~1 b:2.0(高斯分布) k:(3 - L) L为状态向量的维度
