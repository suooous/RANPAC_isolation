# Flow Classification with RanPAC and Isolation Forest

## 项目简介
这是一个结合RanPAC和孤立森林的流量分类系统,支持增量学习和未知流量识别。

## 主要功能
1. 已知类别流量识别
2. 未知类别流量检测和分类
3. 混合流量场景处理

## 系统要求
- Python 3.8+
- PyTorch 1.8+
- scikit-learn
- numpy

## 使用方法
```python
# 初始化分类器
classifier = FlowClassifier(args)

# 训练
classifier.fit_initial(X_init, y_init)

# 预测
predictions, is_known = classifier.predict_scenario1(X)  # 场景1
new_label = classifier.predict_scenario2(X)             # 场景2
predictions, is_known = classifier.predict_scenario3(X)  # 场景3
```

## 文件结构
- flow_classifier.py: 流量分类器实现
- RanPAC.py: RanPAC核心实现
- inc_net.py: 网络结构定义
- main.py: 主程序入口
- utils/data.py: 数据处理工具