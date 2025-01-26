import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.ensemble import IsolationForest
from RanPAC import BaseLearner
from sklearn.cluster import KMeans

class FlowClassifier:
    def __init__(self, args, threshold=0.5):
        self.ranpac = BaseLearner(args)
        self.iforest = IsolationForest(contamination=0.1)
        self.threshold = threshold
        self.known_classes = set()
        self.next_unknown_label = 0
        self.device = args["device"][0]
        self.progress_callback = None
        
    def set_progress_callback(self, callback):
        """设置进度回调函数"""
        self.progress_callback = callback
        self.ranpac.set_progress_callback(callback)
        
    def fit_initial(self, train_dataset):
        """初始训练"""
        train_loader = DataLoader(
            train_dataset, 
            batch_size=32,
            shuffle=True
        )
        
        features, labels = [], []
        for feature, label in train_loader:
            if isinstance(feature, tuple):
                feature = feature[0]
            if torch.is_tensor(feature):
                feature = feature.numpy()
            if torch.is_tensor(label):
                label = label.numpy()
            
            features.append(feature)
            labels.append(label)
            
        X = np.concatenate(features)
        y = np.concatenate(labels)
        
        # 训练孤立森林
        self.iforest.fit(X)
        # 训练RanPAC
        self.ranpac.incremental_train(train_loader)
        # 记录已知类别
        self.known_classes.update(set(y))

    def predict_scenario1(self, test_dataset):
        """场景1: 已知类别识别"""
        test_loader = DataLoader(test_dataset, batch_size=32)
        features = []
        for feature, _ in test_loader:
            if isinstance(feature, tuple):
                feature = feature[0]
            if torch.is_tensor(feature):
                feature = feature.numpy()
            features.append(feature)
        X = np.concatenate(features)
        
        # 使用孤立森林检测异常
        anomaly_scores = -self.iforest.score_samples(X)
        predictions = self.ranpac.predict(test_loader)
        is_known = anomaly_scores < self.threshold
        
        return predictions, is_known
        
    def predict_scenario2(self, test_dataset):
        """场景2: 未知类别识别
        
        Args:
            test_dataset: 测试数据集
            
        Returns:
            np.ndarray: 新的类别标签
        """
        test_loader = DataLoader(test_dataset, batch_size=32)
        features = []
        for feature, _ in test_loader:
            features.append(feature.numpy())
        X = np.concatenate(features)
        
        # 使用孤立森林检测异常
        anomaly_scores = -self.iforest.score_samples(X)
        is_unknown = anomaly_scores >= self.threshold
        
        # 为未知类别分配新标签
        new_labels = np.zeros_like(anomaly_scores, dtype=int)
        if np.any(is_unknown):
            # 对未知样本进行聚类
            unknown_features = X[is_unknown]
            n_clusters = min(3, len(unknown_features))  # 最多3个新类别
            kmeans = KMeans(n_clusters=n_clusters)
            cluster_labels = kmeans.fit_predict(unknown_features)
            
            # 分配新的类别标签
            for i, is_unk in enumerate(is_unknown):
                if is_unk:
                    new_labels[i] = self.next_unknown_label + cluster_labels[np.sum(is_unknown[:i])]
            
            self.next_unknown_label += n_clusters
            
        return new_labels
        
    def predict_scenario3(self, test_dataset):
        """场景3: 混合识别
        
        Args:
            test_dataset: 测试数据集
            
        Returns:
            tuple: (预测标签, 是否为已知类别)
        """
        # 先检测是否为已知类别
        predictions, is_known = self.predict_scenario1(test_dataset)
        
        # 对未知类别进行处理
        unknown_labels = self.predict_scenario2(test_dataset)
        
        # 合并结果
        final_predictions = np.where(is_known, predictions, unknown_labels)
        
        return final_predictions, is_known