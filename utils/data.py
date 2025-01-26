import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd

class FlowDataset(Dataset):
    """流量数据集类
    
    Args:
        data_path: CSV文件路径
        transform: 数据转换方法
    """
    def __init__(self, data_path, transform=None):
        # 读取CSV文件
        df = pd.read_csv(data_path)
        
        # 移除flow_key列
        if 'flow_key' in df.columns:
            df = df.drop('flow_key', axis=1)
            
        # 标签编码
        unique_labels = sorted(df['Label'].unique())  # 获取唯一标签并排序
        self.label_to_id = {label: idx for idx, label in enumerate(unique_labels)}  # 创建标签映射
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}  # 反向映射
        
        # 分离特征和标签
        self.labels = np.array([self.label_to_id[label] for label in df['Label'].values])  # 将字符串标签转换为数字
        features = df.drop('Label', axis=1).values.astype(np.float32)
        
        # 数据标准化
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(features)
        
        # 增强参数
        self.noise_std = 0.01
        self.feature_mask_prob = 0.1
        self.mixup_alpha = 0.2
        self.training = True
        
        # 保存标签信息
        self.num_classes = len(unique_labels)
        self.class_names = unique_labels

    def train(self):
        """设置为训练模式"""
        self.training = True
        return self

    def eval(self):
        """设置为评估模式"""
        self.training = False
        return self

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        
        if self.training:
            # 1. 添加高斯噪声
            noise = np.random.normal(0, self.noise_std, feature.shape)
            feature = feature + noise
            
            # 2. 特征遮蔽
            mask = np.random.rand(*feature.shape) > self.feature_mask_prob
            feature = feature * mask
            
            # 3. Mixup增强
            if np.random.rand() < 0.5:
                mix_idx = np.random.randint(len(self.features))
                mix_lambda = np.random.beta(self.mixup_alpha, self.mixup_alpha)
                mix_feature = self.features[mix_idx]
                mix_label = self.labels[mix_idx]
                
                feature = mix_lambda * feature + (1 - mix_lambda) * mix_feature
                label = mix_lambda * label + (1 - mix_lambda) * mix_label
        
        return torch.FloatTensor(feature), torch.LongTensor([label])[0]
        
    def get_class_name(self, class_id):
        """获取类别名称"""
        return self.id_to_label[class_id]
    
    def get_num_classes(self):
        """获取类别数量"""
        return self.num_classes

class StandardScaler:
    """标准化处理类"""
    def __init__(self):
        self.mean = None
        self.std = None
        
    def fit_transform(self, data):
        """计算并应用标准化"""
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        return (data - self.mean) / (self.std + 1e-8)
    
    def transform(self, data):
        """应用标准化"""
        return (data - self.mean) / (self.std + 1e-8)

def load_flow_data(data_path, train_ratio=0.8):
    """加载并分割数据集"""
    # 读取全部数据
    dataset = FlowDataset(data_path)
    
    # 计算分割点
    n_samples = len(dataset)
    split_idx = int(n_samples * train_ratio)
    
    # 随机打乱索引
    indices = np.random.permutation(n_samples)
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    # 创建训练集和测试集
    train_dataset = torch.utils.data.Subset(dataset.train(), train_indices)  # 训练集设为训练模式
    test_dataset = torch.utils.data.Subset(dataset.eval(), test_indices)    # 测试集设为评估模式
    
    return train_dataset, test_dataset