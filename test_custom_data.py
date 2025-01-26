import torch
from flow_classifier import FlowClassifier
from utils.data import load_flow_data, FlowDataset
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

def test_custom_data(train_path, test_path, model_path):
    """测试自定义数据集
    
    Args:
        train_path: 训练数据路径 (CSV文件)
        test_path: 测试数据路径 (CSV文件)
        model_path: 预训练模型路径 (.pth文件)
    """
    # 1. 配置参数
    args = {
        "device": ["cuda:0"] if torch.cuda.is_available() else ["cpu"],
        "num_tasks": 10,
        "increment": 10,
        "batch_size": 128,
        "learning_rate": 0.001,
        "convnet_type": "resnet18",
        "model_name": "ranpac",
        "weight_decay": 0.01,
        "min_lr": 1e-6,
        "tuned_epoch": 150,
        "use_RP": True,
        "M": 100,
        "body_lr": 0.001,
        "head_lr": 0.001,
        "pretrained": False,
        "dropout": 0.3
    }
    
    # 2. 初始化分类器
    print("\n初始化分类器...")
    classifier = FlowClassifier(args)
    
    # 3. 加载预训练模型
    print(f"加载预训练模型: {model_path}")
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        classifier.ranpac._network.load_state_dict(checkpoint['model_state_dict'])
        print(f"加载的模型验证准确率: {checkpoint['val_acc']:.2f}%")
    else:
        print(f"错误: 找不到模型文件 {model_path}")
        return
    
    # 4. 加载训练数据
    print(f"\n加载训练数据: {train_path}")
    train_dataset = FlowDataset(train_path)
    print(f"训练集大小: {len(train_dataset)}")
    
    # 5. 训练孤立森林
    print("\n训练IsolationForest...")
    train_loader = DataLoader(train_dataset, batch_size=32)
    features = []
    for feature, _ in train_loader:
        if isinstance(feature, tuple):
            feature = feature[0]
        if torch.is_tensor(feature):
            feature = feature.numpy()
        features.append(feature)
    X = np.concatenate(features)
    classifier.iforest.fit(X)
    
    # 6. 加载测试数据
    print(f"\n加载测试数据: {test_path}")
    test_dataset = FlowDataset(test_path)
    print(f"测试集大小: {len(test_dataset)}")
    
    # 7. 进行预测
    print("\n开始预测...")
    predictions, is_known = classifier.predict_scenario1(test_dataset)
    
    # 8. 输出结果
    print("\n预测结果:")
    # 显示前10个预测结果
    class_names = [test_dataset.get_class_name(pred) for pred in predictions[:10]]
    print(f"预测标签示例: {class_names}")
    print(f"是否已知类别示例: {is_known[:10]}")
    print(f"\n已知类别比例: {np.mean(is_known)*100:.2f}%")
    
    # 9. 计算准确率
    test_labels = [label for _, label in test_dataset]
    known_mask = is_known
    if len(known_mask) > 0:
        known_acc = np.mean(predictions[known_mask] == np.array(test_labels)[known_mask]) * 100
        print(f"已知类别准确率: {known_acc:.2f}%")
    
    # 10. 输出详细分类报告
    from sklearn.metrics import classification_report
    print("\n详细分类报告:")
    
    # 获取唯一的类别标签并转换为整数
    unique_labels = sorted(list(set([label.item() if torch.is_tensor(label) else label 
                                   for _, label in test_dataset])))
    
    # 获取类别名称
    class_names = [test_dataset.get_class_name(int(label)) for label in unique_labels]
    
    # 过滤掉未知类别的样本
    y_true = np.array(test_labels)[known_mask]
    y_pred = predictions[known_mask]
    
    print(classification_report(
        y_true, 
        y_pred,
        target_names=class_names,
        zero_division=0
    ))
    
    # 11. 输出每个类别的详细统计
    print("\n各类别统计:")
    for label, name in zip(unique_labels, class_names):
        # 该类别的样本
        mask = (np.array(test_labels)[known_mask] == label)
        if np.sum(mask) > 0:
            total_samples = np.sum(mask)
            correct_samples = np.sum(y_pred[mask] == label)
            acc = (correct_samples / total_samples) * 100
            print(f"{name}:")
            print(f"  样本数: {total_samples}")
            print(f"  正确预测: {correct_samples}")
            print(f"  准确率: {acc:.2f}%")
            print()

if __name__ == "__main__":
    # 设置文件路径
    train_path = "data/processed/known_traffic.csv"  # 训练数据路径
    test_path = "data/processed/known_test.csv"      # 测试数据路径
    model_path = "checkpoints\RMSprop_acc_81.71_20250126_115641.pth"  # 模型路径
    
    # 运行测试
    test_custom_data(train_path, test_path, model_path) 