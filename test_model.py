import torch
from flow_classifier import FlowClassifier
from utils.data import load_flow_data
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

def test_saved_model():
    # 配置参数
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
        "dropout": 0.3,
        "label_smoothing": 0.1,
        "gradient_clip": 1.0,
        "mixup_alpha": 0.2,
        "feature_mask_prob": 0.1,
        "noise_std": 0.01,
    }
    
    # 初始化分类器
    print("\n初始化分类器...")
    classifier = FlowClassifier(args)
    
    # 加载预训练模型
    print("加载预训练模型...")
    pretrained_path = 'checkpoints/RMSprop_acc_84.15_20250126_194319.pth'
    checkpoint = torch.load(pretrained_path)
    classifier.ranpac._network.load_state_dict(checkpoint['model_state_dict'])
    print(f"加载的模型验证准确率: {checkpoint['val_acc']:.2f}%")
    
    # 加载测试数据
    print("\n加载测试数据...")
    _, test_dataset = load_flow_data("data/processed/known_test.csv")
    
    # 提取特征用于训练IsolationForest
    print("训练IsolationForest...")
    train_dataset, _ = load_flow_data("data/processed/known_traffic.csv")
    train_loader = DataLoader(train_dataset, batch_size=32)
    features = []
    for feature, _ in train_loader:
        if isinstance(feature, tuple):
            feature = feature[0]
        # 确保特征是2D的
        feature = feature.reshape(feature.shape[0], -1)  # 展平特征
        if torch.is_tensor(feature):
            feature = feature.numpy()
        features.append(feature)
    X = np.concatenate(features)
    
    # 重新训练IsolationForest
    classifier.iforest.fit(X)
    
    # 进行预测
    print("\n开始预测...")
    predictions, is_known = classifier.predict_scenario1(test_dataset)
    
    # 输出结果
    print("\n预测结果:")
    class_names = [test_dataset.dataset.get_class_name(pred) for pred in predictions[:10]]
    print(f"预测标签示例: {class_names}")
    print(f"是否已知类别示例: {is_known[:10]}")
    print(f"\n已知类别比例: {np.mean(is_known)*100:.2f}%")
    
    # 计算准确率
    test_labels = [label for _, label in test_dataset]
    known_mask = is_known
    if len(known_mask) > 0:
        known_acc = np.mean(predictions[known_mask] == np.array(test_labels)[known_mask]) * 100
        print(f"已知类别准确率: {known_acc:.2f}%")
    
    # 输出混淆矩阵
    from sklearn.metrics import classification_report
    print("\n详细分类报告:")
    print(classification_report(np.array(test_labels)[known_mask], 
                              predictions[known_mask],
                              target_names=test_dataset.dataset.classes))

if __name__ == "__main__":
    test_saved_model() 