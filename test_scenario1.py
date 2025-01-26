import numpy as np
from flow_classifier import FlowClassifier
from utils.data import load_flow_data
import torch
from tqdm import tqdm, trange

def test_scenario1():
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
        # 训练的轮次
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
    classifier = FlowClassifier(args)
    
    # 1. 加载初始数据并训练
    print("\n加载初始训练数据...")
    train_dataset, _ = load_flow_data("data/processed/known_traffic.csv")
    
    # 使用tqdm包装训练过程
    with tqdm(total=args["tuned_epoch"], desc="Training Progress", 
             bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
        def update_pbar(epoch, train_loss, val_loss, train_acc, val_acc, lr):
            pbar.set_postfix({
                'Train Loss': f'{train_loss:.4f}',
                'Val Loss': f'{val_loss:.4f}',
                'Train Acc': f'{train_acc:.2f}%',
                'Val Acc': f'{val_acc:.2f}%',
                'LR': f'{lr:.6f}'
            })
            pbar.update(1)
        
        # 将update_pbar函数传递给分类器
        classifier.set_progress_callback(update_pbar)
        classifier.fit_initial(train_dataset)
    
    print("\n初始训练完成")
    
    # 2. 加载测试数据
    print("加载测试数据...")
    _, test_dataset = load_flow_data("data/processed/known_test.csv")
    
    # 3. 进行预测
    print("开始预测...")
    predictions, is_known = classifier.predict_scenario1(test_dataset)
    
    # 4. 输出结果
    print("\n预测结果:")
    # 获取类别名称
    class_names = [train_dataset.dataset.get_class_name(pred) for pred in predictions[:10]]
    print(f"预测标签: {class_names}...")  # 显示前10个预测的类别名称
    print(f"是否已知类别: {is_known[:10]}...")  # 只显示前10个结果
    print(f"\n已知类别比例: {np.mean(is_known)*100:.2f}%")
    
    # 5. 如果测试数据有真实标签,可以计算准确率
    test_labels = [label for _, label in test_dataset]
    known_mask = is_known
    if len(known_mask) > 0:
        known_acc = np.mean(predictions[known_mask] == np.array(test_labels)[known_mask]) * 100
        print(f"已知类别准确率: {known_acc:.2f}%")

if __name__ == "__main__":
    test_scenario1() 