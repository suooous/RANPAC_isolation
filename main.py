import numpy as np
from flow_classifier import FlowClassifier
from utils.data import load_flow_data

def main():
    # 配置参数
    args = {
        "device": ["cuda:0"],
        "num_tasks": 10,
        "increment": 10,
        "batch_size": 32,
        "learning_rate": 0.001
    }
    
    # 初始化分类器
    classifier = FlowClassifier(args)
    
    # 加载初始数据并训练
    train_dataset, test_dataset = load_flow_data("data/flow_init.csv")
    classifier.fit_initial(train_dataset)
    
    # 场景1: 已知类别识别
    _, test_dataset = load_flow_data("data/flow_known.csv")
    predictions, is_known = classifier.predict_scenario1(test_dataset)
    print("场景1结果:", predictions, is_known)
    
    # 场景2: 未知类别识别
    _, test_dataset = load_flow_data("data/flow_unknown.csv")
    new_label = classifier.predict_scenario2(test_dataset)
    print("场景2结果:", new_label)
    
    # 场景3: 混合识别
    _, test_dataset = load_flow_data("data/flow_mixed.csv")
    predictions, is_known = classifier.predict_scenario3(test_dataset)
    print("场景3结果:", predictions, is_known)

if __name__ == '__main__':
    main()