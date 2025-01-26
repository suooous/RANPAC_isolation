import copy
import logging
import numpy as np
import os
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import time

from inc_net import ResNetCosineIncrementalNet,SimpleVitNet
from utils.toolkit import target2onehot, tensor2numpy, accuracy

num_workers = 8

# 基类
class BaseLearner(object):
    def __init__(self, args):
        self._cur_task = -1
        self._known_classes = 0
        self._classes_seen_so_far = 0
        self.class_increments = []
        self._device = args["device"][0]
        self._multiple_gpus = args["device"]
        
        # 初始化网络，不使用预训练
        if 'resnet' in args.get('convnet_type', ''):
            self._network = ResNetCosineIncrementalNet(args, pretrained=False)
        else:
            self._network = SimpleVitNet(args, pretrained=False)
            
        # 将网络移动到指定设备
        self._network = self._network.to(self._device)
        
        # 其他初始化参数
        self._batch_size = args.get("batch_size", 128)
        self.weight_decay = args.get("weight_decay", 0.0005)
        self.min_lr = args.get("min_lr", 1e-8)
        self.args = args
        self.progress_callback = None

    def eval_task(self):
        y_pred, y_true = self._eval_cnn(self.test_loader)
        acc_total,grouped = self._evaluate(y_pred, y_true)
        return acc_total,grouped,y_pred[:,0],y_true

    def _eval_cnn(self, loader):
        """评估CNN模型性能的函数
        
        Args:
            loader: 数据加载器,包含测试数据
            
        Returns:
            tuple: 包含预测标签和真实标签的两个numpy数组
        """
        # 将网络设置为评估模式
        self._network.eval()
        y_pred, y_true = [], []
        # 遍历数据加载器中的每个batch
        for _, (_, inputs, targets) in enumerate(loader):
            # 将输入数据移至指定设备(GPU/CPU)
            inputs = inputs.to(self._device)
            # 使用torch.no_grad()避免计算梯度
            with torch.no_grad():
                # 获取模型输出的logits
                outputs = self._network(inputs)["logits"]
            # 获取每个样本预测概率最大的类别索引
            predicts = torch.topk(outputs, k=1, dim=1, largest=True, sorted=True)[1] 
            # 将预测结果和真实标签转换为numpy数组并存储
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        # 连接所有batch的结果并返回
        return np.concatenate(y_pred), np.concatenate(y_true)  
    
    def _evaluate(self, y_pred, y_true):
        """评估模型预测性能的函数
        
        Args:
            y_pred: 模型预测的标签
            y_true: 真实的标签
            
        Returns:
            acc_total: 总体准确率
            grouped: 按类别分组的准确率
        """
        ret = {}  # 创建空字典用于存储结果(当前未使用)
        # 计算总体准确率和按类别分组的准确率
        acc_total,grouped = accuracy(y_pred.T[0], y_true, self._known_classes,self.class_increments)
        return acc_total,grouped 
    
    def _compute_accuracy(self, model, loader):
        """计算模型在给定数据加载器上的准确率
        
        Args:
            model: 要评估的模型
            loader: 包含测试数据的数据加载器
            
        Returns:
            float: 模型的准确率(百分比形式,保留2位小数)
        """
        # 将模型设置为评估模式
        model.eval()
        # 初始化正确预测数和总样本数的计数器
        correct, total = 0, 0
        # 遍历数据加载器中的每个batch
        for i, (_, inputs, targets) in enumerate(loader):
            # 将输入数据移至指定设备(GPU/CPU)
            inputs = inputs.to(self._device)
            # 使用torch.no_grad()避免计算梯度
            with torch.no_grad():
                # 获取模型输出的logits
                outputs = model(inputs)["logits"]
            # 获取每个样本预测概率最大的类别索引
            predicts = torch.max(outputs, dim=1)[1]
            # 累加正确预测的样本数
            correct += (predicts.cpu() == targets).sum()
            # 累加总样本数
            total += len(targets)

        # 计算准确率(转换为百分比并保留2位小数)
        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def incremental_train(self, train_loader):
        """增量训练方法"""
        self._network.train()
        
        # 分割验证集
        dataset_size = len(train_loader.dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.2 * dataset_size))
        
        train_indices, val_indices = indices[split:], indices[:split]
        train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
        valid_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
        
        train_loader = torch.utils.data.DataLoader(
            train_loader.dataset, 
            batch_size=128,
            sampler=train_sampler,
            num_workers=4
        )
        val_loader = torch.utils.data.DataLoader(
            train_loader.dataset,
            batch_size=128,
            sampler=valid_sampler,
            num_workers=4
        )
        
        # 使用RMSprop优化器
        optimizer = optim.RMSprop(
            self._network.parameters(),
            lr=0.001,              # 初始学习率
            alpha=0.99,           # 平滑常数/动量
            eps=1e-8,            # 数值稳定性常数
            weight_decay=0.01,    # 权重衰减
            momentum=0.9         # 额外的动量项
        )
        
        # 使用余弦退火学习率调度器
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        
        # 训练过程
        best_val_loss = float('inf')
        best_state = None
        best_val_acc = 0.0
        patience = 20
        no_improve = 0
        best_checkpoint = None  # 用于存储最佳检查点信息
        
        scaler = torch.cuda.amp.GradScaler()
        
        # 创建保存模型的目录
        os.makedirs('checkpoints', exist_ok=True)
        
        for epoch in range(150):
            # 训练阶段
            self._network.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for features, targets in train_loader:
                features = features.to(self._device)
                targets = targets.to(self._device)
                
                with torch.cuda.amp.autocast():
                    outputs = self._network(features)["logits"]
                    loss = F.cross_entropy(outputs, targets)
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self._network.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()
            
            # 验证阶段
            self._network.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for features, targets in val_loader:
                    features = features.to(self._device)
                    targets = targets.to(self._device)
                    
                    outputs = self._network(features)["logits"]
                    loss = F.cross_entropy(outputs, targets)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
            
            # 计算指标
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_acc = 100. * train_correct / train_total
            val_acc = 100. * val_correct / val_total
            
            # 更新学习率
            scheduler.step()
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(self._network.state_dict())
                # 更新最佳检查点信息
                best_checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': best_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': best_val_acc,
                    'val_loss': avg_val_loss,
                    'train_acc': train_acc,
                    'train_loss': avg_train_loss,
                }
                print(f"\nRMSprop with validation accuracy: {val_acc:.2f}%")
                no_improve = 0
            else:
                no_improve += 1
            
            # 使用回调函数更新进度条
            if self.progress_callback:
                self.progress_callback(
                    epoch,
                    avg_train_loss,
                    avg_val_loss,
                    train_acc,
                    val_acc,
                    optimizer.param_groups[0]["lr"]
                )
            elif epoch % 5 == 0:
                # 如果没有回调函数，使用原来的打印方式
                print(f'Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, '
                      f'Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, '
                      f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # 训练结束后保存最佳模型
        if best_checkpoint is not None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            checkpoint_path = f'checkpoints/RMSprop_acc_{best_val_acc:.2f}_{timestamp}.pth'
            torch.save(best_checkpoint, checkpoint_path)
            print(f"\nSaved best model with validation accuracy: {best_val_acc:.2f}%")
        
        # 加载最佳模型
        self._network.load_state_dict(best_state)

    def _smooth_cross_entropy(self, pred, targets, smoothing=0.1):
        """标签平滑的交叉熵损失"""
        n_classes = pred.size(1)
        
        # 创建one-hot编码
        with torch.no_grad():
            targets_one_hot = torch.zeros_like(pred)
            targets_one_hot.scatter_(1, targets.unsqueeze(1), 1.)
        
        # 应用标签平滑
        targets_smooth = (1. - smoothing) * targets_one_hot + smoothing / n_classes
        
        # 计算损失
        log_prob = F.log_softmax(pred, dim=1)
        loss = (-targets_smooth * log_prob).sum(dim=1).mean()
        return loss

    def predict(self, test_loader):
        """预测方法"""
        self._network.eval()
        predictions = []
        
        with torch.no_grad():
            for features, _ in test_loader:
                features = features.to(self._device)
                outputs = self._network(features)["logits"]
                _, predicted = outputs.max(1)
                predictions.extend(predicted.cpu().numpy())
        
        return np.array(predictions)

    def set_progress_callback(self, callback):
        """设置进度回调函数"""
        self.progress_callback = callback

    # def continue_train(self, train_loader, pretrained_path):
    #     """继续训练方法"""
    #     self._network.train()
        
    #     # 加载预训练模型
    #     if os.path.exists(pretrained_path):
    #         checkpoint = torch.load(pretrained_path)
    #         self._network.load_state_dict(checkpoint['model_state_dict'])
    #         print(f"\nLoaded pretrained model with validation accuracy: {checkpoint['val_acc']:.2f}%")
    #         best_val_acc = checkpoint['val_acc']
    #     else:
    #         print("\nNo pretrained model found, starting from scratch")
    #         best_val_acc = 0.0
        
    #     # 分割验证集
    #     dataset_size = len(train_loader.dataset)
    #     indices = list(range(dataset_size))
    #     split = int(np.floor(0.2 * dataset_size))
        
    #     train_indices, val_indices = indices[split:], indices[:split]
    #     train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    #     valid_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
        
    #     train_loader = torch.utils.data.DataLoader(
    #         train_loader.dataset, 
    #         batch_size=128,
    #         sampler=train_sampler,
    #         num_workers=4
    #     )
    #     val_loader = torch.utils.data.DataLoader(
    #         train_loader.dataset,
    #         batch_size=128,
    #         sampler=valid_sampler,
    #         num_workers=4
    #     )
        
    #     # 使用RMSprop优化器，降低学习率
    #     optimizer = optim.RMSprop(
    #         self._network.parameters(),
    #         lr=0.0005,              # 降低学习率继续训练
    #         alpha=0.99,
    #         eps=1e-8,
    #         weight_decay=0.01,
    #         momentum=0.9
    #     )
        
    #     # 使用余弦退火学习率调度器
    #     scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #         optimizer,
    #         T_0=10,
    #         T_mult=2,
    #         eta_min=1e-6
    #     )
        
    #     # 训练过程
    #     best_state = checkpoint['model_state_dict'] if os.path.exists(pretrained_path) else None
    #     patience = 20
    #     no_improve = 0
    #     best_checkpoint = None
        
    #     scaler = torch.cuda.amp.GradScaler()
        
    #     print(f"\nStarting training from accuracy: {best_val_acc:.2f}%")
        
    #     for epoch in range(50):
    #         # 训练阶段
    #         self._network.train()
    #         train_loss = 0
    #         train_correct = 0
    #         train_total = 0
            
    #         for features, targets in train_loader:
    #             features = features.to(self._device)
    #             targets = targets.to(self._device)
                
    #             with torch.cuda.amp.autocast():
    #                 outputs = self._network(features)["logits"]
    #                 loss = F.cross_entropy(outputs, targets)
                
    #             optimizer.zero_grad()
    #             scaler.scale(loss).backward()
    #             scaler.unscale_(optimizer)
    #             torch.nn.utils.clip_grad_norm_(self._network.parameters(), max_norm=1.0)
    #             scaler.step(optimizer)
    #             scaler.update()
                
    #             train_loss += loss.item()
    #             _, predicted = outputs.max(1)
    #             train_total += targets.size(0)
    #             train_correct += predicted.eq(targets).sum().item()
            
    #         # 验证阶段
    #         self._network.eval()
    #         val_loss = 0
    #         val_correct = 0
    #         val_total = 0
            
    #         with torch.no_grad():
    #             for features, targets in val_loader:
    #                 features = features.to(self._device)
    #                 targets = targets.to(self._device)
                    
    #                 outputs = self._network(features)["logits"]
    #                 loss = F.cross_entropy(outputs, targets)
                    
    #                 val_loss += loss.item()
    #                 _, predicted = outputs.max(1)
    #                 val_total += targets.size(0)
    #                 val_correct += predicted.eq(targets).sum().item()
            
    #         # 计算指标
    #         avg_train_loss = train_loss / len(train_loader)
    #         avg_val_loss = val_loss / len(val_loader)
    #         train_acc = 100. * train_correct / train_total
    #         val_acc = 100. * val_correct / val_total
            
    #         # 更新学习率
    #         scheduler.step()
            
    #         # 保存最佳模型
    #         if val_acc > best_val_acc:
    #             best_val_acc = val_acc
    #             best_state = copy.deepcopy(self._network.state_dict())
    #             best_checkpoint = {
    #                 'epoch': epoch,
    #                 'model_state_dict': best_state,
    #                 'optimizer_state_dict': optimizer.state_dict(),
    #                 'val_acc': best_val_acc,
    #                 'val_loss': avg_val_loss,
    #                 'train_acc': train_acc,
    #                 'train_loss': avg_train_loss,
    #             }
    #             print(f"\nNew best accuracy: {val_acc:.2f}% (improved by {val_acc - checkpoint['val_acc']:.2f}%)")
    #             no_improve = 0
    #         else:
    #             no_improve += 1
    #             if no_improve >= patience:
    #                 print(f"\nEarly stopping at epoch {epoch}")
    #                 print(f"Best validation accuracy: {best_val_acc:.2f}%")
    #                 print(f"Total improvement: {best_val_acc - checkpoint['val_acc']:.2f}%")
    #                 break
            
    #         # 使用回调函数更新进度条
    #         if self.progress_callback:
    #             self.progress_callback(
    #                 epoch,
    #                 avg_train_loss,
    #                 avg_val_loss,
    #                 train_acc,
    #                 val_acc,
    #                 optimizer.param_groups[0]["lr"]
    #             )
        
    #     # 训练结束后保存最佳模型
    #     if best_checkpoint is not None:
    #         timestamp = time.strftime("%Y%m%d_%H%M%S")
    #         checkpoint_path = f'checkpoints/RMSprop_acc_{best_val_acc:.2f}_{timestamp}.pth'
    #         torch.save(best_checkpoint, checkpoint_path)
    #         print(f"\nSaved best model with validation accuracy: {best_val_acc:.2f}%")
    #         print(f"Total improvement: {best_val_acc - checkpoint['val_acc']:.2f}%")
        
    #     # 加载最佳模型
    #     if best_state is not None:
    #         self._network.load_state_dict(best_state)
    #     else:
    #         print("\nWarning: No best state found, keeping current model state")

# 继承基类
class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        if args["model_name"]!='ncm':
            if args["model_name"]=='adapter' and '_adapter' not in args["convnet_type"]:
                raise NotImplementedError('Adapter requires Adapter backbone')
            if args["model_name"]=='ssf' and '_ssf' not in args["convnet_type"]:
                raise NotImplementedError('SSF requires SSF backbone')
            if args["model_name"]=='vpt' and '_vpt' not in args["convnet_type"]:
                raise NotImplementedError('VPT requires VPT backbone')

            if 'resnet' in args['convnet_type']:
                self._network = ResNetCosineIncrementalNet(args, True)
                self._batch_size=128
            else:
                self._network = SimpleVitNet(args, True)
                self._batch_size= args["batch_size"]
            
            self.weight_decay=args["weight_decay"] if args["weight_decay"] is not None else 0.0005
            self.min_lr=args['min_lr'] if args['min_lr'] is not None else 1e-8
        else:
            self._network = SimpleVitNet(args, True)
            self._batch_size= args["batch_size"]
        self.args=args

    def after_task(self):
        self._known_classes = self._classes_seen_so_far
    
    def replace_fc(self, trainloader):
        # 提取特征
        Features_f = []
        for _, data, label in trainloader:
            embedding = self._network.convnet(data)
            Features_f.append(embedding)
        
        # 应用随机投影
        if self.args['use_RP']:
            Features_h = torch.nn.functional.relu(Features_f @ self._network.fc.W_rand)
            # 更新权重
            self.Q = self.Q + Features_h.T @ Y
            self.G = self.G + Features_h.T @ Features_h

    # 优化岭参数的函数
    def optimise_ridge_parameter(self,Features,Y):
        ridges=10.0**np.arange(-8,9)
        num_val_samples=int(Features.shape[0]*0.8)
        losses=[]
        Q_val=Features[0:num_val_samples,:].T @ Y[0:num_val_samples,:]
        G_val=Features[0:num_val_samples,:].T @ Features[0:num_val_samples,:]
        for ridge in ridges:
            Wo=torch.linalg.solve(G_val+ridge*torch.eye(G_val.size(dim=0)),Q_val).T #better nmerical stability than .inv
            Y_train_pred=Features[num_val_samples::,:]@Wo.T
            losses.append(F.mse_loss(Y_train_pred,Y[num_val_samples::,:]))
        ridge=ridges[np.argmin(np.array(losses))]
        logging.info("Optimal lambda: "+str(ridge))
        return ridge
    
    # 冻结backbone的参数
    def freeze_backbone(self,is_first_session=False):
        # Freeze the parameters for ViT.
        if 'vit' in self.args['convnet_type']:
            if isinstance(self._network.convnet, nn.Module):
                for name, param in self._network.convnet.named_parameters():
                    if is_first_session:
                        if "head." not in name and "ssf_scale" not in name and "ssf_shift_" not in name: 
                            param.requires_grad = False
                    else:
                        param.requires_grad = False
        else:
            if isinstance(self._network.convnet, nn.Module):
                for name, param in self._network.convnet.named_parameters():
                    if is_first_session:
                        if "ssf_scale" not in name and "ssf_shift_" not in name: 
                            param.requires_grad = False
                    else:
                        param.requires_grad = False

    # 显示参数数量
    def show_num_params(self,verbose=False):
        # show total parameters and trainable parameters
        total_params = sum(p.numel() for p in self._network.parameters())
        logging.info(f'{total_params:,} total parameters.')
        total_trainable_params = sum(p.numel() for p in self._network.parameters() if p.requires_grad)
        logging.info(f'{total_trainable_params:,} training parameters.')
        if total_params != total_trainable_params and verbose:
            for name, param in self._network.named_parameters():
                if param.requires_grad:
                    print(name, param.numel())

    # 训练函数
    def _train(self, train_loader, test_loader, train_loader_for_CPs):
        self._network.to(self._device)
        if self._cur_task == 0 and self.args["model_name"] in ['ncm','joint_linear']:
             self.freeze_backbone()
        if self.args["model_name"] in ['joint_linear','joint_full']: 
            #this branch updates using SGD on all tasks and should be using classes and does not use a RP head
            if self.args["model_name"] =='joint_linear':
                assert self.args['body_lr']==0.0
            self.show_num_params()
            optimizer = optim.SGD([{'params':self._network.convnet.parameters()},{'params':self._network.fc.parameters(),'lr':self.args['head_lr']}], 
                                        momentum=0.9, lr=self.args['body_lr'],weight_decay=self.weight_decay)
            scheduler=optim.lr_scheduler.MultiStepLR(optimizer,milestones=[100000])
            logging.info("Starting joint training on all data using "+self.args["model_name"]+" method")
            self._init_train(train_loader, test_loader, optimizer, scheduler)
            self.show_num_params()
        else:
            #this branch is either CP updates only, or SGD on a PETL method first task only
            if self._cur_task == 0 and self.dil_init==False:
                if 'ssf' in self.args['convnet_type']:
                    self.freeze_backbone(is_first_session=True)
                if self.args["model_name"] != 'ncm':
                    #this will be a PETL method. Here, 'body_lr' means all parameters
                    self.show_num_params()
                    optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.args['body_lr'],weight_decay=self.weight_decay)
                    scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args['tuned_epoch'], eta_min=self.min_lr)
                    #train the PETL method for the first task:
                    logging.info("Starting PETL training on first task using "+self.args["model_name"]+" method")
                    self._init_train(train_loader, test_loader, optimizer, scheduler)
                    self.freeze_backbone()
                if self.args['use_RP'] and self.dil_init==False:
                    self.setup_RP()
            if self.is_dil and self.dil_init==False:
                self.dil_init=True
                self._network.fc.weight.data.fill_(0.0)
            self.replace_fc(train_loader_for_CPs)
            self.show_num_params()
        
    
    # 设置RP
    def setup_RP(self):
        self.initiated_G=False
        self._network.fc.use_RP=True
        if self.args['M']>0:
            #RP with M > 0
            M=self.args['M']
            self._network.fc.weight = nn.Parameter(torch.Tensor(self._network.fc.out_features, M).to(device='cuda')) #num classes in task x M
            self._network.fc.reset_parameters()
            self._network.fc.W_rand=torch.randn(self._network.fc.in_features,M).to(device='cuda')
            self.W_rand=copy.deepcopy(self._network.fc.W_rand) #make a copy that gets passed each time the head is replaced
        else:
            #no RP, only decorrelation
            M=self._network.fc.in_features #this M is L in the paper
        self.Q=torch.zeros(M,self.total_classnum)
        self.G=torch.zeros(M,M)

    # 初始训练函数
    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        for epoch in range(self.args['tuned_epoch']):
            # 训练模式
            self._network.train()
            
            for _, inputs, targets in train_loader:
                # 前向传播
                logits = self._network(inputs)["logits"]
                # 计算损失
                loss = F.cross_entropy(logits, targets)
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            # 学习率调整
            scheduler.step()

        logging.info(info)
        
    

   