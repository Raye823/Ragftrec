import argparse
import torch
import os
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender.ftragrec import FTragrec
from recbole.model.sequential_recommender.duorec import DuoRec
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger
from recbole.utils.utils import set_color
from logging import getLogger

def run_ftragrec(model=None, dataset=None, config_file_list=None, config_dict=None, saved=True, duorec_model_path=None):
    # 配置和参数初始化
    if config_dict is None:
        config_dict = {}
    
    # 确保log_dir存在
    if 'log_dir' not in config_dict:
        import os
        # 修改日志目录格式：log/FTragrec/dataset/ 替代 log/FTragrec_dataset/
        log_dir = os.path.join("./log", f"{model}", f"{dataset}")
        # 使用正斜杠替换反斜杠，避免Windows路径问题
        log_dir = log_dir.replace('\\', '/')
        
        # 确保目录存在
        os.makedirs(log_dir, exist_ok=True)
        
        config_dict['log_dir'] = log_dir
        
    config = Config(
        model=model,
        dataset=dataset,
        config_file_list=config_file_list,
        config_dict=config_dict
    )
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    
    # 记录检索器参数配置
    logger = getLogger()
    logger.info("检索器编码器配置:")
    logger.info(f"  - 层数: {config['retriever_layers'] if 'retriever_layers' in config else 2}")
    logger.info(f"  - 内部大小: {config['retriever_inner_size'] if 'retriever_inner_size' in config else config['inner_size']}")
    logger.info(f"  - 激活函数: {config['retriever_act'] if 'retriever_act' in config else config['hidden_act']}")
    logger.info(f"  - Dropout: {config['retriever_dropout'] if 'retriever_dropout' in config else 0.1}")
    logger.info(f"  - 层归一化参数: {config['retriever_layer_norm_eps'] if 'retriever_layer_norm_eps' in config else config['layer_norm_eps']}")
    logger.info(f"  - 温度参数: {config['retriever_temperature'] if 'retriever_temperature' in config else 0.1}")
    logger.info(f"  - KL损失权重: {config['kl_weight'] if 'kl_weight' in config else 0.05}")
    
    # 数据集准备
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    
    # 模型初始化
    ftragrec = FTragrec(config, train_data.dataset).to(config['device'])
    
    # 加载DuoRec预训练权重
    if duorec_model_path is not None and os.path.exists(duorec_model_path):
        print(f"Loading pretrained DuoRec weights from {duorec_model_path}")
        
        # 加载DuoRec模型权重
        try:
            checkpoint = torch.load(duorec_model_path, map_location=config['device'])
            print(f"成功加载预训练模型文件，检查点类型: {type(checkpoint)}")
            
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                duorec_state_dict = checkpoint['state_dict']
                print(f"从检查点中获取state_dict，包含 {len(duorec_state_dict)} 个参数")
            else:
                duorec_state_dict = checkpoint
                print(f"直接使用检查点作为state_dict，包含 {len(duorec_state_dict) if isinstance(checkpoint, dict) else '未知数量'} 个参数")
                
            # 打印预训练模型的关键参数信息
            for key in list(duorec_state_dict.keys())[:5]:  # 只打印前5个键作为示例
                print(f"预训练参数示例: {key}, 形状: {duorec_state_dict[key].shape if isinstance(duorec_state_dict[key], torch.Tensor) else '非张量'}")
            
            # 检查FTragrec模型的参数
            ftragrec_keys = set(ftragrec.state_dict().keys())
            print(f"FTragrec模型包含 {len(ftragrec_keys)} 个参数")
            
            # 找出共享的参数键
            duorec_keys = set(duorec_state_dict.keys())
            shared_keys = ftragrec_keys.intersection(duorec_keys)
            print(f"找到 {len(shared_keys)} 个共享参数")
            
            # 直接加载整个模型
            missing_keys, unexpected_keys = ftragrec.load_state_dict(duorec_state_dict, strict=False)
            print(f"直接加载结果: {len(missing_keys)}个缺失键, {len(unexpected_keys)}个意外键")
            print(f"缺失的键示例: {missing_keys[:5] if missing_keys else '无'}")
            print(f"意外的键示例: {unexpected_keys[:5] if unexpected_keys else '无'}")
            
            # 仅冻结已加载的参数
            frozen_count = 0
            trainable_count = 0
            
            for name, param in ftragrec.named_parameters():
                if name in shared_keys:
                    param.requires_grad = False
                    frozen_count += 1
                    print(f"冻结参数: {name}")
                elif 'retriever_encoder' in name:
                    param.requires_grad = True
                    trainable_count += 1
                    print(f"可训练参数: {name}")
            
            print(f"冻结了 {frozen_count} 个参数，保留 {trainable_count} 个可训练参数")
            print("DuoRec weights loaded successfully!")
            
           
            # 初始评估完成后开始预缓存知识
            print("开始预缓存知识...")
            
            # 预缓存知识
            print("Precaching knowledge for FAISS index...")
            ftragrec.precached_knowledge(train_data)
            
        except Exception as e:
            print(f"加载预训练模型时出错: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("No pretrained DuoRec weights found or path not provided. Training from scratch.")
        
        # 预缓存知识
        print("Precaching knowledge for FAISS index...")
        ftragrec.precached_knowledge(train_data)
    
    # 创建自定义Trainer
    class FTragrecTrainer(Trainer):
        def __init__(self, config, model):
            super(FTragrecTrainer, self).__init__(config, model)
            
            # 确保模型保存在日志目录中
            self.saved_model_file = os.path.join(config['log_dir'], 'model.pth')
            
            # 记录检索器更新间隔
            self.retriever_update_interval = self.model.retriever_update_interval
            self.logger.info(f"检索器更新间隔: 每 {self.retriever_update_interval} 个epoch更新一次FAISS索引")
            
        def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
            # 调用父类方法进行训练
            result = super()._train_epoch(train_data, epoch_idx, loss_func, show_progress)
            
            # 检查是否需要更新FAISS索引
            if (epoch_idx + 1) % self.retriever_update_interval == 0:
                self.logger.info(f"Epoch {epoch_idx + 1}: 更新FAISS索引...")
                self.model.update_faiss_index(train_data)
                
                # 在索引更新后分析分布差异
                if hasattr(self.model, 'summarize_distribution_analysis'):
                    self.logger.info("分析检索器与推荐器分布差异...")
                    self.model.summarize_distribution_analysis()
            
            return result
            
        def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
            # 调用父类方法进行训练
            result = super().fit(train_data, valid_data, verbose, saved, show_progress, callback_fn)
            
            # 训练结束后进行最终的分布差异分析
            if hasattr(self.model, 'summarize_distribution_analysis'):
                self.logger.info("训练结束，进行最终的检索器与推荐器分布差异分析...")
                self.model.summarize_distribution_analysis()
                
            return result
    
    # 使用自定义Trainer
    trainer = FTragrecTrainer(config, ftragrec)
    
    # 开始训练
    trainer.fit(
        train_data, 
        valid_data, 
        saved=saved, 
        show_progress=config['show_progress']
    )
    

    # 在测试前设置是否使用检索增强
    if 'use_retrieval_for_predict' in config:
        ftragrec.use_retrieval_for_predict = config['use_retrieval_for_predict']
        trainer.logger.info(f"预测阶段检索增强: {'启用' if ftragrec.use_retrieval_for_predict else '禁用'}")
        if ftragrec.use_retrieval_for_predict:
            trainer.logger.info(f"预测阶段检索混合比例: {ftragrec.predict_retrieval_alpha}")
            trainer.logger.info(f"预测阶段检索温度参数: {ftragrec.predict_retrieval_temperature}")
            
    # 测试最佳模型
    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config['show_progress'])
    trainer.logger.info(f"best valid : {trainer.best_valid_result}")
    trainer.logger.info(f"test result: {test_result}")
    
    return {
        'best_valid_score': trainer.best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': trainer.best_valid_result,
        'test_result': test_result
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='ftragrec.yaml', help='配置文件路径')
    parser.add_argument('--dataset', type=str, default='ml-100k', help='数据集名称')
    parser.add_argument('--duorec_path', type=str, default=None, help='DuoRec预训练模型路径')
    # 添加一些常用参数，但不必添加所有可能的参数
    parser.add_argument('--gpu_id', type=str, default='0', help='GPU ID')
    parser.add_argument('--model', type=str, default='FTragrec', help='模型名称')
    parser.add_argument('--log_dir', type=str, default=None, help='日志目录')
    
    # 新增检索器参数
    parser.add_argument('--retriever_layers', type=int, default=None, help='检索器层数')
    parser.add_argument('--retriever_inner_size', type=int, default=None, help='检索器内部大小')
    parser.add_argument('--retriever_act', type=str, default=None, help='检索器激活函数')
    parser.add_argument('--retriever_dropout', type=float, default=None, help='检索器Dropout')
    parser.add_argument('--kl_weight', type=float, default=None, help='KL损失权重')
    
    # 使用parse_known_args代替parse_args，这样可以接受额外的参数
    args, unknown_args = parser.parse_known_args()
    
    # 创建配置字典，包括传递的所有参数
    config_dict = {}
    
    # 将已知参数添加到config_dict
    if args.gpu_id:
        config_dict['gpu_id'] = args.gpu_id
    
    if args.log_dir:
        config_dict['log_dir'] = args.log_dir
        
    # 添加检索器参数
    if args.retriever_layers is not None:
        config_dict['retriever_layers'] = args.retriever_layers
    
    if args.retriever_inner_size is not None:
        config_dict['retriever_inner_size'] = args.retriever_inner_size
    
    if args.retriever_act is not None:
        config_dict['retriever_act'] = args.retriever_act
    
    if args.retriever_dropout is not None:
        config_dict['retriever_dropout'] = args.retriever_dropout
        
    if args.kl_weight is not None:
        config_dict['kl_weight'] = args.kl_weight
    
    # 处理未知参数，将它们添加到config_dict
    i = 0
    while i < len(unknown_args):
        if unknown_args[i].startswith('--'):
            param_name = unknown_args[i][2:]  # 去掉前面的'--'
            
            # 检查是否还有下一个参数作为值
            if i + 1 < len(unknown_args) and not unknown_args[i+1].startswith('--'):
                param_value = unknown_args[i+1]
                i += 2  # 跳过参数名和值
            else:
                # 如果没有值，则视为布尔标志
                param_value = True
                i += 1  # 只跳过参数名
                
            # 尝试将字符串值转换为适当的类型
            try:
                # 如果看起来像列表或字典，使用eval
                if param_value.startswith('[') or param_value.startswith('{'):
                    config_dict[param_name] = eval(param_value)
                # 尝试转换为数字
                elif param_value.replace('.', '', 1).isdigit():
                    if '.' in param_value:
                        config_dict[param_name] = float(param_value)
                    else:
                        config_dict[param_name] = int(param_value)
                # 处理布尔值
                elif param_value.lower() in ('true', 'false'):
                    config_dict[param_name] = param_value.lower() == 'true'
                # 其他情况保持为字符串
                else:
                    config_dict[param_name] = param_value
            except:
                # 如果转换失败，保留原始字符串
                config_dict[param_name] = param_value
        else:
            # 如果参数不以--开头，跳过
            i += 1
    
    run_ftragrec(
        model=args.model, 
        dataset=args.dataset, 
        config_file_list=[args.config],
        config_dict=config_dict,
        duorec_model_path=args.duorec_path
    ) 