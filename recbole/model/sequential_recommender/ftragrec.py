# -*- coding: utf-8 -*-
# @Time    : 2025/1/1
# @Author  : FTragrec Developer

"""
FTragrec
################################################

FTragrec combines Retrieval-based and Transformer-based recommendation
with a specialized RetrieverEncoder that can be fine-tuned for better retrieval quality.
"""

import torch, heapq, scipy, random, math
from torch import nn
import torch.nn.functional as F
import numpy as np
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder, FeedForward, activation_layer, MLPLayers, CrossMultiHeadAttention
from recbole.model.loss import BPRLoss


# 添加可微分记忆模块
class DifferentiableMemory(nn.Module):
    """可微分记忆模块，完全替代FAISS索引功能
    
    该模块维护一组可学习的记忆键、值向量和用户ID信息，
    实现端到端可微分的检索，并支持用户ID过滤
    """
    
    def __init__(self, hidden_size, memory_size=4096, temperature=0.1, device="cuda"):
        super(DifferentiableMemory, self).__init__()
        # 初始化记忆向量
        self.memory_keys = nn.Parameter(torch.randn(memory_size, hidden_size))
        self.memory_values = nn.Parameter(torch.randn(memory_size, hidden_size))
        
        # 初始化参数
        nn.init.normal_(self.memory_keys, mean=0.0, std=0.02)
        nn.init.normal_(self.memory_values, mean=0.0, std=0.02)
        
        # 用于记录每个记忆项对应的用户ID和序列长度
        self.register_buffer("user_ids", torch.zeros(memory_size, dtype=torch.long))
        self.register_buffer("seq_lens", torch.zeros(memory_size, dtype=torch.long))
        
        # 记录当前记忆模块使用情况
        self.register_buffer("usage_count", torch.zeros(memory_size, dtype=torch.long))
        self.current_position = 0
        self.memory_size = memory_size
        self.is_initialized = False
        
        # 温度参数和设备
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.device = device
        
    def normalize_keys(self):
        """L2归一化记忆键以提高检索效率"""
        with torch.no_grad():
            keys_norm = torch.norm(self.memory_keys, p=2, dim=1, keepdim=True)
            self.memory_keys.data = self.memory_keys.data / (keys_norm + 1e-8)
    
    def update_memories(self, query_vectors, value_vectors, user_ids, seq_lens, update_positions=None):
        """更新记忆模块中的特定位置
        
        Args:
            query_vectors: 查询向量 [batch_size, hidden_size]
            value_vectors: 值向量 [batch_size, hidden_size]
            user_ids: 用户ID [batch_size]
            seq_lens: 序列长度 [batch_size]
            update_positions: 指定更新的位置，如果为None则使用循环缓冲区策略
        """
        batch_size = query_vectors.size(0)
        with torch.no_grad():
            # 如果未指定更新位置，使用循环缓冲区策略
            if update_positions is None:
                update_positions = []
                for i in range(batch_size):
                    pos = self.current_position
                    update_positions.append(pos)
                    self.current_position = (self.current_position + 1) % self.memory_size
            
            # 更新指定位置的记忆
            for i, pos in enumerate(update_positions):
                if i >= batch_size:
                    break
                    
                self.memory_keys.data[pos] = query_vectors[i].data
                self.memory_values.data[pos] = value_vectors[i].data
                self.user_ids[pos] = user_ids[i]
                self.seq_lens[pos] = seq_lens[i]
                self.usage_count[pos] = 0  # 重置使用计数
            
            # 增加所有记忆项的使用计数
            self.usage_count += 1
            
            # 标记为已初始化
            if not self.is_initialized and batch_size > 0:
                self.is_initialized = True
                
            # 归一化记忆键
            self.normalize_keys()
    
    def forward(self, queries, user_ids=None, seq_lens=None, top_k=5, filter_same_user=True):
        """从记忆库中检索与查询相关的记忆项，支持用户ID过滤
        
        Args:
            queries: 查询向量 [batch_size, hidden_size]
            user_ids: 当前用户ID [batch_size]，用于过滤
            seq_lens: 当前序列长度 [batch_size]，用于过滤
            top_k: 返回的记忆项数量
            filter_same_user: 是否过滤相同用户的记忆项
            
        Returns:
            retrieved_keys: 检索的记忆键 [batch_size, top_k, hidden_size]
            retrieved_values: 检索的记忆值 [batch_size, top_k, hidden_size]
            attention_scores: 注意力分数 [batch_size, top_k]
        """
        if not self.is_initialized:
            # 如果记忆模块未初始化，返回空结果
            batch_size = queries.size(0)
            empty_keys = torch.zeros((batch_size, top_k, queries.size(1)), device=queries.device)
            empty_vals = torch.zeros((batch_size, top_k, queries.size(1)), device=queries.device)
            empty_weights = torch.ones((batch_size, top_k), device=queries.device) / top_k
            return empty_keys, empty_vals, empty_weights
            
        # 归一化查询向量
        queries_norm = F.normalize(queries, p=2, dim=1)
        
        # 计算查询与所有记忆键的相似度
        similarity = torch.matmul(queries_norm, self.memory_keys.transpose(0, 1))  # [batch_size, memory_size]
        
        # 应用温度缩放
        similarity = similarity / self.temperature
        
        # 创建掩码，排除相同用户的记忆(如果需要)
        batch_size = queries.size(0)
        if filter_same_user and user_ids is not None:
            user_masks = []
            for i in range(batch_size):
                if i < len(user_ids):
                    # 创建掩码：相同用户ID且序列长度大于等于当前序列长度的项为False，其他为True
                    current_user = user_ids[i]
                    current_seq_len = seq_lens[i] if seq_lens is not None else 0
                    
                    user_mask = ~((self.user_ids == current_user) & (self.seq_lens >= current_seq_len))
                    user_masks.append(user_mask)
                else:
                    # 为超出范围的批次项创建全True掩码
                    user_masks.append(torch.ones_like(self.user_ids, dtype=torch.bool))
            
            # 合并所有用户掩码
            user_mask = torch.stack(user_masks)  # [batch_size, memory_size]
            
            # 将掩码应用到相似度上（将要排除的项设为很大的负值）
            similarity = similarity.masked_fill(~user_mask, -1e9)
        
        # 获取top_k相似度及索引
        topk_similarity, topk_indices = torch.topk(similarity, min(top_k, similarity.size(1)), dim=1)
        
        # 计算softmax得到注意力权重
        topk_weights = F.softmax(topk_similarity, dim=1)
        
        # 收集top_k记忆键和值
        retrieved_keys = self.memory_keys[topk_indices]  # [batch_size, top_k, hidden_size]
        retrieved_values = self.memory_values[topk_indices]  # [batch_size, top_k, hidden_size]
        
        return retrieved_keys, retrieved_values, topk_weights


class FTragrec(SequentialRecommender):
    r"""
    FTragrec: Fine-Tuned Retrieval-Augmented Recommendation

    FTragrec implements a retrieval-augmented recommendation system with a
    specialized RetrieverEncoder that can be fine-tuned to align with the recommendation model's preferences.
    """

    def __init__(self, config, dataset):
        super(FTragrec, self).__init__(config, dataset)

        # 共同基础参数
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']
        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']
        
        # 检索相关参数
        self.len_lower_bound = config["len_lower_bound"] if "len_lower_bound" in config else -1
        self.len_upper_bound = config["len_upper_bound"] if "len_upper_bound" in config else -1
        self.len_bound_reverse = config["len_bound_reverse"] if "len_bound_reverse" in config else True
        self.topk = config['top_k'] if 'top_k' in config else 5
        
        # RetrieverEncoder相关参数
        self.retriever_layers = config['retriever_layers'] if 'retriever_layers' in config else 2
        self.retriever_temperature = config['retriever_temperature'] if 'retriever_temperature' in config else 0.1
        self.retriever_dropout = config['retriever_dropout'] if 'retriever_dropout' in config else 0.1
        self.retriever_update_interval = config['retriever_update_interval'] if 'retriever_update_interval' in config else 5
        self.kl_weight = config['kl_weight'] if 'kl_weight' in config else 0.05  # 降低KL损失权重，因为现在是双向的
        
        # 可微分记忆模块参数 - 现在作为主要检索机制
        self.memory_size = config['memory_size'] if 'memory_size' in config else 8192
        self.use_diff_memory = True  # 始终使用可微分记忆模块
        self.memory_weight = config['memory_weight'] if 'memory_weight' in config else 0.3
        self.diff_temperature = config['diff_temperature'] if 'diff_temperature' in config else 0.1
        self.filter_same_user = config['filter_same_user'] if 'filter_same_user' in config else True
        
        # 训练相关参数
        self.batch_size = config['train_batch_size']
        self.current_epoch = 0

        # 定义基础层和损失函数
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        
        # 序列编码器
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )
        
        # 检索器编码器 - 基于Contriever设计
        self.retriever_encoder_layers = nn.ModuleList([
            FeedForward(
                self.hidden_size, 
                self.inner_size, 
                self.retriever_dropout, 
                self.hidden_act, 
                self.layer_norm_eps
            ) for _ in range(self.retriever_layers)
        ])
        
        # 可微分记忆模块 - 用于端到端训练
        self.diff_memory = DifferentiableMemory(
            hidden_size=self.hidden_size,
            memory_size=self.memory_size,
            temperature=self.diff_temperature,
            device=config['device'] if 'device' in config else 'cuda'
        )
        
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # 参数初始化
        self.apply(self._init_weights)
        
        # 数据集
        self.dataset = dataset
        
        # 统计已处理的样本数
        self.processed_samples = 0

        # 新增预测阶段检索增强相关参数
        self.use_retrieval_for_predict = config['use_retrieval_for_predict'] if 'use_retrieval_for_predict' in config else True
        self.predict_retrieval_alpha = config['predict_retrieval_alpha'] if 'predict_retrieval_alpha' in config else 0.5
        self.predict_retrieval_temperature = config['predict_retrieval_temperature'] if 'predict_retrieval_temperature' in config else 0.1
        
        # 混合权重alpha参数
        self.alpha = config['alpha'] if 'alpha' in config else 0.5
        
        # 增强推荐损失权重参数
        self.enhanced_rec_weight = config['enhanced_rec_weight'] if 'enhanced_rec_weight' in config else 0.8
        
        # 添加基于注意力的增强层 - 单通道增强
        self.attention_temperature = config['attention_temperature'] if 'attention_temperature' in config else 0.2
        self.dropout_rate = config['dropout_rate'] if 'dropout_rate' in config else 0.1
        
        # 序列增强的注意力层
        self.seq_tar_attention = CrossMultiHeadAttention(
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            hidden_dropout_prob=self.dropout_rate,
            attn_dropout_prob=self.dropout_rate,
            layer_norm_eps=self.layer_norm_eps,
            attn_tau=self.attention_temperature
        )
        
        # 序列增强的前馈网络
        self.seq_tar_fnn = FeedForward(
            self.hidden_size, 
            self.inner_size, 
            self.dropout_rate, 
            self.hidden_act, 
            self.layer_norm_eps
        )
        
        # 统计模型参数量
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'\nFTragrec模型总参数量: {total_params:,}')
        print(f'可训练参数量: {trainable_params:,}')
        
    def _init_weights(self, module):
        """ 初始化权重 """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_attention_mask(self, item_seq):
        """生成从左到右单向注意力掩码"""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # 单向掩码
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, item_seq, item_seq_len):
        """序列编码器前向传播"""
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        # 确保输出需要梯度
        return output.requires_grad_(True)

    def retriever_forward(self, seq_output):
        """检索器编码器前向传播 - 对序列表示进行非线性变换"""
        try:
            retriever_output = seq_output
            for layer in self.retriever_encoder_layers:
                retriever_output = layer(retriever_output)
            
            return retriever_output
            
        except Exception as e:
            print(f"retriever_forward方法出错: {e}")
            # 失败时返回原始输入，确保程序不会崩溃
            return seq_output

    def precached_knowledge(self, train_dataloader=None):
        """初始化可微分记忆模块 - 预填充部分训练样本"""
        print("开始初始化可微分记忆模块...")
        
        # 使用提供的dataloader或默认dataset
        dataloader_to_use = train_dataloader if train_dataloader is not None else self.dataset
        
        # 遍历数据集中的一部分交互来初始化记忆
        max_init_samples = min(self.memory_size, 2000)  # 限制初始化样本数量
        batch_count = 0
        valid_samples = 0
        
        for batch_idx, interaction in enumerate(dataloader_to_use):
            try:
                if valid_samples >= max_init_samples:
                    break
                    
                interaction = interaction.to(self.diff_memory.device)
                batch_size = interaction[self.ITEM_SEQ].shape[0]
                
                # 根据序列长度过滤
                if self.len_lower_bound != -1 or self.len_upper_bound != -1:
                    if self.len_lower_bound != -1 and self.len_upper_bound != -1:
                        look_up_indices = (interaction[self.ITEM_SEQ_LEN]>=self.len_lower_bound) * (interaction[self.ITEM_SEQ_LEN]<=self.len_upper_bound)
                    elif self.len_upper_bound != -1:
                        look_up_indices = interaction[self.ITEM_SEQ_LEN]<self.len_upper_bound
                    else:
                        look_up_indices = interaction[self.ITEM_SEQ_LEN]>self.len_lower_bound
                    if self.len_bound_reverse:
                        look_up_indices = ~look_up_indices
                else:
                    look_up_indices = interaction[self.ITEM_SEQ_LEN]>-1
                
                # 如果没有有效样本，跳过这个批次
                if look_up_indices.sum().item() == 0:
                    continue
                
                # 获取经过过滤的数据
                item_seq = interaction[self.ITEM_SEQ][look_up_indices]
                item_seq_len = interaction[self.ITEM_SEQ_LEN][look_up_indices]
                user_ids = interaction[self.USER_ID][look_up_indices]
                tar_items = interaction[self.POS_ITEM_ID][look_up_indices]
                
                # 获取序列表示
                seq_output = self.forward(item_seq, item_seq_len)
                
                # 使用检索器编码器获取查询向量
                retriever_output = self.retriever_forward(seq_output)
                
                # 获取目标物品表示
                tar_items_emb = self.item_embedding(tar_items)
                
                # 更新可微分记忆模块
                current_batch_size = retriever_output.size(0)
                update_size = min(current_batch_size, max_init_samples - valid_samples)
                
                self.diff_memory.update_memories(
                    retriever_output[:update_size], 
                    tar_items_emb[:update_size],
                    user_ids[:update_size].cpu(),
                    item_seq_len[:update_size].cpu()
                )
                
                valid_samples += update_size
                batch_count += 1
                
                # 打印进度
                if batch_count % 10 == 0:
                    print(f"已初始化 {valid_samples}/{max_init_samples} 个样本")
                    
            except Exception as e:
                print(f"处理批次 {batch_idx} 时出错: {e}")
                continue
                
        print(f"记忆模块初始化完成: 共处理 {batch_count} 个批次，初始化了 {valid_samples} 个样本")
        print(f"记忆模块大小: {self.memory_size}")
        
        # 归一化记忆键以提高检索效率
        self.diff_memory.normalize_keys()

    def update_memory(self, batch_seqs, batch_targets, user_ids=None, seq_lens=None):
        """更新可微分记忆模块
        
        该方法在训练过程中周期性调用，用于将当前批次的序列表示和目标物品表示
        作为知识更新到可微分记忆模块中。
        
        Args:
            batch_seqs: 批次序列表示 [batch_size, hidden_size]
            batch_targets: 批次目标物品表示 [batch_size, hidden_size]
            user_ids: 用户ID [batch_size]，用于记录所有者信息
            seq_lens: 序列长度 [batch_size]，用于记录序列长度
        """
        if not self.use_diff_memory or not hasattr(self, 'diff_memory'):
            return
            
        # 处理序列表示
        retriever_output = self.retriever_forward(batch_seqs)
        
        # 从当前批次中随机选择样本更新记忆模块 (防止单个批次占据过多记忆空间)
        batch_size = batch_seqs.size(0)
        
        # 计算会被更新的样本数
        update_size = min(batch_size, max(1, self.memory_size // 100))  # 每次最多更新记忆库大小的1%
        
        # 随机选择更新的样本索引
        if batch_size > update_size:
            indices = torch.randperm(batch_size)[:update_size]
            update_queries = retriever_output[indices]
            update_values = batch_targets[indices]
            
            if user_ids is not None:
                update_users = user_ids[indices]
            else:
                update_users = torch.zeros(update_size, dtype=torch.long, device=batch_seqs.device)
                
            if seq_lens is not None:
                update_seqlens = seq_lens[indices]
            else:
                update_seqlens = torch.zeros(update_size, dtype=torch.long, device=batch_seqs.device)
        else:
            update_queries = retriever_output
            update_values = batch_targets
            update_users = user_ids if user_ids is not None else torch.zeros(batch_size, dtype=torch.long, device=batch_seqs.device)
            update_seqlens = seq_lens if seq_lens is not None else torch.zeros(batch_size, dtype=torch.long, device=batch_seqs.device)
        
        # 更新记忆模块
        # 选择使用率最低的记忆项进行更新
        self.diff_memory.update_memories(
            update_queries, 
            update_values,
            update_users.cpu() if update_users.device != torch.device('cpu') else update_users,
            update_seqlens.cpu() if update_seqlens.device != torch.device('cpu') else update_seqlens
        )
        
        # 增加处理的样本计数
        self.processed_samples += batch_size

    def update_memory_index(self):
        """更新可微分记忆模块索引"""
        print("开始优化可微分记忆模块...")
        
        # 简单地归一化记忆键以保持检索效率
        self.diff_memory.normalize_keys()
        
        # 打印当前状态
        print(f"记忆模块状态: 已处理样本数={self.processed_samples}")
        
        # 可以在这里添加优化逻辑，如移除最少使用的项等
        # 这里我们保持简单，只进行归一化

    def retrieve_seq_tar(self, queries, batch_user_id, batch_seq_len, topk=5):
        """检索相似的序列和目标项，使用可微分记忆模块
        
        该方法使用可微分记忆模块进行端到端可训练的检索
        """
        try:
            # 使用检索器编码器处理查询
            retriever_queries = self.retriever_forward(queries)
            
            # 将batch_user_id和batch_seq_len转换为tensor (如果它们是列表)
            if isinstance(batch_user_id, list):
                user_ids = torch.tensor(batch_user_id, device=retriever_queries.device)
            else:
                user_ids = batch_user_id
                
            if isinstance(batch_seq_len, list):
                seq_lens = torch.tensor(batch_seq_len, device=retriever_queries.device)
            else:
                seq_lens = batch_seq_len
            
            # 从可微分记忆中检索
            memory_keys, memory_values, memory_weights = self.diff_memory(
                retriever_queries, 
                user_ids=user_ids,
                seq_lens=seq_lens,
                top_k=min(topk, self.memory_size),
                filter_same_user=self.filter_same_user
            )
            
            # 确保记忆结果保留梯度信息
            memory_keys.requires_grad_(True)
            memory_values.requires_grad_(True)
            
            return memory_keys, memory_values
            
        except Exception as e:
            print(f"可微分记忆检索出错: {e}")
            # 出错时返回空结果
            batch_size = queries.size(0)
            empty_seqs = torch.zeros((batch_size, 1, self.hidden_size), device=queries.device, requires_grad=True)
            empty_tars = torch.zeros((batch_size, 1, self.hidden_size), device=queries.device, requires_grad=True)
            return empty_seqs, empty_tars

    def compute_retrieval_scores(self, query_vectors, candidate_vectors):
        """计算检索似然分布 - 计算查询序列与检索序列的相似度并转换为概率分布
        
        Args:
            query_vectors: 查询向量 [batch_size, 1, hidden_size]
            candidate_vectors: 候选向量 [batch_size, n_candidates, hidden_size]
            
        Returns:
            retrieval_probs: 相似度概率分布 [batch_size, 1, n_candidates]
        """
        # 计算相似度分数
        similarity = torch.bmm(query_vectors, candidate_vectors.transpose(1, 2))
        
        # 应用温度缩放并转换为概率分布
        retrieval_logits = similarity / self.predict_retrieval_temperature
        retrieval_probs = torch.softmax(retrieval_logits, dim=-1)
        
        return retrieval_probs  # [batch_size, 1, topk]

    def compute_recommendation_scores(self, seq_output, retrieved_seqs, retrieved_tars, pos_items=None):
        """计算推荐模型的评分分布 - 基于每个检索目标嵌入增强的表示对目标项的预测概率"""
        batch_size, n_retrieved, hidden_size = retrieved_tars.size()
        device = seq_output.device
        
        # 获取目标项的嵌入向量
        if pos_items is None:
            # 如果未提供目标项，则需要从外部获取
            # 这部分需要在调用函数时传入正确的pos_items
            raise ValueError("目标物品信息(pos_items)不能为空，需要用于计算推荐分布")
        
        pos_items_emb = self.item_embedding(pos_items)  # [batch_size, hidden_size]
        
        # 初始化保存每个检索结果对应的打分
        scores_list = []
        
        # 对每个检索到的目标嵌入单独处理
        for i in range(n_retrieved):
            # 获取当前检索结果的目标嵌入
            current_tar_emb = retrieved_tars[:, i, :]  # [batch_size, hidden_size]
            
            # 将目标嵌入与原始序列表示混合(使用未经过retrieverencoder的原始序列表示)
            temp_alpha = self.alpha
            enhanced_seq = (1 - temp_alpha) * seq_output + temp_alpha * current_tar_emb
            
            # 计算增强表示与真实目标项的相似度(点积)
            similarity = torch.sum(enhanced_seq * pos_items_emb, dim=-1)  # [batch_size]
            
            # 应用sigmoid函数将相似度转换为概率
            prob = torch.sigmoid(similarity)
            scores_list.append(prob)
        
        # 将所有检索结果的评分堆叠
        combined_scores = torch.stack(scores_list, dim=1)  # [batch_size, n_retrieved]
        
        # 为每个用户归一化评分为概率分布
        recommendation_probs = combined_scores / (combined_scores.sum(dim=1, keepdim=True) + 1e-8)
        
        return recommendation_probs

    def compute_kl_loss(self, retrieval_probs, recommendation_probs):
        """计算JS散度损失（双向KL散度的平均）
        
        Args:
            retrieval_probs: 检索分布 [batch_size, n_retrieved]
            recommendation_probs: 推荐分布 [batch_size, n_retrieved]
        
        Returns:
            js_div: JS散度损失
        """
        # 数值稳定性
        epsilon = 1e-8
        retrieval_probs = retrieval_probs + epsilon
        recommendation_probs = recommendation_probs + epsilon
        
        # 计算平均分布
        mean_probs = 0.5 * (retrieval_probs + recommendation_probs)
        
        # KL(retrieval||mean) - 使用数值稳定的计算方式
        kl_retrieval = torch.sum(retrieval_probs * (torch.log(retrieval_probs) - torch.log(mean_probs)), dim=-1)
        
        # KL(recommendation||mean) - 使用数值稳定的计算方式
        kl_recommendation = torch.sum(recommendation_probs * (torch.log(recommendation_probs) - torch.log(mean_probs)), dim=-1)
        
        # JS散度 = 0.5 * (KL(P||M) + KL(Q||M))
        js_div = 0.5 * (kl_retrieval + kl_recommendation)
        
        return js_div.mean()

    def calculate_loss(self, interaction):
        """计算模型损失，包括推荐损失、JS散度损失和记忆模块损失"""
        # 获取序列和长度
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        batch_user_id = list(interaction[self.USER_ID].detach().cpu().numpy())
        batch_seq_len = list(item_seq_len.detach().cpu().numpy())
        
        # 获取序列表示
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        
        # 第一部分：JS散度损失计算 - 用于优化检索器
        js_loss = 0.0
        # 检索相似序列和目标项
        retrieved_seqs, retrieved_tars = self.retrieve_seq_tar(
            seq_output, batch_user_id, batch_seq_len, topk=self.topk
        )
        
        # 只有在成功检索到序列时才计算JS散度损失
        if retrieved_seqs.size(0) > 0 and torch.sum(retrieved_seqs).item() > 1e-6:
            # 计算检索似然分布
            retrieval_probs = self.compute_retrieval_scores(
                seq_output.unsqueeze(1), retrieved_seqs
            ).squeeze(1)
            
            # 计算推荐模型的评分分布
            recommendation_probs = self.compute_recommendation_scores(
                seq_output, retrieved_seqs, retrieved_tars, pos_items
            )
            
            # 计算JS散度损失
            js_loss = self.compute_kl_loss(retrieval_probs, recommendation_probs)
        
        # 第二部分：可微分记忆模块的对比学习损失
        memory_loss = 0.0
        if self.use_diff_memory and hasattr(self, 'diff_memory'):
            try:
                # 获取当前批次的表示
                retriever_output = self.retriever_forward(seq_output)
                pos_items_emb = self.item_embedding(pos_items)
                
                # 从可微分记忆模块中检索
                memory_keys, memory_values, memory_weights = self.diff_memory(
                    retriever_output, top_k=min(5, self.memory_size)
                )
                
                # 计算与正例的相似度
                pos_sim = torch.bmm(
                    pos_items_emb.unsqueeze(1),  # [batch_size, 1, hidden_size]
                    memory_values.transpose(1, 2)  # [batch_size, hidden_size, top_k]
                ).squeeze(1)  # [batch_size, top_k]
                
                # 最大化正例相似度的对数似然
                log_probs = torch.log_softmax(pos_sim / 0.1, dim=1)  # 使用温度参数0.1
                memory_loss = -torch.mean(torch.sum(memory_weights * log_probs, dim=1))
                
            except Exception as e:
                print(f"计算记忆模块损失时出错: {e}")
        
        # 第三部分：序列增强 - 使用注意力机制的单通道增强
        enhanced_seq_output = self.sequence_augmentation(seq_output, batch_user_id, batch_seq_len)
        
        # 使用增强后的序列表示计算推荐损失
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(enhanced_seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(enhanced_seq_output * neg_items_emb, dim=-1)  # [B]
            rec_loss = self.loss_fct(pos_score, neg_score)
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(enhanced_seq_output, test_item_emb.transpose(0, 1))
            rec_loss = self.loss_fct(logits, pos_items)
        
        # 总损失 = 推荐损失 + KL权重 * JS散度损失 + 记忆权重 * 记忆模块损失
        total_loss = rec_loss
        
        if js_loss != 0.0:
            total_loss = total_loss + self.kl_weight * js_loss
            
        if memory_loss != 0.0:
            total_loss = total_loss + self.memory_weight * memory_loss
            
        # 记录各部分损失
        if hasattr(self, 'logger'):
            self.logger.debug(f"Rec Loss: {rec_loss.item():.4f}")
            if js_loss != 0.0:
                self.logger.debug(f"JS Loss: {js_loss.item():.4f}")
            if memory_loss != 0.0:
                self.logger.debug(f"Memory Loss: {memory_loss.item():.4f}")
            self.logger.debug(f"Total Loss: {total_loss.item():.4f}")
        
        return total_loss

    def full_sort_predict(self, interaction):
        """使用检索增强的全排序预测，支持可微分记忆模块"""
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        
        # 获取基础序列表示
        seq_output = self.forward(item_seq, item_seq_len)
        
        # 检索增强处理
        if self.use_retrieval_for_predict and (
            (hasattr(self, 'seq_emb_index') and self.seq_emb_index is not None) or
            (self.use_diff_memory and hasattr(self, 'diff_memory'))
        ):
            try:
                # 获取用户ID和序列长度用于检索过滤
                batch_user_id = list(interaction[self.USER_ID].detach().cpu().numpy())
                batch_seq_len = list(item_seq_len.detach().cpu().numpy())
                
                # 使用注意力机制的序列增强 (结合FAISS和可微分记忆)
                enhanced_seq_output = self.sequence_augmentation(seq_output, batch_user_id, batch_seq_len)
                
                # 使用增强后的表示计算分数
                test_items_emb = self.item_embedding.weight
                scores = torch.matmul(enhanced_seq_output, test_items_emb.transpose(0, 1))
                
                # 确保分数保留梯度信息
                scores.requires_grad_(True)
                return scores
                
            except Exception as e:
                print(f"预测时检索增强失败: {e}")
                # 发生错误时回退到基本预测
    
        # 如果检索失败或没有检索索引，使用基本预测
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        
        # 确保分数保留梯度信息
        scores.requires_grad_(True)
        return scores

    def predict(self, interaction):
        """使用检索增强的单物品预测，支持可微分记忆模块"""
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        
        # 获取基础序列表示
        seq_output = self.forward(item_seq, item_seq_len)
        
        # 检索增强处理
        if self.use_retrieval_for_predict and (
            (hasattr(self, 'seq_emb_index') and self.seq_emb_index is not None) or
            (self.use_diff_memory and hasattr(self, 'diff_memory'))
        ):
            try:
                # 获取用户ID和序列长度用于检索过滤
                batch_user_id = list(interaction[self.USER_ID].detach().cpu().numpy())
                batch_seq_len = list(item_seq_len.detach().cpu().numpy())
                
                # 使用注意力机制的序列增强 (结合FAISS和可微分记忆)
                enhanced_seq_output = self.sequence_augmentation(seq_output, batch_user_id, batch_seq_len)
                
                # 使用增强后的表示计算分数
                test_item_emb = self.item_embedding(test_item)
                scores = torch.mul(enhanced_seq_output, test_item_emb).sum(dim=1)
                
                # 确保分数保留梯度信息
                scores.requires_grad_(True)
                return scores
                
            except Exception as e:
                print(f"预测时检索增强失败: {e}")
                # 发生错误时回退到基本预测
        
        # 如果检索失败或没有检索索引，使用基本预测
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        
        # 确保分数保留梯度信息
        scores.requires_grad_(True)
        return scores

    def sequence_augmentation(self, seq_output, batch_user_id, batch_seq_len, topk=None):
        """使用注意力机制对序列表示进行增强，结合端到端可微检索
        
        Args:
            seq_output: 原始序列表示 [batch_size, hidden_size]
            batch_user_id: 用户ID列表
            batch_seq_len: 序列长度列表
            topk: 检索的最近邻数量，默认使用self.topk
            
        Returns:
            增强后的序列表示 [batch_size, hidden_size]
        """
        if topk is None:
            topk = self.topk
        
        # 部分一：从可微分记忆和/或FAISS索引中检索
        retrieved_seqs, retrieved_tars = self.retrieve_seq_tar(
            seq_output, batch_user_id, batch_seq_len, topk=topk
        )
        
        # 如果检索到的序列为空或只有一个很小的值(检索失败情况)，直接返回原始表示
        if retrieved_seqs.size(0) == 0 or (retrieved_seqs.size(1) <= 1 and torch.sum(retrieved_seqs).item() < 1e-6):
            return seq_output
            
        # 使用注意力机制融合信息
        # 以序列作为query，检索序列作为key，检索目标项作为value
        # 将序列扩展为3D张量 [batch_size, 1, hidden_size]
        query = seq_output.unsqueeze(1)
        
        # 应用注意力层: query-key-value注意力
        augmented_output = self.seq_tar_attention(query, retrieved_seqs, retrieved_tars)
        
        # 通过前馈网络进一步处理
        augmented_output = self.seq_tar_fnn(augmented_output)
        
        # 混合原始表示和增强表示
        alpha = self.alpha
        enhanced_seq_output = (1 - alpha) * seq_output + alpha * augmented_output
        
        # 确保增强后的表示保留梯度信息
        enhanced_seq_output.requires_grad_(True)
        
        return enhanced_seq_output