# -*- coding: utf-8 -*-
# @Time    : 2025/1/1
# @Author  : FTragrec Developer

"""
FTragrec
################################################

FTragrec combines Retrieval-based and Transformer-based recommendation
with a specialized RetrieverEncoder that can be fine-tuned for better retrieval quality.
"""

import torch, heapq, scipy, faiss, random, math
from faiss import normalize_L2
from torch import nn
import numpy as np
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder, FeedForward, activation_layer, MLPLayers
from recbole.model.loss import BPRLoss


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
        self.nprobe = config['nprobe'] if 'nprobe' in config else 8
        self.topk = config['top_k'] if 'top_k' in config else 5
        
        # RetrieverEncoder相关参数
        self.retriever_layers = config['retriever_layers'] if 'retriever_layers' in config else 1
        self.retriever_temperature = config['retriever_temperature'] if 'retriever_temperature' in config else 0.1
        self.retriever_dropout = config['retriever_dropout'] if 'retriever_dropout' in config else 0.1
        self.retriever_update_interval = config['retriever_update_interval'] if 'retriever_update_interval' in config else 5
        self.kl_weight = config['kl_weight'] if 'kl_weight' in config else 0.1
        
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
        
        # 数据集和检索索引
        self.dataset = dataset
        self.seq_emb_knowledge = None
        self.tar_emb_knowledge = None
        self.user_id_list = None
        self.item_seq_len_all = None

        # 新增预测阶段检索增强相关参数
        self.use_retrieval_for_predict = config['use_retrieval_for_predict'] if 'use_retrieval_for_predict' in config else True
        self.predict_retrieval_alpha = config['predict_retrieval_alpha'] if 'predict_retrieval_alpha' in config else 0.5
        self.predict_retrieval_temperature = config['predict_retrieval_temperature'] if 'predict_retrieval_temperature' in config else 0.1
        
        # 新增: 从配置中获取混合权重alpha参数，如果不存在则使用默认值0.5
        self.alpha = config['alpha'] if 'alpha' in config else 0.5
        
        # 新增: 从配置中获取KL散度小权重参数，如果不存在则使用默认值0.1
        self.kl_small_weight = config['kl_small_weight'] if 'kl_small_weight' in config else 0.1
        
        # 新增: 从配置中获取增强推荐损失权重参数，如果不存在则使用默认值1.0
        self.enhanced_rec_weight = config['enhanced_rec_weight'] if 'enhanced_rec_weight' in config else 1.0
        
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
        return output  # [B H]

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
        """预缓存知识 - 构建检索索引"""
        print("开始预缓存知识...")
        
        # 使用提供的dataloader或默认dataset
        dataloader_to_use = train_dataloader if train_dataloader is not None else self.dataset
        
        seq_emb_knowledge, tar_emb_knowledge, user_id_list = None, None, None
        item_seq_all = None
        item_seq_len_all = None
        
        # 遍历数据集中的所有交互
        batch_count = 0
        valid_batch_count = 0
        total_samples_before_filter = 0
        total_samples_after_filter = 0
        
        for batch_idx, interaction in enumerate(dataloader_to_use):
            batch_count += 1
            try:
                interaction = interaction.to("cuda")
                batch_size = interaction[self.ITEM_SEQ].shape[0]
                total_samples_before_filter += batch_size
                
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
                
                # 统计过滤后的样本数量
                valid_samples = look_up_indices.sum().item()
                total_samples_after_filter += valid_samples
                
                if valid_samples == 0:
                    continue
                    
                valid_batch_count += 1
                
                # 获取序列和长度信息
                item_seq = interaction[self.ITEM_SEQ][look_up_indices]
                
                if item_seq_all is None:
                    item_seq_all = item_seq
                else:
                    item_seq_all = torch.cat((item_seq_all, item_seq), dim=0)
                    
                item_seq_len = interaction[self.ITEM_SEQ_LEN][look_up_indices]
                item_seq_len_list = list(interaction[self.ITEM_SEQ_LEN][look_up_indices].detach().cpu().numpy())
                
                if isinstance(item_seq_len_all, list):
                    item_seq_len_all.extend(item_seq_len_list)
                else:
                    item_seq_len_all = item_seq_len_list
                    
                # 获取序列表示和目标项表示
                seq_output = self.forward(item_seq, item_seq_len)
                
                # 使用检索器编码器进行非线性变换
                try:
                    retriever_output = self.retriever_forward(seq_output)
                except Exception as e:
                    print(f"retriever_forward执行失败: {e}")
                    retriever_output = seq_output
                
                tar_items = interaction[self.POS_ITEM_ID][look_up_indices]
                tar_items_emb = self.item_embedding(tar_items)
                
                user_id_cans = list(interaction[self.USER_ID][look_up_indices].detach().cpu().numpy())
                
                # 累积序列表示和目标项表示
                try:
                    retriever_output_np = retriever_output.detach().cpu().numpy()
                    
                    if isinstance(seq_emb_knowledge, np.ndarray):
                        seq_emb_knowledge = np.concatenate((seq_emb_knowledge, retriever_output_np), 0)
                    else:
                        seq_emb_knowledge = retriever_output_np
                    
                    tar_items_emb_np = tar_items_emb.detach().cpu().numpy()
                    if isinstance(tar_emb_knowledge, np.ndarray):
                        tar_emb_knowledge = np.concatenate((tar_emb_knowledge, tar_items_emb_np), 0)
                    else:
                        tar_emb_knowledge = tar_items_emb_np
                    
                    if isinstance(user_id_list, list):
                        user_id_list.extend(user_id_cans)
                    else:
                        user_id_list = user_id_cans
                except Exception as e:
                    print(f"处理批次 {batch_idx} 时出错: {e}")
                    continue
            except Exception as e:
                print(f"处理批次 {batch_idx} 时出错: {e}")
                
        print(f"数据收集完成: 总批次={batch_count}, 有效批次={valid_batch_count}, 过滤后样本数={total_samples_after_filter}/{total_samples_before_filter}")
        
        # 数据验证
        if seq_emb_knowledge is None or len(seq_emb_knowledge) == 0:
            print("错误: seq_emb_knowledge 为空! 请检查数据集和过滤条件是否正确。")
            return
            
        print(f"收集到的嵌入: seq_emb_knowledge形状={seq_emb_knowledge.shape}, tar_emb_knowledge形状={tar_emb_knowledge.shape}")
        
        # 保存数据
        self.user_id_list = user_id_list
        self.item_seq_all = item_seq_all
        self.item_seq_len_all = item_seq_len_all
        self.seq_emb_knowledge = seq_emb_knowledge
        self.tar_emb_knowledge = tar_emb_knowledge
        
        # 构建FAISS索引
        d = 64  # 使用固定维度值64，与RaSeRec一致
        
        # 根据数据量调整nlist
        n_samples = len(seq_emb_knowledge)
        nlist = min(128, max(1, n_samples // 39))  # 确保nlist不大于样本数量的1/39
        print(f"构建FAISS索引: 样本数={n_samples}, nlist={nlist}, 维度={d}")
        
        try:
            # 创建seq_emb索引
            seq_emb_knowledge_copy = np.array(seq_emb_knowledge, copy=True)
            normalize_L2(seq_emb_knowledge_copy)
            
            seq_emb_quantizer = faiss.IndexFlatL2(d) 
            self.seq_emb_index = faiss.IndexIVFFlat(seq_emb_quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT) 
            self.seq_emb_index.train(seq_emb_knowledge_copy)
            self.seq_emb_index.add(seq_emb_knowledge_copy)    
            self.seq_emb_index.nprobe = self.nprobe

            # 创建tar_emb索引
            tar_emb_knowledge_copy = np.array(tar_emb_knowledge, copy=True)
            normalize_L2(tar_emb_knowledge_copy)
            tar_emb_quantizer = faiss.IndexFlatL2(d) 
            self.tar_emb_index = faiss.IndexIVFFlat(tar_emb_quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT) 
            self.tar_emb_index.train(tar_emb_knowledge_copy)
            self.tar_emb_index.add(tar_emb_knowledge_copy) 
            self.tar_emb_index.nprobe = self.nprobe
            
            print("FAISS索引构建完成")
            
        except Exception as e:
            print(f"FAISS索引构建过程中出错: {e}")
            return

    def update_faiss_index(self):
        print("开始更新FAISS索引...")
        
        # 检查数据是否有效
        if self.seq_emb_knowledge is None or self.tar_emb_knowledge is None:
            print("错误: 没有可用的知识嵌入，无法更新索引")
            return
            
        # 构建FAISS索引
        d = 64  # 使用固定维度值64，与RaSeRec一致
        
        # 根据数据量调整nlist
        n_samples = len(self.seq_emb_knowledge)
        nlist = min(128, max(1, n_samples // 39))  # 确保nlist不大于样本数量的1/39
        print(f"更新FAISS索引: 样本数={n_samples}, nlist={nlist}, 维度={d}")
        
        try:
            # 创建seq_emb索引
            seq_emb_knowledge_copy = np.array(self.seq_emb_knowledge, copy=True)
            normalize_L2(seq_emb_knowledge_copy)
            
            seq_emb_quantizer = faiss.IndexFlatL2(d) 
            self.seq_emb_index = faiss.IndexIVFFlat(seq_emb_quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT) 
            self.seq_emb_index.train(seq_emb_knowledge_copy)
            self.seq_emb_index.add(seq_emb_knowledge_copy)    
            self.seq_emb_index.nprobe = self.nprobe

            # 创建tar_emb索引
            tar_emb_knowledge_copy = np.array(self.tar_emb_knowledge, copy=True)
            normalize_L2(tar_emb_knowledge_copy)
            tar_emb_quantizer = faiss.IndexFlatL2(d) 
            self.tar_emb_index = faiss.IndexIVFFlat(tar_emb_quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT) 
            self.tar_emb_index.train(tar_emb_knowledge_copy)
            self.tar_emb_index.add(tar_emb_knowledge_copy) 
            self.tar_emb_index.nprobe = self.nprobe
            
            print("FAISS索引更新完成")
            
        except Exception as e:
            print(f"FAISS索引更新过程中出错: {e}")
            return

    def retrieve_seq_tar(self, queries, batch_user_id, batch_seq_len, topk=5):
        """检索相似的序列和目标项"""
        # 使用检索器编码器处理查询
        retriever_queries = self.retriever_forward(queries)
        
        # 将查询转换为CPU张量并进行L2归一化
        queries_cpu = retriever_queries.detach().cpu().numpy()
        normalize_L2(queries_cpu)
        
        # 使用FAISS索引搜索相似序列
        _, I1 = self.seq_emb_index.search(queries_cpu, 4*topk)
        I1_filtered = []
        
        # 过滤结果
        for i, I_entry in enumerate(I1):
            current_user = batch_user_id[i]
            current_length = batch_seq_len[i]
            filtered_indices = [idx for idx in I_entry if self.user_id_list[idx] != current_user or (self.user_id_list[idx] == current_user and self.item_seq_len_all[idx] < current_length)]
            I1_filtered.append(filtered_indices[:topk])
            
        I1_filtered = np.array(I1_filtered)
        
        # 获取检索到的序列和目标项表示
        retrieval_seq = self.seq_emb_knowledge[I1_filtered]
        retrieval_tar = self.tar_emb_knowledge[I1_filtered]
        
        return torch.tensor(retrieval_seq).to(queries.device), torch.tensor(retrieval_tar).to(queries.device)

    def compute_retrieval_scores(self, query_vectors, candidate_vectors):
        """计算检索似然分布 - 检索器认为序列d在给定输入x的条件下被选中的概率分布"""
        # 计算相似度分数
        similarity = torch.bmm(query_vectors, candidate_vectors.transpose(1, 2))
        
        # 应用温度缩放并转换为概率分布
        retrieval_logits = similarity / self.retriever_temperature
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
        """计算检索分布与推荐分布之间的KL散度损失"""
        # 避免数值问题
        epsilon = 1e-8
        retrieval_probs = retrieval_probs + epsilon
        recommendation_probs = recommendation_probs + epsilon
        
        # KL散度计算: KL(retrieval_probs || recommendation_probs)
        # 优化检索分布使其接近推荐分布
        kl_div = torch.sum(retrieval_probs * torch.log(retrieval_probs / recommendation_probs), dim=-1)
        
        return kl_div.mean()

    def calculate_loss(self, interaction):
        """计算模型损失"""
        # 获取序列和长度
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        batch_user_id = list(interaction[self.USER_ID].detach().cpu().numpy())
        batch_seq_len = list(item_seq_len.detach().cpu().numpy())
        
        # 获取序列表示
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        
        # 基础推荐损失计算
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            rec_loss = self.loss_fct(pos_score, neg_score)
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            rec_loss = self.loss_fct(logits, pos_items)
        
        # 检索器优化部分
        if hasattr(self, 'seq_emb_index'):
            # 检索相似序列和目标项
            retrieved_seqs, retrieved_tars = self.retrieve_seq_tar(
                seq_output, batch_user_id, batch_seq_len, topk=self.topk
            )
            
            # 使用检索器编码器处理序列表示
            retriever_output = self.retriever_forward(seq_output)
            
            # 计算检索似然分布
            retrieval_probs = self.compute_retrieval_scores(
                retriever_output.unsqueeze(1), retrieved_seqs
            )
            
            # 计算推荐模型的评分分布
            recommendation_probs = self.compute_recommendation_scores(
                seq_output, retrieved_seqs, retrieved_tars, pos_items
            )
            
            # 计算KL散度损失
            kl_loss = self.compute_kl_loss(retrieval_probs.squeeze(1), recommendation_probs)
            
            # 新增: 使用检索增强表示计算额外的推荐损失，直接优化推荐质量
            if retrieved_seqs.size(0) > 0:
                batch_size, n_retrieved, dim = retrieved_seqs.size()
                
                # 扩展查询表示以匹配检索结果维度
                expanded_seq = seq_output.unsqueeze(1).expand(-1, n_retrieved, -1)
                
                # 计算相似度作为注意力分数
                attention_scores = torch.sum(expanded_seq * retrieved_seqs, dim=-1) / self.retriever_temperature
                attention_weights = torch.softmax(attention_scores, dim=-1).unsqueeze(-1)
                
                # 加权汇总检索到的目标表示
                retrieved_targets_weighted = retrieved_tars * attention_weights
                retrieved_knowledge = torch.sum(retrieved_targets_weighted, dim=1)
                
                # 混合原始序列表示和检索增强表示
                alpha = self.alpha  # 使用混合权重参数
                enhanced_seq_output = (1 - alpha) * seq_output + alpha * retrieved_knowledge
                
                # 使用增强表示计算额外的推荐损失
                if self.loss_type == 'BPR':
                    enhanced_pos_score = torch.sum(enhanced_seq_output * pos_items_emb, dim=-1)
                    enhanced_neg_score = torch.sum(enhanced_seq_output * neg_items_emb, dim=-1)
                    enhanced_rec_loss = self.loss_fct(enhanced_pos_score, enhanced_neg_score)
                else:  # self.loss_type = 'CE'
                    enhanced_logits = torch.matmul(enhanced_seq_output, test_item_emb.transpose(0, 1))
                    enhanced_rec_loss = self.loss_fct(enhanced_logits, pos_items)
                
                # 小权重给KL损失以保持检索器和推荐器之间的一致性
                small_weight = self.kl_small_weight
                # 使用配置的增强推荐损失权重
                total_loss = rec_loss + self.enhanced_rec_weight * enhanced_rec_loss + self.kl_weight * kl_loss
            else:
                # 如果没有检索到足够的序列，只使用原始损失和KL损失
                total_loss = rec_loss + self.kl_weight * kl_loss
            
            # 更新FAISS索引
            self.current_epoch += 1
            if self.current_epoch % self.retriever_update_interval == 0:
                # 在训练循环中更新FAISS索引
                # 实际更新会在外部trainer中处理
                pass
            
            return total_loss
        else:
            # 如果没有检索索引，只返回基础推荐损失
            return rec_loss

    def full_sort_predict(self, interaction):
        """使用检索增强的全排序预测"""
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        
        # 获取基础序列表示
        seq_output = self.forward(item_seq, item_seq_len)
        
        # 检索增强处理
        if hasattr(self, 'seq_emb_index') and self.seq_emb_index is not None:
            try:
                # 获取用户ID和序列长度用于检索过滤
                batch_user_id = list(interaction[self.USER_ID].detach().cpu().numpy())
                batch_seq_len = list(item_seq_len.detach().cpu().numpy())
                
                # 使用检索器编码器处理序列表示
                retriever_output = self.retriever_forward(seq_output)
                
                # 检索相似序列和目标项
                retrieved_seqs, retrieved_tars = self.retrieve_seq_tar(
                    seq_output, batch_user_id, batch_seq_len, topk=self.topk
                )
                
                if retrieved_seqs.size(0) > 0:
                    # 计算查询序列与检索序列的注意力权重
                    batch_size, n_retrieved, dim = retrieved_seqs.size()
                    
                    # 扩展查询表示以匹配检索结果维度
                    expanded_seq = seq_output.unsqueeze(1).expand(-1, n_retrieved, -1)
                    
                    # 计算相似度作为注意力分数
                    attention_scores = torch.sum(expanded_seq * retrieved_seqs, dim=-1) / self.retriever_temperature
                    attention_weights = torch.softmax(attention_scores, dim=-1).unsqueeze(-1)
                    
                    # 加权汇总检索到的目标表示
                    retrieved_targets_weighted = retrieved_tars * attention_weights
                    retrieved_knowledge = torch.sum(retrieved_targets_weighted, dim=1)
                    
                    # 混合原始序列表示和检索增强表示
                    alpha = self.alpha  # 使用混合权重参数
                    enhanced_seq_output = (1 - alpha) * seq_output + alpha * retrieved_knowledge
                    
                    # 使用增强后的表示计算分数
                    test_items_emb = self.item_embedding.weight
                    scores = torch.matmul(enhanced_seq_output, test_items_emb.transpose(0, 1))
                    return scores
                    
            except Exception as e:
                print(f"预测时检索增强失败: {e}")
                # 发生错误时回退到基本预测
    
        # 如果检索失败或没有检索索引，使用基本预测
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores

    def predict(self, interaction):
        """使用检索增强的单物品预测"""
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        
        # 获取基础序列表示
        seq_output = self.forward(item_seq, item_seq_len)
        
        # 检索增强处理
        if hasattr(self, 'seq_emb_index') and self.seq_emb_index is not None:
            try:
                # 获取用户ID和序列长度用于检索过滤
                batch_user_id = list(interaction[self.USER_ID].detach().cpu().numpy())
                batch_seq_len = list(item_seq_len.detach().cpu().numpy())
                
                # 使用检索器编码器处理序列表示
                retriever_output = self.retriever_forward(seq_output)
                
                # 检索相似序列和目标项
                retrieved_seqs, retrieved_tars = self.retrieve_seq_tar(
                    seq_output, batch_user_id, batch_seq_len, topk=self.topk
                )
                
                if retrieved_seqs.size(0) > 0:
                    # 计算查询序列与检索序列的注意力权重
                    batch_size, n_retrieved, dim = retrieved_seqs.size()
                    
                    # 扩展查询表示以匹配检索结果维度
                    expanded_seq = seq_output.unsqueeze(1).expand(-1, n_retrieved, -1)
                    
                    # 计算相似度作为注意力分数
                    attention_scores = torch.sum(expanded_seq * retrieved_seqs, dim=-1) / self.retriever_temperature
                    attention_weights = torch.softmax(attention_scores, dim=-1).unsqueeze(-1)
                    
                    # 加权汇总检索到的目标表示
                    retrieved_targets_weighted = retrieved_tars * attention_weights
                    retrieved_knowledge = torch.sum(retrieved_targets_weighted, dim=1)
                    
                    # 混合原始序列表示和检索增强表示
                    alpha = self.alpha  # 使用混合权重参数
                    enhanced_seq_output = (1 - alpha) * seq_output + alpha * retrieved_knowledge
                    
                    # 使用增强后的表示计算分数
                    test_item_emb = self.item_embedding(test_item)
                    scores = torch.mul(enhanced_seq_output, test_item_emb).sum(dim=1)
                    return scores
                    
            except Exception as e:
                print(f"预测时检索增强失败: {e}")
                # 发生错误时回退到基本预测
        
        # 如果检索失败或没有检索索引，使用基本预测
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores