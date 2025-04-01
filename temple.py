def compute_retrieval_scores(self, query_vectors, candidate_vectors, temperature=0.5):
        """
        计算检索似然分布 - 检索器认为序列d在给定输入x的条件下被选中的概率分布
        
        Args:
            query_vectors: 查询向量 (用户序列表示)
            candidate_vectors: 候选向量 (检索到的序列表示)
            temperature: 温度参数，控制分布的平滑程度
            
        Returns:
            检索概率分布
        """
        # 计算相似度分数
        similarity = torch.bmm(query_vectors, candidate_vectors.transpose(1, 2))
        
        # 应用温度缩放并转换为概率分布
        retrieval_logits = similarity / temperature
        retrieval_probs = torch.softmax(retrieval_logits, dim=-1)
        
        return retrieval_probs

    def compute_recommendation_scores(self, seq_output, retrieved_seqs, retrieved_tars):
        """
        计算推荐模型的评分分布 - 在给定输入序列x和检索序列d的情况下的目标概率
        
        Args:
            seq_output: 序列表示
            retrieved_seqs: 检索到的序列表示
            retrieved_tars: 检索到的目标项表示
            
        Returns:
            推荐概率分布
        """
        batch_size = seq_output.size(0)
        
        # 使用RAM模块增强序列表示
        seq_output_saug = self.seq_tar_ram(seq_output.unsqueeze(1), retrieved_seqs, retrieved_tars)
        seq_output_saug = self.seq_tar_ram_fnn(seq_output_saug)
        seq_output_saug = self.seq_tar_ram_1(seq_output_saug.unsqueeze(1), retrieved_seqs, retrieved_tars)
        
        # 计算增强序列表示与目标项的相似度
        similarity = torch.bmm(seq_output_saug.unsqueeze(1), retrieved_tars.transpose(1, 2))
        
        # 转换为概率分布
        recommendation_probs = torch.softmax(similarity.squeeze(1), dim=-1)
        
        return recommendation_probs

    def compute_kl_loss(self, retrieval_probs, recommendation_probs):
        """
        计算检索分布与推荐分布之间的KL散度损失
        
        Args:
            retrieval_probs: 检索概率分布
            recommendation_probs: 推荐概率分布
            
        Returns:
            KL散度损失
        """
        # 避免数值问题
        epsilon = 1e-8
        retrieval_probs = retrieval_probs + epsilon
        recommendation_probs = recommendation_probs + epsilon
        
        # KL散度计算方向: KL(retrieval_probs || recommendation_probs)
        # 优化检索分布使其接近推荐分布
        kl_div = torch.sum(retrieval_probs * torch.log(retrieval_probs / recommendation_probs), dim=-1)
        
        return kl_div.mean()

    def retriever_fine_tuning_step(self, interaction):
        """
        执行一次检索器微调步骤
        
        Args:
            interaction: 交互数据
            
        Returns:
            检索器优化损失
        """
        # 获取序列和长度
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        batch_user_id = list(interaction[self.USER_ID].detach().cpu().numpy())
        batch_seq_len = list(item_seq_len.detach().cpu().numpy())
        
        # 获取序列表示
        seq_output = self.forward(item_seq, item_seq_len)
        
        # 检索相似序列和目标项
        torch_retrieval_seq_embs, torch_retrieval_tar_embs, _, _ = self.retrieve_seq_tar(
            seq_output, batch_user_id, batch_seq_len, topk=self.topk, mode="train"
        )
        
        # 计算检索似然分布
        retrieval_scores = self.compute_retrieval_scores(
            seq_output.unsqueeze(1), torch_retrieval_seq_embs, temperature=self.retriever_temperature
        )
        
        # 计算推荐模型的评分分布
        recommendation_scores = self.compute_recommendation_scores(
            seq_output, torch_retrieval_seq_embs, torch_retrieval_tar_embs
        )
        
        # 计算KL散度损失
        kl_loss = self.compute_kl_loss(retrieval_scores, recommendation_scores)
        
        return kl_loss