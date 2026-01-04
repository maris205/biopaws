## 论文对应代码

### 1 基础数据: `1-data`

- `1-get_sample_uniprot_sprot.ipynb`：获得10000条采样蛋白质数据  
- `2-get_non_homologous_pairs.ipynb`：获得非同源蛋白质序列  
- `3-get_homology_pairs.ipynb`：获得同源蛋白质序列  
- `4-get_distant homology_pairs.ipynb`：获得远程同源序列  
- `mysql_part`：基于 MySQL 表的工程化实现，主要解决速度问题，可直接导入 MySQL 数据文件

### 2 GPT2 相关的微调和可解释实验: `2-gpt_ft_test_explain`

- `1-gpt2_ft_en_test_protein_confusion.ipynb`：GPT2 基于英文 PAWS-X 微调，测试蛋白质序列，提供混淆矩阵  
- `2-gpt2_test_protein.ipynb`：GPT2 直接测试蛋白质序列，提供混淆矩阵  
- `3-acc distribution.ipynb`：准确率（acc）的分布统计，包括微调和未微调模型  
- `4-explain_***`：语言能力迁移的可解释性分析  
- `batch_run`：批量运行代码

### 3 LLaMA3 微调测试：`3-llama_sft_test`

- `1-llama_sft_**`：LLaMA 3.1 微调代码，使用不同的量化策略  
- `2-llama_sft_test.py`：微调模型，测试蛋白质同源性  
- `3-llama**`：官方预训练模型和微调模型的测试结果  
- `4-*_standard_protein`：SOTA 大模型在常见同源蛋白质判定任务上的表现  
- `5-*_remote_protein`：SOTA 大模型在远程同源蛋白质判定任务上的表现  
- `6-qwen3_explain-`：基于思维链（Chain-of-Thought）的可解释性分析

### 4 BioPAWS 数据集测试：`4-biopaws`

- `1-qwen3_dna`：DNA 同源判定  
- `2-qwen3_dna_protein`：DNA-蛋白质编码关系判定  
- `3-qwen3_dna_single`：DNA 单序列分类问题  
- `4-qwen3_protein_single`：蛋白质单序列分类问题

--- 

> 注：原始 notebook 中的星号（`*`）和下划线命名已按语义还原为通配符或描述性文字，便于阅读。

如需生成完整的 `.md` 文件或添加 GitHub 风格链接/图标，也可告知！