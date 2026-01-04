## Code Repository for the Paper

### 1. Basic Data Preparation: `1-data`

- `1-get_sample_uniprot_sprot.ipynb`: Sample 10,000 protein sequences from UniProtKB/Swiss-Prot  
- `2-get_non_homologous_pairs.ipynb`: Generate non-homologous protein sequence pairs  
- `3-get_homology_pairs.ipynb`: Generate homologous protein sequence pairs  
- `4-get_distant_homology_pairs.ipynb`: Generate distantly homologous protein sequence pairs  
- `mysql_part`: Engineering implementation using MySQL tables to accelerate data processing; includes ready-to-import SQL dump files

### 2. GPT-2 Fine-tuning and Interpretability Experiments: `2-gpt_ft_test_explain`

- `1-gpt2_ft_en_test_protein_confusion.ipynb`: Fine-tune GPT-2 on English PAWS-X dataset and evaluate on protein sequences (with confusion matrix)  
- `2-gpt2_test_protein.ipynb`: Directly test pretrained GPT-2 on protein homology tasks (with confusion matrix)  
- `3-acc_distribution.ipynb`: Accuracy distribution analysis for both fine-tuned and base models  
- `4-explain_***`: Interpretability studies on cross-domain language capability transfer  
- `batch_run`: Scripts for batch execution of experiments

### 3. LLaMA-3 Fine-tuning and Evaluation: `3-llama_sft_test`

- `1-llama_sft_**`: Fine-tuning code for LLaMA-3.1 with various quantization strategies  
- `2-llama_sft_test.py`: Evaluate fine-tuned models on protein homology classification  
- `3-llama**`: Benchmark results using official pretrained and fine-tuned LLaMA models  
- `4-*_standard_protein`: Performance of state-of-the-art (SOTA) large models on standard protein homology detection  
- `5-*_remote_protein`: Performance of SOTA large models on **distant homology** detection  
- `6-qwen3_explain-`: Chain-of-Thought (CoT)-based interpretability analysis

### 4. BioPAWS Dataset Evaluation: `4-biopaws`

- `1-qwen3_dna`: DNA sequence homology classification  
- `2-qwen3_dna_protein`: Assessing DNA–protein coding relationship  
- `3-qwen3_dna_single`: Single DNA sequence classification  
- `4-qwen3_protein_single`: Single protein sequence classification

---

> **Note**: Wildcards (`*`) in original notebook filenames have been preserved or generalized for clarity while maintaining semantic meaning.

---