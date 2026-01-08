## Emergence of Biological Structural Discovery in General-Purpose Language Models

### Abstract
Large language models (LLMs) are evolving into engines for scientific discovery, yet the assumption that biological understanding requires domain-specific pre-training remains unchallenged. Here, we report that general-purpose LLMs possess an emergent capability for biological structural discovery. First, we demonstrate that a small-scale GPT-2, fine-tuned solely on English paraphrasing, achieves ~84% zero-shot accuracy in protein homology detection, where network-based interpretability confirms a deep structural isomorphism between human language and the language of life. Scaling to massive models (e.g., Qwen-3) reveals a phase transition, achieving near-perfect accuracy (~100%) on standard tasks while maintaining 75% precision on specially constructed remote homology datasets. Chain-of-Thought interpretability reveals that these models transcend simple sequence alignment, leveraging implicit structural knowledge to perform reasoning akin to "mental folding." We formalize this cross-modal universality through the BioPAWS benchmark. Our work establishes a minimalist paradigm for AI for Science, proving that abstract logical structures distilled from human language constitute a powerful cognitive prior for decoding the complex syntax of biology.

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

### gemini chat history
- paper: https://gemini.google.com/share/98b9335c3ddb
- data: https://gemini.google.com/share/37c12b897bbc
- some codes (grok): https://grok.com/share/c2hhcmQtMw_da2af51b-7586-46ff-bd4b-319e19ca37d7

### cite paper
Emergence of Biological Structural Discovery in General-Purpose Language Models
Liang Wang
bioRxiv 2026.01.03.697478; doi: https://doi.org/10.64898/2026.01.03.697478

---

> **Note**: Wildcards (`*`) in original notebook filenames have been preserved or generalized for clarity while maintaining semantic meaning.

---
