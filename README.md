### TL;DR
```
conda activate hipporag
python src/hipporag/document_processor.py  2>>index.log
python generate_abstracts.py outputs/Harmony_docs_zh_cn/markdown_parse/structure.json outputs/Harmony_docs_zh_cn/markdown_parse/abstract.json 2>> index.log
python extract_entities_triples.py outputs/Harmony_docs_zh_cn/markdown_parse/abstract.json outputs/Harmony_docs_zh_cn/markdown_parse/triples.json 2>> index.log
```

---

## 中文使用指南 / Chinese Usage Guide

### 知识库构建流程 (Knowledge Base Construction)

本项目已完成知识库构建功能的实现，支持从Markdown文档构建层次化知识图谱。完整流程包括：

#### 步骤1: 文档处理 (Document Processing)
使用 `document_processor.py` 处理Markdown文档，提取文档结构、代码块、表格等信息：

```bash
python src/hipporag/document_processor.py [input_dir] [output_dir] 2>>index.log
```

该步骤会生成 `structure.json`，包含文档的层次化结构。

#### 步骤2: 生成摘要 (Abstract Generation)
使用 `generate_abstracts.py` 为文档、段落、代码块和表格生成摘要：

```bash
python generate_abstracts.py \
    outputs/Harmony_docs_zh_cn/markdown_parse/structure.json \
    outputs/Harmony_docs_zh_cn/markdown_parse/abstract.json \
    2>> index.log
```

#### 步骤3: 提取实体和三元组 (Entity and Triple Extraction)
使用 `extract_entities_triples.py` 从文档中提取实体和关系三元组：

```bash
python extract_entities_triples.py \
    outputs/Harmony_docs_zh_cn/markdown_parse/abstract.json \
    outputs/Harmony_docs_zh_cn/markdown_parse/triples.json \
    2>> index.log
```

该步骤会生成包含实体和三元组的 `triples.json` 文件。

### 检索功能 (Retrieval Functionality)

检索功能已实现，支持以下两种检索模式：

#### 1. 传统段落检索 (Traditional Passage Retrieval)

```python
from src.hipporag import HippoRAG
from src.hipporag.utils.config_utils import BaseConfig

# 配置
config = BaseConfig(
    save_dir='outputs',
    llm_name='gpt-4o-mini',
    embedding_model_name='nvidia/NV-Embed-v2'
)

# 初始化HippoRAG
hipporag = HippoRAG(global_config=config)

# 索引文档
docs = ["文档1", "文档2", "文档3"]
hipporag.index(docs)

# 检索
queries = ["查询问题1", "查询问题2"]
retrieval_results = hipporag.retrieve(queries=queries, num_to_retrieve=5)

# 检索增强生成
qa_results = hipporag.rag_qa(queries=queries)
```

#### 2. 层次化索引检索 (Hierarchical Index Retrieval)

使用层次化JSON结构进行索引和检索：

```python
import json
from src.hipporag import HippoRAG
from src.hipporag.utils.config_utils import BaseConfig

# 配置
config = BaseConfig(
    save_dir='outputs/Harmony_docs_zh_cn',
    llm_name='gpt-4o-mini',
    embedding_model_name='Qwen/Qwen2.5-7B-Instruct'
)

# 初始化HippoRAG
hipporag = HippoRAG(global_config=config)

# 加载层次化JSON结构（包含文件、段落、代码块、表格等）
with open('outputs/Harmony_docs_zh_cn/markdown_parse/triples.json', 'r', encoding='utf-8') as f:
    json_structure = json.load(f)

# 执行层次化索引
hipporag.index_from_json(json_structure)

# 检索查询
queries = ["如何配置API?", "代码示例在哪里?"]
retrieval_results = hipporag.retrieve(queries=queries, num_to_retrieve=10)

# 检索增强生成
qa_results, responses, metadata = hipporag.rag_qa(queries=queries)
```

#### 检索特性 (Retrieval Features)

1. **多步骤检索流程**:
   - 事实检索：基于查询向量与事实向量的相似度
   - 认知记忆：使用重排序器筛选高质量事实
   - 图搜索：基于PersonalizedPageRank在知识图谱上传播权重
   - 结果合成：结合图搜索和密集检索的结果

2. **支持的节点类型**:
   - 文件节点：完整文档级别的检索
   - 段落节点：文档段落检索
   - 代码块节点：代码片段检索
   - 表格节点：结构化数据检索
   - 实体节点：语义实体检索

3. **评估功能**:
   - 支持检索评估（Recall@K）
   - 支持问答评估（精确匹配和F1分数）

### 完整示例脚本 (Complete Example Script)

查看 `demo_retrieval.py` 了解完整的检索使用示例。

---




<h1 align="center">HippoRAG 2: From RAG to Memory</h1>
<p align="center">
    <img src="https://github.com/OSU-NLP-Group/HippoRAG/raw/main/images/hippo_brain.png" width="55%" style="max-width: 300px;">
</p>

[<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/drive/1nuelysWsXL8F5xH6q4JYJI8mvtlmeM9O#scrollTo=TjHdNe2KC81K)

[<img align="center" src="https://img.shields.io/badge/arXiv-2502.14802 HippoRAG 2-b31b1b" />](https://arxiv.org/abs/2502.14802)
[<img align="center" src="https://img.shields.io/badge/🤗 Dataset-HippoRAG 2-yellow" />](https://huggingface.co/datasets/osunlp/HippoRAG_2/tree/main)
[<img align="center" src="https://img.shields.io/badge/arXiv-2405.14831 HippoRAG 1-b31b1b" />](https://arxiv.org/abs/2405.14831)
[<img align="center" src="https://img.shields.io/badge/GitHub-HippoRAG 1-blue" />](https://github.com/OSU-NLP-Group/HippoRAG/tree/legacy)

### HippoRAG 2 is a powerful memory framework for LLMs that enhances their ability to recognize and utilize connections in new knowledge—mirroring a key function of human long-term memory.

Our experiments show that HippoRAG 2 improves associativity (multi-hop retrieval) and sense-making (the process of integrating large and complex contexts) in even the most advanced RAG systems, without sacrificing their performance on simpler tasks.

Like its predecessor, HippoRAG 2 remains cost and latency efficient in online processes, while using significantly fewer resources for offline indexing compared to other graph-based solutions such as GraphRAG, RAPTOR, and LightRAG.

<p align="center">
  <img align="center" src="https://github.com/OSU-NLP-Group/HippoRAG/raw/main/images/intro.png" />
</p>
<p align="center">
  <b>Figure 1:</b> Evaluation of continual learning capabilities across three key dimensions: factual memory (NaturalQuestions, PopQA), sense-making (NarrativeQA), and associativity (MuSiQue, 2Wiki, HotpotQA, and LV-Eval). HippoRAG 2 surpasses other methods across all
categories, bringing it one step closer to true long-term memory.
</p>

<p align="center">
  <img align="center" src="https://github.com/OSU-NLP-Group/HippoRAG/raw/main/images/methodology.png" />
</p>
<p align="center">
  <b>Figure 2:</b> HippoRAG 2 methodology.
</p>

#### Check out our papers to learn more:

* [**HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models**](https://arxiv.org/abs/2405.14831) [NeurIPS '24].
* [**From RAG to Memory: Non-Parametric Continual Learning for Large Language Models**](https://arxiv.org/abs/2502.14802) [ICML '25].

----

## Installation

```sh
conda create -n hipporag python=3.10
conda activate hipporag
pip install hipporag
```
Initialize the environmental variables and activate the environment:

```sh
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HOME=<path to Huggingface home directory>
export OPENAI_API_KEY=<your openai api key>   # if you want to use OpenAI model

conda activate hipporag
```

## Quick Start

### OpenAI Models

This simple example will illustrate how to use `hipporag` with any OpenAI model:

```python
from hipporag import HippoRAG

# Prepare datasets and evaluation
docs = [
    "Oliver Badman is a politician.",
    "George Rankin is a politician.",
    "Thomas Marwick is a politician.",
    "Cinderella attended the royal ball.",
    "The prince used the lost glass slipper to search the kingdom.",
    "When the slipper fit perfectly, Cinderella was reunited with the prince.",
    "Erik Hort's birthplace is Montebello.",
    "Marina is bom in Minsk.",
    "Montebello is a part of Rockland County."
]

save_dir = 'outputs'# Define save directory for HippoRAG objects (each LLM/Embedding model combination will create a new subdirectory)
llm_model_name = 'gpt-4o-mini' # Any OpenAI model name
embedding_model_name = 'nvidia/NV-Embed-v2'# Embedding model name (NV-Embed, GritLM or Contriever for now)

#Startup a HippoRAG instance
hipporag = HippoRAG(save_dir=save_dir, 
                    llm_model_name=llm_model_name,
                    embedding_model_name=embedding_model_name) 

#Run indexing
hipporag.index(docs=docs)

#Separate Retrieval & QA
queries = [
    "What is George Rankin's occupation?",
    "How did Cinderella reach her happy ending?",
    "What county is Erik Hort's birthplace a part of?"
]

retrieval_results = hipporag.retrieve(queries=queries, num_to_retrieve=2)
qa_results = hipporag.rag_qa(retrieval_results)

#Combined Retrieval & QA
rag_results = hipporag.rag_qa(queries=queries)

#For Evaluation
answers = [
    ["Politician"],
    ["By going to the ball."],
    ["Rockland County"]
]

gold_docs = [
    ["George Rankin is a politician."],
    ["Cinderella attended the royal ball.",
    "The prince used the lost glass slipper to search the kingdom.",
    "When the slipper fit perfectly, Cinderella was reunited with the prince."],
    ["Erik Hort's birthplace is Montebello.",
    "Montebello is a part of Rockland County."]
]

rag_results = hipporag.rag_qa(queries=queries, 
                              gold_docs=gold_docs,
                              gold_answers=answers)
```

#### Example (OpenAI Compatible Embeddings)

If you want to use LLMs and Embeddings Compatible to OpenAI, please use the following methods.</p>
    
```python
hipporag = HippoRAG(save_dir=save_dir, 
    llm_model_name='Your LLM Model name',
    llm_base_url='Your LLM Model url',
    embedding_model_name='Your Embedding model name',  
    embedding_base_url='Your Embedding model url')
```

### Local Deployment (vLLM)

This simple example will illustrate how to use `hipporag` with any vLLM-compatible locally deployed LLM.

1. Run a local [OpenAI-compatible vLLM server](https://docs.vllm.ai/en/latest/getting_started/quickstart.html#quickstart-online) with specified GPUs (make sure you leave enough memory for your embedding model).

```sh
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME=<path to Huggingface home directory>

conda activate hipporag  # vllm should be in this environment

# Tune gpu-memory-utilization or max_model_len to fit your GPU memory, if OOM occurs
vllm serve meta-llama/Llama-3.3-70B-Instruct --tensor-parallel-size 2 --max_model_len 4096 --gpu-memory-utilization 0.95 
```

2. Now you can use very similar code to the one above to use `hipporag`: 

```python
save_dir = 'outputs'# Define save directory for HippoRAG objects (each LLM/Embedding model combination will create a new subdirectory)
llm_model_name = # Any OpenAI model name
embedding_model_name = # Embedding model name (NV-Embed, GritLM or Contriever for now)
llm_base_url= # Base url for your deployed LLM (i.e. http://localhost:8000/v1)

hipporag = HippoRAG(save_dir=save_dir,
                    llm_model_name=llm_model,
                    embedding_model_name=embedding_model_name,
                    llm_base_url=llm_base_url)

# Same Indexing, Retrieval and QA as running OpenAI models above
```

## Testing

When making a contribution to HippoRAG, please run the scripts below to ensure that your changes do not result in unexpected behavior from our core modules. 

These scripts test for indexing, graph loading, document deletion and incremental updates to a HippoRAG object.

### OpenAI Test

To test HippoRAG with an OpenAI LLM and embedding model, simply run the following. 
The cost of this test will be negligible.

```sh
export OPENAI_API_KEY=<your openai api key> 

conda activate hipporag

python tests_openai.py
```

### Local Test

To test locally, you must deploy a vLLM instance. We choose to deploy a smaller 8B model `Llama-3.1-8B-Instruct` for cheaper testing.

```sh
export CUDA_VISIBLE_DEVICES=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME=<path to Huggingface home directory>

conda activate hipporag  # vllm should be in this environment

# Tune gpu-memory-utilization or max_model_len to fit your GPU memory, if OOM occurs
vllm serve meta-llama/Llama-3.1-8B-Instruct --tensor-parallel-size 2 --max_model_len 4096 --gpu-memory-utilization 0.95 --port 6578
```

Then, we run the following test script:

```sh
CUDA_VISIBLE=1 python tests_local.py
```

## Reproducing our Experiments

To use our code to run experiments we recommend you clone this repository and follow the structure of the `main.py` script.

### Data for Reproducibility

We evaluated several sampled datasets in our paper, some of which are already included in the `reproduce/dataset` directory of this repo. For the complete set of datasets, please visit
our [HuggingFace dataset](https://huggingface.co/datasets/osunlp/HippoRAG_v2) and place them under `reproduce/dataset`. We also provide the OpenIE results for both `gpt-4o-mini` and `Llama-3.3-70B-Instruct` for our `musique` sample under `outputs/musique`.

To test your environment is properly set up, you can use the small dataset `reproduce/dataset/sample.json` for debugging as shown below.

### Running Indexing & QA

Initialize the environmental variables and activate the environment:

```sh
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HOME=<path to Huggingface home directory>
export OPENAI_API_KEY=<your openai api key>   # if you want to use OpenAI model

conda activate hipporag
```

### Run with OpenAI Model

```sh
dataset=sample  # or any other dataset under `reproduce/dataset`

# Run OpenAI model
python main.py --dataset $dataset --llm_base_url https://api.openai.com/v1 --llm_name gpt-4o-mini --embedding_name nvidia/NV-Embed-v2
```

### Run with vLLM (Llama)

1. As above, run a local [OpenAI-compatible vLLM server](https://docs.vllm.ai/en/latest/getting_started/quickstart.html#quickstart-online) with specified GPU.

```sh
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME=<path to Huggingface home directory>

conda activate hipporag  # vllm should be in this environment

# Tune gpu-memory-utilization or max_model_len to fit your GPU memory, if OOM occurs
vllm serve meta-llama/Llama-3.3-70B-Instruct --tensor-parallel-size 2 --max_model_len 4096 --gpu-memory-utilization 0.95 
```

2. Use another GPUs to run the main program in another terminal.

```sh
export CUDA_VISIBLE_DEVICES=2,3  # set another GPUs while vLLM server is running
export HF_HOME=<path to Huggingface home directory>
dataset=sample

python main.py --dataset $dataset --llm_base_url http://localhost:8000/v1 --llm_name meta-llama/Llama-3.3-70B-Instruct --embedding_name nvidia/NV-Embed-v2
```

#### Advanced: Run with vLLM offline batch

vLLM offers an [offline batch mode](https://docs.vllm.ai/en/latest/getting_started/quickstart.html#offline-batched-inference) for faster inference, which could bring us more than 3x faster indexing compared to vLLM online server. 

1. Use the following command to run the main program with vLLM offline batch mode.

```sh
export CUDA_VISIBLE_DEVICES=0,1,2,3 # use all GPUs for faster offline indexing
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME=<path to Huggingface home directory>
export OPENAI_API_KEY=''
dataset=sample

python main.py --dataset $dataset --llm_name meta-llama/Llama-3.3-70B-Instruct --openie_mode offline --skip_graph
```

2. After the first step, OpenIE result is saved to file. Go back to run vLLM online server and main program as described in the `Run with vLLM (Llama)` main section.

## Debugging Note

- `/reproduce/dataset/sample.json` is a small dataset specifically for debugging.
- When debugging vLLM offline mode, set `tensor_parallel_size` as `1` in `hipporag/llm/vllm_offline.py`.
- If you want to rerun a particular experiment, remember to clear the saved files, including OpenIE results and knowledge graph, e.g.,

```sh
rm reproduce/dataset/openie_results/openie_sample_results_ner_meta-llama_Llama-3.3-70B-Instruct_3.json
rm -rf outputs/sample/sample_meta-llama_Llama-3.3-70B-Instruct_nvidia_NV-Embed-v2
```
### Custom Datasets

To setup your own custom dataset for evaluation, follow the format and naming convention shown in `reproduce/dataset/sample_corpus.json` (your dataset's name should be followed by `_corpus.json`). If running an experiment with pre-defined questions, organize your query corpus according to the query file `reproduce/dataset/sample.json`, be sure to also follow our naming convention.

The corpus and optional query JSON files should have the following format:

#### Retrieval Corpus JSON

```json
[
  {
    "title": "FIRST PASSAGE TITLE",
    "text": "FIRST PASSAGE TEXT",
    "idx": 0
  },
  {
    "title": "SECOND PASSAGE TITLE",
    "text": "SECOND PASSAGE TEXT",
    "idx": 1
  }
]
```

#### (Optional) Query JSON

```json

[
  {
    "id": "sample/question_1.json",
    "question": "QUESTION",
    "answer": [
      "ANSWER"
    ],
    "answerable": true,
    "paragraphs": [
      {
        "title": "{FIRST SUPPORTING PASSAGE TITLE}",
        "text": "{FIRST SUPPORTING PASSAGE TEXT}",
        "is_supporting": true,
        "idx": 0
      },
      {
        "title": "{SECOND SUPPORTING PASSAGE TITLE}",
        "text": "{SECOND SUPPORTING PASSAGE TEXT}",
        "is_supporting": true,
        "idx": 1
      }
    ]
  }
]
```

#### (Optional) Chunking Corpus

When preparing your data, you may need to chunk each passage, as longer passage may be too complex for the OpenIE process.

## Code Structure

```
📦 .
│-- 📂 src/hipporag
│   ├── 📂 embedding_model          # Implementation of all embedding models
│   │   ├── __init__.py             # Getter function for get specific embedding model classes
|   |   ├── base.py                 # Base embedding model class `BaseEmbeddingModel` to inherit and `EmbeddingConfig`
|   |   ├── NVEmbedV2.py            # Implementation of NV-Embed-v2 model
|   |   ├── ...
│   ├── 📂 evaluation               # Implementation of all evaluation metrics
│   │   ├── __init__.py
|   |   ├── base.py                 # Base evaluation metric class `BaseMetric` to inherit
│   │   ├── qa_eval.py              # Eval metrics for QA
│   │   ├── retrieval_eval.py       # Eval metrics for retrieval
│   ├── 📂 information_extraction  # Implementation of all information extraction models
│   │   ├── __init__.py
|   |   ├── openie_openai_gpt.py    # Model for OpenIE with OpenAI GPT
|   |   ├── openie_vllm_offline.py  # Model for OpenIE with LLMs deployed offline with vLLM
│   ├── 📂 llm                      # Classes for inference with large language models
│   │   ├── __init__.py             # Getter function
|   |   ├── base.py                 # Config class for LLM inference and base LLM inference class to inherit
|   |   ├── openai_gpt.py           # Class for inference with OpenAI GPT
|   |   ├── vllm_llama.py           # Class for inference using a local vLLM server
|   |   ├── vllm_offline.py         # Class for inference using the vLLM API directly
│   ├── 📂 prompts                  # Prompt templates and prompt template manager class
|   │   ├── 📂 dspy_prompts         # Prompts for filtering
|   │   │   ├── ...
|   │   ├── 📂 templates            # All prompt templates for template manager to load
|   │   │   ├── README.md           # Documentations of usage of prompte template manager and prompt template files
|   │   │   ├── __init__.py
|   │   │   ├── triple_extraction.py
|   │   │   ├── ...
│   │   ├── __init__.py
|   |   ├── linking.py              # Instruction for linking
|   |   ├── prompt_template_manager.py  # Implementation of prompt template manager
│   ├── 📂 utils                    # All utility functions used across this repo (the file name indicates its relevant usage)
│   │   ├── config_utils.py         # We use only one config across all modules and its setup is specified here
|   |   ├── ...
│   ├── __init__.py
│   ├── HippoRAG.py          # Highest level class for initiating retrieval, question answering, and evaluations
│   ├── embedding_store.py   # Storage database to load, manage and save embeddings for passages, entities and facts.
│   ├── rerank.py            # Reranking and filtering methods
│-- 📂 examples
│   ├── ...
│   ├── ...
│-- 📜 README.md
│-- 📜 requirements.txt   # Dependencies list
│-- 📜 .gitignore         # Files to exclude from Git


```

## Contact

Questions or issues? File an issue or contact 
[Bernal Jiménez Gutiérrez](mailto:jimenezgutierrez.1@osu.edu),
[Yiheng Shu](mailto:shu.251@osu.edu),
[Yu Su](mailto:su.809@osu.edu),
The Ohio State University

## Citation

If you find this work useful, please consider citing our papers:

### HippoRAG 2
```
@misc{gutiérrez2025ragmemorynonparametriccontinual,
      title={From RAG to Memory: Non-Parametric Continual Learning for Large Language Models}, 
      author={Bernal Jiménez Gutiérrez and Yiheng Shu and Weijian Qi and Sizhe Zhou and Yu Su},
      year={2025},
      eprint={2502.14802},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.14802}, 
}
```

### HippoRAG

```
@inproceedings{gutiérrez2024hipporag,
      title={HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models}, 
      author={Bernal Jiménez Gutiérrez and Yiheng Shu and Yu Gu and Michihiro Yasunaga and Yu Su},
      booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
      year={2024},
      url={https://openreview.net/forum?id=hkujvAPVsg}
 ```

## TODO:

- [x] Add support for more embedding models
- [x] Add support for embedding endpoints
- [ ] Add support for vector database integration

Please feel free to open an issue or PR if you have any questions or suggestions.
