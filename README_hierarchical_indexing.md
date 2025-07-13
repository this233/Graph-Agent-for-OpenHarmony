# HippoRAG层次化文档索引功能

本文档介绍HippoRAG新增的层次化文档索引功能，支持处理复杂的多层级文档结构。

## 功能概述

新的`index_from_json`方法支持从层次化JSON结构构建知识图谱，包括：

### 支持的节点类型

1. **文件节点** (`file-xxxx`)
   - 文件摘要的向量化存储
   - 文件路径和内容信息
   - 用于文档级别的检索

2. **段落节点** (`chunk-xxxx`)  
   - 段落摘要或内容的向量化
   - 支持多级段落嵌套结构
   - 主要的检索目标，作为RAG上下文

3. **代码块节点** (`code-xxxx`)
   - 代码块及上下文的摘要向量化
   - 支持编程相关问题的精确检索

4. **表格节点** (`table-xxxx`)
   - 表格及上下文的摘要向量化
   - 支持结构化数据的检索

5. **语义实体节点** (`entity-xxxx`)
   - 从文本中提取的专有名词
   - 实体名称和解释的组合向量化
   - 支持实体间的语义关系

### 支持的边类型

#### 结构性边
- **包含关系**: 父节点 → 子节点 (权重: 1.0)
  - 文件 → 段落
  - 段落 → 子段落
  - 段落 → 代码块/表格
  - 段落 → 实体

- **跳转关系**: 段落 → 引用文件 (权重: 1.0)
  - 支持文档间的超链接关系

#### 语义边
- **实体关系**: 基于提取的三元组
  - 实体 ↔ 实体 (双向连接)
  - 权重基于关系频次

- **同义词边**: 基于向量相似度
  - 语义相似的实体间连接
  - 增强检索的召回率

## JSON数据格式

```json
{
  "file-xxxx": {
    "abstract": "文件摘要（用于向量化）",
    "file_path": "文件相对路径",
    "content": "完整文件内容",
    "chunks": {
      "chunk-xxxx": {
        "abstract": "段落摘要（用于向量化）", 
        "content": "完整段落内容（含标题层级）",
        "jump": {
          "file-yyyy": {
            "file_path": "跳转目标文件路径",
            "jump_name": "跳转显示文本"
          }
        },
        "codes": {
          "code-xxxx": {
            "abstract": "代码块摘要（用于向量化）",
            "content": "代码块及上下文内容"
          }
        },
        "tables": {
          "table-xxxx": {
            "abstract": "表格摘要（用于向量化）", 
            "content": "表格及上下文内容"
          }
        },
        "filter_chunk": {
          "content": "去除代码和表格后的文本",
          "extracted_entities": [
            ["实体名", "实体解释"],
            ["实体名2", "实体解释2"]
          ],
          "extracted_triples": [
            ["主体实体", "关系", "客体实体"],
            ["实体A", "关系类型", "实体B"]
          ]
        },
        "chunks": {
          "chunk-yyyy": {
            // 递归的子段落结构
          }
        }
      }
    }
  }
}
```

## 使用方法

### 基本用法

```python
from src.hipporag.HippoRAG import HippoRAG
from src.hipporag.utils.config_utils import BaseConfig
import json

# 配置HippoRAG
config = BaseConfig()
config.save_dir = "output/hierarchical_demo"
config.llm_name = "gpt-3.5-turbo"
config.embedding_model_name = "text-embedding-ada-002"

# 初始化HippoRAG
hippo_rag = HippoRAG(global_config=config)

# 加载JSON数据
with open("triples.json", "r", encoding="utf-8") as f:
    json_data = json.load(f)

# 执行层次化索引
hippo_rag.index_from_json(json_data)

# 查看图统计信息
graph_info = hippo_rag.get_graph_info()
print(f"文件节点: {graph_info['num_file_nodes']}")
print(f"段落节点: {graph_info['num_chunk_nodes']}")
print(f"代码节点: {graph_info['num_code_nodes']}")
print(f"表格节点: {graph_info['num_table_nodes']}")
print(f"实体节点: {graph_info['num_entity_nodes']}")
print(f"事实数量: {graph_info['num_extracted_facts']}")

# 执行检索
results = hippo_rag.retrieve(["查询问题"], num_to_retrieve=5)
```

### 高级配置

```python
# 同义词边配置
config.synonymy_edge_topk = 50  # 每个实体的候选同义词数量
config.synonymy_edge_sim_threshold = 0.8  # 相似度阈值

# 图搜索配置  
config.linking_top_k = 20  # 链接的顶级实体数量
config.passage_node_weight = 0.05  # 段落节点权重

# PageRank配置
config.damping = 0.5  # 阻尼因子
```

## 检索增强

层次化索引提供了多层次的检索能力：

### 1. 多粒度检索
- **文件级**: 文档整体相关性
- **段落级**: 具体内容片段  
- **代码级**: 编程相关查询
- **表格级**: 结构化数据查询
- **实体级**: 概念和术语查询

### 2. 图增强检索
- **结构导航**: 沿层次关系传播权重
- **跳转关系**: 跨文档关联发现
- **语义关系**: 基于实体关系的推理
- **同义词扩展**: 概念相似性匹配

### 3. 混合排序
- **事实检索**: 基于实体关系的相关性
- **密集检索**: 基于向量相似度
- **图搜索**: 基于PersonalizedPageRank
- **认知记忆**: DSPy过滤器重排序

## 性能优化

### 批量处理
- 支持大规模文档的批量索引
- 并行处理多个嵌入计算
- 增量更新现有图结构

### 内存管理
- 分离式存储不同类型的嵌入
- 按需加载检索所需数据
- 高效的图结构表示

### 可扩展性
- 支持动态添加新的节点类型
- 可配置的边权重计算策略
- 模块化的处理流程

## 示例脚本

运行提供的示例脚本：

```bash
python example_hierarchical_index.py
```

该脚本展示了：
- JSON数据加载和验证
- 层次化索引执行
- 图统计信息显示
- 多轮检索测试

## 注意事项

1. **数据质量**: 确保JSON结构完整，实体和关系提取准确
2. **计算资源**: 大规模文档需要充足的内存和计算时间
3. **模型选择**: 选择合适的嵌入模型以获得最佳检索效果
4. **参数调优**: 根据具体应用场景调整各项参数

## 故障排除

### 常见问题

1. **JSON格式错误**
   - 检查文件编码（UTF-8）
   - 验证JSON语法正确性
   - 确认必需字段存在

2. **嵌入计算失败**
   - 检查API密钥和配置
   - 确认网络连接正常
   - 验证模型名称正确

3. **图构建异常**
   - 检查节点ID格式一致性
   - 验证边关系的有效性
   - 确认内存资源充足

### 调试模式

启用详细日志以排查问题：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

这将显示详细的处理过程和潜在错误信息。 