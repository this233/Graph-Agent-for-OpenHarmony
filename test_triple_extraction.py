# 测试三元组提取功能

import os
import json
import logging
from typing import List, Dict, Any

from src.hipporag import HippoRAG
from src.hipporag.information_extraction.openie_openai import OpenIE
from src.hipporag.llm.openai_gpt import CacheOpenAI
from src.hipporag.utils.config_utils import BaseConfig

# 设置API密钥
os.environ['OPENAI_API_KEY'] = 'sk-0xdfGKYi0W6KOzcGC4B3958f6b6b482f8616A7E05eCa7aEb'

def test_triple_extraction_with_hipporag():
    """使用HippoRAG实例测试三元组提取功能"""
    print("=== 测试三元组提取功能 (使用HippoRAG) ===")
    
    # 测试数据：段落和对应的实体
    test_data = [
        {
            "passage": "Albert Einstein was born in Germany in 1879. He developed the theory of relativity.",
            "entities": ["Albert Einstein", "Germany", "1879", "theory of relativity"]
        },
        {
            "passage": "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976 in Cupertino, California.",
            "entities": ["Apple Inc.", "Steve Jobs", "Steve Wozniak", "Ronald Wayne", "April 1976", "Cupertino", "California"]
        },
        {
            "passage": "The Great Wall of China was built during the Ming Dynasty and extends over 13,000 miles.",
            "entities": ["Great Wall of China", "Ming Dynasty", "13,000 miles"]
        }
    ]
    
    save_dir = 'outputs/triple_test'
    llm_model_name = 'qwen-plus-latest'
    embedding_model_name = 'text-embedding-3-large'
    
    # 初始化HippoRAG实例
    hipporag = HippoRAG(
        save_dir=save_dir,
        llm_model_name=llm_model_name,
        llm_base_url='https://api.vveai.com/v1',
        embedding_base_url='https://api.vveai.com/v1',
        embedding_model_name=embedding_model_name
    )
    
    for i, data in enumerate(test_data):
        print(f"\n--- 测试案例 {i+1} ---")
        print(f"输入文本: {data['passage']}")
        print(f"预设实体: {data['entities']}")
        
        try:
            chunk_key = f"test_chunk_{i+1}"
            
            # 调用三元组提取功能
            triple_result = hipporag.openie.triple_extraction(
                chunk_key=chunk_key, 
                passage=data['passage'], 
                named_entities=data['entities']
            )
            
            print(f"提取的三元组数量: {len(triple_result.triples)}")
            print("提取的三元组:")
            for j, triple in enumerate(triple_result.triples):
                print(f"  {j+1}. {triple}")
            
            print(f"元数据: {triple_result.metadata}")
            
            # 验证三元组质量
            quality_analysis = analyze_triple_quality(triple_result.triples, data['entities'])
            print(f"三元组质量分析: {quality_analysis}")
            
        except Exception as e:
            print(f"三元组提取出错: {str(e)}")

def test_triple_extraction_direct():
    """直接使用OpenIE类测试三元组提取功能"""
    print("\n\n=== 测试三元组提取功能 (直接使用OpenIE) ===")
    
    test_data = [
        {
            "passage": "Barack Obama served as the 44th President of the United States from 2009 to 2017.",
            "entities": ["Barack Obama", "44th President", "United States", "2009", "2017"]
        },
        {
            "passage": "Tesla, Inc. is an American electric vehicle company founded by Elon Musk in 2003.",
            "entities": ["Tesla, Inc.", "American", "electric vehicle company", "Elon Musk", "2003"]
        },
        {
            "passage": "The Eiffel Tower is located in Paris, France and was completed in 1889 by Gustave Eiffel.",
            "entities": ["Eiffel Tower", "Paris", "France", "1889", "Gustave Eiffel"]
        }
    ]
    
    # 创建基础配置
    config = BaseConfig()
    config.llm_name = 'qwen-plus-latest'
    config.llm_base_url = 'https://api.vveai.com/v1'
    config.save_dir = 'outputs/triple_direct_test'
    
    # 初始化LLM模型
    llm_model = CacheOpenAI.from_experiment_config(config)
    
    # 初始化OpenIE实例
    openie = OpenIE(llm_model=llm_model)
    
    for i, data in enumerate(test_data):
        print(f"\n--- 直接测试案例 {i+1} ---")
        print(f"输入文本: {data['passage']}")
        print(f"输入实体: {data['entities']}")
        
        try:
            chunk_key = f"direct_test_chunk_{i+1}"
            
            # 直接调用三元组提取功能
            triple_result = openie.triple_extraction(
                chunk_key=chunk_key, 
                passage=data['passage'], 
                named_entities=data['entities']
            )
            
            print(f"提取的三元组: {triple_result.triples}")
            
            # 分析关系类型
            relation_analysis = analyze_relation_types(triple_result.triples)
            print(f"关系类型分析: {relation_analysis}")
            
        except Exception as e:
            print(f"直接三元组提取出错: {str(e)}")

def test_complete_openie_pipeline():
    """测试完整的OpenIE流程（NER + 三元组提取）"""
    print("\n\n=== 测试完整OpenIE流程 ===")
    
    # test_passages = [
    #     "Microsoft Corporation was founded by Bill Gates and Paul Allen in 1975 in Redmond, Washington.",
    #     "The iPhone was introduced by Apple Inc. in 2007 and revolutionized the smartphone industry."
    # ]

    test_passages = [
    ]
    with open('tmp.input', 'r') as f:
        test_passages.append(f.read())
        # test_passages.extend(f.readlines())
    

    
    config = BaseConfig()
    config.llm_name = 'qwen-plus-latest'
    config.llm_base_url = 'https://api.vveai.com/v1'
    config.save_dir = 'outputs/complete_openie_test'
    
    llm_model = CacheOpenAI.from_experiment_config(config)
    openie = OpenIE(llm_model=llm_model)
    
    for i, passage in enumerate(test_passages):
        print(f"\n--- 完整流程测试 {i+1} ---")
        print(f"输入文本: {passage}")
        
        try:
            chunk_key = f"complete_test_chunk_{i+1}"
            
            # 步骤1: NER
            print("步骤1: 执行NER...")
            ner_result = openie.ner(chunk_key=chunk_key, passage=passage)
            print(f"NER结果: {ner_result.unique_entities}， 数量{len(ner_result.unique_entities)}")
            
            # 步骤2: 三元组提取
            print("步骤2: 执行三元组提取...")
            triple_result = openie.triple_extraction(
                chunk_key=chunk_key, 
                passage=passage, 
                named_entities=ner_result.unique_entities
            )
            print(f"三元组提取结果: {triple_result.triples}，数量{len(triple_result.triples)}")
            
            # 步骤3: 使用openie方法（一次性完成）
            # print("步骤3: 一次性OpenIE...")
            # complete_result = openie.openie(chunk_key=chunk_key, passage=passage)
            # print(f"一次性NER结果: {complete_result['ner'].unique_entities}")
            # print(f"一次性三元组结果: {complete_result['triplets'].triples}")
            
            # # 比较结果
            # print("结果比较:")
            # print(f"  NER一致性: {set(ner_result.unique_entities) == set(complete_result['ner'].unique_entities)}")
            # print(f"  三元组一致性: {triple_result.triples == complete_result['triplets'].triples}")
            
        except Exception as e:
            print(f"完整流程测试出错: {str(e)}")

def analyze_triple_quality(triples: List[List[str]], entities: List[str]) -> Dict[str, Any]:
    """分析三元组质量"""
    if not triples:
        return {"valid_triples": 0, "entity_coverage": 0, "relation_diversity": 0}
    
    valid_triples = [t for t in triples if len(t) == 3]
    entity_set = set(entities)
    
    # 计算实体覆盖率
    entities_in_triples = set()
    for triple in valid_triples:
        entities_in_triples.update([triple[0], triple[2]])  # 主语和宾语
    
    entity_coverage = len(entities_in_triples.intersection(entity_set)) / len(entity_set) if entity_set else 0
    
    # 计算关系多样性
    relations = [triple[1] for triple in valid_triples]
    relation_diversity = len(set(relations))
    
    return {
        "total_triples": len(triples),
        "valid_triples": len(valid_triples),
        "entity_coverage": round(entity_coverage, 2),
        "relation_diversity": relation_diversity,
        "unique_relations": list(set(relations))
    }

def analyze_relation_types(triples: List[List[str]]) -> Dict[str, List[str]]:
    """分析关系类型"""
    relation_types = {
        "identity": [],      # is, are, was, were
        "possession": [],    # has, owns, possesses
        "location": [],      # in, at, located, based
        "temporal": [],      # during, in, from, to
        "action": [],        # founded, created, built, developed
        "others": []
    }
    
    for triple in triples:
        if len(triple) == 3:
            relation = triple[1].lower()
            
            if any(word in relation for word in ["is", "are", "was", "were"]):
                relation_types["identity"].append(triple)
            elif any(word in relation for word in ["has", "owns", "possess"]):
                relation_types["possession"].append(triple)
            elif any(word in relation for word in ["in", "at", "located", "based"]):
                relation_types["location"].append(triple)
            elif any(word in relation for word in ["during", "from", "to", "since", "until"]):
                relation_types["temporal"].append(triple)
            elif any(word in relation for word in ["founded", "created", "built", "developed", "established"]):
                relation_types["action"].append(triple)
            else:
                relation_types["others"].append(triple)
    
    # 只返回非空的类型
    return {k: v for k, v in relation_types.items() if v}

def test_triple_extraction_performance():
    """测试三元组提取性能"""
    print("\n\n=== 三元组提取性能测试 ===")
    
    # 复杂长文本
    complex_passage = """
    The Amazon rainforest, also known as Amazonia, is a moist broadleaf tropical rainforest in the Amazon biome 
    that covers most of the Amazon basin of South America. This basin encompasses 7,000,000 square kilometers, 
    of which 5,500,000 square kilometers are covered by the rainforest. This region includes territory belonging 
    to nine nations and 3,344 formally acknowledged indigenous territories. The majority of the forest is contained 
    within Brazil, with 60% of the rainforest, followed by Peru with 13%, Colombia with 10%, and with minor amounts 
    in Bolivia, Ecuador, French Guiana, Guyana, Suriname, and Venezuela.
    """
    
    entities = [
        "Amazon rainforest", "Amazonia", "moist broadleaf tropical rainforest", "Amazon biome", 
        "Amazon basin", "South America", "7,000,000 square kilometers", "5,500,000 square kilometers",
        "nine nations", "3,344 formally acknowledged indigenous territories", "Brazil", "60%",
        "Peru", "13%", "Colombia", "10%", "Bolivia", "Ecuador", "French Guiana", 
        "Guyana", "Suriname", "Venezuela"
    ]
    
    config = BaseConfig()
    config.llm_name = 'qwen-plus-latest'
    config.llm_base_url = 'https://api.vveai.com/v1'
    config.save_dir = 'outputs/triple_performance_test'
    
    llm_model = CacheOpenAI.from_experiment_config(config)
    openie = OpenIE(llm_model=llm_model)
    
    print(f"测试文本长度: {len(complex_passage)} 字符")
    print(f"输入实体数量: {len(entities)}")
    
    try:
        import time
        start_time = time.time()
        
        triple_result = openie.triple_extraction(
            chunk_key="performance_test", 
            passage=complex_passage, 
            named_entities=entities
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"处理时间: {processing_time:.2f} 秒")
        print(f"提取的三元组数量: {len(triple_result.triples)}")
        
        quality_analysis = analyze_triple_quality(triple_result.triples, entities)
        print(f"质量分析: {quality_analysis}")
        
        print("提取的三元组示例:")
        for i, triple in enumerate(triple_result.triples[:5]):  # 只显示前5个
            print(f"  {i+1}. {triple}")
        
        if triple_result.metadata:
            print(f"Token使用情况: {triple_result.metadata}")
            
    except Exception as e:
        print(f"性能测试出错: {str(e)}")

def main():
    """主测试函数"""
    print("开始三元组提取功能测试...")
    
    # 测试1: 使用HippoRAG实例
    # test_triple_extraction_with_hipporag()
    
    # 测试2: 直接使用OpenIE
    # test_triple_extraction_dir  ect()
    
    # 测试3: 完整OpenIE流程
    test_complete_openie_pipeline()
    
    # 测试4: 性能测试
    # test_triple_extraction_performance()
    
    print("\n=== 三元组提取测试完成 ===")

if __name__ == "__main__":
    main() 