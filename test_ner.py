# 测试NER（命名实体识别）功能

import os
import json
import logging
from typing import List

from src.hipporag import HippoRAG
from src.hipporag.information_extraction.openie_openai import OpenIE
from src.hipporag.llm.openai_gpt import CacheOpenAI
from src.hipporag.utils.config_utils import BaseConfig

# 设置API密钥
os.environ['OPENAI_API_KEY'] = 'sk-0xdfGKYi0W6KOzcGC4B3958f6b6b482f8616A7E05eCa7aEb'

def test_ner_with_hipporag():
    """使用HippoRAG实例测试NER功能"""
    print("=== 测试NER功能 (使用HippoRAG) ===")
    
    # 测试段落
    test_passages = [
        "Albert Einstein was born in Germany in 1879. He developed the theory of relativity.",
        "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976 in Cupertino, California.",
        "The Great Wall of China was built during the Ming Dynasty and extends over 13,000 miles.",
        "Microsoft Corporation was established by Bill Gates and Paul Allen in Albuquerque, New Mexico in 1975.",
        "The Amazon rainforest is located in South America and covers approximately 5.5 million square kilometers."
    ]
    
    save_dir = 'outputs/ner_test'
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
    
    # 使用HippoRAG的内部OpenIE模块进行NER测试
    for i, passage in enumerate(test_passages):
        print(f"\n--- 测试段落 {i+1} ---")
        print(f"输入文本: {passage}")
        
        try:
            # 创建一个假的chunk_key
            chunk_key = f"test_chunk_{i+1}"
            
            # 调用NER功能
            ner_result = hipporag.openie.ner(chunk_key=chunk_key, passage=passage)
            
            print(f"提取的实体: {ner_result.unique_entities}")
            print(f"原始响应: {ner_result.response}")
            print(f"元数据: {ner_result.metadata}")
            
        except Exception as e:
            print(f"NER处理出错: {str(e)}")

def test_ner_direct():
    """直接使用OpenIE类测试NER功能"""
    print("\n\n=== 测试NER功能 (直接使用OpenIE) ===")
    
    # 测试段落
    test_passages = [
        "Barack Obama served as the 44th President of the United States from 2009 to 2017.",
        "Tesla, Inc. is an American electric vehicle and clean energy company founded by Elon Musk.",
        "The Eiffel Tower is located in Paris, France and was completed in 1889."
    ]
    
    # 创建基础配置
    config = BaseConfig()
    config.llm_name = 'qwen-plus-latest'
    config.llm_base_url = 'https://api.vveai.com/v1'
    config.save_dir = 'outputs/ner_direct_test'
    
    # 初始化LLM模型
    llm_model = CacheOpenAI.from_experiment_config(config)
    
    # 初始化OpenIE实例
    openie = OpenIE(llm_model=llm_model)
    
    for i, passage in enumerate(test_passages):
        print(f"\n--- 直接测试段落 {i+1} ---")
        print(f"输入文本: {passage}")
        
        try:
            chunk_key = f"direct_test_chunk_{i+1}"
            
            # 直接调用NER功能
            ner_result = openie.ner(chunk_key=chunk_key, passage=passage)
            
            print(f"提取的实体: {ner_result.unique_entities}")
            print(f"实体数量: {len(ner_result.unique_entities)}")
            print(f"响应长度: {len(ner_result.response) if ner_result.response else 0}")
            
            # 显示实体类型分析
            if ner_result.unique_entities:
                entity_analysis = analyze_entity_types(ner_result.unique_entities)
                print(f"实体类型分析: {entity_analysis}")
            
        except Exception as e:
            print(f"直接NER处理出错: {str(e)}")

def analyze_entity_types(entities: List[str]) -> dict:
    """简单分析实体类型"""
    analysis = {
        'persons': [],
        'organizations': [],
        'locations': [],
        'dates': [],
        'others': []
    }
    
    for entity in entities:
        entity_lower = entity.lower()
        
        # 简单的启发式分类
        if any(keyword in entity_lower for keyword in ['president', 'ceo', 'founder', 'served as']):
            analysis['persons'].append(entity)
        elif any(keyword in entity_lower for keyword in ['inc', 'corp', 'company', 'corporation']):
            analysis['organizations'].append(entity)
        elif any(keyword in entity_lower for keyword in ['tower', 'city', 'country', 'states', 'paris', 'france', 'california']):
            analysis['locations'].append(entity)
        elif any(char.isdigit() for char in entity) and len(entity) <= 10:
            analysis['dates'].append(entity)
        else:
            analysis['others'].append(entity)
    
    # 移除空列表
    return {k: v for k, v in analysis.items() if v}

def test_ner_performance():
    """测试NER功能的性能"""
    print("\n\n=== NER性能测试 ===")
    
    # 长文本测试
    long_passage = """
    In 1955, Rosa Parks, an African American woman, refused to give up her seat to a white passenger on a Montgomery city bus in Alabama. 
    This act of defiance sparked the Montgomery Bus Boycott, which lasted for 381 days and was led by Martin Luther King Jr. 
    The boycott was a pivotal moment in the American Civil Rights Movement. Rosa Parks worked as a seamstress at Montgomery Fair department store. 
    She was also secretary of the Montgomery chapter of the NAACP (National Association for the Advancement of Colored People). 
    The boycott ended in December 1956 when the United States Supreme Court ruled that segregation on public buses was unconstitutional.
    """
    
    config = BaseConfig()
    config.llm_name = 'qwen-plus-latest'
    config.llm_base_url = 'https://api.vveai.com/v1'
    config.save_dir = 'outputs/ner_performance_test'
    
    llm_model = CacheOpenAI.from_experiment_config(config)
    openie = OpenIE(llm_model=llm_model)
    
    print(f"测试文本长度: {len(long_passage)} 字符")
    
    try:
        import time
        start_time = time.time()
        
        ner_result = openie.ner(chunk_key="performance_test", passage=long_passage)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"处理时间: {processing_time:.2f} 秒")
        print(f"提取的实体数量: {len(ner_result.unique_entities)}")
        print(f"提取的实体: {ner_result.unique_entities}")
        
        if ner_result.metadata:
            print(f"Token使用情况: {ner_result.metadata}")
        
    except Exception as e:
        print(f"性能测试出错: {str(e)}")

def main():
    """主测试函数"""
    print("开始NER功能测试...")
    
    # 测试1: 使用HippoRAG实例
    test_ner_with_hipporag()
    
    # 测试2: 直接使用OpenIE
    test_ner_direct()
    
    # 测试3: 性能测试
    test_ner_performance()
    
    print("\n=== NER测试完成 ===")

if __name__ == "__main__":
    main() 