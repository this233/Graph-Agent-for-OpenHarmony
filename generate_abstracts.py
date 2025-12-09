#!/usr/bin/env python3
"""
摘要生成器使用示例脚本

该脚本展示如何使用AbstractGenerator为document_processor生成的JSON结构补充摘要字段。

特点：
- 使用 deepseek-v3.2-exp 模型（支持深度思考模式）
- 支持从备份文件中复用已有摘要（调用失败时优先尝试复用）
- 支持备用配置
"""

import json
import os
import argparse
import logging
import requests
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 禁用SSL警告
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# API配置
MAAS_API_URL = "https://api.modelarts-maas.com/v2/chat/completions"
MAAS_API_KEY = "BQm_Gkd1EoTcHkJfVf31dTWfMIOsW3_mKIDfM5j-MvvwNM5jNl9XnLOjvNjEOuDiIWoKb-DIphdRWt2gOoNwBw"
MODEL_NAME = "deepseek-v3.2-exp"

os.environ['OPENAI_API_KEY'] = MAAS_API_KEY

# 设置路径以便导入HippoRAG模块
import sys
sys.path.append('src')

from hipporag.abstract_generator import AbstractGenerator
from hipporag.utils.config_utils import BaseConfig

def call_maas_api(content: str, content_type: str, max_retries: int = 3, 
                   enable_thinking: bool = False) -> Optional[str]:
    """
    直接调用 MAAS API 生成摘要
    
    Args:
        content: 待生成摘要的内容
        content_type: 内容类型
        max_retries: 最大重试次数
        enable_thinking: 是否启用深度思考模式
        
    Returns:
        str: 生成的摘要，失败返回None
    """
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {MAAS_API_KEY}'
    }
    
    # 根据内容类型构建提示词
    type_prompts = {
        'file': "请为以下文档内容生成一个简洁的摘要（不超过200字），概括主要内容和关键信息：",
        'chunk': "请为以下文档片段生成一个简洁的摘要（不超过150字），概括主要内容：",
        'code': "请为以下代码片段生成一个简洁的摘要（不超过100字），说明代码的功能和用途：",
        'table': "请为以下表格内容生成一个简洁的摘要（不超过100字），概括表格包含的信息："
    }
    
    prompt = type_prompts.get(content_type, type_prompts['chunk'])
    
    # 截断过长的内容
    max_content_length = 8000
    if len(content) > max_content_length:
        content = content[:max_content_length] + "\n... [内容已截断]"
    
    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "你是一个专业的技术文档摘要助手，擅长提取关键信息并生成简洁准确的摘要。"},
            {"role": "user", "content": f"{prompt}\n\n{content}"}
        ],
        "chat_template_kwargs": {
            "thinking": enable_thinking
        },
        "max_tokens": 300
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                MAAS_API_URL,
                headers=headers,
                data=json.dumps(data),
                verify=False,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    abstract = result['choices'][0]['message']['content']
                    return abstract.strip()
            else:
                logging.warning(f"API调用失败 (尝试 {attempt + 1}/{max_retries}): {response.status_code}")
                
        except Exception as e:
            logging.warning(f"API调用异常 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
            
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)
            
    return None


class FallbackAbstractGenerator(AbstractGenerator):
    """
    带有备用配置的摘要生成器
    
    当主配置的LLM调用失败时，优先尝试从备份文件复用，然后切换到备用配置重试
    """
    
    def __init__(self, primary_config: BaseConfig, fallback_config: Optional[BaseConfig] = None,
                 backup_file: Optional[str] = None):
        """
        初始化带备用配置的摘要生成器
        
        Args:
            primary_config: 主配置
            fallback_config: 备用配置，当主配置失败时使用
            backup_file: 备份JSON文件路径，用于复用已有摘要
        """
        super().__init__(global_config=primary_config)
        self.primary_config = primary_config
        self.fallback_config = fallback_config
        self.fallback_generator = None
        self.logger = logging.getLogger(__name__)
        
        # 存储API密钥
        self.primary_api_key = os.environ.get('OPENAI_API_KEY')
        self.fallback_api_key = 'sk-0xdfGKYi0W6KOzcGC4B3958f6b6b482f8616A7E05eCa7aEb'
        
        # 统计fallback使用次数
        self.fallback_used_count = 0
        self.backup_used_count = 0
        
        # 加载备份摘要
        self.backup_abstracts = {}
        if backup_file and os.path.exists(backup_file):
            self._load_backup_abstracts(backup_file)
        
        if fallback_config:
            try:
                # 临时切换到备用API密钥来初始化备用配置
                original_key = os.environ.get('OPENAI_API_KEY')
                os.environ['OPENAI_API_KEY'] = self.fallback_api_key
                
                try:
                    self.fallback_generator = AbstractGenerator(global_config=fallback_config)
                    self.logger.info("备用配置初始化成功")
                finally:
                    # 恢复原始API密钥
                    if original_key:
                        os.environ['OPENAI_API_KEY'] = original_key
                        
            except Exception as e:
                self.logger.warning(f"备用配置初始化失败: {str(e)}")
                
    def _load_backup_abstracts(self, backup_file: str):
        """
        从备份文件加载已有的摘要
        
        Args:
            backup_file: 备份JSON文件路径
        """
        try:
            self.logger.info(f"加载备份文件: {backup_file}")
            with open(backup_file, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)
                
            # 递归收集所有摘要
            def collect_abstracts(obj: Any, parent_id: str = ""):
                if isinstance(obj, dict):
                    # 收集文件级摘要
                    for key, value in obj.items():
                        if key.startswith('file-') and isinstance(value, dict):
                            abstract = value.get('abstract', '')
                            content = value.get('content', '')
                            if abstract and content:
                                content_hash = self._compute_content_hash(content)
                                self.backup_abstracts[content_hash] = abstract
                                
                        # 收集chunk、code、table级摘要
                        if isinstance(value, dict):
                            if 'abstract' in value and 'content' in value:
                                abstract = value.get('abstract', '')
                                content = value.get('content', '')
                                if abstract and content:
                                    content_hash = self._compute_content_hash(content)
                                    self.backup_abstracts[content_hash] = abstract
                            collect_abstracts(value, key)
                    
                    # 处理 chunks, codes, tables 字典
                    for dict_key in ['chunks', 'codes', 'tables']:
                        if dict_key in obj and isinstance(obj[dict_key], dict):
                            for item_id, item_value in obj[dict_key].items():
                                if isinstance(item_value, dict):
                                    abstract = item_value.get('abstract', '')
                                    content = item_value.get('content', '')
                                    if abstract and content:
                                        content_hash = self._compute_content_hash(content)
                                        self.backup_abstracts[content_hash] = abstract
                                    collect_abstracts(item_value, item_id)
                                    
            collect_abstracts(backup_data)
            self.logger.info(f"从备份文件加载了 {len(self.backup_abstracts)} 个摘要")
            
        except Exception as e:
            self.logger.warning(f"加载备份文件失败: {str(e)}")
            
    def _compute_content_hash(self, content: str) -> str:
        """计算内容的哈希值"""
        import hashlib
        return hashlib.md5(content.encode()).hexdigest()
        
    def _get_backup_abstract(self, content: str) -> Optional[str]:
        """
        尝试从备份中获取摘要
        
        Args:
            content: 内容
            
        Returns:
            str: 备份中的摘要，如果没有则返回None
        """
        content_hash = self._compute_content_hash(content)
        return self.backup_abstracts.get(content_hash)
    
    def generate_abstract_with_fallback(self, content: str, content_type: str, **kwargs) -> str:
        """
        使用备用配置机制生成摘要
        
        优先级：
        1. 首先尝试主配置（MAAS API）
        2. 如果失败，尝试从备份文件复用
        3. 如果备份没有，尝试备用配置
        
        Args:
            content: 待生成摘要的内容
            content_type: 内容类型 ('file', 'chunk', 'code', 'table')
            **kwargs: 额外参数（如文件路径等）
            
        Returns:
            生成的摘要文本
        """
        primary_failed = False
        
        # 首先尝试主配置（直接调用 MAAS API）
        try:
            result = call_maas_api(content, content_type)
            if result and not result.startswith("摘要生成失败"):
                return result
            else:
                primary_failed = True
                self.logger.warning(f"⚠️  主配置 {content_type} 摘要生成失败")
        except Exception as e:
            primary_failed = True
            content_preview = content[:100] + "..." if len(content) > 100 else content
            self.logger.warning(f"❌ 主配置 {content_type} 摘要生成异常:")
            self.logger.warning(f"  错误: {str(e)}")
            self.logger.warning(f"  内容长度: {len(content)} 字符")
        
        # 如果主配置失败，尝试从备份复用
        if primary_failed:
            backup_result = self._get_backup_abstract(content)
            if backup_result:
                self.backup_used_count += 1
                self.logger.info(f"✅ 从备份复用 {content_type} 摘要")
                return backup_result
        
        # 如果备份也没有，尝试备用配置
        if self.fallback_generator and primary_failed:
            try:
                content_preview = content[:200] + "..." if len(content) > 200 else content
                self.logger.info(f"=== 使用备用配置重试 {content_type} 摘要生成 ===")
                self.logger.info(f"Chunk类型: {content_type}")
                self.logger.info(f"内容长度: {len(content)} 字符")
                if 'file_path' in kwargs:
                    self.logger.info(f"文件路径: {kwargs['file_path']}")
                
                # 临时切换到备用API密钥
                original_key = os.environ.get('OPENAI_API_KEY')
                os.environ['OPENAI_API_KEY'] = self.fallback_api_key
                
                try:
                    result = self.fallback_generator.generate_abstract(content, content_type, **kwargs)
                    self.fallback_used_count += 1
                    
                    if not result.startswith("摘要生成失败:"):
                        result_preview = result[:100] + "..." if len(result) > 100 else result
                        self.logger.info(f"✅ 备用配置成功生成 {content_type} 摘要")
                        self.logger.info(f"摘要预览: {result_preview}")
                        if hasattr(self, 'stats') and self.stats['failed_calls'] > 0:
                            self.stats['failed_calls'] -= 1
                        return result
                finally:
                    if original_key:
                        os.environ['OPENAI_API_KEY'] = original_key
                    
            except Exception as e:
                self.logger.warning(f"备用配置调用也失败: {str(e)}")
        
        # 所有配置都失败，返回错误信息
        return f"摘要生成失败: 主配置、备份和备用配置都无法完成任务"
    
    def _process_single_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理单个摘要生成任务（重写父类方法以使用fallback机制）
        
        Args:
            task: 任务信息
            
        Returns:
            更新后的任务信息（包含生成的摘要）
        """
        task_type = task['type']
        content = task['content']
        
        # 生成摘要
        kwargs = {}
        if task_type == 'file' and 'file_path' in task:
            kwargs['file_path'] = task['file_path']
            
        abstract = self.generate_abstract_with_fallback(content, task_type, **kwargs)
        
        # 更新任务信息
        task['abstract'] = abstract
        return task

def create_fallback_config():
    """创建备用配置"""
    fallback_config = BaseConfig()
    fallback_config.llm_name = 'gpt-4.1'
    # fallback_config.llm_base_url = 'http://api.v36.cm/v1/'
    fallback_config.llm_base_url = 'https://api.vveai.com/v1'
    fallback_config.temperature = 0.1
    fallback_config.max_new_tokens = 200
    
    # 不在这里修改API密钥，而是在实际使用时临时切换
    return fallback_config

def setup_logging(level=logging.INFO):
    """设置日志配置"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('abstract_generation.log')
        ]
    )

def validate_input_file(file_path: str) -> bool:
    """验证输入文件格式"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 简单验证：检查是否包含文件级别的结构
        if not isinstance(data, dict):
            return False
            
        # 检查是否有文件级别的条目
        for key, value in data.items():
            if key.startswith('file-') and isinstance(value, dict):
                if 'content' in value and 'file_path' in value:
                    return True
        
        return False
        
    except Exception as e:
        print(f"验证输入文件失败: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="为文档结构JSON生成摘要",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python generate_abstracts.py input.json output.json
  python generate_abstracts.py input.json output.json --max-workers 5 --llm-name gpt-4
  
支持的LLM模型:
  - gpt-3.5-turbo (默认)
  - gpt-4
  - gpt-4-turbo
  - 其他OpenAI兼容的模型
        """
    )
    
    parser.add_argument(
        "input_file", 
        help="输入的文档结构JSON文件路径"
    )
    
    parser.add_argument(
        "output_file", 
        help="输出的文档结构JSON文件路径"
    )
    
    parser.add_argument(
        "--max-workers", 
        type=int, 
        default=20, 
        help="最大并发工作线程数 (默认: 3)"
    )
    
    parser.add_argument(
        "--llm-name", 
        default="gpt-3.5-turbo", 
        help="LLM模型名称 (默认: gpt-3.5-turbo)"
    )
    
    parser.add_argument(
        "--llm-base-url", 
        help="LLM服务的基础URL（用于自定义API端点）"
    )
    
    parser.add_argument(
        "--save-dir", 
        default="./outputs", 
        help="保存目录 (默认: ./outputs)"
    )
    
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.1, 
        help="LLM生成温度 (默认: 0.1)"
    )
    
    parser.add_argument(
        "--max-tokens", 
        type=int, 
        default=200, 
        help="每个摘要的最大token数 (默认: 200)"
    )
    
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="只分析任务，不实际生成摘要"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="启用详细日志输出"
    )
    
    parser.add_argument(
        "--disable-fallback", 
        action="store_true", 
        help="禁用备用配置，只使用主配置"
    )
    
    parser.add_argument(
        "--backup",
        help="备份JSON文件路径，调用失败时优先尝试复用已有摘要"
    )
    
    args = parser.parse_args()
    
    # 设置日志级别
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    
    logger = logging.getLogger(__name__)
    
    # 验证输入文件
    if not os.path.exists(args.input_file):
        logger.error(f"输入文件不存在: {args.input_file}")
        sys.exit(1)
    
    if not validate_input_file(args.input_file):
        logger.error(f"输入文件格式无效: {args.input_file}")
        logger.error("请确保输入文件是由document_processor生成的有效JSON结构")
        sys.exit(1)
    
    # 创建输出目录
    output_dir = Path(args.output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载文档结构
    logger.info(f"加载文档结构: {args.input_file}")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        doc_structure = json.load(f)
    
    logger.info(f"文档结构包含 {len(doc_structure)} 个文件")
    
    # 创建主配置
    config = BaseConfig()
    config.llm_name = args.llm_name
    config.temperature = args.temperature
    config.max_new_tokens = args.max_tokens
    config.save_dir = args.save_dir
    
    if args.llm_base_url:
        config.llm_base_url = args.llm_base_url
    
    # config.save_dir = 'outputs/complete_openie_test'
    config.llm_name = 'DeepSeek-V3'
    config.llm_base_url = 'https://api.modelarts-maas.com/v1'

    # 创建备用配置
    fallback_config = None
    if not args.disable_fallback:
        try:
            fallback_config = create_fallback_config()
            logger.info("备用配置已创建")
        except Exception as e:
            logger.warning(f"创建备用配置失败: {str(e)}")

    # 创建摘要生成器
    logger.info("初始化摘要生成器...")
    if fallback_config and not args.disable_fallback:
        generator = FallbackAbstractGenerator(
            primary_config=config, 
            fallback_config=fallback_config,
            backup_file=args.backup
        )
        logger.info("使用带备用配置的摘要生成器")
        if args.backup:
            logger.info(f"备份文件: {args.backup}")
    else:
        generator = AbstractGenerator(global_config=config)
        logger.info("使用标准摘要生成器")
    
    if args.dry_run:
        # 只分析任务，不实际生成
        logger.info("=== 执行预分析 ===")
        tasks = generator._collect_abstract_tasks(doc_structure)
        
        task_counts = {}
        for task in tasks:
            task_type = task['type']
            task_counts[task_type] = task_counts.get(task_type, 0) + 1
        
        logger.info(f"发现的摘要生成任务：")
        for task_type, count in task_counts.items():
            logger.info(f"  {task_type}: {count} 个")
        
        logger.info(f"总计需要生成 {len(tasks)} 个摘要")
        
        # 估算耗时（假设每个摘要需要3秒）
        estimated_time = len(tasks) * 3 / args.max_workers
        logger.info(f"预估处理时间: {estimated_time:.1f} 秒")
        
        return
    
    # 处理文档结构
    logger.info("开始生成摘要...")
    try:
        updated_structure = generator.process_document_structure(
            doc_structure, 
            max_workers=args.max_workers
        )
        
        # 保存结果
        logger.info(f"保存结果到: {args.output_file}")
        generator.save_updated_structure(updated_structure, args.output_file)
        
        logger.info("摘要生成完成！")
        
        # 输出统计信息
        stats = generator.stats
        total_processed = (stats['files_processed'] + stats['chunks_processed'] + 
                          stats['codes_processed'] + stats['tables_processed'])
        total_tasks = total_processed + stats['failed_calls']
        
        logger.info(f"处理统计:")
        logger.info(f"  总任务数: {total_tasks}")
        logger.info(f"  成功完成: {total_processed} 项")
        logger.info(f"  失败: {stats['failed_calls']} 项")
        if total_tasks > 0:
            logger.info(f"  总成功率: {total_processed/total_tasks*100:.1f}%")
        
        # 输出备用配置使用统计
        if isinstance(generator, FallbackAbstractGenerator):
            backup_count = generator.backup_used_count
            fallback_count = generator.fallback_used_count
            if backup_count > 0:
                logger.info(f"  其中备份复用成功: {backup_count} 项")
            if fallback_count > 0:
                logger.info(f"  其中备用配置成功: {fallback_count} 项")
            main_success = total_processed - backup_count - fallback_count
            logger.info(f"  主配置成功: {main_success} 项")
        
    except KeyboardInterrupt:
        logger.warning("用户中断了处理过程")
        sys.exit(1)
    except Exception as e:
        logger.error(f"处理过程中发生错误: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 