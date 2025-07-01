import json
import os
import logging
from typing import Dict, Any, List, Optional, Tuple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from .llm import _get_llm_class, BaseLLM
from .utils.config_utils import BaseConfig
from .utils.llm_utils import TextChatMessage

logger = logging.getLogger(__name__)

class AbstractGenerator:
    """
    摘要生成器：为文档结构中的各种内容生成摘要
    
    支持为以下类型的内容生成摘要：
    - 文件级别摘要
    - 文档块(chunk)摘要  
    - 代码块摘要
    - 表格摘要
    """
    
    def __init__(self, global_config: Optional[BaseConfig] = None, llm_model: Optional[BaseLLM] = None):
        """
        初始化摘要生成器
        
        Args:
            global_config: 全局配置对象
            llm_model: 预初始化的LLM模型，如果为None则根据配置创建
        """
        if global_config is None:
            self.global_config = BaseConfig()
        else:
            self.global_config = global_config
            
        if llm_model is None:
            self.llm_model: BaseLLM = _get_llm_class(self.global_config)
        else:
            self.llm_model = llm_model
            
        # 摘要生成的提示模板
        self.prompts = {
            'file': self._get_file_prompt(),
            'chunk': self._get_chunk_prompt(), 
            'code': self._get_code_prompt(),
            'table': self._get_table_prompt()
        }
        
        # 统计信息
        self.stats = {
            'files_processed': 0,
            'chunks_processed': 0,
            'codes_processed': 0,
            'tables_processed': 0,
            'total_llm_calls': 0,
            'failed_calls': 0
        }
    
    def _get_file_prompt(self) -> str:
        """获取文件级别摘要的提示模板"""
        return """你是一个专业的文档分析助手。请为以下Markdown文件生成一个简洁的摘要。

要求：
1. 摘要长度控制在100-200字
2. 概括文件的主要内容和主题
3. 突出文件的核心价值和用途
4. 使用清晰、专业的语言
5. 不要包含格式标记

文件路径：{file_path}

文件内容：
{content}

请生成摘要："""

    def _get_chunk_prompt(self) -> str:
        """获取文档块摘要的提示模板"""
        return """你是一个专业的文档分析助手。请为以下文档片段生成一个简洁的摘要。

要求：
1. 摘要长度控制在50-100字
2. 概括该片段的主要内容
3. 突出关键信息和要点
4. 使用清晰、准确的语言
5. 不要包含格式标记

文档片段内容：
{content}

请生成摘要："""

    def _get_code_prompt(self) -> str:
        """获取代码块摘要的提示模板"""
        return """你是一个专业的代码分析助手。请为以下代码块生成一个简洁的摘要。

要求：
1. 摘要长度控制在30-80字
2. 说明代码的主要功能和用途
3. 如果能识别编程语言，请说明
4. 突出代码的核心逻辑或算法
5. 使用技术性但易懂的语言

代码块内容：
{content}

请生成摘要："""

    def _get_table_prompt(self) -> str:
        """获取表格摘要的提示模板"""
        return """你是一个专业的数据分析助手。请为以下表格生成一个简洁的摘要。

要求：
1. 摘要长度控制在30-80字
2. 说明表格的主要内容和结构
3. 突出表格中的关键数据或信息
4. 如果是配置表、参数表等，请说明用途
5. 使用清晰、准确的语言

表格内容：
{content}

请生成摘要："""

    def generate_abstract(self, content: str, content_type: str, **kwargs) -> str:
        """
        为指定内容生成摘要
        
        Args:
            content: 待生成摘要的内容
            content_type: 内容类型 ('file', 'chunk', 'code', 'table')
            **kwargs: 额外参数（如文件路径等）
            
        Returns:
            生成的摘要文本
        """
        if content_type not in self.prompts:
            raise ValueError(f"不支持的内容类型: {content_type}")
            
        # 构建提示
        prompt_template = self.prompts[content_type]
        
        # 准备模板参数
        template_params = {'content': content}
        if content_type == 'file' and 'file_path' in kwargs:
            template_params['file_path'] = kwargs['file_path']
            
        prompt = prompt_template.format(**template_params)
        
        try:
            # 构建消息格式 - 使用TextChatMessage格式
            messages: List[TextChatMessage] = [
                TextChatMessage(role="system", content="你是一个专业的文档分析助手，擅长生成简洁准确的摘要。"),
                TextChatMessage(role="user", content=prompt)
            ]
            
            # 调用LLM生成摘要
            try:
                response, metadata, cache_hit = self.llm_model.infer(messages)
            except ValueError:
                # 如果没有cache_hit返回值，则只有2个返回值
                response, metadata = self.llm_model.infer(messages)
            
            # 更新统计信息
            self.stats['total_llm_calls'] += 1
            
            # 清理响应文本 - response实际上是字符串，虽然类型注解显示为List[TextChatMessage]
            abstract = str(response).strip()
            
            logger.debug(f"Generated abstract for {content_type}: {abstract[:50]}...")
            return abstract
            
        except Exception as e:
            self.stats['failed_calls'] += 1
            logger.error(f"生成摘要失败 ({content_type}): {str(e)}")
            return f"摘要生成失败: {str(e)}"
    
    def process_document_structure(self, doc_structure: Dict[str, Any], 
                                 max_workers: int = 3,
                                 save_progress: bool = True,
                                 progress_file: Optional[str] = None) -> Dict[str, Any]:
        """
        处理整个文档结构，为所有空缺的摘要字段生成内容
        
        Args:
            doc_structure: 文档结构字典
            max_workers: 并发处理的最大工作线程数
            save_progress: 是否保存处理进度
            progress_file: 进度保存文件路径
            
        Returns:
            更新后的文档结构字典
        """
        logger.info("开始处理文档结构，生成摘要...")
        
        # 收集所有需要生成摘要的任务
        tasks = self._collect_abstract_tasks(doc_structure)
        
        logger.info(f"发现 {len(tasks)} 个需要生成摘要的任务")
        
        if len(tasks) == 0:
            logger.info("没有发现需要生成摘要的内容")
            return doc_structure
        
        # 批量处理任务
        completed_tasks = self._process_tasks_batch(tasks, max_workers)
        
        # 更新文档结构
        updated_structure = self._update_document_structure(doc_structure, completed_tasks)
        
        # 输出统计信息
        self._print_statistics()
        
        return updated_structure
    
    def _collect_abstract_tasks(self, doc_structure: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        递归收集所有需要生成摘要的任务
        
        Args:
            doc_structure: 文档结构字典
            
        Returns:
            任务列表，每个任务包含类型、内容、路径等信息
        """
        tasks = []
        
        for file_id, file_info in doc_structure.items():
            # 处理文件级别的摘要
            if file_info.get('abstract', '') == '':
                tasks.append({
                    'type': 'file',
                    'path': [file_id],
                    'content': file_info.get('content', ''),
                    'file_path': file_info.get('file_path', ''),
                    'id': file_id
                })
            
            # 递归处理chunks
            if 'chunks' in file_info:
                chunk_tasks = self._collect_chunk_tasks(file_info['chunks'], [file_id, 'chunks'])
                tasks.extend(chunk_tasks)
        
        return tasks
    
    def _collect_chunk_tasks(self, chunks_dict: Dict[str, Any], base_path: List[str]) -> List[Dict[str, Any]]:
        """
        递归收集chunk相关的摘要任务
        
        Args:
            chunks_dict: chunks字典
            base_path: 当前路径
            
        Returns:
            任务列表
        """
        tasks = []
        
        for chunk_id, chunk_info in chunks_dict.items():
            current_path = base_path + [chunk_id]
            
            # 处理chunk摘要
            if chunk_info.get('abstract', '') == '':
                tasks.append({
                    'type': 'chunk',
                    'path': current_path,
                    'content': chunk_info.get('content', ''),
                    'id': chunk_id
                })
            
            # 处理代码块摘要
            if 'codes' in chunk_info:
                for code_id, code_info in chunk_info['codes'].items():
                    if code_info.get('abstract', '') == '':
                        tasks.append({
                            'type': 'code',
                            'path': current_path + ['codes', code_id],
                            'content': code_info.get('content', ''),
                            'id': code_id
                        })
            
            # 处理表格摘要
            if 'tables' in chunk_info:
                for table_id, table_info in chunk_info['tables'].items():
                    if table_info.get('abstract', '') == '':
                        tasks.append({
                            'type': 'table',
                            'path': current_path + ['tables', table_id],
                            'content': table_info.get('content', ''),
                            'id': table_id
                        })
            
            # 递归处理嵌套的chunks
            if 'chunks' in chunk_info:
                nested_tasks = self._collect_chunk_tasks(chunk_info['chunks'], current_path + ['chunks'])
                tasks.extend(nested_tasks)
        
        return tasks
    
    def _process_tasks_batch(self, tasks: List[Dict[str, Any]], max_workers: int) -> List[Dict[str, Any]]:
        """
        批量处理摘要生成任务
        
        Args:
            tasks: 任务列表
            max_workers: 最大并发数
            
        Returns:
            完成的任务列表
        """
        completed_tasks = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_task = {}
            for task in tasks:
                future = executor.submit(self._process_single_task, task)
                future_to_task[future] = task
            
            # 收集结果
            with tqdm(total=len(tasks), desc="生成摘要") as pbar:
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        result = future.result()
                        completed_tasks.append(result)
                        
                        # 更新进度统计
                        task_type = result['type']
                        if task_type == 'file':
                            self.stats['files_processed'] += 1
                        elif task_type == 'chunk':
                            self.stats['chunks_processed'] += 1
                        elif task_type == 'code':
                            self.stats['codes_processed'] += 1
                        elif task_type == 'table':
                            self.stats['tables_processed'] += 1
                            
                    except Exception as e:
                        logger.error(f"任务处理失败: {str(e)}")
                        self.stats['failed_calls'] += 1
                    
                    pbar.update(1)
        
        return completed_tasks
    
    def _process_single_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理单个摘要生成任务
        
        Args:
            task: 任务信息
            
        Returns:
            更新后的任务信息（包含生成的摘要）
        """
        task_type = task['type']
        content = task['content']
        
        # # 限制内容长度以避免过长的输入
        # max_content_length = {
        #     'file': 8000,
        #     'chunk': 4000,
        #     'code': 2000,
        #     'table': 2000
        # }
        
        # if len(content) > max_content_length[task_type]:
        #     content = content[:max_content_length[task_type]] + "...[内容过长，已截断]"
        
        # 生成摘要
        kwargs = {}
        if task_type == 'file' and 'file_path' in task:
            kwargs['file_path'] = task['file_path']
            
        abstract = self.generate_abstract(content, task_type, **kwargs)
        
        # 更新任务信息
        task['abstract'] = abstract
        return task
    
    def _update_document_structure(self, doc_structure: Dict[str, Any], 
                                 completed_tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        将生成的摘要更新到文档结构中
        
        Args:
            doc_structure: 原始文档结构
            completed_tasks: 完成的任务列表
            
        Returns:
            更新后的文档结构
        """
        # 深度复制以避免修改原始数据
        import copy
        updated_structure = copy.deepcopy(doc_structure)
        
        for task in completed_tasks:
            path = task['path']
            abstract = task['abstract']
            
            # 根据路径更新摘要
            current_dict = updated_structure
            for i, key in enumerate(path):
                if i == len(path) - 1:
                    # 最后一级，更新abstract字段
                    if key in current_dict:
                        current_dict[key]['abstract'] = abstract
                else:
                    # 中间级，继续导航
                    current_dict = current_dict.get(key, {})
        
        return updated_structure
    
    def _print_statistics(self):
        """打印处理统计信息"""
        logger.info("=== 摘要生成统计 ===")
        logger.info(f"文件摘要: {self.stats['files_processed']}")
        logger.info(f"文档块摘要: {self.stats['chunks_processed']}")
        logger.info(f"代码块摘要: {self.stats['codes_processed']}")
        logger.info(f"表格摘要: {self.stats['tables_processed']}")
        logger.info(f"总LLM调用: {self.stats['total_llm_calls']}")
        logger.info(f"失败调用: {self.stats['failed_calls']}")
        logger.info("================")
    
    def save_updated_structure(self, doc_structure: Dict[str, Any], output_path: str):
        """
        保存更新后的文档结构到JSON文件
        
        Args:
            doc_structure: 文档结构字典
            output_path: 输出文件路径
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(doc_structure, f, ensure_ascii=False, indent=2)
                
            logger.info(f"更新后的文档结构已保存到: {output_path}")
            
        except Exception as e:
            logger.error(f"保存文档结构失败: {str(e)}")
            raise


def main():
    """
    主函数示例
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="生成文档结构摘要")
    parser.add_argument("input_file", help="输入的文档结构JSON文件路径")
    parser.add_argument("output_file", help="输出的文档结构JSON文件路径")
    parser.add_argument("--max-workers", type=int, default=3, help="最大并发工作线程数")
    
    args = parser.parse_args()
    
    # 加载文档结构
    with open(args.input_file, 'r', encoding='utf-8') as f:
        doc_structure = json.load(f)
    
    # 创建摘要生成器
    generator = AbstractGenerator()
    
    # 处理文档结构
    updated_structure = generator.process_document_structure(
        doc_structure, 
        max_workers=args.max_workers
    )
    
    # 保存结果
    generator.save_updated_structure(updated_structure, args.output_file)


if __name__ == "__main__":
    main() 