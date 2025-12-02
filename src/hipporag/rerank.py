import json
import difflib
from pydantic import BaseModel, Field, TypeAdapter
from openai import OpenAI
from copy import deepcopy
from typing import Union, Optional, List, Dict, Any, Tuple, Literal
import re
import ast
from .prompts.filter_default_prompt import best_dspy_prompt
from .utils.llm_utils import fix_broken_generated_json

class Fact(BaseModel):
    fact: list[list[str]] = Field(description="A list of facts, each fact is a list of 3 strings: [subject, predicate, object]")


class DSPyFilter:
    def __init__(self, hipporag):
        """
        Initializes the object with the necessary configurations and templates for processing input and output messages.

        Parameters:
        hipporag : An object that provides the global configuration and the LLM model required for inference.

        Attributes:
        dspy_file_path : The file path for reranking as specified in the global configuration.
        one_input_template : A string template for formatting the input message with placeholders for specific fields.
        one_output_template : A string template for formatting the output message with specific fields.
        message_template : A template generated using the specified dspy file path.
        llm_infer_fn : A function reference for making inferences using the provided LLM model.
        model_name : The name of the language model as specified in the global configuration.
        default_gen_kwargs : A dictionary for storing the default generation keyword arguments.
        """
        dspy_file_path = hipporag.global_config.rerank_dspy_file_path
        self.one_input_template = """[[ ## question ## ]]\n{question}\n\n[[ ## fact_before_filter ## ]]\n{fact_before_filter}\n\nRespond with the corresponding output fields, starting with the field `[[ ## fact_after_filter ## ]]` (must be formatted as a valid Python Fact), and then ending with the marker for `[[ ## completed ## ]]`."""
        self.one_output_template = """[[ ## fact_after_filter ## ]]\n{fact_after_filter}\n\n[[ ## completed ## ]]"""
        self.message_template = self.make_template(dspy_file_path)
        self.llm_infer_fn = hipporag.llm_model.infer
        self.model_name = hipporag.global_config.llm_name
        self.default_gen_kwargs = {}

    def make_template(self, dspy_file_path):
        if dspy_file_path is not None:
            dspy_saved = json.load(open(dspy_file_path, 'r'))
        else:
            dspy_saved = best_dspy_prompt

        system_prompt = dspy_saved['prog']['system']
        message_template = [
            {"role": "system", "content": system_prompt},
        ]
        demos = dspy_saved["prog"]["demos"]
        for demo in demos:
            message_template.append({"role": "user", "content": self.one_input_template.format(question=demo["question"], fact_before_filter=demo["fact_before_filter"])})
            message_template.append({"role": "assistant", "content": self.one_output_template.format(fact_after_filter=demo["fact_after_filter"])})
        return message_template

    def parse_filter(self, response):
        sections = [(None, [])]
        field_header_pattern = re.compile('\\[\\[ ## (\\w+) ## \\]\\]')
        for line in response.splitlines():
            match = field_header_pattern.match(line.strip())
            if match:
                sections.append((match.group(1), []))
            else:
                sections[-1][1].append(line)

        sections = [(k, "\n".join(v).strip()) for k, v in sections]
        parsed = []
        for k, value in sections:
            if k == "fact_after_filter":
                try:
                    # fields[k] = parse_value(v, signature.output_fields[k].annotation) if _parse_values else v
                    try:
                        parsed_value = json.loads(value)
                    except json.JSONDecodeError:
                        # 尝试修复被截断的JSON
                        try:
                            fixed_value = fix_broken_generated_json(value)
                            parsed_value = json.loads(fixed_value)
                        except json.JSONDecodeError:
                            try:
                                parsed_value = ast.literal_eval(value)
                            except (ValueError, SyntaxError):
                                # 最后尝试修复后用ast解析
                                try:
                                    fixed_value = fix_broken_generated_json(value)
                                    parsed_value = ast.literal_eval(fixed_value)
                                except (ValueError, SyntaxError):
                                    parsed_value = value
                    parsed = TypeAdapter(Fact).validate_python(parsed_value).fact
                except Exception as e:
                    print(
                        f"Error parsing field {k}: {e}.\n\n\t\tOn attempting to parse the value\n```\n{value}\n```"
                    )

        return parsed

    def llm_call(self, question, fact_before_filter):
        # make prompt
        messages = deepcopy(self.message_template)
        messages.append({"role": "user", "content": self.one_input_template.format(question=question, fact_before_filter=fact_before_filter)})
        # call openai

        # 增加token限制，避免返回内容被截断导致JSON解析失败
        self.default_gen_kwargs['max_completion_tokens'] = 2048

        response = self.llm_infer_fn(
            messages=messages,
            model=self.model_name,
            **self.default_gen_kwargs
        )

        if len(response) > 1:
            return response[0]
        return response

    def __call__(self, *args, **kwargs):
        return self.rerank(*args, **kwargs)

    def rerank(self,
               query: str,
               candidate_items: List[Tuple],
               candidate_indices: List[int],
               len_after_rerank: int =None) -> Tuple[List[int], List[Tuple], dict]:
        fact_before_filter = {"fact": [list(candidate_item) for candidate_item in candidate_items]}
        try:
            # prediction = self.program(question=query, fact_before_filter=json.dumps(fact_before_filter))
            response = self.llm_call(query, json.dumps(fact_before_filter))
            generated_facts = self.parse_filter(response)
        except Exception as e:
            print('exception', e)
            generated_facts = []
        result_indices = []
        for generated_fact in generated_facts:
            closest_matched_fact = difflib.get_close_matches(str(generated_fact), [str(i) for i in candidate_items], n=1, cutoff=0.0)[0]
            try:
                result_indices.append(candidate_items.index(eval(closest_matched_fact)))
            except Exception as e:
                print('result_indices exception', e)

        sorted_candidate_indices = [candidate_indices[i] for i in result_indices]
        sorted_candidate_items = [candidate_items[i] for i in result_indices]
        return sorted_candidate_indices[:len_after_rerank], sorted_candidate_items[:len_after_rerank], {'confidence': None}

    def rerank_contents(self,
                        query: str,
                        contents: List[str],
                        content_type: str,
                        len_after_rerank: int = 5) -> Tuple[List[int], List[str], dict]:
        """
        对内容列表进行LLM重排序
        
        Args:
            query: 查询字符串
            contents: 内容列表
            content_type: 内容类型 ('chunk', 'table', 'code')
            len_after_rerank: 重排序后保留的数量
            
        Returns:
            (top_indices, top_contents, rerank_log)
        """
        if len(contents) == 0:
            return [], [], {'error': 'empty contents'}
        
        # 构建带编号的内容列表
        numbered_contents = []
        for i, content in enumerate(contents):
            # 截断过长的内容
            truncated = content[:1500] + "..." if len(content) > 1500 else content
            numbered_contents.append(f"[{i}] {truncated}")
        
        content_list_str = "\n\n".join(numbered_contents)
        
        # 构建提示
        type_desc = {
            'chunk': '文档段落',
            'table': '表格',
            'code': '代码块'
        }.get(content_type, '内容')
        
        system_prompt = f"""你是一个专业的信息检索助手。你的任务是从候选{type_desc}列表中选择与用户问题最相关的{len_after_rerank}个{type_desc}。

请按相关性从高到低的顺序返回选中{type_desc}的编号列表（仅返回编号，用逗号分隔）。

规则：
1. 只返回编号，不要返回其他内容
2. 编号用逗号分隔，如: 3,7,1,5,2
3. 最多返回{len_after_rerank}个编号
4. 按相关性从高到低排序"""

        user_prompt = f"""用户问题: {query}

候选{type_desc}列表:
{content_list_str}

请返回最相关的{len_after_rerank}个{type_desc}的编号（按相关性排序，逗号分隔）:"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self.llm_infer_fn(
                messages=messages,
                model=self.model_name,
                max_completion_tokens=100
            )
            
            # 处理各种返回格式
            if isinstance(response, tuple):
                response = response[0] if len(response) > 0 else ""
            if isinstance(response, list) and len(response) > 0:
                response = response[0]
            if not isinstance(response, str):
                response = str(response)
            
            # 解析返回的编号
            response = response.strip()
            # 提取数字
            numbers = re.findall(r'\d+', response)
            indices = []
            for num in numbers:
                idx = int(num)
                if 0 <= idx < len(contents) and idx not in indices:
                    indices.append(idx)
                    if len(indices) >= len_after_rerank:
                        break
            
            if len(indices) == 0:
                # 降级：返回前几个
                indices = list(range(min(len_after_rerank, len(contents))))
            
            top_contents = [contents[i] for i in indices]
            return indices, top_contents, {'response': response}
            
        except Exception as e:
            print(f"Error in rerank_contents: {e}")
            # 降级：返回前几个
            indices = list(range(min(len_after_rerank, len(contents))))
            return indices, [contents[i] for i in indices], {'error': str(e)}