from .ner_zh import one_shot_ner_paragraph, one_shot_ner_output
from ...utils.llm_utils import convert_format_to_template

ner_conditioned_re_system = """您的任务是从给定的段落和命名实体列表构建开源鸿蒙(OpenHarmony)知识图谱的RDF（资源描述框架）三元组。
这些三元组将用于构建开源鸿蒙知识图谱，助力开发者掌握鸿蒙技术栈、高效开发，参与开源社区共建。

请用JSON三元组列表回复，每个三元组代表RDF图中的一个关系。

请注意以下要求：
- 每个三元组应该包含至少一个，最好是两个，来自命名实体列表中的开源鸿蒙相关实体
- 重点构建技术组件间的关系、架构层次关系、支持关系、实现关系等
- 清楚地将代词解析为其具体名称以保持清晰度
- 关系描述应准确反映开源鸿蒙技术栈中的实际关系

"""


ner_conditioned_re_frame = """将段落转换为JSON字典，包含命名实体列表和三元组列表。
段落：
```
{passage}
```

{named_entity_json}
"""


ner_conditioned_re_input = ner_conditioned_re_frame.format(passage=one_shot_ner_paragraph, named_entity_json=one_shot_ner_output)


ner_conditioned_re_output = """{"triples": [
            ["OpenHarmony", "包含", "分布式软总线"],
            ["分布式软总线", "是", "统一基座"],
            ["分布式软总线", "提供", "分布式通信能力"],
            ["分布式软总线", "支持", "Arm Cortex-A"],
            ["OpenHarmony", "支持", "轻量系统"],
            ["OpenHarmony", "支持", "标准系统"],
            ["分布式软总线", "协同", "分布式数据管理"],
            ["分布式软总线", "协同", "分布式任务调度"],
            ["分布式数据管理", "属于", "分布式能力"],
            ["分布式任务调度", "属于", "分布式能力"],
            ["Ability框架", "实现", "一次开发多端部署"],
            ["UI框架", "实现", "一次开发多端部署"],
            ["Ability框架", "属于", "OpenHarmony"],
            ["UI框架", "属于", "OpenHarmony"]
    ]
}
"""


prompt_template = [
    {"role": "system", "content": ner_conditioned_re_system},
    {"role": "user", "content": ner_conditioned_re_input},
    {"role": "assistant", "content": ner_conditioned_re_output},
    {"role": "user", "content": convert_format_to_template(original_string=ner_conditioned_re_frame, placeholder_mapping=None, static_values=None)}
] 