from .ner_zh import one_shot_ner_paragraph, one_shot_ner_output
from ...utils.llm_utils import convert_format_to_template

ner_conditioned_re_system = """您的任务是从给定的段落和命名实体列表构建开源鸿蒙(OpenHarmony)知识图谱的RDF（资源描述框架）三元组。
这些三元组将用于构建开源鸿蒙知识图谱，助力开发者掌握鸿蒙技术栈、高效开发，参与开源社区共建。

请用JSON三元组列表回复，每个三元组代表RDF图中的一个关系，非专业术语尽量用中文回答。

请注意以下要求：
- 每个三元组应该包含至少一个，最好是两个，来自命名实体列表中的开源鸿蒙相关实体
- 清楚地将代词解析为其具体名称以保持清晰度
- 关系描述应准确反映开源鸿蒙技术栈中的实际关系，涉及到的实体应该是准确具体的技术实体

"""

# - 提取的三元组数不应该太多，重点提取关键事实即可，不应超过50个
ner_conditioned_re_frame = """将段落转换为JSON字典，包含命名实体列表和三元组列表。
段落：
```
{passage}
```

{named_entity_json}
"""


ner_conditioned_re_input = ner_conditioned_re_frame.format(passage=one_shot_ner_paragraph, named_entity_json=one_shot_ner_output)


ner_conditioned_re_output = """{"triples": [
            ["Context", "提供", "startAbility"],
            ["startAbility", "用于", "启动应用"],
            ["startAbility", "需要", "ohos.permission.START_ABILITIES"],
            ["startAbility", "接受", "Want"],
            ["startAbility", "返回", "Promise<void>"],
            ["startAbility", "失败时抛出", "BusinessError"],
            ["BusinessError", "包含", "16000001"],
            ["16000001", "表示", "无效参数"],
            ["UIAbilityContext", "继承自", "Context"],
            ["UIAbilityContext", "提供", "terminateSelf"],
            ["terminateSelf", "用于", "结束当前Ability"],
            ["Want", "是", "参数对象"],
            ["Promise<void>", "是", "返回类型"],
            ["BusinessError", "是", "异常类型"]
    ]
}
"""


prompt_template = [
    {"role": "system", "content": ner_conditioned_re_system},
    {"role": "user", "content": ner_conditioned_re_input},
    {"role": "assistant", "content": ner_conditioned_re_output},
    {"role": "user", "content": convert_format_to_template(original_string=ner_conditioned_re_frame, placeholder_mapping=None, static_values=None)}
] 