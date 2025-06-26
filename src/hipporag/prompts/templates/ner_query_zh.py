ner_system = """您是一个非常有效的实体提取系统。
"""

query_prompt_one_shot_input = """请提取所有对解决下面问题重要的命名实体。
请将命名实体以json格式放置。

问题：Arthur's Magazine 和 First for Women 哪个杂志先创刊？

"""
query_prompt_one_shot_output = """
{"named_entities": ["First for Women", "Arthur's Magazine"]}
"""
# query_prompt_template = """
# 问题：{}

# """
prompt_template = [
    {"role": "system", "content": ner_system},
    {"role": "user", "content": query_prompt_one_shot_input},
    {"role": "assistant", "content": query_prompt_one_shot_output},
    {"role": "user", "content": "问题：${query}"}
] 