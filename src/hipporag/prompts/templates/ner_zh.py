ner_system = """您的任务是从开源鸿蒙(OpenHarmony)技术文档中精确提取专有名词作为命名实体，用于构建技术知识图谱。

**提取原则：**
- 只提取具体的技术术语，忽略通用词汇和描述性语言
- 优先提取API相关的核心概念
- 保持实体名称的原始形式，不要修改或翻译
- 每个实体应该是独立的、有意义的技术概念

**重点提取的实体类型：**

**1. API核心实体（最高优先级）**
- 模块/命名空间：API模块标识符
- 类型定义：类名、接口名、枚举
- 方法/函数：具体的方法名、构造函数
- 属性/字段：对象属性、常量、配置项
- 参数/返回值：参数名称、返回类型

**2. 系统与框架实体**
- 系统组件：核心服务、框架组件
- 开发工具：编译器、开发环境
- 架构概念：应用模型、设计模式

**3. 技术规范实体**
- 权限标识：具体的权限名称
- 错误码：错误类型、状态码
- 版本标识：API版本号、兼容性标记

**输出要求：**
- 格式：{"named_entities": [("实体名", "简短类型说明"), ...]}
- 每个实体的类型说明控制在3-8个汉字
- 按重要性排序，API相关实体优先
- 避免重复和过于泛化的概念

**不要提取：**
- 通用形容词（如"强大"、"高效"）
- 业务描述词（如"用户体验"、"开发效率"）
- 过于抽象的概念（如"生态系统"、"技术栈"）
"""

# - 只需提取关键实体，不要超过30个
one_shot_ner_paragraph = """Context类提供startAbility方法启动应用，需要ohos.permission.START_ABILITIES权限。
方法接受Want参数对象，返回Promise<void>类型。
失败时抛出BusinessError异常，错误码为16000001表示无效参数。
UIAbilityContext继承自Context，提供terminateSelf方法结束当前Ability。"""

one_shot_ner_output = """{"named_entities": [
    ("Context", "基础类"),
    ("startAbility", "启动方法"),
    ("ohos.permission.START_ABILITIES", "权限标识"),
    ("Want", "参数对象"),
    ("Promise<void>", "返回类型"),
    ("BusinessError", "异常类型"),
    ("16000001", "错误码"),
    ("UIAbilityContext", "上下文类"),
    ("terminateSelf", "生命周期方法"),
    ("Ability", "组件类型")
]}"""

prompt_template = [
    {"role": "system", "content": ner_system},
    {"role": "user", "content": one_shot_ner_paragraph},
    {"role": "assistant", "content": one_shot_ner_output},
    {"role": "user", "content": "${passage}"}
] 