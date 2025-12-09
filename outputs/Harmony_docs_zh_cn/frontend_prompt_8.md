# 用户问题
What is OpenHarmony's development language?

# 核心回答
OpenHarmony的默认开发语言是ArkTS，这是一种基于TypeScript生态扩展的官方高级语言。ArkTS在保持TypeScript基本风格的同时，通过强化静态类型检查、禁止运行时对象布局改变、限制运算符语义等规范，提升了代码的健壮性和执行性能。自API version 10起，ArkTS进一步增强了静态检查和分析能力。除了ArkTS，OpenHarmony还支持TypeScript和JavaScript开发，特别是通过类Web开发范式。此外，系统还支持C/C++语言的三方组件，可通过Node-API实现与ArkTS的跨语言交互。ArkTS兼容TS/JS生态，开发者可以复用现有代码，同时通过方舟编译器（ArkCompiler）实现高效编译和运行。

## 关键要点
- ArkTS是OpenHarmony应用的默认开发语言
- ArkTS基于TypeScript生态扩展
- 支持TypeScript和JavaScript开发
- 支持C/C++语言的三方组件
- 通过Node-API实现跨语言交互
- 方舟编译器支持多语言编译运行

# 详细内容

## 1. ArkTS语言概述

ArkTS是OpenHarmony应用的默认开发语言，也是官方推荐的高级语言。它在TypeScript生态基础上做了进一步扩展，保持了TypeScript的基本风格，同时通过规范定义强化了开发期的静态检查和分析。这种设计不仅提升了代码的健壮性，还实现了更好的程序执行稳定性和性能表现。ArkTS的主要特性包括强制使用静态类型、禁止在运行时改变对象布局、限制运算符语义等。从API version 10开始，ArkTS进一步通过规范强化了静态检查和分析能力。ArkTS兼容TS/JS生态，开发者可以使用TS/JS进行开发或复用已有代码。ArkTS运行时是OpenHarmony上应用的默认语言运行时，支持ArkTS、TS和JS语言的字节码及相关标准库。它提供解释器、AOT和JIT高效执行方式，并通过Node-API实现完善的跨语言调用接口，支持多语言混合开发。未来，ArkTS会结合应用开发/运行的需求持续演进，逐步增强并行和并发能力、扩展系统类型，以及引入分布式开发范式等更多特性。

**本节要点：**
- OpenHarmony官方推荐的高级语言
- 基于TypeScript生态扩展
- 强化静态检查和分析
- 兼容TS/JS生态

**来源：**
- 文件: `https://gitee.com/openharmony/docs/raw/master/zh-cn/application-dev/quick-start/arkts-get-started.md`
  定位: 1 初识ArkTS语言
  相关性: 明确说明ArkTS是OpenHarmony应用的默认开发语言
- 文件: `https://gitee.com/openharmony/docs/raw/master/zh-cn/application-dev/arkts-utils/arkts-overview.md`
  定位: 1 ArkTS简介
  相关性: 详细描述ArkTS作为官方高级语言的特性和优势

## 2. ArkTS与TypeScript/JavaScript的关系

ArkTS在TypeScript生态基础上做了扩展，保持TS的基本风格，但通过规范定义强化了开发期的静态检查和分析。与标准TypeScript相比，ArkTS的主要差异包括：强制使用静态类型，确保程序中变量的类型在运行前就是确定的，这有助于编译器验证代码正确性，减少运行时类型检查，从而提升性能。禁止在运行时改变对象布局，为实现最优性能，ArkTS禁止在程序执行期间更改对象布局。限制运算符语义，为获得更好的性能并鼓励编写清晰的代码，ArkTS限制了部分运算符的语义，例如一元加法运算符仅能作用于数字。不支持Structural typing，当前ArkTS不支持该特性，但根据实际场景需求和反馈，后续会重新考虑是否支持。在API version 11上，OpenHarmony SDK中的TypeScript版本为4.9.5，target字段为es2017。应用中支持使用ECMA2017及更高版本的语法进行TS/JS开发，但有一些应用环境限制，包括强制使用严格模式、禁止使用eval()、禁止使用with(){}、禁止以字符串为代码创建函数、禁止循环依赖。方舟运行时兼容TS/JS，支持JSON数字格式中的科学计数法，而标准TS/JS中这类语法会导致SyntaxError。

**本节要点：**
- 基于TypeScript生态扩展
- 强制使用静态类型
- 禁止运行时对象布局改变
- 限制运算符语义

**来源：**
- 文件: `https://gitee.com/openharmony/docs/raw/master/zh-cn/application-dev/quick-start/arkts-get-started.md`
  定位: 1 初识ArkTS语言
  相关性: 详细说明ArkTS与标准TypeScript的差异
- 文件: `https://gitee.com/openharmony/docs/raw/master/zh-cn/application-dev/quick-start/arkts-migration-background.md`
  定位: 1 ArkTS语法适配背景 > 1.5 方舟运行时兼容TS/JS
  相关性: 说明方舟运行时对TS/JS的兼容性和限制

## 3. 其他支持的语言

除了ArkTS作为主要开发语言外，OpenHarmony还支持其他编程语言。TypeScript和JavaScript是重要的支持语言，特别是在类Web开发范式中使用。根据UI框架的对比，类Web开发范式使用JS语言，适用于界面较为简单的程序应用和卡片开发，主要面向Web前端开发人员。而声明式开发范式使用ArkTS语言，适用于复杂度较大、团队合作度较高的程序，主要面向移动系统应用开发人员和系统应用开发人员。OpenHarmony三方组件根据开发语言分为两种：一种是使用JavaScript和TypeScript语言的三方组件，通常以源码或OpenHarmony HAR的方式引入，在应用开发中使用。另一种是C和C++语言的三方组件，通常以源码或OpenHarmony hpm包的方式引入，在应用开发中以NAPI的方式使用，或直接编译在OpenHarmony操作系统镜像中。当前OpenHarmony提供了UI、动画、图片、多媒体、文件数据、网络、安全、工具等多种类型的三方组件。开发者还可以通过Node-API实现ArkTS与C/C++(Native)的跨语言交互能力。OpenHarmony的Node-API是基于Node.js社区版本的扩展实现，支持更灵活的ArkTS交互和自定义对象创建。

**本节要点：**
- 支持TypeScript和JavaScript
- 支持C/C++语言的三方组件
- 通过Node-API实现跨语言交互

**来源：**
- 文件: `https://gitee.com/openharmony/docs/raw/master/zh-cn/application-dev/quick-start/start-overview.md`
  定位: 1 开发准备 > 1.1 基本概念 > 1.1.1 UI框架
  相关性: 对比声明式开发范式和类Web开发范式使用的语言
- 文件: `https://gitee.com/openharmony/docs/raw/master/zh-cn/third-party-components/third-party-components-introduction.md`
  定位: 1 OpenHarmony三方组件
  相关性: 说明OpenHarmony支持的其他开发语言

## 4. 开发范式与语言选择

OpenHarmony提供了两种主要的UI开发范式，分别对应不同的语言选择。声明式开发范式使用ArkTS语言，采用数据驱动更新的方式，适用于复杂度较大、团队合作度较高的程序，主要面向移动系统应用开发人员和系统应用开发人员。这种范式直接与UI后端引擎交互，省略了JS框架，使得渲染更新链路更加精简，占用内存更少。类Web开发范式使用JS语言，同样采用数据驱动更新的方式，但适用于界面较为简单的程序应用和卡片开发，主要面向Web前端开发人员。这种范式通过JS框架进行页面DOM管理。两种范式都基于方舟开发框架（ArkUI框架），为开发者提供应用UI开发所必需的能力，包括多种组件、布局计算、动画能力、UI交互、绘制等功能。在实际开发中，ArkTS对并发编程API和能力进行了增强，提供了TaskPool和Worker两种并发API供开发者选择。针对TS/JS并发能力支持有限的问题，ArkTS进一步提出了Sendable的概念来支持对象在并发实例间的引用传递，提升ArkTS对象在并发实例间的通信性能。方舟编译运行时（ArkCompiler）支持ArkTS、TS和JS的编译运行，目前主要分为ArkTS编译工具链和ArkTS运行时两部分。ArkTS编译工具链负责将高级语言编译为方舟字节码文件（*.abc），ArkTS运行时则负责在设备侧运行字节码文件，执行程序逻辑。

**本节要点：**
- 声明式开发范式使用ArkTS
- 类Web开发范式使用JS
- 两种范式都基于ArkUI框架
- ArkTS增强并发编程能力

**来源：**
- 文件: `https://gitee.com/openharmony/docs/raw/master/zh-cn/application-dev/quick-start/start-overview.md`
  定位: 1 开发准备 > 1.1 基本概念 > 1.1.1 UI框架
  相关性: 详细说明两种开发范式及其对应的语言选择
- 文件: `https://gitee.com/openharmony/docs/raw/master/zh-cn/application-dev/arkts-utils/arkts-overview.md`
  定位: 1 ArkTS简介
  相关性: 说明ArkTS在并发编程方面的增强

## 5. 实际应用与示例

在实际的OpenHarmony应用开发中，ArkTS被广泛应用于各种场景。从OpenHarmony 3.1 Beta到4.0 Release的各个版本中，新增的Samples和Codelabs大多使用ArkTS语言开发。例如在OpenHarmony 3.2 Release中，新增的无障碍功能示例、企业设备管理示例和任务延时调度示例都使用ArkTS语言。在OpenHarmony 4.0 Release中，文件管理、多端部署、卡片和安全控件等功能的示例也都使用ArkTS语言。这些示例展示了ArkTS在不同领域的应用能力，包括分布式数据管理、UI组件开发、动画效果实现等。在开发过程中，ArkTS支持模块化开发，开发者可以通过export/import语法在不同模块间共享ArkTS页面、ts/js方法和静态资源。例如，可以在OpenHarmony ohpm模块中通过export导出ArkTS页面结构体或ts/js方法，然后在其他模块中通过import引入使用。这种模块化机制支持组件化和模块化开发，提高了代码的复用性和开发效率。ArkTS还支持与C/C++的跨语言交互，通过Node-API实现Native能力的开发和封装，为复杂应用开发提供了更多可能性。

**本节要点：**
- 广泛应用于各种Samples和Codelabs
- 支持模块化开发
- 支持跨语言交互

**来源：**
- 文件: `https://gitee.com/openharmony/docs/raw/master/zh-cn/release-notes/OpenHarmony-v3.2-release.md`
  定位: 1 OpenHarmony 3.2 Release > 1.5 更新说明 > 1.5.3 Samples
  相关性: 展示ArkTS在实际示例中的应用
- 文件: `https://gitee.com/openharmony/docs/raw/master/zh-cn/release-notes/OpenHarmony-v4.0-release.md`
  定位: 1 OpenHarmony 4.0 Release > 1.5 更新说明 > 1.5.3 Samples
  相关性: 说明ArkTS在最新版本中的应用情况

# 相关图片

## 图片 1
![ArkTS语言特性演进图，展示了从API9到API10再到演进阶段的技术发展路径](https://gitee.com/openharmony/docs/raw/master/zh-cn/application-dev/quick-start/figures/arkts.png)

**说明：** ArkTS语言特性演进图，展示了从API9到API10再到演进阶段的技术发展路径

**上下文：** 该图片出现在ArkTS语言介绍章节，用于直观展示ArkTS在不同API版本中的特性和优势

**与问题的相关性：** 直接展示了OpenHarmony主要开发语言ArkTS的技术演进

**尺寸：** 2560x1236像素
**建议显示方式：** full-width

**来源文件：** `/root/code/docs/zh-cn/application-dev/quick-start/arkts-get-started.md`
**定位：** 1 初识ArkTS语言

# 相关表格

## 表格 1: UI开发范式对比

**描述：** 对比OpenHarmony两种UI开发范式在语言生态、UI更新方式、适用场景和适用人群方面的差异

| **开发范式名称** | **语言生态** | **UI更新方式** | **适用场景** | **适用人群** |
| ---------------- | ------------ | -------------- | -------------------------------- | -------------------------------------- |
| 声明式开发范式 | ArkTS语言 | 数据驱动更新 | 复杂度较大、团队合作度较高的程序 | 移动系统应用开发人员、系统应用开发人员 |
| 类Web开发范式 | JS语言 | 数据驱动更新 | 界面较为简单的程序应用和卡片 | Web前端开发人员 |

**关键数据：**
- 声明式开发范式使用ArkTS语言
- 类Web开发范式使用JS语言
- 两种范式都采用数据驱动更新方式

**来源文件：** `https://gitee.com/openharmony/docs/raw/master/zh-cn/application-dev/quick-start/start-overview.md`
**定位：** 1 开发准备 > 1.1 基本概念 > 1.1.1 UI框架

# 相关代码

## 代码 1: ArkTS泛型函数示例

**功能说明：** 这段代码展示了ArkTS中泛型函数的正确写法。与TypeScript不同，ArkTS不支持泛型箭头函数，必须使用function关键字定义泛型函数。示例中定义了一个泛型函数generic_func，它接受一个类型参数T，该参数必须继承自String类型。函数返回类型也是T，确保类型安全。

**使用注意：** ArkTS要求使用function关键字定义泛型函数，不支持泛型箭头函数语法

**涉及API：** 泛型类型约束

```typescript
function generic_func<T extends String>(x: T): T {
  return x;
}

generic_func<String>('string');
```

**来源文件：** `https://gitee.com/openharmony/docs/raw/master/zh-cn/release-notes/changelogs/OpenHarmony_5.0.0.25/changelogs-arkcompiler.md`
**定位：** 1 ArkCompiler子系统变更说明 > 1.1 cl.ArkCompiler.1 ArkTS Linter规则变更

## 代码 2: ArkTS模块导入示例

**功能说明：** 这段代码展示了ArkTS中模块导入的正确语法。ArkTS不支持TypeScript中的`import default as ...`语法，必须使用显式的`import ... from ...`语法。

**使用注意：** ArkTS要求使用标准的import语法，不支持某些TypeScript特有的导入方式

**涉及API：** 模块导入语法

```typescript
import d from 'mod'
```

**来源文件：** `https://gitee.com/openharmony/docs/raw/master/zh-cn/release-notes/changelogs/OpenHarmony_5.0.0.25/changelogs-arkcompiler.md`
**定位：** 1 ArkCompiler子系统变更说明 > 1.1 cl.ArkCompiler.1 ArkTS Linter规则变更

# 相关概念

- TypeScript
- JavaScript
- ArkCompiler
- Node-API
- 声明式UI
- 类Web开发范式
- 方舟字节码

# 推荐阅读

- **ArkTS学习路线**: 深入了解ArkTS语言的学习路径和进阶知识
- **从TypeScript到ArkTS的适配规则**: 帮助TypeScript开发者快速迁移到ArkTS开发
- **ArkTS跨语言交互指南**: 学习如何在ArkTS中与C/C++等语言进行交互
