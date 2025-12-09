# 用户问题
What is OpenHarmony's development process?

# 核心回答
OpenHarmony的开发流程是一个完整的端到端过程，涵盖从代码获取到最终部署的多个阶段。首先，开发者需要获取源码，可以通过四种方式：从gitcode或github仓库获取最新代码、通过DevEco Marketplace网站获取发行版、从镜像站点下载归档版本。然后进入开发准备阶段，包括Fork主干仓、Clone到本地、创建开发分支。开发过程支持两种操作方式：基于IDE的图形化开发和基于命令行的传统开发。编译阶段使用build.sh脚本或hb工具进行，支持Debug版本和模块级编译。最后是烧录和运行阶段，将编译结果部署到目标设备上。整个流程支持三种基础系统类型：轻量系统（MCU处理器，128KiB内存）、小型系统（应用处理器，1MiB内存）和标准系统（应用处理器，128MiB内存），开发者可根据目标硬件能力选择相应的系统组件进行集成开发。

## 关键要点
- 支持四种源码获取方式：gitcode、github、DevEco Marketplace、镜像站点
- 开发准备包括Fork、Clone、创建分支三个步骤
- 提供IDE和命令行两种开发方式
- 使用build.sh或hb工具进行编译构建
- 支持三种基础系统类型：轻量系统、小型系统、标准系统
- 完整的开发流程：环境搭建→获取源码→开发→编译→烧录→运行

# 详细内容

## 1. 源码获取方式

OpenHarmony的源码获取提供了四种灵活的方式，满足不同开发者的需求。第一种方式是从gitcode代码仓库获取，通过repo或git工具直接下载，这种方式可以获得最新的代码更新，适合需要紧跟社区发展的开发者。第二种方式是通过DevEco Marketplace网站获取，开发者可以访问该网站查找满足需求的开源发行版，直接下载或定制后下载，再通过hpm-cli命令工具将所需的组件及工具链下载安装到本地，这种方式更加便捷，适合快速开始项目开发。第三种方式是从镜像站点下载归档后的发行版压缩文件，这种方式下载速度较快，特别适合网络环境不佳的情况，同时也可以获取旧版本的源码。第四种方式是从github代码仓库获取，同样通过repo或git工具下载，为习惯使用github的开发者提供了便利。对于参与孵化仓开发的场景，需要使用特定的repo init命令来初始化和下载代码，其他下载步骤与主线相同。这种多元化的源码获取方式确保了开发者可以根据自己的实际情况选择最合适的方法开始OpenHarmony的开发工作。

**本节要点：**
- 四种源码获取方式：gitcode、DevEco Marketplace、镜像站点、github
- 不同方式适用于不同场景和需求
- 孵化仓开发使用特定的初始化命令

**来源：**
- 文件: `https://gitee.com/openharmony/docs/raw/master/zh-cn/device-dev/get-code/sourcecode-acquire.md`
  定位: 1 获取源码 > 1.2 获取源码概述
  相关性: 详细说明了OpenHarmony源码的四种获取方式
- 文件: `https://gitee.com/openharmony/docs/raw/master/zh-cn/device-dev/porting/Readme-CN.md`
  定位: 2 代码准备
  相关性: 说明了孵化仓开发的代码获取方式

## 2. 开发准备与分支管理

在开始OpenHarmony开发之前，需要进行充分的开发准备工作。首先是Fork主干仓，开发者在OpenHarmony gitcode组织下找到感兴趣的仓库，点击右上角的'Fork'按钮，创建一份个人名下的代码副本（个人仓）。这一步是开源项目贡献的标准流程，确保主仓库的稳定性。接下来是将个人仓Clone到本地，使用git clone命令将代码克隆到本地计算机，作为本地工作空间。这个步骤需要替换命令中的{your_gitcode_id}和{repository_name}为实际的用户名和仓库名。最后是创建开发分支，基于最新的主干代码创建一个新的本地分支，专门用于本次贡献的开发工作。这种分支管理策略有助于保持代码的整洁性和可维护性，同时便于后续的代码审查和合并操作。整个开发准备过程体现了开源协作的最佳实践，确保每个开发者都能在独立的环境中进行开发，同时又能与社区保持同步。

**本节要点：**
- Fork主干仓创建个人副本
- Clone到本地建立工作空间
- 基于主干创建开发分支

**来源：**
- 文件: `https://gitee.com/openharmony/docs/raw/master/zh-cn/contribute/贡献流程.md`
  定位: 1 OpenHarmony 社区贡献流程 > 1.1 贡献准备 (代码下载) > 1.1.2 获取代码
  相关性: 详细说明了开发准备的三步流程：Fork、Clone、创建分支

## 3. 开发方式与工具选择

OpenHarmony为开发者提供了两种主要的开发方式，以适应不同的开发习惯和需求。第一种是基于IDE的开发方式，使用DevEco Device Tool进行一站式开发。这种方式采用Windows+Ubuntu混合开发环境：在Windows上主要进行代码开发、代码调试、烧录等操作，而在Ubuntu环境中实现源码编译。DevEco Device Tool提供界面化的操作接口，为开发者提供更快捷的开发体验，特别适合不熟悉命令行操作或习惯界面化操作的开发者。第二种是基于命令行的开发方式，通过命令行工具包进行操作。这种方式通过命令行下载安装编译依赖工具，在Linux系统中进行编译时，所有相关操作都通过命令实现；在Windows系统中使用开发板厂商提供的工具进行代码烧录。命令行方式提供了简便统一的工具链安装方式，适合习惯使用命令行操作的开发者。两种方式各有优势，IDE方式更加直观便捷，命令行方式更加灵活高效，开发者可以根据自己的技术背景和项目需求选择最适合的开发方式。

**本节要点：**
- IDE方式：DevEco Device Tool，Windows+Ubuntu混合环境
- 命令行方式：通过命令操作，工具链统一安装
- 两种方式适应不同开发习惯

**来源：**
- 文件: `https://gitee.com/openharmony/docs/raw/master/zh-cn/device-dev/quick-start/quickstart-overview.md`
  定位: 1 快速入门概述 > 1.1 操作方式
  相关性: 详细对比了IDE和命令行两种开发方式的特点和适用人群

## 4. 编译构建流程

OpenHarmony的编译构建过程提供了两种主要的方法：使用build.sh脚本和hb工具。对于Debug版本的编译，可以使用'./build.sh --product-name {product_name} --gn-args is_debug=true'命令。需要注意的是，Debug全版本因镜像大小限制，全量编译可能无法烧录，因此建议进行单模块编译Debug二进制。单模块编译使用'./build.sh --product-name {product_name} --gn-args is_debug=true --build-target {target_name}'命令，其中{product_name}为当前版本支持的平台，如hispark_taurus_standard等。编译完成后，结果镜像保存在out/{device_name}/packages/phone/images/目录下。build.sh脚本支持丰富的参数选项，包括-h/--help显示帮助信息、--product-name指定产品名、--device-name指定装置名称、--target-cpu指定CPU、--target-os指定操作系统、-T/--build-target指定编译目标、--gn-args指定GN参数、--ninja-args指定Ninja参数等。hb是OpenHarmony的命令行工具，用来执行编译命令，其中'hb set'命令用于设置要编译的产品。编译构建过程支持多种配置选项，包括日志级别设置（debug、info、error）、设备类型指定、设备操作模式设置等，为开发者提供了灵活的编译控制能力。

**本节要点：**
- 两种编译方法：build.sh脚本和hb工具
- Debug版本和模块级编译支持
- 丰富的编译参数选项

**来源：**
- 文件: `https://gitee.com/openharmony/docs/raw/master/zh-cn/device-dev/subsystems/subsys-build-all.md`
  定位: 1 编译构建指导 > 1.3 编译构建使用指导 > 1.3.2 编译命令
  相关性: 详细说明了OpenHarmony的编译构建方法和命令选项

## 5. 系统类型与开发流程

OpenHarmony定义了三种基础系统类型，设备开发者通过选择基础系统类型完成必选组件集配置后，便可实现最小系统的开发。第一种是轻量系统（mini system），使用MCU类处理器（如Arm Cortex-M、RISC-V 32位的设备），最小内存要求为128KiB，提供多种轻量级网络协议、轻量级的图形框架以及丰富的IOT总线读写部件等，可支撑智能家居领域的连接类模组、传感器设备、穿戴类设备等产品。第二种是小型系统（small system），使用应用处理器（如Arm Cortex-A的设备），最小内存要求为1MiB，提供更高的安全能力、标准的图形框架、视频编解码的多媒体能力，可支撑智能家居领域的IP Camera、电子猫眼、路由器以及智慧出行域的行车记录仪等产品。第三种是标准系统（standard system），同样使用应用处理器，最小内存要求为128MiB，提供增强的交互能力、3D GPU以及硬件合成能力、更多控件以及动效更丰富的图形能力、完整的应用框架，可支撑高端冰箱显示屏等产品。整个开发流程包括开发环境搭建、编译、烧录、调测以及运行'Hello World'等步骤，引导开发者快速熟悉OpenHarmony设备开发的基本流程和方法。

**本节要点：**
- 三种基础系统类型：轻量系统、小型系统、标准系统
- 不同系统类型对应不同的处理器和内存要求
- 完整的开发流程：环境搭建→编译→烧录→运行

**来源：**
- 文件: `https://gitee.com/openharmony/docs/raw/master/zh-cn/device-dev/quick-start/quickstart-overview.md`
  定位: 1 快速入门概述
  相关性: 详细说明了OpenHarmony的三种基础系统类型及其开发流程

# 相关图片

## 图片 1
![OpenHarmony系统在伙伴硬件平台上的移植适配流程，展示从移植准备到移植验证的四个主要步骤](https://gitee.com/openharmony/docs/raw/master/zh-cn/device-dev/porting/figures/zh-cn_image_0000001378282213.png)

**说明：** OpenHarmony系统在伙伴硬件平台上的移植适配流程，展示从移植准备到移植验证的四个主要步骤

**上下文：** 该图片出现在移植适配概述部分，用于说明OpenHarmony系统在不同硬件平台上的完整移植流程

**与问题的相关性：** 展示了OpenHarmony开发流程中的移植适配环节，帮助理解系统在不同硬件上的部署过程

**尺寸：** 630x362像素
**建议显示方式：** inline

**来源文件：** `/root/code/docs/zh-cn/device-dev/porting/porting-minichip-overview.md`
**定位：** 1 概述 > 1.2 适配流程

## 图片 2
![OpenHarmony中SysCap（系统能力）在设备开发和应用开发中的应用流程](https://gitee.com/openharmony/docs/raw/master/zh-cn/design/figures/Component-SysCap.png)

**说明：** OpenHarmony中SysCap（系统能力）在设备开发和应用开发中的应用流程

**上下文：** 该图片出现在部件设计和开发指南中，用于说明系统能力从设备到应用的完整流转过程

**与问题的相关性：** 展示了OpenHarmony开发流程中系统能力的定义和应用，体现了组件化设计的核心理念

**尺寸：** 903x410像素
**建议显示方式：** inline

**来源文件：** `/root/code/docs/zh-cn/design/OpenHarmony部件设计和开发指南.md`
**定位：** 1 OpenHarmony部件设计和开发指南 > 1.2 部件设计 > 1.2.3 系统能力（SysCap）

## 图片 3
![OpenHarmony部件化设计的核心理念，展示系统能力如何分解为独立部件](https://gitee.com/openharmony/docs/raw/master/zh-cn/design/figures/Component-Definition.png)

**说明：** OpenHarmony部件化设计的核心理念，展示系统能力如何分解为独立部件

**上下文：** 该图片出现在部件定义部分，用于说明OpenHarmony组件化架构的基本概念

**与问题的相关性：** 展示了OpenHarmony开发流程的架构基础，帮助理解系统组件如何灵活组合适应不同设备

**尺寸：** 760x459像素
**建议显示方式：** inline

**来源文件：** `/root/code/docs/zh-cn/design/OpenHarmony部件设计和开发指南.md`
**定位：** 1 OpenHarmony部件设计和开发指南 > 1.1 基本概念 > 1.1.1 部件定义

# 相关表格

## 表格 1: 基础类型系统简介

**描述：** 介绍OpenHarmony三种基础系统类型的处理器要求、内存要求和能力特点

| 类型 | 处理器 | 最小内存 | 能力 |
| :--------: | :--------: | -------- | -------- |
| 轻量系统（mini system） | MCU类处理器（例如Arm Cortex-M、RISC-V 32位的设备） | 128KiB | 提供多种轻量级网络协议，轻量级的图形框架，以及丰富的IOT总线读写部件等。可支撑的产品如智能家居领域的连接类模组、传感器设备、穿戴类设备等。 |
| 小型系统（small system） | 应用处理器（例如Arm Cortex-A的设备） | 1MiB | 提供更高的安全能力、标准的图形框架、视频编解码的多媒体能力。可支撑的产品如智能家居领域的IP Camera、电子猫眼、路由器以及智慧出行域的行车记录仪等。 |
| 标准系统（standard system） | 应用处理器（例如Arm Cortex-A的设备） | 128MiB | 提供增强的交互能力、3D GPU以及硬件合成能力、更多控件以及动效更丰富的图形能力、完整的应用框架。可支撑的产品如高端的冰箱显示屏等。 |

**关键数据：**
- 轻量系统：MCU处理器，128KiB内存，适用于传感器设备
- 小型系统：应用处理器，1MiB内存，适用于摄像头设备
- 标准系统：应用处理器，128MiB内存，适用于高端显示屏

**来源文件：** `https://gitee.com/openharmony/docs/raw/master/zh-cn/device-dev/quick-start/quickstart-overview.md`
**定位：** 1 快速入门概述

## 表格 2: 入门方式对比

**描述：** 对比OpenHarmony两种开发方式的特点、工具和适用人群

| 方式 | 工具 | 特点 | 适用人群 |
| -------- | -------- | -------- | -------- |
| 基于IDE入门 | IDE（DevEco Device Tool） | 完全采用IDE进行一站式开发，编译依赖工具的安装及编译、烧录、运行都通过IDE进行操作。<br/>DevEco Device Tool采用Windows+Ubuntu混合开发环境：<br/>- 在Windows上主要进行代码开发、代码调试、烧录等操作。<br/>- 在Ubuntu环境实现源码编译。<br/>DevEco Device Tool提供界面化的操作接口，可以为您提供更快捷的开发体验。 | 不熟悉命令行操作的开发者<br/>习惯界面化操作的开发者 |
| 基于命令行入门 | 命令行工具包 | 通过命令行方式下载安装编译依赖工具，在Linux系统中进行编译时，相关操作通过命令实现；在Windows系统中使用开发板厂商提供的工具进行代码烧录。<br/>命令行方式提供了简便统一的工具链安装方式。 | 习惯使用命令行操作的开发者 |

**关键数据：**
- IDE方式：一站式开发，混合环境，界面化操作
- 命令行方式：命令实现，工具链统一安装
- 不同方式适合不同开发习惯的开发者

**来源文件：** `https://gitee.com/openharmony/docs/raw/master/zh-cn/device-dev/quick-start/quickstart-overview.md`
**定位：** 1 快速入门概述 > 1.1 操作方式

## 表格 3: 源码目录结构

**描述：** 介绍OpenHarmony源码的主要目录结构及其功能描述

| 目录名 | 描述 |
| -------- | -------- |
| applications | 应用程序样例，包括camera等 |
| base | 基础软件服务子系统集&硬件服务子系统集 |
| build | 组件化编译、构建和配置脚本 |
| docs | 说明文档 |
| domains | 增强软件服务子系统集 |
| drivers | 驱动子系统 |
| foundation | 系统基础能力子系统集 |
| kernel | 内核子系统 |
| prebuilts | 编译器及工具链子系统 |
| test | 测试子系统 |
| third_party | 开源第三方组件 |
| utils | 常用的工具集 |
| vendor | 厂商提供的软件 |
| build.py | 编译脚本文件 |

**关键数据：**
- applications：应用程序样例
- build：编译构建脚本
- kernel：内核子系统
- vendor：厂商软件

**来源文件：** `https://gitee.com/openharmony/docs/raw/master/zh-cn/device-dev/get-code/sourcecode-acquire.md`
**定位：** 1 获取源码 > 1.7 源码目录简介

# 相关代码

## 代码 1: 代码获取命令

**功能说明：** 该命令用于将OpenHarmony代码从个人仓库克隆到本地工作空间。首先使用git clone命令克隆指定仓库，然后进入该仓库目录准备后续开发工作。这是开源项目贡献的标准流程，确保开发者可以在独立的环境中进行修改和测试。

**使用注意：** 需要将{your_gitcode_id}和{repository_name}替换为实际的用户名和仓库名

**涉及API：** git clone, cd

```shell
git clone git@gitcode.com:{your_gitcode_id}/{repository_name}.git
cd {repository_name}
```

**来源文件：** `https://gitee.com/openharmony/docs/raw/master/zh-cn/contribute/贡献流程.md`
**定位：** 1 OpenHarmony 社区贡献流程 > 1.1 贡献准备 (代码下载) > 1.1.2 获取代码

## 代码 2: 编译构建命令

**功能说明：** 这些命令用于OpenHarmony系统的编译构建。第一条命令用于编译Debug版本，第二条命令用于单独编译指定模块的Debug二进制。由于Debug全版本镜像大小限制，建议使用模块级编译。

**使用注意：** {product_name}为当前版本支持的平台，如hispark_taurus_standard等

**涉及API：** build.sh

```shell
./build.sh --product-name {product_name} --gn-args is_debug=true
./build.sh --product-name {product_name} --gn-args is_debug=true --build-target {target_name}
```

**来源文件：** `https://gitee.com/openharmony/docs/raw/master/zh-cn/device-dev/subsystems/subsys-build-all.md`
**定位：** 1 编译构建指导 > 1.3 编译构建使用指导 > 1.3.2 编译命令

## 代码 3: 孵化仓代码初始化

**功能说明：** 该命令用于初始化OpenHarmony孵化仓的代码下载环境。通过指定特定的manifest文件和分支，确保获取正确的孵化仓代码版本。

**使用注意：** 专门用于参与孵化仓开发的场景，其他下载步骤与主线相同

**涉及API：** repo init

```shell
repo init -u https://gitee.com/openharmony-sig/manifest.git -b master -m devboard.xml --no-repo-verify
```

**来源文件：** `https://gitee.com/openharmony/docs/raw/master/zh-cn/device-dev/porting/Readme-CN.md`
**定位：** 2 代码准备

# 相关概念

- 开源贡献流程
- 组件化设计
- 分布式操作系统
- 系统能力（SysCap）
- 混合开发环境
- 编译构建系统

# 推荐阅读

- **OpenHarmony项目贡献指南**: 深入了解OpenHarmony社区的完整贡献流程和规范
- **OpenHarmony部件设计和开发指南**: 学习OpenHarmony组件化架构的设计理念和实现方法
- **设备开发快速入门**: 通过实际案例掌握OpenHarmony设备开发的具体步骤
