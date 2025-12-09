# 用户问题
What is OpenHarmony's development environment?

# 核心回答
OpenHarmony的开发环境主要采用Windows+Ubuntu混合开发模式，提供两种主要开发方式：基于IDE的图形化开发和基于命令行的开发。IDE方式使用DevEco Device Tool，在Windows上进行代码开发、调试和烧录，在Ubuntu环境进行源码编译。命令行方式通过hb工具进行编译构建，支持多种编译选项。此外，OpenHarmony还提供Docker编译环境来简化环境配置，支持标准系统、小型系统和轻量系统三种不同类型的设备开发。开发环境的选择取决于目标设备类型和开发者的操作习惯，其中Ubuntu环境主要用于编译，Windows环境主要用于代码编辑和烧录操作。

## 关键要点
- Windows+Ubuntu混合开发环境
- 两种开发方式：IDE和命令行
- 支持三种系统类型：轻量系统、小型系统、标准系统
- 提供Docker环境简化配置
- DevEco Device Tool作为主要IDE工具
- hb工具用于命令行编译构建

# 详细内容

## 1. 开发环境架构与模式

OpenHarmony的开发环境采用创新的Windows+Ubuntu混合开发架构，这种设计充分考虑了开发者的实际使用习惯和不同操作系统的优势。在嵌入式开发中，很多开发者习惯于使用Windows进行代码编辑，比如使用Windows的Visual Studio Code进行OpenHarmony代码开发。然而，当前阶段大部分开发板源码（如Hi3861、Hi3516系列开发板）还不支持在Windows环境下进行编译，因此必须使用Ubuntu的编译环境对源码进行编译。同时，开发板的烧录操作需要在Windows环境中进行。这种混合环境架构允许开发者在Windows平台上使用DevEco Device Tool的可视化界面进行相关操作，通过远程连接的方式对接Ubuntu下的DevEco Device Tool，然后对Ubuntu下的源码进行开发、编译、烧录等操作。对于没有Ubuntu系统的开发者，可以在Windows系统中通过虚拟机方式搭建Ubuntu系统，然后根据指导完成Ubuntu基础环境配置，再进行DevEco Device Tool工具的安装。这种架构设计既保证了开发效率，又确保了编译环境的稳定性。

**本节要点：**
- Windows+Ubuntu混合开发模式
- Windows用于代码编辑和烧录
- Ubuntu用于源码编译
- 支持虚拟机方式搭建Ubuntu环境

**来源：**
- 文件: `https://gitee.com/openharmony/docs/raw/master/zh-cn/device-dev/quick-start/quickstart-ide-env-ubuntu.md`
  定位: 1 搭建Ubuntu环境
  相关性: 详细说明了混合开发环境的架构和优势
- 文件: `https://gitee.com/openharmony/docs/raw/master/zh-cn/device-dev/quick-start/quickstart-pkg-prepare.md`
  定位: 1 准备开发环境
  相关性: 解释了为什么需要混合环境以及具体的使用场景

## 2. 两种开发方式对比

OpenHarmony为开发者提供了两种主要的入门指导方式，以适应不同的开发习惯和需求。第一种是基于IDE的开发方式，使用DevEco Device Tool作为开发环境。这种方式完全采用IDE进行一站式开发，编译依赖工具的安装及编译、烧录、运行都通过IDE进行操作。DevEco Device Tool采用Windows+Ubuntu混合开发环境的具体实现是：在Windows上主要进行代码开发、代码调试、烧录等操作，而在Ubuntu环境实现源码编译。DevEco Device Tool提供界面化的操作接口，可以为开发者提供更快捷的开发体验，特别适合不熟悉命令行操作或习惯界面化操作的开发者。第二种是基于命令行的开发方式，使用命令行工具包。这种方式通过命令行方式下载安装编译依赖工具，在Linux系统中进行编译时，相关操作通过命令实现；在Windows系统中使用开发板厂商提供的工具进行代码烧录。命令行方式提供了简便统一的工具链安装方式，适合习惯使用命令行操作的开发者。在基于命令行方式开发的过程中，除了Windows环境要求和Ubuntu环境要求外，不对开发设备做另外的要求，开发者需要自行准备Windows环境和Ubuntu环境。

**本节要点：**
- IDE方式：DevEco Device Tool一站式开发
- 命令行方式：hb工具链
- 根据开发者习惯选择合适的方式

**来源：**
- 文件: `https://gitee.com/openharmony/docs/raw/master/zh-cn/device-dev/quick-start/quickstart-overview.md`
  定位: 1 快速入门概述 > 1.1 操作方式
  相关性: 详细对比了两种开发方式的特点和适用人群

## 3. Docker编译环境

OpenHarmony为开发者提供了Docker编译环境，这在很大程度上简化了编译前的环境配置工作。Docker环境主要分为两种类型：独立Docker环境和基于HPM的Docker环境。独立Docker环境适用于直接基于Ubuntu、Windows操作系统平台进行版本编译的场景，支持标准系统、小型系统和轻量系统在Ubuntu/Windows平台编译。基于HPM的Docker环境适用于使用HPM工具进行发行版编译的场景，主要针对轻量和小型系统。对于小型系统，获取Docker镜像的命令为：docker pull swr.cn-south-1.myhuaweicloud.com/openharmony-docker/docker_oh_small:3.2。对于轻量系统，获取镜像的命令为：docker pull swr.cn-south-1.myhuaweicloud.com/openharmony-docker/docker_oh_mini:3.2。进入Docker构建环境的操作步骤根据系统不同而有所区别。在Ubuntu系统中，进入小型系统Docker构建环境的命令是：docker run -it -v $(pwd):/home/openharmony swr.cn-south-1.myhuaweicloud.com/openharmony-docker/docker_oh_small:3.2；进入轻量系统Docker构建环境的命令是：docker run -it -v $(pwd):/home/openharmony swr.cn-south-1.myhuaweicloud.com/openharmony-docker/docker_oh_mini:3.2。在Windows系统中，假设源码目录为D:\OpenHarmony，进入小型系统Docker构建环境的命令是：docker run -it -v D:\OpenHarmony:/home/openharmony swr.cn-south-1.myhuaweicloud.com/openharmony-docker/docker_oh_small:3.2；进入轻量系统Docker构建环境的命令是：docker run -it -v D:\OpenHarmony:/home/openharmony swr.cn-south-1.myhuaweicloud.com/openharmony-docker/docker_oh_mini:3.2。

**本节要点：**
- 两种Docker环境：独立和基于HPM
- 支持Ubuntu和Windows平台
- 简化环境配置流程

**来源：**
- 文件: `https://gitee.com/openharmony/docs/raw/master/zh-cn/device-dev/get-code/gettools-acquire.md`
  定位: 1 Docker编译环境 > 1.3 独立Docker环境 > 1.3.1 搭建Docker环境（轻量系统和小型系统）
  相关性: 提供了详细的Docker环境搭建命令和操作步骤

## 4. 编译构建工具与命令

OpenHarmony的编译构建主要使用hb工具，这是一个专门为OpenHarmony项目设计的命令行工具。hb工具包含多个常用命令：hb set用于设置要编译的产品，支持设置代码根目录和产品配置；hb env用于查看当前设置信息，包括根路径、板级配置、内核类型、产品信息等详细环境配置；hb build用于编译产品、部件、模块或芯片解决方案；hb clean用于清除编译产物。hb build命令支持丰富的编译选项，包括构建类型（release或debug版本）、编译器指定、测试套件编译、目标CPU选择等。具体选项包括：-b BUILD_TYPE指定构建类型，-c COMPILER指定编译器，-t [TEST [TEST ...]]编译测试套件，-cpu TARGET_CPU选择CPU，--gn-args GN_ARGS指定gn构建参数，--log-level LOG_LEVEL指定编译期间的日志级别（debug、info、error三个级别可选），--fast-rebuild启用快速重建等。当hb build后无参数时，会按照设置好的代码路径、产品进行编译，编译选项使用与之前保持一致。-f选项将删除当前产品所有编译产品，等同于hb clean + hb build。hb build {component_name}可以基于设置好的产品对应的单板、内核，单独编译部件。hb build -p ipcamera@hisilicon支持免set编译产品，该命令可以跳过set步骤直接编译产品。在device/board/device_company下单独执行hb build会进入内核选择界面，选择完成后会根据当前路径的单板、选择的内核编译出仅包含内核、驱动的镜像。

**本节要点：**
- hb工具：主要编译构建工具
- 支持多种编译选项和参数
- 可单独编译部件或完整产品

**来源：**
- 文件: `https://gitee.com/openharmony/docs/raw/master/zh-cn/device-dev/subsystems/subsys-build-all.md`
  定位: 1 编译构建指导 > 1.3 编译构建使用指导 > 1.3.2 编译命令
  相关性: 详细说明了hb工具的各种命令和编译选项

## 5. 系统类型与环境要求

OpenHarmony定义了三种基础系统类型，每种类型对应不同的硬件要求和开发环境配置。轻量系统（mini system）面向MCU类处理器，如Arm Cortex-M、RISC-V 32位的设备，硬件资源极其有限，支持的设备最小内存为128KiB，提供多种轻量级网络协议、轻量级图形框架和丰富的IOT总线读写部件，适用于智能家居领域的连接类模组、传感器设备、穿戴类设备等。用户态和LiteOS-A的内核态编译均使用llvm编译器编译，安装方法在搭建基础环境中已提供。小型系统（small system）面向应用处理器，如Arm Cortex-A的设备，支持的设备最小内存为1MiB，提供更高的安全能力、标准图形框架和视频编解码的多媒体能力，适用于IP Camera、电子猫眼、路由器等产品。标准系统（standard system）同样面向应用处理器，支持的设备最小内存为128MiB，提供增强的交互能力、3D GPU及硬件合成能力、更多控件及动效更丰富的图形能力、完整应用框架，适用于高端冰箱显示屏等产品。对于选择移植linux内核的情况，需要执行命令安装gcc-arm-linux-gnueabi交叉编译工具链：sudo apt-get install gcc-arm-linux-gnueabi，用于编译linux内核态镜像。开发者需要根据目标设备的系统类型选择相应的开发环境和工具链配置。

**本节要点：**
- 三种系统类型：轻量、小型、标准
- 不同内存和处理器要求
- 针对不同应用场景

**来源：**
- 文件: `https://gitee.com/openharmony/docs/raw/master/zh-cn/device-dev/quick-start/quickstart-overview.md`
  定位: 1 快速入门概述
  相关性: 详细说明了三种系统类型的硬件要求和应用场景
- 文件: `https://gitee.com/openharmony/docs/raw/master/zh-cn/device-dev/porting/porting-smallchip-prepare-building.md`
  定位: 1 编译构建 > 1.1 编译环境搭建
  相关性: 说明了不同系统类型对应的编译器要求

# 相关图片

## 图片 1
![华为DevEco Device Tool的欢迎界面，展示HarmonyOS智能设备集成开发环境的功能介绍](https://gitee.com/openharmony/docs/raw/master/zh-cn/device-dev/quick-start/figures/zh-cn_image_0000001340557741.png)

**说明：** 华为DevEco Device Tool的欢迎界面，展示HarmonyOS智能设备集成开发环境的功能介绍

**上下文：** 在搭建Ubuntu环境章节中展示DevEco Device Tool的安装界面

**与问题的相关性：** 展示了OpenHarmony主要IDE工具的用户界面

**尺寸：** 719x447像素
**建议显示方式：** inline

**来源文件：** `/root/code/docs/zh-cn/device-dev/quick-start/quickstart-ide-env-ubuntu.md`
**定位：** 1 搭建Ubuntu环境 > 1.2 操作步骤

## 图片 2
![OpenHarmony OS不同版本和解决方案的SDK包下载信息界面](https://gitee.com/openharmony/docs/raw/master/zh-cn/device-dev/subsystems/figures/ohos_sdk_download.png)

**说明：** OpenHarmony OS不同版本和解决方案的SDK包下载信息界面

**上下文：** 在Rust工具链使用说明中展示SDK下载页面

**与问题的相关性：** 展示了开发环境中SDK资源的获取方式

**尺寸：** 840x369像素
**建议显示方式：** inline

**来源文件：** `/root/code/docs/zh-cn/device-dev/subsystems/subsys-build-rust-toolchain.md`
**定位：** 1 Rust 工具链使用说明 > 1.3 操作指导 > 1.3.2 非OpenHarmony 社区代码编译 > 1.3.2.2 安装 OpenHarmony OS Clang 工具

# 相关表格

## 表格 1: Docker镜像介绍

**描述：** 列出了OpenHarmony提供的不同系统类型对应的Docker镜像仓库和版本标签

| 系统类型 | 运行平台 | Docker镜像仓库 | 标签 |
| -------- | -------- | -------- | -------- |
| 标准系统（独立Docker环境） | Ubuntu/Windows | swr.cn-south-1.myhuaweicloud.com/openharmony-docker/docker_oh_standard | 3.2 |
| 小型系统（独立Docker环境） | Ubuntu/Windows | swr.cn-south-1.myhuaweicloud.com/openharmony-docker/docker_oh_small | 3.2 |
| 轻量系统（独立Docker环境） | Ubuntu/Windows | swr.cn-south-1.myhuaweicloud.com/openharmony-docker/docker_oh_mini | 3.2 |
| 轻量和小型系统（HPM Docker环境） | Ubuntu/Windows | swr.cn-south-1.myhuaweicloud.com/openharmony-docker/openharmony-docker | 0.0.3 |

**关键数据：**
- 标准系统Docker镜像：docker_oh_standard:3.2
- 小型系统Docker镜像：docker_oh_small:3.2
- 轻量系统Docker镜像：docker_oh_mini:3.2
- HPM Docker环境镜像：openharmony-docker:0.0.3

**来源文件：** `https://gitee.com/openharmony/docs/raw/master/zh-cn/device-dev/get-code/gettools-acquire.md`
**定位：** 1 Docker编译环境 > 1.1 Docker环境介绍

## 表格 2: 基础类型系统简介

**描述：** 详细说明了OpenHarmony三种基础系统类型的处理器要求、最小内存配置和主要能力

| 类型 | 处理器 | 最小内存 | 能力 |
| :--------: | :--------: | -------- | -------- |
| 轻量系统（mini&nbsp;system） | MCU类处理器（例如Arm&nbsp;Cortex-M、RISC-V&nbsp;32位的设备） | 128KiB | 提供多种轻量级网络协议，轻量级的图形框架，以及丰富的IOT总线读写部件等。可支撑的产品如智能家居领域的连接类模组、传感器设备、穿戴类设备等。 |
| 小型系统（small&nbsp;system） | 应用处理器（例如Arm&nbsp;Cortex-A的设备） | 1MiB | 提供更高的安全能力、标准的图形框架、视频编解码的多媒体能力。可支撑的产品如智能家居领域的IP&nbsp;Camera、电子猫眼、路由器以及智慧出行域的行车记录仪等。 |
| 标准系统（standard&nbsp;system） | 应用处理器（例如Arm&nbsp;Cortex-A的设备） | 128MiB | 提供增强的交互能力、3D&nbsp;GPU以及硬件合成能力、更多控件以及动效更丰富的图形能力、完整的应用框架。可支撑的产品如高端的冰箱显示屏。 |

**关键数据：**
- 轻量系统：MCU处理器，128KiB内存
- 小型系统：应用处理器，1MiB内存
- 标准系统：应用处理器，128MiB内存

**来源文件：** `https://gitee.com/openharmony/docs/raw/master/zh-cn/device-dev/quick-start/quickstart-overview.md`
**定位：** 1 快速入门概述

# 相关代码

## 代码 1: hb工具编译命令

**功能说明：** 这是OpenHarmony编译工具hb的build命令详细说明，展示了所有可用的编译选项和参数。包括构建类型选择、编译器指定、测试套件编译、目标CPU选择、产品指定、完整编译、详细模式、日志级别控制、快速重建等功能。开发者可以使用这些选项来定制编译过程，满足不同的开发需求。

**使用注意：** hb build后无参数会按照设置好的代码路径和产品进行编译；-f选项会删除所有编译产物并重新编译；可以单独编译指定部件；支持免set直接编译产品

**涉及API：** hb set, hb env, hb clean

```shell
hb build -h
usage: hb build [-h] [-b BUILD_TYPE] [-c COMPILER] [-t [TEST [TEST ...]]] [-cpu TARGET_CPU] [--dmverity] [--tee]
[-p PRODUCT] [-f] [-n] [-T [TARGET [TARGET ...]]] [-v] [-shs] [--patch] [--compact-mode]
[--gn-args GN_ARGS] [--keep-ninja-going] [--build-only-gn] [--log-level LOG_LEVEL] [--fast-rebuild]
[--device-type DEVICE_TYPE] [--build-variant BUILD_VARIANT]
[component [component ...]]

positional arguments:
component             name of the component, mini/small only

optional arguments:
-h, --help            show this help message and exit
-b BUILD_TYPE, --build_type BUILD_TYPE
release or debug version, mini/small only
-c COMPILER, --compiler COMPILER
specify compiler, mini/small only
-t [TEST [TEST ...]], --test [TEST [TEST ...]]
compile test suit
-cpu TARGET_CPU, --target-cpu TARGET_CPU
select cpu
--dmverity            enable dmverity
--tee                 Enable tee
-p PRODUCT, --product PRODUCT
build a specified product with {product_name}@{company}
-f, --full            full code compilation
-n, --ndk             compile ndk
-T [TARGET [TARGET ...]], --target [TARGET [TARGET ...]]
compile single target
-v, --verbose         show all command lines while building
-shs, --sign_haps_by_server
sign haps by server
--patch               apply product patch before compiling
--compact-mode        compatible with standard build system set to false if we use build.sh as build entrance
--gn-args GN_ARGS     specifies gn build arguments, eg: --gn-args="foo="bar" enable=true blah=7"
--keep-ninja-going    keeps ninja going until 1000000 jobs fail
--build-only-gn       only do gn parse, do not run ninja
--log-level LOG_LEVEL
specifies the log level during compilationyou can select three levels: debug, info and error
--fast-rebuild        it will skip prepare, preloader, gn_gen steps so we can enable it only when there is no change
for gn related script
--device-type DEVICE_TYPE
specifies device type
--build-variant BUILD_VARIANT
specifies device operating mode
```

**来源文件：** `https://gitee.com/openharmony/docs/raw/master/zh-cn/device-dev/subsystems/subsys-build-all.md`
**定位：** 1 编译构建指导 > 1.3 编译构建使用指导 > 1.3.2 编译命令

## 代码 2: Docker环境搭建命令

**功能说明：** 这些命令用于搭建OpenHarmony的Docker编译环境，包括获取不同系统类型的Docker镜像和在Ubuntu/Windows系统中启动对应Docker容器的操作。Docker环境可以大大简化编译前的环境配置工作，为开发者提供标准化的编译环境。

**使用注意：** 在获取镜像后需要创建新的Docker容器并进入其中；Ubuntu系统使用$(pwd)表示当前目录；Windows系统需要指定具体的源码目录路径

**涉及API：** docker pull, docker run

```shell
# 获取小型系统镜像的命令为：
docker pull swr.cn-south-1.myhuaweicloud.com/openharmony-docker/docker_oh_small:3.2

# 获取轻量系统镜像的命令为：
docker pull swr.cn-south-1.myhuaweicloud.com/openharmony-docker/docker_oh_mini:3.2

# Ubuntu系统进入小型系统Docker构建环境：
docker run -it -v $(pwd):/home/openharmony swr.cn-south-1.myhuaweicloud.com/openharmony-docker/docker_oh_small:3.2

# Ubuntu系统进入轻量系统Docker构建环境：
docker run -it -v $(pwd):/home/openharmony swr.cn-south-1.myhuaweicloud.com/openharmony-docker/docker_oh_mini:3.2

# Windows系统进入小型系统Docker构建环境：
docker run -it -v D:\OpenHarmony:/home/openharmony swr.cn-south-1.myhuaweicloud.com/openharmony-docker/docker_oh_small:3.2

# Windows系统进入轻量系统Docker构建环境：
docker run -it -v D:\OpenHarmony:/home/openharmony swr.cn-south-1.myhuaweicloud.com/openharmony-docker/docker_oh_mini:3.2
```

**来源文件：** `https://gitee.com/openharmony/docs/raw/master/zh-cn/device-dev/get-code/gettools-acquire.md`
**定位：** 1 Docker编译环境 > 1.3 独立Docker环境 > 1.3.1 搭建Docker环境（轻量系统和小型系统）

# 相关概念

- DevEco Device Tool
- hb编译工具
- Docker环境
- 交叉编译工具链
- Windows+Ubuntu混合开发

# 推荐阅读

- **快速入门环境搭建章节**: 提供OpenHarmony基础环境搭建的详细指导
- **Ubuntu基础环境配置**: 详细说明Ubuntu系统的环境配置要求
- **编译构建使用指导**: 深入理解OpenHarmony的编译构建机制
