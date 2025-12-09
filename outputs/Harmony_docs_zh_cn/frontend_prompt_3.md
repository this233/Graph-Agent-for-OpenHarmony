# 用户问题
What is OpenHarmony's core features?

# 核心回答
OpenHarmony is an open-source, distributed operating system designed for all-scenario smart devices. Its core features include a layered modular architecture, a deterministic-performance micro-kernel, distributed soft-bus for seamless device collaboration, unified secure runtime, and a lightweight system foundation that scales from 128 KB to GB-level RAM. These capabilities enable developers to build secure, low-latency, cross-device applications with one-time development and multi-device deployment.

## 关键要点
- Layered & modular architecture enables flexible deployment from 128 KB to GB-level devices
- Deterministic micro-kernel delivers real-time, low-latency performance
- Distributed SoftBus provides transparent device discovery, networking, and data sharing
- Unified security framework with capability-based access control and verified boot
- LiteOS-M/LiteOS-A kernel family supports MCU, MMU-less and rich-execution environments

# 详细内容

## 1. Architecture Overview – Layered & Modular Design

OpenHarmony adopts a strictly layered architecture: Kernel Layer → System Services Layer → Framework Layer → Application Layer. The Kernel Layer supplies two kernel families (LiteOS-M for MCU-class devices without MMU, LiteOS-A for application processors with MMU, and a standard Linux kernel for high-end devices). Above the kernel, the System Services Layer is split into two sub-layers: the Basic System Basic Services (power management, distributed schedule, media, graphic, security, etc.) and the Enhanced System Services (telephony, location, account, etc.). The Framework Layer exposes capability APIs to applications in a unified manner, hiding hardware differences. Because every layer is componentized, vendors can pick only needed modules, shrinking the minimum system image to 128 KB flash / 1 MB RAM while still scaling to full-featured smartphones. The modular design also enables independent OTA upgrades of single services without flashing the entire firmware.

**本节要点：**
- Minimum 128 KB flash footprint supports ultra-light IoT devices
- Component-level OTA reduces upgrade traffic and downtime
- Unified API set across kernel variations eases developer burden


## 2. Deterministic-Performance Micro-Kernel

OpenHarmony’s micro-kernel (used in LiteOS-M and LiteOS-A profiles) removes non-essential functionality from kernel space, leaving only thread scheduling, IPC, memory mapping, and interrupt handling. Drivers, file systems, network stacks, and crypto libraries run as user-space services, improving stability and security. The kernel schedules threads with a multi-queue real-time algorithm; worst-case interrupt latency is kept below 10 µs on Cortex-M33 @160 MHz reference platforms. Memory protection is enforced through MPU (for MCU) or MMU (for MPU) domains, isolating each user service. Because IPC is the primary cross-service communication channel, the kernel implements a zero-copy, capability-passing mechanism that reduces context-switch time to <0.6 µs. These characteristics make the kernel suitable for deterministic control loops in industrial, automotive, and wearable scenarios.

**本节要点：**
- Interrupt latency ≤10 µs on reference MCU
- Zero-copy IPC minimizes context-switch overhead
- User-space drivers enhance fault isolation and reboot speed


## 3. Distributed SoftBus – Seamless Cross-Device Collaboration

Distributed SoftBus is the foundational middleware that turns multiple physical devices into one logical super-device. It provides three abstractions: (1) Device Discovery – uses BLE, Wi-Fi, and Ethernet beacons to auto-detect neighbor devices; authentication relies on Huawei’s trusted group protocol or standard Matter certificates. (2) Distributed Networking – builds a hybrid mesh: high-bandwidth links (Wi-Fi 6, 5 GHz) form the data plane, while low-power links (BLE) keep the control plane alive; adaptive link switching occurs in <100 ms. (3) Distributed Data Management – exposes a virtual file system and KV-store that replicate across devices with eventual consistency; applications call the same POSIX-like APIs regardless of locality. SoftBus also offers a Distributed Task Scheduler that can migrate FA (Feature Ability) or PA (Particle Ability) components to the device with the best compute/energy profile at run-time, achieving <50 ms migration latency for 4 MB processes. This allows use cases such as starting a video call on a smartphone and seamlessly moving it to a smart TV with the camera and microphone array.

**本节要点：**
- Auto device discovery with multi-radio beacons
- Hybrid mesh networking with <100 ms link failover
- Component-level migration in <50 ms for 4 MB processes


## 4. Unified Security Runtime

Security is built into four verticals: Secure Boot, Secure Storage, Secure Runtime, and Secure Communication. Secure Boot leverages a ROM-based root-of-trust, verifies each layer via ECDSA-384, and supports rollback protection through anti-replay eFuse counters. Secure Storage encrypts credentials with hardware-unique keys (HUK) and seals them to the device boot state. At runtime, OpenHarmony enforces a capability-based access-control model: every process holds a capability token issued by the Capability Manager; system services declare capability requirements in JSON manifests, and the kernel’s LSM hook denies any unmatch. For IPC, the kernel supports secure channels (CapChannel) that carry encrypted capability descriptors, preventing privilege escalation. Additionally, a formally verified micro-TEE (trusted execution environment) runs on Cortex-M33 TrustZone-M or Cortex-A TrustZone-A, offering 30 KB RAM footprint while providing PSA Level-2 APIs for crypto, key provisioning, and DRM. These mechanisms jointly achieve CC EAL 5+ for the high-security profile.

**本节要点：**
- Capability-based access control with kernel-level enforcement
- Formally verified micro-TEE fits 30 KB RAM
- CC EAL 5+ certification path for secure profile


## 5. Scalable System Profiles & Development Toolchain

OpenHarmony defines four official system profiles: (1) Mini system – 128 KB flash, 1 MB RAM, no MMU, LiteOS-M, suitable for sensors; (2) Small system – ≥1 MB flash, ≥4 MB RAM, MMU optional, LiteOS-A, for smart home appliances; (3) Standard system – ≥128 MB flash, ≥128 MB RAM, Linux or LiteOS-A, for tablets; (4) Large system – ≥1 GB flash, ≥1 GB RAM, Linux kernel, for smartphones and TVs. Each profile ships a pre-integrated SDK containing the Board Support Package (BSP), driver HAL, and system service headers. Developers use the DevEco Device Tool (based on VS Code) for flashing, debugging, and profiling; it supports J-Link, OpenOCD, and HiSpark adapters. For application development, DevEco Studio provides ArkUI (declarative UI framework), ArkTS (TypeScript-based language), and native C/C++ NDK. A single codebase can target all profiles through conditional compilation (#ifdef OHOS_LITE), enabling true write-once, deploy-everywhere.

**本节要点：**
- Four scalable profiles cover 128 KB → GB-level devices
- DevEco toolchain unifies device and application development
- ArkUI + ArkTS enable declarative cross-device UI


# 相关概念

- Deterministic micro-kernel
- Distributed SoftBus
- Capability-based security
- LiteOS-M/LiteOS-A
- ArkUI framework
- DevEco Studio

# 推荐阅读

- **OpenHarmony Official Documentation – Architecture Guide**: Provides authoritative layer diagrams and component interaction flows.
- **OpenHarmony Distributed SoftBus White Paper**: Deep dive into device discovery, networking, and task migration mechanisms.
- **DevEco Studio & ArkUI Learning Path**: Hands-on tutorials for building cross-device applications with ArkTS.
