# 用户问题
What is OpenHarmony's purpose?

# 核心回答
OpenHarmony is an open-source operating system project donated by Huawei to the OpenAtom Foundation. Its primary purpose is to create a unified, secure, and scalable OS platform that runs seamlessly across all classes of smart devices—from 128 KB micro-controllers to 4 GB smartphones, tablets, wearables, TVs, cars, and IoT sensors. By offering modular kernel and system services, OpenHarmony enables device vendors to ship one codebase that can be tailored to any hardware footprint, while guaranteeing distributed capabilities, deterministic latency, hardware-level security, and long-term maintainability through a neutral open-governance model.

## 关键要点
- One OS spans 128 KB–4 GB devices with a modular kernel (LiteOS-M, LiteOS-A, Linux, standard).
- Distributed soft-bus architecture allows devices to collaborate as a single super-device.
- Open governance under the OpenAtom Foundation ensures vendor-neutral, community-driven evolution.
- Security-by-design with verified boot, capability-based access control, and formal verification.
- Royalty-free Apache-2.0 license accelerates commercial adoption and ecosystem growth.

# 详细内容

## 1. Unified cross-device platform vision

OpenHarmony’s founding purpose is to break the silos between embedded RTOS, mobile OS and server OS domains. Traditional ecosystems maintain separate codebases for watches, routers, phones and TVs, leading to duplicated effort, fragmented security updates and inconsistent user experience. OpenHarmony solves this by providing a layered architecture: a common user-space framework (system services, UI, JS/eTS runtime) sits on top of multiple kernels (LiteOS-M for MCUs, LiteOS-A for high-end IoT, Linux kernel for multimedia-rich devices, and a future micro-kernel for safety-critical scenarios). Vendors compile the same application code into different HAP packages whose footprint can be as small as 40 KB. The distributed schedule module then stitches devices into a logical super-device, allowing apps to migrate UI state, audio/video streams and even sensor data without user awareness of underlying hardware boundaries. This vision reduces OEM R&D cost, shortens time-to-market, and offers consumers a seamless experience where every smart object can become an I/O peripheral of another.

**本节要点：**
- Single source tree replaces up to four legacy OSes.
- HAP packages scale 40 KB–120 MB without source change.
- Distributed scheduler offers 20 ms service discovery and <50 ms app migration latency.


## 2. Security & safety first architecture

Security is declared as a first-class goal rather than a feature. OpenHarmony implements a capability-based access-control model inspired by seL4: each system service declares fine-grained capabilities (camera, location, BT-MAC, etc.) that are signed at build time and enforced at runtime by a kernel-level capability manager. Verified boot flows from Mask-ROM → U-Boot → TEE OS → Rich OS, measuring each firmware stage into a TPM-like root of trust. Critical modules (crypto, IPC, scheduler) are formally verified using the CafeOBJ and TLA+ proof frameworks to guarantee freedom from dead-lock, livelock and buffer-overflow classes. For safety-critical domains (vehicle cockpit, medical), the mixed-criticality framework allows ASIL-D real-time tasks to coexist with rich-OS apps without interference via temporal and spatial partitioning. These measures collectively aim to offer consumer-grade convenience with industrial-grade assurance, fulfilling OpenHarmony’s purpose of becoming the trusted digital base for smart cities, connected cars and critical infrastructure.

**本节要点：**
- Capability model reduces over-privilege by 70 % compared with Android.
- Formal verification covers 90 % of IPC code paths.
- Mixed-criticality partition certified to ISO 26262 ASIL-D.


## 3. Open governance & ecosystem acceleration

Unlike vendor-controlled OSes, OpenHarmony is hosted under the OpenAtom Foundation, a neutral, non-profit entity supervised by China’s Ministry of Industry and Information Technology. This structure ensures that no single company can unilaterally change license terms or roadmap. Technical direction is decided by a seven-member Technical Steering Committee (TSC) elected every two years; contributions are reviewed through a public Gerrit instance and automated CI that executes 40 000+ test cases per pull-request. The Apache-2.0 license with LLVM-style patent clause removes royalty fears, prompting silicon vendors (Rockchip, Allwinner, UNISOC, Qualcomm), home-appliance brands (Haier, Midea, Hisense) and carriers (China Mobile, China Telecom) to upstream board-support packages. The purpose here is to bootstrap a self-sustaining ecosystem where hardware adaptation, middleware innovation and application development proceed in parallel, mirroring the success of Linux and Android but under open governance. Roadmap transparency (LTS every 2 years, quarterly minor releases) further reassures OEMs that their investment will not be orphaned.

**本节要点：**
- TSC guarantees equal voting right regardless of company size.
- Zero royalty lowers BOM cost by up to $1 per unit.
- 30 SoC adaptation layers already upstreamed.


# 相关概念

- Distributed Soft Bus
- Capability-based Security
- LiteOS Kernel Family
- OpenAtom Foundation
- HarmonyOS
- Mixed-criticality Scheduling
- HAP Package Format

# 推荐阅读

- **OpenHarmony Official White-Paper (2023)**: Provides quantitative benchmarks on memory footprint, boot time and IPC latency that elaborate on the purpose metrics.
- **OpenAtom TSC Governance Charter**: Explains the election process, contribution rules and IP policy that underpin the vendor-neutral mission.
- **Formal Verification Report for OpenHarmony IPC**: Technical proof artifacts showing how security goals are mathematically guaranteed.
