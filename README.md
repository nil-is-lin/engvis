# engvis

工程可视化框架 — 基于 Rust 和 wgpu 的高性能 3D 工程可视化工具，以**三周期极小曲面 (TPMS)** 为核心。

## 项目简介

engvis 是一个以**三周期极小曲面 (Triply Periodic Minimal Surface, TPMS)** 为核心的工程可视化框架，基于 Rust 和 wgpu 构建。TPMS 是一类在三维空间中无限周期延伸、平均曲率为零的曲面，广泛应用于增材制造、生物支架、热交换器和轻量化结构设计。

---

## 三周期极小曲面 (TPMS)

### 数学定义

三周期极小曲面定义为光滑函数 $f: \mathbb{R}^3 \to \mathbb{R}$ 的零等值面：

```math
\mathcal{S} = \{\, \mathbf{x} \in \mathbb{R}^3 \mid f(\mathbf{x}) = 0 \,\}
```

该曲面将空间划分为 $`f > 0`$ 和 $`f < 0`$ 两个区域，形成双连续的三维网络结构。

### 支持曲面（11 种）

| 曲面 | 隐式方程 $`f(kx, ky, kz) = 0`$ |
|---|---|
| **Gyroid** | $`\sin x \cos y + \sin y \cos z + \sin z \cos x`$ |
| **Schwarz P** | $`\cos x + \cos y + \cos z`$ |
| **Schwarz D** | $`\sin x \sin y \sin z + \sin x \cos y \cos z + \cos x \sin y \cos z + \cos x \cos y \sin z`$ |
| **Schoen IWP** | $`2(\cos x \cos y + \cos y \cos z + \cos z \cos x) - (\cos 2x + \cos 2y + \cos 2z)`$ |
| **Neovius** | $`3(\cos x + \cos y + \cos z) + 4 \cos x \cos y \cos z`$ |
| **F-RD** | $`4 \cos x \cos y \cos z - (\cos 2x \cos 2y + \cos 2y \cos 2z + \cos 2z \cos 2x)`$ |
| **Lidinoid** | $`\tfrac{1}{2}(\sin 2x \cos y \sin z + \sin 2y \cos z \sin x + \sin 2z \cos x \sin y) - \tfrac{1}{2}(\cos 2x \cos 2y + \cos 2y \cos 2z + \cos 2z \cos 2x) + 0.15`$ |
| **Split-P** | $`1.1(\cos 2x + \cos 2y + \cos 2z) - 0.2(\cos 2x \cos 2y + \cos 2y \cos 2z + \cos 2z \cos 2x) - 0.4(\cos 4x + \cos 4y + \cos 4z)`$ |
| **Fischer-Koch S** | $`\cos 2x \sin y \cos z + \cos 2y \sin z \cos x + \cos 2z \sin x \cos y`$ |
| **Fischer-Koch Y** | $`2 \cos x \cos y \cos z + \sin 2x \sin y + \sin 2y \sin z + \sin 2z \sin x`$ |
| **Fischer-Koch CP** | $`\cos x + \cos y + \cos z + 4 \cos x \cos y \cos z`$ |

### 三种形态模式

| 模式 | 方程 | 描述 |
|---|---|---|
| **Minimal Surface** | $`f(\mathbf{x}) = 0`$ | 经典极小曲面——开放薄片 |
| **Shell** | $`f^2 - \delta^2 \le 0`$ | 厚壁空心结构，材料覆盖在极小曲面两侧 |
| **Skeletal** | $`f(\mathbf{x}) - C = 0`$ | 固态支柱网络，等值面偏移 |

**Shell 模式**采用光滑场 $`g = f^2 - \delta^2`$（而非 $`|f| - \delta`$），避免了 $`C^1`$ 尖点，保证 Marching Cubes 在近零区域正确拼接。壁厚 $`t`$ 通过 $`\delta = \tfrac{1}{2} t k`$ 映射为物理几何厚度。

**Skeletal 模式**通过体积分数 $`\varphi \in (0, 1)`$ 参数化：用户在 UI 中设置 $`\varphi`$，程序通过二分查找求解对应的等值面偏移 $`C`$，使

```math
\frac{|\{\mathbf{x} : f(\mathbf{x}) < C\}|}{|\text{domain}|} = \varphi
```

同一 $`\varphi`$ 值在所有 TPMS 上具有一致的固/空比语义（$`\varphi = 0.5`$ 始终对应对称极小曲面）。

### 多晶胞域扩展

支持三个维度的独立晶胞堆叠数 $`(n_x, n_y, n_z)`$，总晶胞数为 $`n_x \times n_y \times n_z`$。内在周期 $`k`$ 控制单个晶胞内的空间频率（Gyroid 默认 $`k=4`$，多数为 $`k=3`$，Fischer-Koch 为 $`k=2`$）。

---

## 网格生成

- **MC33 (Marching Cubes 33)**：规则网格采样，自适应分辨率（1–512），Shell 模式最少 96³
- **Dual Contouring (DC)**：自适应八叉树，可调深度，保留尖锐特征
- **JIT 编译求值**：通过 [Fidget](https://github.com/nil-is-lin/fidget) 将隐式曲面的表达式树编译为平台原生代码，实时采样性能媲美手写函数
- **边界封闭 (Boundary Capping)**：Shell 和 Skeletal 网格通过 CSG 与包围盒相交，结合边界环扇形补面算法生成无边界边的封闭实体
- **Newton 投影**：Skeletal 模式下将交界顶点投影到 TPMS 与盒子面的精确交线上，消除阶梯锯齿
- **异步构建**：高分辨率网格在后台线程构建，保持 UI 响应

---

## 主要特性

- **GPU 加速渲染**：基于 wgpu 实现跨平台 GPU 渲染（Vulkan / Metal / DirectX 12 / WebGPU）
- **JIT 隐式曲面求值**：通过 fidget 将表达式树编译为原生代码
- **自定义表达式**：输入任意 Rhai 脚本（如 `sin(4*x)*cos(4*y)+...`）实时生成网格
- **PBR 材质系统**：物理正确的金属/粗糙度材质渲染
- **灵活的相机控制**：轨道相机，前 / 顶 / 右 / 等轴测视图，Fit 对焦
- **拓扑分析**：实时显示 $`V, E, F, \chi = V-E+F`$，边界边、非流形边、连通分量、水密性
- **网格 I/O**：导入 / 导出 OBJ、STL、PLY
- **glTF 模型支持**：加载外部 3D 模型
- **实时 UI**：基于 egui 的四面板布局，参数实时调整

---

## 项目结构

```
engvis/
├── crates/
│   ├── engvis-core/      # 核心数据类型：Scene、Mesh、Camera、Topology、Aabb 等
│   ├── engvis-surface/   # 纯数学库：SurfaceType 枚举、TPMS 公式、隐式曲面求值
│   ├── engvis-mesher/    # 网格生成：MC33、Dual Contouring、边界封闭
│   └── engvis-renderer/  # GPU 渲染：wgpu 管线、PBR、egui 集成
├── examples/
│   ├── hello_viewer/     # 最小 EngvisApp 示例
│   └── colorbar3d/       # 色标 3D 演示
├── fidget-demo/          # 独立隐式曲面查看器（fidget-core + fidget-mesh）
├── formulas/             # SVG 公式图和 Typst 源文件
├── doc/                  # LaTeX 技术文档
└── src/
    └── main.rs           # 主应用程序入口
```

### engvis-surface

纯数学库，无 GPU 依赖，提供：
- `SurfaceType` 枚举：类型安全地表示所有内置曲面（`Gyroid`、`SchwarzP`、`Custom(String)` 等）
- TPMS 隐式公式求值（基于 fidget 表达式树）
- `TreeParams` 参数结构体：周期、晶胞数、幅值、偏移、旋转、混合等
- `GradientField` / `GradientMode`：梯度场权重生成，用于混合与偏移
- `Morphology` 枚举：`MinimalSurface` / `Shell` / `Skeletal`
- 单元测试：15 个测试覆盖全部 TPMS 公式和辅助方法

### engvis-mesher

网格生成库，提供：
- Marching Cubes 33（MC33）：完整 Bourke 查找表、边界平滑、绕向修复
- Dual Contouring（DC）：自适应八叉树，保留尖锐特征
- 边界封闭（Boundary Capping）：边界环扇形补面算法
- 拓扑分析（Topology）：半边算法，计算 $`V, E, F, \chi`$

### engvis-core

核心库，提供：
- 相机系统（`OrbitCamera`）
- 网格数据结构（`Mesh`、`SubMesh`、`MeshVertex`）
- 材质系统（`PbrMaterial`）
- 光照系统（`AmbientLight`、`DirectionalLight`、`PointLight`）
- 场景管理（`Scene`、`SceneNode`）
- 输入处理（`InputState`）
- 包围盒（`Aabb`）
- 注解图元（球体线框、盒子线框）

### engvis-renderer

渲染库，提供：
- GPU 上下文管理（`GpuContext`、`GpuResources`）
- 网格渲染器（`MeshRenderer`）
- 材质管线（`MaterialPipeline`）
- 网格渲染（`GridRenderer`）
- 覆盖层渲染（`OverlayRenderer`）
- 光照缓冲（`LightingBuffer`）
- 纹理缓存（`TextureCache`）
- glTF 加载器
- egui 集成
- 后处理管线（`PostprocessPipeline`）

---

## 快速开始

### 环境要求

- Rust 1.80 或更高版本
- 支持 Vulkan、Metal 或 DirectX 12 的 GPU

### 运行

```bash
git clone https://github.com/nil-is-lin/engvis.git
cd engvis
cargo run --release
```

### 交互式参数调整

在右侧 **Source** 面板中可实时调整：
- 曲面选择（下拉菜单，显示隐式方程）
- 周期 $`k`$（1–10 滑块）
- 各方向晶胞数 $`n_x, n_y, n_z`$（1–10 滑块）
- 形态模式（MinimalSurface / Shell / Skeletal）
- Shell 模式下壁厚 $`t`$
- Skeletal 模式下的体积分数 $`\varphi`$（以及求解后的 $`C`$ 值显示）
- 网格分辨率与后端（MC33 / DC）
- 自定义隐式表达式（Rhai 脚本）

---

## 技术栈

| 库 | 用途 |
|---|---|
| [wgpu](https://github.com/gfx-rs/wgpu) | 跨平台 GPU 抽象层 |
| [winit](https://github.com/rust-windowing/winit) | 窗口管理和事件处理 |
| [egui](https://github.com/emilk/egui) | 即时模式 GUI |
| [fidget](https://github.com/nil-is-lin/fidget) | JIT 隐式曲面求值 |
| [glam](https://github.com/bitshifter/glam-rs) | 数学库（向量、矩阵、四元数） |
| [rfd](https://github.com/PolyMeilex/rfd) | 原生文件对话框 |
| [bytemuck](https://github.com/Lokathor/bytemuck) | 零拷贝类型转换 |

---

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 作者

nil
