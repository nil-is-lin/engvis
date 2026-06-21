# engvis

工程可视化框架 - 基于Rust和wgpu的高性能3D工程可视化工具

## 项目简介

engvis 是一个以**三周期极小曲面 (Triply Periodic Minimal Surface, TPMS)** 为核心的工程可视化框架，基于 Rust 和 wgpu 构建。TPMS 是一类在三维空间中无限周期延伸、平均曲率为零的曲面，广泛应用于增材制造、生物支架、热交换器和轻量化结构设计。

## 三周期极小曲面 (TPMS)

### 数学定义

三周期极小曲面定义为光滑函数 $f: \mathbb{R}^3 \to \mathbb{R}$ 的零等值面：

$$\mathcal{S} = \{\ \mathbf{x} \in \mathbb{R}^3 \ |\ f(\mathbf{x}) = 0 \ \}$$

该曲面将空间划分为 $f>0$ 和 $f<0$ 两个区域，形成双连续的三维网络结构。

### 支持曲面 (11 种)

| 曲面 | 隐式方程 $f(kx, ky, kz) = 0$ |
|---|---|
| **Gyroid** | $\sin x \cos y + \sin y \cos z + \sin z \cos x$ |
| **Schwarz P** | $\cos x + \cos y + \cos z$ |
| **Schwarz D** | $\sin x \sin y \sin z + \sin x \cos y \cos z + \cos x \sin y \cos z + \cos x \cos y \sin z$ |
| **Schoen IWP** | $2(\cos x \cos y + \cos y \cos z + \cos z \cos x) - (\cos 2x + \cos 2y + \cos 2z)$ |
| **Neovius** | $3(\cos x + \cos y + \cos z) + 4 \cos x \cos y \cos z$ |
| **F-RD** | $4 \cos x \cos y \cos z - (\cos 2x \cos 2y + \cos 2y \cos 2z + \cos 2z \cos 2x)$ |
| **Lidinoid** | $\frac{1}{2}[...] - \frac{1}{2}[...] + 0.15$ |
| **Split-P** | $1.1[...] - 0.2[...] - 0.4[...]$ |
| **Fischer-Koch S** | $\cos 2x \sin y \cos z + \cos 2y \sin z \cos x + \cos 2z \sin x \cos y$ |
| **Fischer-Koch Y** | $2 \cos x \cos y \cos z + \sin 2x \sin y + \sin 2y \sin z + \sin 2z \sin x$ |
| **Fischer-Koch CP** | $\cos x + \cos y + \cos z + 4 \cos x \cos y \cos z$ |

### 三种形态模式

| 模式 | 方程 | 描述 |
|---|---|---|
| **Minimal Surface** | $f(\mathbf{x}) = 0$ | 经典极小曲面——开放薄片 |
| **Shell** | $f^2 - \delta^2 \le 0$ | 厚壁空心结构，材料覆盖在极小曲面两侧 |
| **Skeletal** | $f(\mathbf{x}) - C = 0$ | 固态支柱网络，等值面偏移 |

**Shell 模式**采用光滑场 $g = f^2 - \delta^2$（而非 $|f| - \delta$），避免了 $C^1$ 尖点，保证 Marching Cubes 在近零区域正确拼接。壁厚 $t$ 通过 $\delta = \frac{1}{2} t k$ 映射为物理几何厚度。

**Skeletal 模式**通过体积分数 $\varphi \in (0, 1)$ 参数化：用户在 UI 中设置 $\varphi$，程序通过二分查找求解对应的等值面偏移 $C$，使 $|\{\mathbf{x} : f(\mathbf{x}) < C\}| / |\text{domain}| = \varphi$。同一 $\varphi$ 值在所有 TPMS 上具有一致的固/空比语义（$\varphi = 0.5$ 始终对应对称极小曲面）。

### 多晶胞域扩展

支持三个维度的独立晶胞堆叠数 $(n_x, n_y, n_z)$，总晶胞数为 $n_x \times n_y \times n_z$。内在周期 $k$ 控制单个晶胞内的空间频率（Gyroid 默认 $k=4$，多数为 $k=3$，Fischer-Koch 为 $k=2$）。

### 网格生成

- **MC33 (Marching Cubes 33)**：规则网格采样，自适应分辨率（1-512），Shell 模式最少 96³
- **Dual Contouring (DC)**：自适应八叉树，可调深度，保留尖锐特征
- **JIT 编译求值**：通过 [Fidget](https://github.com/nil-is-lin/fidget) 将隐式曲面的表达式树编译为平台原生代码，实时采样性能媲美手写函数
- **边界封闭 (Boundary Capping)**：Shell 和 Skeletal 网格通过 CSG 与包围盒相交，结合边界环扇形补面算法生成无边界边的封闭实体
- **Newton 投影**：Skeletal 模式下将交界顶点投影到 TPMS 与盒子面的精确交线上，消除阶梯锯齿
- **异步构建**：高分辨率网格在后台线程构建，保持 UI 响应

### 交互式参数调整

在右侧 Source 面板中可实时调整：
- 曲面选择（下拉菜单，显示隐式方程）
- 周期 $k$（1-10 滑块）
- 各方向晶胞数 $n_x, n_y, n_z$（1-10 滑块）
- 形态模式（MinimalSurface / Shell / Skeletal）
- Shell 模式下壁厚 $t$
- Skeletal 模式下的体积分数 $\varphi$（以及求解后的 $C$ 值显示）
- 网格分辨率与后端选择

## 主要特性

- **GPU加速渲染**: 基于wgpu实现跨平台GPU渲染
- **glTF模型支持**: 支持加载glTF格式的3D模型
- **PBR材质系统**: 完整的物理渲染材质系统
- **灵活的相机控制**: 轨道相机，支持多种视图模式（前视图、顶视图、右视图、等轴测视图）
- **顶点和边线渲染**: 可视化显示网格的顶点和边线
- **高级光照系统**: 支持环境光、方向光和点光源
- **实时UI界面**: 基于egui的实时参数调整界面
- **场景管理**: 完整的场景图和节点管理系统

## 项目结构

```
engvis/
├── crates/
│   ├── engvis-core/      # 核心数据类型和算法
│   └── engvis-renderer/  # GPU渲染实现
└── src/
    └── main.rs           # 主应用程序
```

### engvis-core

核心库，提供：
- 相机系统（OrbitCamera）
- 网格数据结构（Mesh, SubMesh, MeshVertex）
- 材质系统（PbrMaterial）
- 光照系统（AmbientLight, DirectionalLight, PointLight）
- 场景管理（Scene, SceneNode）
- 输入处理（InputState）
- 包围盒（Aabb）

### engvis-renderer

渲染库，提供：
- GPU上下文管理（GpuContext, GpuResources）
- 网格渲染器（MeshRenderer）
- 材质管线（MaterialPipeline）
- 网格渲染（GridRenderer）
- 覆盖层渲染（OverlayRenderer）
- 光照缓冲（LightingBuffer）
- 纹理缓存（TextureCache）
- glTF加载器
- egui集成

## 快速开始

### 环境要求

- Rust 1.70或更高版本
- 支持Vulkan、Metal或DirectX 12的GPU

### 运行示例

```bash
# 克隆仓库
git clone https://github.com/nil-is-lin/engvis.git
cd engvis

# 运行程序
cargo run --release
```

### 使用方法

程序启动后会显示一个默认的立方体场景。你可以：

1. **加载glTF模型**:
   - 点击菜单栏 `File` -> 输入模型路径 -> 点击 `Load glTF`

2. **相机控制**:
   - 鼠标左键拖动：旋转视图
   - 鼠标右键拖动：平移视图
   - 鼠标滚轮：缩放
   - 底部面板提供预设视图按钮：Front、Top、Right、Iso、Fit

3. **调整渲染参数**:
   - 左侧面板：场景信息
   - 右侧面板：材质、光照、渲染选项
   - 可以调整表面透明度、边线颜色和宽度、顶点大小等

4. **光照调整**:
   - 在右侧面板调整环境光和方向光的颜色和强度

## 作为库使用

### 添加依赖

在你的 `Cargo.toml` 中添加：

```toml
[dependencies]
engvis-core = "0.1.0"
engvis-renderer = "0.1.0"
```

### ⚠️ 重要注意事项

#### 关键步骤：上传场景数据到GPU

在使用engvis库时，**必须**在创建渲染器后调用`upload_scene`方法将场景数据上传到GPU。如果缺少这一步，渲染管线会因为缺少必要的BindGroup而报错：

```
wgpu error: Validation Error
Caused by:
  In a CommandEncoder
    In a draw command, kind: Draw
      The current set RenderPipeline with 'PBR Solid Pipeline' label expects a BindGroup to be set at index 2
```

**正确的初始化流程：**

```rust
// 1. 创建场景
let mut scene = Scene::default();
scene.meshes.push(your_mesh);
scene.materials.push(PbrMaterial::default());
scene.nodes.push(SceneNode {
    name: "MyObject".to_string(),
    local_transform: Affine3A::IDENTITY,
    mesh_index: Some(0),
    children: Vec::new(),
    visible: true,
});

// 2. 创建渲染器
let mut renderer = Renderer::new(
    &gpu.context.device,
    &gpu.context.queue,
    gpu.surface_format,
    &scene,
    size.width,
    size.height,
);

// 3. 上传场景数据到GPU（关键步骤！）
renderer.upload_scene(&gpu.context.device, &gpu.context.queue, &scene);
```

#### 为什么需要upload_scene？

`upload_scene`方法会：
- 将网格数据（顶点、索引）上传到GPU缓冲区
- 创建材质纹理和BindGroup
- 设置光照系统的GPU资源
- 准备渲染管线所需的所有资源

没有这一步，渲染器无法访问场景数据，导致渲染失败。

### 示例代码

```rust
use engvis_core::{OrbitCamera, Scene, SceneNode, PbrMaterial, mesh::create_cube_mesh};
use engvis_renderer::{Renderer, create_window_and_gpu};
use glam::{Vec3, Affine3A};

// 创建场景
let mut scene = Scene::default();

// 使用内置函数创建网格（推荐）
let cube = create_cube_mesh();

// 添加网格和材质
scene.meshes.push(cube);
scene.materials.push(PbrMaterial {
    name: "Default".to_string(),
    albedo: [0.7, 0.4, 0.3, 1.0],
    metallic: 0.1,
    roughness: 0.6,
    ..Default::default()
});

// 添加场景节点
scene.nodes.push(SceneNode {
    name: "MyObject".to_string(),
    local_transform: Affine3A::from_translation(Vec3::new(0.0, 0.5, 0.0)),
    mesh_index: Some(0),
    children: Vec::new(),
    visible: true,
});

// 创建相机
let camera = OrbitCamera::new(Vec3::ZERO, 5.0);

// 创建窗口和GPU上下文
let (window, gpu) = create_window_and_gpu(event_loop, "My App", 800, 600).await;

// 创建渲染器
let mut renderer = Renderer::new(
    &gpu.context.device,
    &gpu.context.queue,
    gpu.surface_format,
    &scene,
    size.width,
    size.height,
);

// 上传场景数据到GPU（关键步骤！）
renderer.upload_scene(&gpu.context.device, &gpu.context.queue, &scene);

// 渲染循环中...
renderer.render_scene_pass(&device, &queue, view, encoder, &scene, &camera);
```

### 手动创建网格

如果需要手动创建网格，必须提供完整的顶点数据（包括切线）：

```rust
use engvis_core::{Mesh, MeshVertex, SubMesh, Aabb};

let vertices = vec![
    MeshVertex {
        position: [-0.5, -0.5, 0.5],
        normal: [0.0, 0.0, 1.0],
        uv: [0.0, 0.0],
        tangent: [1.0, 0.0, 0.0, 1.0],  // 切线是必需的！
    },
    // ... 更多顶点
];

let indices = vec![
    0, 1, 2, 0, 2, 3,  // 每个面2个三角形
    // ... 更多索引
];

let mesh = Mesh {
    name: "MyMesh".to_string(),
    vertices,
    indices,
    sub_meshes: vec![SubMesh {
        material_index: 0,
        index_offset: 0,
        index_count: indices.len() as u32,
    }],
    aabb: Aabb {
        min: Vec3::new(-0.5, -0.5, -0.5),
        max: Vec3::new(0.5, 0.5, 0.5),
    },
};
```

### 加载glTF模型

```rust
use engvis_renderer::load_gltf;

let scene = load_gltf(
    "path/to/model.gltf",
    &gpu.context.device,
    &gpu.context.queue,
    &mut renderer.texture_cache,
)?;

renderer.upload_scene(&gpu.context.device, &gpu.context.queue, &scene);
```

### 自定义光照

```rust
use engvis_core::{AmbientLight, DirectionalLight, LightingEnvironment};

scene.lighting = LightingEnvironment {
    ambient: AmbientLight {
        color: [0.3, 0.3, 0.3],
        intensity: 0.5,
    },
    directional_lights: vec![
        DirectionalLight {
            direction: Vec3::new(1.0, -1.0, 0.5).normalize(),
            color: [1.0, 1.0, 1.0],
            intensity: 1.0,
        },
    ],
    point_lights: Vec::new(),
};
```

## 技术栈

- **wgpu**: 跨平台GPU抽象层
- **winit**: 窗口管理和事件处理
- **egui**: 即时模式GUI库
- **glam**: 数学库
- **bytemuck**: 零拷贝类型转换

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 贡献

欢迎提交Issue和Pull Request！

## 作者

nil