# engvis

工程可视化框架 - 基于Rust和wgpu的高性能3D工程可视化工具

## 项目简介

engvis是一个用Rust编写的工程可视化框架，专为工程应用设计。它提供了完整的3D渲染管线，支持glTF模型加载、PBR材质渲染、灵活的相机控制和丰富的UI界面。

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
git clone https://github.com/yourusername/engvis.git
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

### 示例代码

```rust
use engvis_core::{OrbitCamera, Scene, SceneNode, Mesh, PbrMaterial};
use engvis_renderer::{Renderer, create_window_and_gpu};
use glam::{Vec3, Affine3A};

// 创建场景
let mut scene = Scene::default();

// 添加网格和材质
scene.meshes.push(your_mesh);
scene.materials.push(PbrMaterial::default());

// 添加场景节点
scene.nodes.push(SceneNode {
    name: "MyObject".to_string(),
    local_transform: Affine3A::IDENTITY,
    mesh_index: Some(0),
    children: Vec::new(),
    visible: true,
});

// 创建相机
let camera = OrbitCamera::new(Vec3::ZERO, 5.0);

// 渲染
// ... 参考examples目录中的完整示例
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

nil (nil_lilin@163.com)