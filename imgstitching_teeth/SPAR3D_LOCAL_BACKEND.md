# SPAR3D Local Backend

这个方案的目标是：在当前这台 Mac 上再备一条本地 3D 生成后端，不依赖 Hunyuan。

## 适用定位

- 主方案跑不起来时的备案
- 单张牙齿全景图到 `GLB` 的备选路线
- 先追求“本机能跑”，再谈更高质量

## 为什么选 SPAR3D

官方仓库：

- [Stability-AI/stable-point-aware-3d](https://github.com/Stability-AI/stable-point-aware-3d)

这个项目的一个实际优点是：

- 官方明确提到了 Apple Silicon / MPS 的实验支持
- 低内存机器可以直接切 CPU
- CLI 用法相对直接，容易包装成独立本地服务

## 本仓库里已经加好的文件

- 本地服务脚本：[`scripts/spar3d_local_service.py`](scripts/spar3d_local_service.py)
- Mac 一键启动脚本：[`scripts/start_spar3d_local.sh`](scripts/start_spar3d_local.sh)

这套服务故意做成和当前 Hunyuan 类似的 HTTP 协议：

- `GET /health`
- `POST /send`
- `GET /status/{uid}`

也就是说，后续如果你想把前端切到它，不需要重新设计任务轮询协议。

## 默认启动方式

在仓库根目录执行：

```bash
chmod +x scripts/start_spar3d_local.sh
./scripts/start_spar3d_local.sh
```

默认行为：

- 克隆 `stable-point-aware-3d` 到 `third_party/stable-point-aware-3d`
- 建立 `.venv-spar3d`
- 安装依赖
- 默认使用 `cpu`
- 默认端口 `8091`
- 默认打开 `low-vram-mode`

## 常用环境变量

### 1. 明确用 CPU

```bash
export SPAR3D_DEVICE=cpu
./scripts/start_spar3d_local.sh
```

### 2. 如果你想试 MPS

```bash
export SPAR3D_DEVICE=mps
./scripts/start_spar3d_local.sh
```

### 3. 改端口

```bash
export SPAR3D_PORT=8091
./scripts/start_spar3d_local.sh
```

### 4. Hugging Face token

SPAR3D 权重是 gated 的。如果第一次启动报权限错误：

```bash
export HF_TOKEN=hf_xxx
./scripts/start_spar3d_local.sh
```

## 健康检查

```bash
curl http://127.0.0.1:8091/health
```

正常会返回类似：

```json
{
  "status": "ok",
  "message": "SPAR3D local single-image service is ready."
}
```

## 日志和 PID

- 日志：`.runtime/spar3d/logs/service.log`
- PID：`.runtime/spar3d/service.pid`

## 当前实现特点

为了尽量降低接入复杂度，这个服务不是把 SPAR3D 常驻加载在内存里，而是：

- 收到任务
- 调它官方 `run.py`
- 把 `mesh.glb` 收回
- 返回 base64 结果

好处：

- 集成简单
- 出错位置清晰
- 不会把仓库强绑到某个内部 Python API

代价：

- 每次推理都会慢一些
- 不是最高性能实现

但作为“本机备案后端”，这个权衡是合理的。

## 如果以后要正式接进当前页面

现在这套服务协议已经和当前 Hunyuan 客户端兼容。

下一步最自然的接法是：

1. 页面里加一个后端选择器 `Hunyuan / SPAR3D`
2. 保留当前 `/send` 和 `/status` 轮询逻辑
3. 根据后端选择不同服务地址

也就是说，这次已经不是只写文档，而是把“可接前端的本地服务骨架”也搭好了。
