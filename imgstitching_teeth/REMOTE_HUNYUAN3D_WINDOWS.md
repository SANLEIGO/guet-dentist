# Windows Remote Hunyuan3D

这份说明是给下面这套网络拓扑准备的：

- 当前这台 Mac / 本机：`100.98.144.100`
- 远程 Windows 主机：`100.69.194.152`

目标是把 Hunyuan3D 的推理搬到 Windows 主机上跑，再让当前仓库通过 HTTP 调用远程服务。

## 先说结论

你的 Windows 主机是：

- `RTX 2060`
- `6GB VRAM`
- `32GB RAM`

这个配置 **不建议默认跑 `Hunyuan3D-2.1`**。我们已经在本地 16GB 统一内存机器上验证过，`2.1` 在模型加载阶段就很容易撑爆资源。

所以：

- 想优先跑通：推荐 `Hunyuan3D-2mini-Turbo`
- 想用 CPU 跑：也推荐先配 `2mini-Turbo`
- 想硬试 `2.1`：可以，但大概率会 OOM、超慢，或者直接卡死

## 你需要准备什么

把当前整个仓库放到 Windows 主机上，假设路径是：

```powershell
D:\imgstitching_teeth
```

要求：

1. 安装 `Python 3.10` 或 `3.11`
2. 安装 `Git`
3. 显卡驱动支持较新的 CUDA 运行时
4. Windows 能访问 Hugging Face 下载模型

## 一键启动脚本

仓库里已经加好了脚本：

[`scripts/windows/Start-Hunyuan3D-Remote.ps1`](scripts/windows/Start-Hunyuan3D-Remote.ps1)

### 你现在要的 CPU 跑法

仓库里已经加好了 CPU 包装脚本：

[`scripts/windows/Start-Hunyuan3D-Remote-CPU.ps1`](scripts/windows/Start-Hunyuan3D-Remote-CPU.ps1)

如果你是双击启动，优先用这个不会闪退的命令包装器：

[`scripts/windows/Start-Hunyuan3D-Remote-CPU.cmd`](scripts/windows/Start-Hunyuan3D-Remote-CPU.cmd)

在 Windows PowerShell 里进入仓库根目录后运行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\windows\Start-Hunyuan3D-Remote-CPU.ps1 -ModelPreset 2mini-turbo -BindHost 0.0.0.0 -Port 8081
```

如果你不想开 PowerShell，直接双击：

```text
scripts\windows\Start-Hunyuan3D-Remote-CPU.cmd
```

它会：

- 用 PowerShell 调起 CPU 脚本
- 保留窗口
- 在失败时把退出码留在屏幕上
- 最后自动 `pause`

这会自动转调主脚本，并固定使用：

- `DevicePreset=cpu`
- `PyTorch CPU wheels`
- `--device cpu`

### 如果你想手动控制 CPU / CUDA

在 Windows PowerShell 里进入仓库根目录后运行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\windows\Start-Hunyuan3D-Remote.ps1 -ModelPreset 2mini-turbo -DevicePreset cpu -BindHost 0.0.0.0 -Port 8081
```

这个脚本会做几件事：

1. 创建 `.venv-hunyuan`
2. 安装 PyTorch CUDA 轮子
3. 安装运行依赖
4. 安装 vendored 的 `third_party/Hunyuan3D-2`
5. 下载对应模型
6. 在后台启动服务

后台日志默认写到：

```text
.runtime\hunyuan3d\logs\service.windows.log
```

### 如果你非要试 2.1 的 CPU 版

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\windows\Start-Hunyuan3D-Remote-CPU.ps1 -ModelPreset 2.1 -BindHost 0.0.0.0 -Port 8081
```

但再次强调：这个更像“测试能不能跑”，不是我推荐的实用方案。

## Mac 这一侧怎么改

在当前这台 Mac 仓库根目录新建或修改 `.env`：

### 如果远程跑 2mini-Turbo

```env
HUNYUAN3D_SERVICE_URL=http://100.69.194.152:8081
HUNYUAN3D_SERVICE_MODE=single_image
HUNYUAN3D_MODEL_PATH=tencent/Hunyuan3D-2mini
HUNYUAN3D_SUBFOLDER=hunyuan3d-dit-v2-mini-turbo
HUNYUAN3D_DEVICE=auto
HUNYUAN3D_REQUEST_TIMEOUT_SEC=20
HUNYUAN3D_POLL_INTERVAL_SEC=3
```

### 如果远程跑 2.1

```env
HUNYUAN3D_SERVICE_URL=http://100.69.194.152:8081
HUNYUAN3D_SERVICE_MODE=single_image
HUNYUAN3D_MODEL_PATH=tencent/Hunyuan3D-2.1
HUNYUAN3D_SUBFOLDER=hunyuan3d-dit-v2-1
HUNYUAN3D_DEVICE=auto
HUNYUAN3D_REQUEST_TIMEOUT_SEC=20
HUNYUAN3D_POLL_INTERVAL_SEC=3
```

## 怎么检查远程服务起来了

在 Windows 主机本机执行：

```powershell
curl http://127.0.0.1:8081/health
```

在当前 Mac 上执行：

```bash
curl http://100.69.194.152:8081/health
```

如果通了，应该能看到类似：

```json
{"status":"ok","message":"Hunyuan3D single-image bridge is ready."}
```

## 常见问题

### 1. PowerShell 不让执行脚本

先运行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

### 2. 端口不通

确认：

1. Windows 防火墙放行 `8081`
2. 服务是用 `-Host 0.0.0.0` 启动的
3. 两台机器之间确实能互相访问

### 3. CPU 太慢或者 2.1 起不来

先别纠结，直接切回：

```powershell
.\scripts\windows\Start-Hunyuan3D-Remote-CPU.ps1 -ModelPreset 2mini-turbo -BindHost 0.0.0.0 -Port 8081
```

### 4. 想改成 CUDA 跑

把主脚本加上 `-DevicePreset cuda`：

```powershell
.\scripts\windows\Start-Hunyuan3D-Remote.ps1 -ModelPreset 2mini-turbo -DevicePreset cuda -BindHost 0.0.0.0 -Port 8081
```

### 5. 想前台看日志

可以加 `-Foreground`：

```powershell
.\scripts\windows\Start-Hunyuan3D-Remote-CPU.ps1 -ModelPreset 2mini-turbo -BindHost 0.0.0.0 -Port 8081 -Foreground
```

这样不会后台化，日志会直接在当前窗口打印。
