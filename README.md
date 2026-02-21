# Astra Pro Plus 侧视 + 手机顶视 Demo（Windows）

本目录是**独立实现**，不会改动你原来的双摄 Demo。

- 目标设备：
  - 侧视：Astra Pro Plus（Depth 走 OpenNI）+ 侧视 UVC 彩色相机（负责手势图像）
- 顶视：手机摄像头（仅用于 3x3 九宫格测试）
  - 当前默认已改为 2x2 四宫格
- 目标系统：Windows 10/11 x64
- 抓取逻辑：捏合阈值 + 深度阈值 + 防抖状态机（IDLE->ARMED->GRAB）
- 交互增强：虚拟小球/小方块（`V` 切换）
- BLE：保留，且支持配置开关

---

## 1. 与 OrbbecSDK 的兼容结论（先看）

结合 Orbbec 官方仓库信息：

- [OrbbecSDK(v1)](https://github.com/orbbec/OrbbecSDK) 中，Astra Pro Plus 归在 OpenNI 协议设备驱动范围内。
- [pyorbbecsdk](https://github.com/orbbec/pyorbbecsdk) 的设备支持表中，Astra Pro Plus 在 `main(v1.x)` 为 `limited maintenance`，在 `v2-main` 标注 `not supported`。

因此本项目默认走 **OpenNI 采集链路**（`from openni import openni2`），对 Astra Pro Plus 是合理路线。

---

## 2. Windows 一步步操作（从零开始）

### 步骤 1：安装 Python

建议 Python 3.9-3.11（64 位）。

```powershell
python --version
```

### 步骤 2：准备项目

```powershell
git clone https://github.com/tianhangzhu03/astra_demo.git
cd astra_demo
```

### 步骤 3：创建虚拟环境并安装基础依赖

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install opencv-python mediapipe bleak numpy
```

### 步骤 4：安装 Orbbec OpenNI 运行时/驱动

1. 按 [OrbbecSDK](https://github.com/orbbec/OrbbecSDK) 文档安装 Windows 驱动（Astra Pro Plus 属于 OpenNI 协议设备）。
2. 安装与该运行时匹配的 Python OpenNI 绑定（要求代码中可执行 `from openni import openni2`）。

### 步骤 4.1：Beta OpenNI 版本（当前项目实测）

如果你使用的是这个 Beta 包：

`OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows_x64`

且 `OpenNI2.dll` 位于：

`...\OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows\samples\bin\OpenNI2.dll`

则 `OPENNI2_REDIST` 应设置为 **bin 目录**（不是 dll 文件本身）：

`...\OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows\samples\bin`

PowerShell 临时设置（当前终端）：

```powershell
$env:OPENNI2_REDIST="C:\Users\Benjamin_Zhu\Downloads\OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows_x64\OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows\samples\bin"
$env:PATH="$env:OPENNI2_REDIST;$env:PATH"
```

PowerShell 持久设置（用户级）：

```powershell
[Environment]::SetEnvironmentVariable("OPENNI2_REDIST","C:\Users\Benjamin_Zhu\Downloads\OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows_x64\OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows\samples\bin","User")
$userPath=[Environment]::GetEnvironmentVariable("Path","User")
if ($userPath -notlike "*OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows*samples\\bin*") {
  [Environment]::SetEnvironmentVariable("Path","C:\Users\Benjamin_Zhu\Downloads\OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows_x64\OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows\samples\bin;$userPath","User")
}
```

设置后请重开终端，再验证：

```powershell
echo $env:OPENNI2_REDIST
Test-Path "$env:OPENNI2_REDIST\OpenNI2.dll"
python -c "from openni import openni2; openni2.initialize(); d=openni2.Device.open_any(); print('openni device ok')"
```

验证命令：

```powershell
python -c "from openni import openni2; print('openni2 ok')"
```

如果这条命令失败，先不要运行 Demo，先修复 OpenNI 安装与 PATH。

### 步骤 5：配置参数（BLE + 顶视相机）

编辑 `astra_demo/config.py`：

- `top_camera_id`（手机摄像头 ID）
- `side_color_camera_id`（侧视 UVC 彩色相机 ID）
- `ble_enabled = True/False`
- `ble_mac_address`
- `ble_uuid`

调试视觉流程时建议先设 `ble_enabled=False`。

可先运行相机探测脚本辅助确定 ID：

```powershell
python -m astra_demo.camera_id_probe
```

操作说明：
- 终端会打印可用 ID 列表
- 预览窗口中按 `0-9` 切换相机
- 按 `q` 退出

### 步骤 6：运行

在仓库根目录执行：

```powershell
python -m astra_demo.main
```

按键：

- `q`：退出
- `v`：切换虚拟物体（BALL/CUBE）

---

## 3. 参数调优（关键）

在 `astra_demo/config.py` 调：

- 捏合：`pinch_enter` / `pinch_exit`
- 深度：`depth_enter_mm` / `depth_exit_mm`
- 防抖：`enter_frames` / `exit_frames`
- 侧面手识别：`side_min_detection_confidence` / `side_min_tracking_confidence` / `side_use_clahe`
- 低延迟：`hand_process_scale` / `smooth_alpha` / `depth_view_width` / `depth_view_height`
- 九宫格：`grid_w_ratio` / `grid_h_ratio` / `grid_y_ratio`
- 深度预览范围：`depth_vis_min_mm` / `depth_vis_max_mm`
- 虚拟物体：`prop_size_follow` / `prop_size_held` / `prop_follow_alpha`

当前仓库默认值已切为 Demo 预设（适合快速演示）：
- 顶部九宫格更大：`grid_w_ratio=0.84`, `grid_h_ratio=0.78`
- 深度门限降权：`depth_enter_mm=760`, `depth_exit_mm=860`
- 虚拟物体更大：`prop_size_follow=42`, `prop_size_held=56`
- 深度可视化为独立清晰窗口：`Depth Range View`
- 默认显示范围已放宽（更容易看清手部）：`depth_vis_min_mm=180`, `depth_vis_max_mm=1200`
- 默认深度不参与抓取判定：`use_depth_gate=False`（仅显示）
- 侧面识别增强默认开启：`side_use_clahe=True`
- 低延迟预设默认开启：`hand_process_scale=0.75`, `smooth_alpha=0.55`
- 虚拟物体默认常驻，初始化为侧面普通位置（不固定九宫格）

判定逻辑（侧视）：

1. 先算捏合距离 `pinch_dist`  
2. 再取侧视中点附近 `5x5` 深度中值 `depth_mm`  
3. 用迟滞阈值更新两个门：
   - `top_pinch_state`：进入用 `pinch_enter`，退出用 `pinch_exit`
   - `depth_gate_state`：进入用 `depth_enter_mm`，退出用 `depth_exit_mm`
4. 状态机：
   - `IDLE -> ARMED`：捏合成立且顶部九宫格命中连续 `enter_frames`
   - `ARMED -> GRAB`：深度门成立连续 `enter_frames`
   - `GRAB -> IDLE`：捏合或深度释放连续 `exit_frames`
5. `GRAB` 时触发 BLE 上升沿，释放时触发下降沿

调参建议（按顺序）：

- 当前深度门逻辑是“距离越近 mm 越小”，满足 `depth <= threshold` 才认为通过。
- 误触发多：先提高 `enter_frames`，再减小 `depth_enter_mm`。
- 释放不及时：先减小 `exit_frames`，再减小 `depth_exit_mm`。
- 捏合漏检：适当增大 `pinch_enter`/`pinch_exit`（保持 `enter < exit`）。
- 深度图过黑/过白：调整 `depth_vis_min_mm`/`depth_vis_max_mm`，让手部区域在彩色预览里有明显对比。

---

## 4. 运行自检与测试

### 语法检查

```powershell
python -m compileall astra_demo
```

### 逻辑单测

```powershell
python -m unittest astra_demo.tests.test_grab_logic
```

---

## 5. 常见问题

1. `ImportError: openni`  
   说明 OpenNI Python 绑定或运行时未装好。

2. 能看到 RGB，看不到深度  
   多数是驱动/流配置问题，先用 Orbbec 官方工具确认深度流可用。

3. BLE 卡住主流程  
   视觉调试阶段直接把 `ble_enabled=False`。

4. 画面有但抓不到  
   先看窗口中的 `PinchDist` 与 `Depth(mm)`，再调阈值。

---

## 6. 目录说明

- `astra_demo/main.py`：主入口
- `astra_demo/camera_source.py`：Astra OpenNI 采集（`start/read/stop`）
- `astra_demo/grab_logic.py`：抓取状态机与 5x5 深度中值采样
- `astra_demo/virtual_prop.py`：虚拟小球/小方块状态与渲染
- `astra_demo/ble_client.py`：BLE 后台线程与边沿触发发包
- `astra_demo/config.py`：参数与硬件配置
- `astra_demo/tests/test_grab_logic.py`：核心逻辑单元测试
