# Windows 顶视 + 侧视抓取 Demo

## 手动硬件控制

新版全英文操作员面板与快捷键说明见仓库根目录的 `README.md`。
新版统一蓝色系，保持宽高比约 4:5 的竖向矩形与四区虚线，小球位于左上区。不做区域命中判断，不显示 IDLE/ZONE/pinch。
CSV 兼容列 `target_key` 固定为 `0`。下文涉及 zone 的说明为旧版实验记录。
请先执行 `python -m pip install -r requirements.txt` 安装新增的 Pillow 字体渲染依赖。
`python -m astra_demo.operator_panel --preview` 可独立预览界面和蓝色球体，无需相机或 BLE。
真实主程序中，`W/A/S/D` 或方向键施加四方向静态力，`Space` 施加法向力，`G` 发出振动提示。
`Esc` / `X` / STOP OUTPUT 关闭输出，`Q` 停止后退出；也支持鼠标点击。法向力不是关闭输出。
右侧 +/- 调节电压、时长与振动频率。默认 0 mV、0.2 s、100 Hz，需主动增加电压才允许输出。
完整参数范围、快捷键、协议方向和停止限制见根目录 README。强度需要实机校准，软件上限不是安全额定值。
原来的抓取自动振动已移除，视觉跟手与抓取日志保留。离线 `--preview` / `--snapshot` 不连接硬件。
下文自动振动及 zone 描述为历史流程，以根目录手动控制说明为准。

本项目现在是**纯视觉 / 纯 UVC 相机**方案，不再依赖 Astra 深度流、OpenNI 运行时或 Orbbec 驱动。

当前结构：
- 顶视：手机摄像头或虚拟摄像头，用于 2x2 四宫格定位与小球显示
- 侧视：任意 UVC 彩色相机，用于拇指-食指捏合判定
- 反馈：BLE 单次震动脉冲
- 日志：每次运行自动生成一份 CSV

## 1. 当前抓取逻辑

抓取判定分两路输入：
- 顶视：只负责 `zone` 判断和小球跟手位置
- 侧视：只负责 `pinch` 判断

状态机：
- `IDLE -> ARMED`：侧视 pinch 连续 `enter_frames` 成立，且顶视手存在
- `ARMED -> GRAB`：再次连续 `enter_frames` 成立
- `GRAB -> IDLE`：pinch 打开连续 `exit_frames` 成立

当前默认参数见 `astra_demo/config.py`：
- `pinch_enter = 0.085`
- `pinch_exit = 0.094`
- `enter_frames = 2`
- `exit_frames = 1`
- `pinch_missing_hold_frames = 4`
- `pinch_median_window = 3`

## 2. Windows 安装步骤

### 步骤 1：安装 Python

建议 Python 3.9-3.11（64 位）。

```powershell
python --version
```

### 步骤 2：克隆仓库

```powershell
git clone https://github.com/tianhangzhu03/astra_demo.git
cd astra_demo
```

### 步骤 3：创建虚拟环境并安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

依赖只有：
- `opencv-python`
- `mediapipe`
- `bleak`
- `numpy`

不需要安装：
- OpenNI
- OrbbecSDK
- Astra 深度驱动

### 步骤 4：确认相机 ID

运行：

```powershell
python -m astra_demo.camera_id_probe
```

操作：
- 按 `0-9` 切换相机
- 按 `q` 退出

然后修改 `astra_demo/config.py`：
- `top_camera_id`：顶视手机 / Camo 相机 ID
- `side_color_camera_id`：侧视彩色相机 ID

如果你使用 iPhone + Camo：
- 顶视通常是 Camo 暴露出来的虚拟摄像头
- 侧视通常是笔记本外接相机或 Astra 的 RGB/UVC 彩色相机

### 步骤 5：配置 BLE（可选）

在 `astra_demo/config.py` 中设置：
- `ble_enabled`
- `ble_mac_address`
- `ble_uuid`
- `ble_fixed_volts`
- `ble_fixed_freq`
- `ble_pulse_ms`

如果只做视觉调试，先设：

```python
ble_enabled = False
```

### 步骤 6：运行

```powershell
python -m astra_demo.main --subject P01
```

按键：
- `q`：退出

## 3. 输出内容

每次运行会自动生成一个 CSV：
- 输出目录：`session_logs/`
- 文件名示例：`grab_log_P01_20260322_214500.csv`

字段：
- `subject_id`
- `run_id`
- `timestamp_iso`
- `timestamp_ms`
- `target_key`
- `pinch_norm`
- `pinch_cm`

其中：

```text
pinch_cm = pinch_norm * pinch_plane_width_cm
```

默认：
- `pinch_plane_width_cm = 62.0`

## 4. 关键参数

主要在 `astra_demo/config.py` 里调整：

抓取判定：
- `pinch_enter`
- `pinch_exit`
- `enter_frames`
- `exit_frames`
- `pinch_missing_hold_frames`
- `pinch_median_window`

手部检测与低延迟：
- `hand_process_scale`
- `side_min_detection_confidence`
- `side_min_tracking_confidence`
- `side_use_clahe`
- `smooth_alpha`

顶视：
- `top_hand_process_scale`
- `top_min_detection_confidence`
- `top_min_tracking_confidence`
- `top_smooth_alpha`
- `top_predict_beta`
- `top_visual_hold_frames`
- `top_grab_hold_frames`
- `hover_key_hold_frames`
- `grid_w_ratio`
- `grid_h_ratio`
- `grid_y_ratio`

虚拟小球：
- `prop_size_follow`
- `prop_size_held`
- `prop_follow_alpha`
- `prop_idle_visible`

实验记录：
- `pinch_plane_width_cm`
- `default_subject_id`
- `session_log_dir`

## 5. 当前默认行为

- 不再使用深度相机做任何判定
- 不再显示深度窗口
- 不再要求 OpenNI / Orbbec 驱动
- 顶视画面只显示四宫格和虚拟小球
- 小球抓住前后不再改变大小
- 小球位置锚定在顶视拇指-食指 pinch 位置，而不是掌心
- 抓取成功时只发固定一组 BLE 震动脉冲

## 6. 自检命令

语法检查：

```powershell
python -m py_compile astra_demo/config.py astra_demo/main.py astra_demo/grab_logic.py
```

单元测试：

```powershell
python -B -m unittest astra_demo.tests.test_grab_logic
```

## 7. 目录说明

- `astra_demo/main.py`：主入口
- `astra_demo/grab_logic.py`：抓取状态机
- `astra_demo/virtual_prop.py`：虚拟小球状态与渲染
- `astra_demo/ble_client.py`：BLE 后台线程与发包
- `astra_demo/session_logger.py`：每次运行的抓取日志
- `astra_demo/config.py`：参数配置
- `astra_demo/camera_id_probe.py`：Windows 相机 ID 探测
- `astra_demo/tests/test_grab_logic.py`：核心逻辑单测
