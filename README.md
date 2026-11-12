# Windows 顶视 + 侧视抓取 Demo

## 手动硬件控制（2026-11）

新增全英文 `Haptic Guidance | Operator` 面板：左侧顶视画面，右侧侧视预览与手动指令按钮。
界面统一蓝色系，不显示 IDLE、ZONE 或 pinch 调试数值。
工作平面使用宽高比约 3:2 的横向矩形（X 轴长、Y 轴短）与四区虚线，不执行区域命中计算。蓝色小球初始位于左上区，选中状态同时使用文字和边框提示。旧 CSV 的 `target_key` 列保留兼容，
新运行写入 `0`，不再代表目标区域。下文的四宫格实验说明仅适用于旧版。
**主程序的按钮和键盘现已发送真实 BLE 指令。原来的抓取上升沿自动振动已关闭。**
视觉抓取、小球跟手和抓取 CSV 保留；按 G 是向参与者发出振动提示，不是强制把视觉状态设为 GRAB。

| 按钮 / 键盘 | 硬件指令 |
| --- | --- |
| Forward / W / 上箭头 | Static, Up (direction=1) |
| Back / S / 下箭头 | Static, Down (direction=2) |
| Left / A / 左箭头 | Static, Left (direction=3) |
| Right / D / 右箭头 | Static, Right (direction=4) |
| Normal force / Space | Static, Normal (direction=0)，用于提示参与者停手 |
| Vibration / G | Vibration, Normal，提示开始抓取 |
| STOP OUTPUT / Esc / X | 关闭输出，不是施加法向力 |
| Exit / Q | 尝试停止输出，然后退出 |

键盘需要 OpenCV 窗口获得焦点；Windows 下按住方向/振动键不会连续重触发，松开后再按。
**Up/Down 是协议方向，不保证等于桌面前后；需按磁铁安装朝向实测确认映射。**

### 手动参数

界面右侧 +/- 按钮可调节，下列快捷键也可用。参数作用于下一次指令，不改变正在执行的脉冲。

| 参数 | 初始值 | 范围 / 步长 | 减 / 加 |
| --- | --- | --- | --- |
| Voltage | 0 mV | 0–5000 mV / 100 mV | `[` / `]` |
| Duration | 0.2 s | 0.1–5.0 s / 0.1 s | `-` / `=` |
| Vibration frequency | 100 Hz | 10–200 Hz / 10 Hz | `,` / `.` |

0 mV 时拒绝执行力/振动指令，需由操作者主动提高电压。上限只是软件范围，不代表装置的安全工作范围。
Static 力指令的频率字段固定为 0；频率调节只影响振动。初始值在 `astra_demo/config.py` 的
`manual_voltage_mv`、`manual_duration_s`、`manual_frequency_hz` 中修改；旧 `ble_fixed_*` 和 `ble_pulse_ms` 不再驱动主程序输出。

### BLE 和停止机制

沿用 CEL_GUI 的 9 字节协议、毫伏单位和小端字节序。目标配置仍是 `ble_mac_address`、`ble_uuid`、`ble_enabled`，
默认特征为 `120062c4-b99e-4141-9439-c4f9db977899`，须具有 write 或 write-without-response 属性。
后台线程串行发送，主视觉循环不等待 BLE。一次仅执行一个脉冲，忙时的新动作丢弃，不排队；停止命令优先清除未发送动作。
每次连接先停止输出，再允许新动作；断线重连不重放旧指令。界面区分 Requested、Sent、错误状态，Sent 仅表示写入成功，不证明实际产生了力。

时长由电脑计时，并非写入硬件的数据包：结束时先发 Static Normal 0 mV，约 100 ms 后发 `FF 06 00 00 00 00 00 00 00`。
所以时长是发送零输出前的目标时长，不是已校准的机械力持续时间，通信和释放过程会增加偏差。
退出时也尝试同样的停止流程。**断电、电脑崩溃或 BLE 断线时不能保证停止包送达；硬件是否有超时保护需确认，实测须保留物理断电手段。**
不要同时运行同事的 CEL_GUI 和本程序连接同一设备。

字体使用系统 TrueType 字体并缓存文字贴图；Windows 优先使用 Bahnschrift。
先更新依赖以安装新增的 Pillow，再预览布局和球体（无需相机、BLE 或 MediaPipe）：

```powershell
python -m pip install -r requirements.txt
python -m astra_demo.operator_panel --preview
```

Windows 双摄像头与真实硬件运行（会尝试连接 BLE，默认电压为 0）：

```powershell
python -m astra_demo.main --subject P01
```

默认显示操作员面板和干净的顶视窗口；侧视画面已嵌入面板，不再另开第三个窗口。
`show_operator_window` 控制面板显示，`operator_panel_fps=20` 限制面板合成频率，
`show_side_color_window` 可开启独立侧视调试窗口。检测仍在每轮主循环执行。

`--preview` 和 `--snapshot` 始终离线，不创建 BLE 控制器、不连接设备。下文保留的自动振动、zone 配置说明属于历史实验流程，以本节手动控制说明为准。

小球采用预计算的球面光照、柔和高光和局部阴影，逐帧仅做局部透明合成。
抓取前后半径保持为 44 px；界面不新增实验数据记录，预览选择不会写入抓取日志。

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
