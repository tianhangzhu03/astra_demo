# Physical Object Pilot

用 **AirPods 耳机盒**做实体道具的视觉试验：看真实接触、遮挡和搬运是否自然，保留 WASD/G 的 BLE 提示，并记录录像和人工抓放事件。Top／Side 各自显示紫色**二维掌部轨迹**，不检测物体、不做 3D 重建、读取深度或估计球心，也不自动判定成功。

这是独立的实验入口。代码全部在 `physical_demo/`，不 import `astra_demo`，不使用其配置、抓取状态机或日志目录。原来的 `Start_Visual_Demo.cmd` 和轻量虚拟球代码保持原样。CEL_GUI transport 是原工作区通信代码的独立快照；之后两边的修改不会相互传播。

## 启动

先退出原 demo，释放相机及 BLE，再双击仓库根目录 **`Start_Physical_Demo.cmd`**。默认 Top/Camo=2、Side/Astra RGB=3，BLE 开启，设定电压从 **0 mV** 开始；实际硬件会拒绝 0 mV 的输出请求。

```powershell
# 在仓库根目录运行；等价于启动器
python -m physical_demo.main --top-camera 2 --side-camera 3

# 同一实体版，只试画面和人工标记，不扫描或连接 BLE
python -m physical_demo.main --no-ble --top-camera 2 --side-camera 3

# 为本次物体补充说明；不要求尺寸或球形假设
python -m physical_demo.main --object-name "AirPods case" --object-notes "Closed case, empty"
```

默认 BLE address/UUID 沿用已配置的设备，独立保存在 `physical_demo/config.py`，也可通过 `--ble-address` / `--ble-uuid` 指定。插拔相机后如编号变化，修改命令参数；不会自动改用别的编号。默认 Windows capture backend 为 MSMF，可显式指定 `--backend dshow`。

沿用当前 Python 环境即可。代码使用 OpenCV、NumPy、Pillow、Bleak 和 MediaPipe；新建独立环境时可安装本目录的 `requirements.txt`。现有环境和原版依赖没有更改。掌部模型独立保存在 `physical_demo/models/hand_landmarker.task`，不读取原 demo 的模型路径；模型文件不入 Git。模型缺失时使用 `--hand-model` 指定本机副本，或按 [MediaPipe 官方说明](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker?tab=Python) 获取 Hand Landmarker 模型。`--no-hand-trace` 可关闭二维轨迹；模型失败不会停止相机录像或触发 BLE。

## 一轮操作

把耳机盒放在 Top 左上 PICKUP 附近，手从右下 START 出发。蓝线只作静态参考；**L** 可以隐藏。紫线是各画面中的实际掌部移动，紫点是当前跟踪位置。N／0 会清空显示轨迹，保留历史 CSV；短暂缺失最多由验证过的 Optical Flow 衔接 350 ms，失去有效位置后断开轨迹，不把遮挡期间的位置编造为测量。两路轨迹都只是像素坐标，Side 的上下变化也不是毫米高度。

两路画面到达后，自动开始三路连续录像。按 **N** 开始 trial；不需要识别手或确认物体位置。第一条成功写入 BLE 的方向指令开始计时；纯视觉模式以第一条模拟方向指令计时，并明确标为 `simulated_direction`。

操作员用 **W / A** 引导到道具上方，按 **Space / LOWER** 提示下降，再用 **G** 提示抓取，实际抓住时按 **F** 标记。下一次 **Space / LIFT** 提示抬起；用 **D / S** 引导返回后，再按 **Space / LOWER** 提示下降放置，实际放下时按 **R** 标记。然后 **E** 结束 trial，或直接 **N** 结束已标记放下的一轮并开始下一轮。**Q** 关闭全部录像并退出。

**实体版 Space 现在表示上下移动提示；X / Esc 才是停止硬件输出。** LOWER 和 LIFT 都发送原有 **Static Normal、direction=0** 的同一个脉冲，参与者通过任务阶段理解含义，不需要辨别两种极性。界面显示 `Next: LOWER/LIFT`；只有收到 `command_sent` 才交替含义，no-BLE 用明确标注的模拟发送推进。拒绝、断线或发送前取消都不推进。**Tab** 仅切换下一次含义，可纠正误按，不发送输出。每个新 trial 默认 LOWER；F/R 不改变 Normal 阶段，也不发确认振动。

Space 不再用于实体版每个转角的“停手”。转角暂停先由操作员口头提示；停止硬件输出本身不等于参与者停止移动。原轻量 demo 的按键含义保持原样。

| 操作 | 行为 |
| --- | --- |
| W / A / S / D、方向键 | Forward / Left / Back / Right；busy 或断线时拒绝，不积压等待发送 |
| Space | 按 Next 标签发送 LOWER 或 LIFT 提示；二者均为单次 Normal mode 0 |
| Tab | 无输出地切换下一次 Normal 含义 |
| G | 单次抓取提示振动；可重复手动提示 |
| F / R | 人工标记 grasp / release，**不发送任何 BLE 指令或自动确认振动** |
| Z | 撤销本 trial 最后一个有效人工标记，原事件仍保留 |
| N | 两路画面新鲜、三路录像就绪后开始 trial；不是任务计时起点 |
| E | 结束当前 trial 并请求停止输出；三路录像继续 |
| 0 | 当前 trial 记为 reset，取消待发送输出；手动把道具放回，再按 N |
| B | 同时暂停／恢复三路录像；暂停会结束当前 trial 并请求停止输出 |
| L | 显示／隐藏控制台 Top 静态路线提示 |
| X / Esc | 请求停止输出，取消尚未发送的指令 |
| Q | 停止输出，封存视频、时间戳和日志后退出 |
| 1 / 2 | 设定电压 − / + 25 mV |
| 3 / 4 | 设定时长 − / + 0.1 s |
| 5 / 6 | 振动频率 − / + 10 Hz |

F/R 记录的是操作员观察，不会改变真实物体或自动触发触觉。如果漏标 F，仍允许 R，并将该轮标为 `grasp_not_marked`。没有方向指令时不计算任务时长。手被道具遮住不会引起软件“掉球”，因为本版本没有虚拟抓取状态机。

## 数据与录像

每次运行在 **`physical_demo/runs/physical_<subject>_<UTC时间>/`** 建立独立 session，不写入旧 `session_logs/`。`--output` 可指定其他目录；程序拒绝原 demo 的代码、日志、output 和 calibration 目录。

| 文件 | 内容 |
| --- | --- |
| `session.json` | 物体名称／备注、相机编号、画面镜像方式、BLE 设置、时钟说明、各录像片段及最终写入统计 |
| `events.csv` | 操作员动作、人工标记／撤销、BLE 请求／拒绝／成功写入／结束／断线、录像与相机状态；即时 flush |
| `trials.json` | 每轮人工标记及撤销记录、显示帧证据、时间、缺失项、所属录像片段 |
| `trials.csv` | 每轮摘要，包括首个方向到人工放下的时间、人工标记的持有时长；无几何误差或成功率 |
| `hand_trace.csv` | 每路二维掌部的原始检测坐标、显示平滑坐标、来源帧／时间、landmark／Optical Flow／缺失状态、分段编号；无 XYZ |
| `videos/segment_001/top.mp4` | 不叠加绘制内容、不镜像的 Top 相机视频 |
| `videos/segment_001/side.mp4` | 不叠加绘制内容、不镜像的 Side 相机视频 |
| `videos/segment_001/operator.mp4` | 完整控制台：两路画面、可选路线、设置、人工标记、触觉状态；Side 默认仅在此镜像 |
| 对应的 `.csv` | 每个编码帧的 session 时间、来源帧编号与收帧时间、重复标识、所属 trial、写入时间和丢弃的待处理更新数 |

相机 MP4 是未经叠加的**有损编码视频，不是 sensor RAW**。控制台录像不包含桌面、鼠标或窗口标题栏。需要改变 Side 显示方向可加 `--side-no-mirror`；原始相机文件始终保持采集方向。

N、E 和 0 不切断三路视频；B 恢复会新建 `segment_002`，不会覆盖上一段。每段三路视频共享起点和名义 30 fps，但相机不是硬件同步曝光，UI 也不是每秒 30 次独立刷新。

对齐以 CSV 的 **session 时间**为准：`video_t_s` 是编码槽时间，`capture_t_s` 是来源采集／UI 渲染完成时间。`operator.csv` 另有 `top_frame_id` / `side_frame_id` 和各自 `*_capture_t_s`，指向画面中的原始相机帧。事件的 `details_json.evidence` 保存按键时最近显示的帧；人工反应延迟没有校正。`trial_id` 跟随来源样本，精确的 trial 边界以事件时间为准。

录像线程只选择编码时刻之前的来源帧；采集慢或中断时重复上一帧，并明确标记。待处理来源帧队列有上限，超出的更新数量记录在 `dropped_updates`；帧编号的跳跃也可用于检查抽样。相机超过 1 s 没有新帧会在控制台标为 STALE，禁止开启新 trial。编码异常或分辨率变化会停止本组录像、结束当前 trial 并请求停止输出。

`command_sent` 只代表 BLE write 成功，不代表实测电流、接触力或执行器响应。Normal 命令通过 `role=normal_lower/normal_lift` 记录语义，wire packet 相同；`normal_intent_selected/advanced` 记录人工切换和发送后的交替。没有实际施力方向或高度测量。人工标记不会自动生成 grasp confirmation；本版本保留 G 的提示脉冲。

## 离线验证与代码入口

```powershell
python -m unittest discover -s physical_demo/tests -q
python -m physical_demo.main --preview
python -m physical_demo.main --smoke
```

`--preview` 只原位更新 `physical_demo/output/operator_preview.png`。`--smoke` 强制使用带标识的 synthetic 输入和 no-BLE，在约 3 s 内生成三路录像及一轮人工标记；这些数据的 `synthetic=true`，不能当作实机实验。上述入口均不连接真实相机或 BLE。

`main.py` 负责试验流程，`panel.py` 负责控制台，`session.py` 负责事件和人工观察，`recording.py` / `cameras.py` 负责独立录像／采集线程，`haptics.py` 是独立 CEL_GUI transport。`hand_trace.py` 在单独 worker 中检测两路掌部，`palm_motion.py` 使用原跟踪算法的独立副本。没有深度或原 demo 的运行时依赖；启动不会自动下载模型。

当前已完成离线协议／流程检查和合成视频解码验证。AirPods 实物的遮挡、画面效果、真实相机负载和 BLE 联动仍需现场验收。
