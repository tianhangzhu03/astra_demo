# Windows 顶视 + 侧视抓取 Demo

## 当前项目：A → B → C 双视角 AR

2026-11-15：天蓝色虚拟障碍、橙红色球和开口向下的 box；Side 去掉障碍，两路去掉实际手轨迹，Operator Top 参考路线为虚线。G 人工抓放，不再要求 Pinch 或严格接近判定。WASD / I、K 单击选择方向，每秒一个 Static 短脉冲，X/Stop 停止；电压从 2500 mV 起，每次调整 500 mV。新版 BLE 核对 CEL v0.2.1，CW=5、CCW=6、STOP=FF06；J/L 请求90°翻手和反向返回，无需F二次确认。

`Start_AR_Demo.cmd` 默认沿用 Top/iPhone 与 Side/Astra RGB，不打开 depth，不读取标定文件。黄色食指标带独立记录：Top XY和Side图像上下Z代理全程连续显示，各视角每个trial固定零点；Shear/Normal只标记阶段，不屏蔽轴、不重新归零。两路原始像素持续保存；漏检只影响该视角并留空，AR/G/BLE照常工作。路线和障碍缩小20%，Top球直径放大25%；Side球为75%不透明度，box为35%。

无需图纸或徒手方向标定。曲线单位a.u.（对应视角画面高度的比例），不是世界XYZ或毫米；Top/Side幅值不能直接作为同一物理尺度比较。相机和缩放保持固定，试录确认方向即可。`--no-ble`禁用硬件，Side P/O继续手动放球/box。

操作和记录定义见 **[ABC.md](astra_demo/ABC.md)**。`python -m astra_demo.phase_check` 仅保留当前 G/J/L 任务流程的短模拟检查，无真实硬件。旧 depth、stereo、affine 标定和相关测试代码已删除；每次修改只做相关的必要检查。黄色胶带搜索限制在食指 DIP–TIP 末节区域。

## 轻量双视角：保留的 baseline 入口

双击根目录 **`Start_Visual_Demo.cmd`**。它使用本机目前的 Top/Camo=2、Side/Astra RGB=3，开启 `--no-ble --linked-views --grasp-mode pinch`，不读取深度、不加载标定文件。若程序已运行，直接使用现有窗口，不要重复启动占用相机。

```powershell
python -m astra_demo.main --no-ble --linked-views --grasp-mode pinch --top-camera 2 --side-camera 3 --subject visual_linked
```

1. 接好 Astra 与手机，电脑打开 Camo Studio，iPhone 解锁并打开 Camo Camera。让两路看见同一只手和实际操作区域。Top View 若显示相机/暂停图标，先恢复手机画面。
2. 将半张开的手放到 Top View 蓝球所在的实际位置。**在 Side View 中点击手指准备抓球的位置一次**，放置侧视蓝球；独立 Side 窗口和控制台的 Side 缩略图都支持点选，最大化/留白不影响坐标映射。右下 START 手形仅提示演示时的建议起始位置，不作启动确认。每次新 trial 会复用这个侧面起始位置；启动新 session 或移动相机后再点一次。正在抓球或 trial 进行中不能改位置。
3. 两路视频正常、**Side View 能识别到手掌和手指**时，按 **N**：开始 trial，并自动启动两路独立 Display 录像。不检查手是否在 START，也不要求 Top View 此刻识别到手或手指处于半张开状态。**W** 向前，靠近 TURN 时 **Space** 停住；**A** 向左，到球旁再 **Space** 停住。
4. 按 **G** 提示抓取，在球附近先稍张开拇指与食指，看到 **PINCH TO GRAB** 后自然握拢，让拇指与食指靠近。其他手指可随之弯曲。默认恢复 Side pinch 判定，Top 只判断当前掌部区域是否到球附近，不要求顶视指尖可靠。两路球同时变成抓取色。位置缺失时保留最后位置，抓取不会自行解除；两路共用一个抓取状态。
5. 保持抓握，按 **D** 提示向右，转弯前 **Space**；按 **S** 提示返回，到 DROP 后 **Space**。放低手并明确张开至少约 0.35 秒：两路同时释放、恢复蓝色，各自在放开位置保留。纯视觉模式的方向键只是屏幕提示，不会主动移动手或驱动硬件。
6. 放下后录像仍继续；**B** 停止两路录像，**Q** 保存并生成报告后退出。下一轮按 **N**。随时按 **数字 0 / Restart** 可以一键重来：停止输出、取消抓取提示和待发送的确认、松开虚拟球、清空当前显示轨迹，两路小球回到各自起始位置，并开启新 trial 和两段独立录像。相机连接、Side 放置位置和控制参数保留；看不到 Side 手掌/手指或相机未就绪时自动等待，恢复后继续，不用再次按 N。等待时 **Esc/X** 可取消重启。新一轮仍由第一条方向指令开始计时、必须重新按 G 才能抓取。中途终止用 **E**；**R** 为辅助释放，会记为 assisted。按 **V** 可在未录制时切换 Display/Raw；需要录到小球和标签时保持 Display。

此模式提供共享抓取状态和两路各自的像素跟踪，不代表同一 calibrated 3D 点。侧视 `side_ball_rise_px` 为初始 y 减当前 y，包含透视影响，不是毫米高度。日志新增 Side 掌心、跟踪点、小球位置与有效性；每轮报告新增 `trial_XXX_side_trajectory.png/.svg`，timeline 增加 Side 图像 y 变化。原有 Top View 轨迹误差、指令参数和视频时间 CSV 保留。输出在 `session_logs/experiment_visual_linked_<时间>/`。

### 当前默认与可选 Wearable 模式

默认恢复 `pinch`：四指关节角版本在本机实际录像中漏掉了握拢动作，因此不再作为启动器默认。保留 G 提示、附近张开→捏合、明确张开释放规则，不回退到未经提示自动吸附。小球叠加在识别之后，Side 识别输入始终是原始相机图像。

针对食指/中指指尖贴磁铁和胶带、手背有 PCB 的布局，`--grasp-mode wearable` 仅使用 Side 的食指、中指、无名指、小指 MCP/PIP/DIP 关节整体判断抓握，不用指尖距离。四条可用链至少三条一致；只有三条可用时取两条，只有两条时须两条一致，少于两条不判断。Top 的手指姿态不参与，也不会否决 Side。关节角使用 MediaPipe 推断的 world landmarks，仅用于手势，不能当作标定后的空间坐标或测量值。

两种模式的 Top 接近条件都使用掌部区域与球相交/接近，容差为球半径 + 20 px。掌部短暂丢失时，可由当前帧 Optical Flow 验证的位置最多衔接 500 ms；旧坐标缓存不能独自允许新抓取。球在抓住瞬间保持原位置，锁定到过滤后掌部的像素偏移，之后不乘手的角度和尺度。原始掌心轨迹继续用于误差评价；session 保存 grasp_mode 和规则说明。

已确认的附近张开状态允许最多 500 ms 的手势缺失；缺失帧不计作闭合，恢复后仍需连续有效闭合。离开球附近、失去有效 Top 抓握中心、取消提示或长时间缺失，均要求重新张开。释放须在同一路持续明确张开至少 350 ms / 3 帧；识别失败不松手，R 保留辅助释放。

默认 `SHOW SIDE FINGERS` 表示 Side pinch 不可靠；可选模式的 `SHOW SIDE GRIP` 表示四指整体姿态不可靠；`SHOW TOP HAND` 表示掌部检测与短时图像跟踪都不可用。CSV schema 5 在原有数据上新增 `top_region_valid/source/age_ms`，明确区分原始检测、当前图像跟踪与缺失。Top openness 仅作诊断，不参与抓握。模型仍可能受 PCB、胶带和遮挡影响；这些规则不代表已验证佩戴状态的检测性能。

若要做四指版本对照，可在退出后运行 `Start_Visual_Demo.cmd --grasp-mode wearable`。不要把它的单元测试通过当成佩戴状态实机成功。

## 当前交互、跟踪与独立录像

窗口可调整大小和最大化，内容等比例缩放；顶部和侧面各有独立窗口。

顶视四宫格使用约 95% 的画面宽度，横屏时高度约 88–90%，随相机宽高比展开。配色取自提供的科研配色参考图：L 形参考路线用蓝色 `#145C8D`，实际掌心轨迹用灰紫色 `#8B5A86`，两者均配细白边以适应相机背景。参考线加粗为原来的约两倍，转弯使用与两段直线相切的小圆弧；起点和小球位置保持不变。新 trial 会将圆角中心线保存到 `reference.outbound_path/return_path`，屏幕、录像和误差计算共用同一路线；线宽不作为误差容差，旧 trial 的路线和评价不变。右下角的五指张开手形表示 START / DROP，蓝色小十字对应拇指与食指之间的建议对准位置；START 不检查位置，DROP 放置成功的默认容差仍为 45 px。参考掌心偏移继续用于轨迹误差与报告计算。小球仍保留原有浅蓝色/抓住后浅绿色及立体渲染。

- **抓取必须同时满足三项条件**：操作员按 `G` 发出提示；顶视抓取点在球附近；提示之后，在球附近观察到手从张开变为闭合（wearable 为整手握拢，pinch 为拇指/食指捏合）。尚未提示、远处闭合、已经握着再移到球旁，都不会吸附。
- 移动阶段可保持拇指与食指自然半张开。到球位先用 `Space / Normal force` 提示停稳，再按 `G` 提示抓取。
- 抓取后保持 `HELD`，小球变绿，随掌部移动。跨区、离开原球位、短暂识别失败都不解除抓取。
- 明确张开手指（wearable 看 Side 四指整体，pinch 看拇指/食指），持续至少 350 ms 且至少 3 个有效检测帧后释放；球留在当前位置并恢复蓝色。
- `R` 手动释放；`C` 取消本次抓取提示及待发送反馈，但不松开已经抓住的球。每次释放后必须重新按 G，并重新完成附近的张开→捏合。H 强制吸附入口已移除。
- 识别不到可靠位置时，最多用 0.7 秒 Optical Flow 衔接；之后冻结位置并显示 `Tracking lost - hold retained`，不会自动掉球。再次识别到手后继续跟随。
- 抓取提示和成功确认**各为一个相同的单次振动**。两次均使用按 G 时的 Voltage、Duration、Frequency，默认 Duration 为 0.2 s；没有双振动编码。
- 正常 BLE 模式在提示指令写入成功后才允许抓取。未连接、0 mV、忙时拒绝的提示不会解锁；写入成功不证明执行器实际振动。
- 抓取过快时，成功确认会等提示脉冲及停止指令完成，再留 150 ms 间隔后发送。等待有时限；取消、释放、停止输出、断线或重连都会清掉未发送反馈，不跨次补发。等待确认期间方向/Normal 指令会提示稍候，Esc / X 始终可取消输出。
- `--no-ble` 中按 G 只模拟已发出提示，仍严格执行位置与手势条件，不扫描或连接 BLE。

### 画面上的触觉指令提示

顶视、侧视窗口和控制面板的相机画面都显示同一条英文 `HAPTIC` 状态，无需额外按键：`FORWARD`、`BACK`、`LEFT`、`RIGHT`、`NORMAL FORCE`、`GRASP CUE`（G 提示抓取）或 `GRASP CONFIRMATION`（抓住后的自动确认）。两种振动仍各为单次短振动。

当前反馈使用紧凑的圆角标签：方向指令为浅蓝 `#AFC9E6`，Normal force 为淡黄 `#F8E7A1`，抓取提示为浅紫 `#C7A2D9`，抓住确认为灰紫 `#8B5A86`。同类方向共用颜色，具体方向仍由英文名称区分。IDLE 恢复中性背景；PENDING / STOPPING / DISCONNECTED 使用浅暖色并保留明确状态文字。不使用闪烁或全宽深色横条。

指令成功写入 BLE 后才显示反馈名称。待发送、停止中、断线分别显示 `PENDING`、`STOPPING`、`DISCONNECTED`；拒绝的指令不会替换当前反馈。名称表示软件发送的指令，不是执行器实测反馈。脉冲结束后当前状态回到 `IDLE`。画面只保留当前反馈和简短抓握状态，不显示 `SIMULATED`、`COMMAND SENT`、`LAST TRIGGER`、设定参数或 trigger 时间。标签和路线图例合并为四宫格上方的一排，窄画面放不下时省略次要抓握标签；完整抓握/跟踪信息仍在控制面板中。

`--no-ble` 的模式标识保留在控制面板的 `VISION ONLY` 和记录文件的 simulated 字段中；相机画面里的指令名称按设定时长显示后回到空闲，不产生实际输出。该模式仍不发送或模拟自动成功确认。默认 **Display** 录制会把这条文字提示写入两个独立 MP4；**Raw** 继续保存原始相机帧。控制面板保留设定电压和时长；精确的软件指令时间及每次指令参数仍完整写入 `events.csv` / `commands.csv`，用于同步分析和 timeline。

### 两路相机与完整 UI 录像

建议用内置录制，无需同时运行两套桌面录屏软件。录像直接取程序画面，不受窗口遮挡、最小化或尺寸影响，也不把两路拼成一幅画面。

| 操作 | 功能 |
| --- | --- |
| `T` / REC Top | 单独开始或停止顶部录像 |
| `Y` / REC Side | 单独开始或停止侧面录像 |
| `B` / REC Both | 同时录两路；都在录制时再次按下同时停止 |
| `U` / Record UI 或 Stop UI | 仅在 trial 内切换 Operator UI 录像；N 启动，E/结束停止 |
| `V` / Video | 停止两路相机录像后切换 Display / Raw；UI 录像可继续 |
| `Q` | 停止输出、结束录像并退出 |

默认 `Display` 保存每个视图的全部内容：顶部包括小球、参考线和状态，侧面包括手部标记及状态。`Raw` 保存未经镜像、绘制的相机帧。另有独立的 `operator_001.mp4`，保存完整 1180 × 900 应用画布，包含两路显示画面、控制按钮、参数和当前状态；它始终录制 UI 显示内容，不受相机 Raw/Display 选项影响，不包含桌面、系统鼠标及窗口标题栏。
文件位于 `session_logs/experiment_<subject>_<run_id>/videos/`，相机分别为 `top_001.mp4`、`side_001.mp4`。UI 与 Top/Side 均只在 trial 内录制：N 成功建立 trial 后启动；E、任务结束、重启等待和 Q 关闭全部视频。每轮保存独立片段。U/T/Y/B 只在 trial 内生效，待机按这些键不会启动录制。`--no-ui-record` 禁用每轮 UI 自动录制。ABC 待机也不逐帧写 XYZ CSV。
每个 MP4 配套一个 CSV，记录视频帧时间、相机帧到达主程序的时间和重复帧标记。
UI 视频与两路相机使用同一 session 时钟，但起点不同。`operator_*.csv` 的 `capture_t_s` 表示界面渲染完成时刻，额外的 `top_capture_t_s`、`side_capture_t_s` 指向其中相机画面的收帧时刻；对齐用时间戳，不用三个文件的相同帧号。`trials.json` 会关联覆盖该 trial 的 UI 文件，`session.json` 列出全部 UI 片段。窗口遮挡、最大化和缩放不会改变保存的完整画布。
录像按真实经过时间输出名义 30 fps；视觉处理较慢或相机暂时无新帧时会重复最后一帧，因此不代表获得了 30 fps 的独立采样。两路共同开始也不等于硬件同步曝光，CSV 时间是电脑收帧时间。

本机此前确认的相机编号为顶视 Camo=1、侧视 Action 4=3；插拔后可能变化：

```powershell
python -m astra_demo.main --no-ble --top-camera 1 --side-camera 3 --subject visual_test
# 按 N 开始 trial 后才录制；待机不录制
```

### Trial、同步记录与自动评价

使用 `python -m pip install -r requirements.txt` 安装依赖（报表使用 Matplotlib）。启动命令保持不变；正式硬件测试不要加 `--no-ble`，该选项仅用于纯视觉或离线测试。接上 Astra 相机后先用 `python -m astra_demo.camera_id_probe` 核对相机编号。

1. 确保两路视频正常、Side View 能识别到手掌和手指，然后按 **N / Trial**。右下 **START / DROP** 手形只提示建议演示位置，不是启动条件；也不要求 Top View 此刻识别到手。程序固定本轮参考路线、将小球放回左上角，并开始两路独立录像。若 Top View 尚未识别到手，掌部参考偏移在首次有效检测时固定。N 不发送硬件指令；正在抓球或输出未结束时不能开始新 trial。轻量双视角需先完成一次 Side 小球放置。
2. 按 **W → Space → A → Space → G** 完成接近与抓取。第一条成功写入 BLE 的方向指令开始任务计时；`--no-ble` 使用第一条模拟方向指令，并在报告中标记为 `vision_only`。
3. 成功确认结束后，按 **D → Space → S → Space** 搬回去。参与者明确张开手指后结束本轮：只有本轮已经抓取、位置可验证、球中心落在 DROP 半径内且没有使用手动释放，才记为成功。默认 DROP 半径为 45 px，可用 `--drop-tolerance-px 60` 调整。
4. 放下后录像继续，便于记录移开手的过程；按 **B** 停止两路录像。按 **N** 开始下一轮时会结束旧录像片段并新建两个片段。**E / End** 将当前 trial 记为 `incomplete`、请求停止输出并停止录像；若手里仍有球，先张开或用 R 释放再开始下一轮。**Q** 停止输出、关闭录像，并自动生成统计表和图片。

**数字 0 / Restart** 可在持球、未完成或完成后直接重来，无需先 R、E、N。未完成的一轮记为 `reset`（不计成功），已完成结果不改写；旧录像正常封存，新 trial 使用新文件。当前画面的轨迹、抓握许可和计时状态重置，磁力/振动不因重启自动触发。这个按钮不会删除历史数据。

没有按 N 时仍会记录轨迹和指令，但 trial ID 为空，不纳入任务评价。R 手动释放记为 `assisted`；非目标位置释放记为 `off_target`；无法验证释放位置记为 `unverified_position`。退出时未完成的 trial 记为 `interrupted`。失败、中止和辅助完成不会写入正常成功的 completion time；另存 attempt duration。N 开始后的参考路线固定，不根据实际轨迹事后拟合。

每个 session 的目录内容：

| 文件 | 内容 |
| --- | --- |
| `session.json` | participant/session ID、时钟锚点、相机与镜像设置、记录模式、算法参数及测量口径 |
| `trials.json` | trial ID、参考路线、起止/抓取/释放时刻、目标半径、结果和关联视频片段 |
| `trajectory.csv` | 原始掌心与抓取点、实际驱动小球的平滑位置、小球位置、顶/侧收帧时刻、跟踪有效性与状态 |
| `events.csv` | 指令请求/拒绝/发送/结束、断线、抓取/释放、录像和 trial 事件；含 command ID、cue role、**设定电压 mV、设定时长 s**及频率 Hz |
| `videos/top_*.mp4`、`side_*.mp4` | 两路独立视频；Display 模式含时间戳、trial ID、参考路线、实际掌部轨迹及原有叠加信息 |
| `videos/operator_*.mp4`、`operator_*.csv` | 每轮 Operator UI 录像及渲染/相机时间戳；随 trial 启停 |
| `videos/*.csv` | 每帧的 `video_t_s`、`capture_t_s`、session/trial ID、Unix 时间和重复帧标记 |
| `reports/trial_summary.csv`、`summary.json` | 各轮完成结果、成功率、成功完成时间、放置误差、轨迹 RMSE/平均/P95 误差、分阶段误差及有效跟踪比例；硬件与纯视觉结果分组 |
| `reports/commands.csv` | 每条指令一行的设定参数与请求/发送/结束时刻、角色和状态；区分抓取提示与自动成功确认 |
| `reports/trial_XXX_trajectory.png/.svg` | 去程和返程各自的参考路线与实际掌部轨迹对比 |
| `reports/trial_XXX_timeline.png/.svg` | 手部 x/y 轨迹、各条指令的设定区间、电压/时长、抓取与释放时刻 |

**评价定义：** completion time 从第一条已发送的方向指令到视觉确认主动释放，包含中间停顿和操作员等待。轨迹误差使用原始检测的 palm center 到对应阶段参考折线的最近欧氏距离，单位为 px；整体 RMSE 按有效采样点计算。参考掌部路线等于预定抓取点路线加上固定 `palm - grasp anchor` 偏移，在 N 时或首次有效 Top 检测时采样一次，并记录来源和时间，不按实际路线事后拟合。偏移尚未确定的采样不计算轨迹误差。跟踪缺失、冻结和 Optical Flow 补点不用于 accuracy；超过 250 ms 的空档不插值、不跨空档累计路径长度。空数据输出空值而非零误差。二维轨迹偏差同时包含视觉检测误差，不能视作物理测量的 ground truth；目前不换算厘米，也不使用旧 grab CSV 的 `pinch_cm` 计算评价结果。

**同步口径：** 视频、指令和轨迹都基于一次锚定的 host monotonic clock；通过视频 CSV 的 `capture_t_s` 对齐真实收帧记录，通过 `video_t_s` 对齐编码后的帧位置。两个相机是软件时间对齐，没有硬件同步曝光。命令时间为软件请求/成功写入 BLE/停止序列结束时间；图中条形长度是设定时长，不代表实测电流、电压、受力或执行器响应。

需要重新出图时（不会连接硬件）：

```powershell
python -m astra_demo.trial_report "session_logs/experiment_P01_<run_id>"
```

默认路线为右下 START → 右上 TURN → 左上 PICKUP → TURN → DROP。可用 `--reference path.json` 指定固定的图像归一化坐标，例如 `{"start":[0.75,0.75],"turn":[0.75,0.25],"pickup":[0.25,0.25],"drop":[0.75,0.75]}`。坐标基于显示后的顶视图，x 向右、y 向下；它与默认工作区内的百分比路线不完全相同。

### 算法与参数

MediaPipe Tasks 使用 VIDEO 模式连续跟踪。Top 使用掌部区域内的特征点和前后向 Optical Flow / RANSAC 检查运动；图像运动有效时降低单帧 landmark 跳动的权重，检测坐标仍持续校正漂移。过滤后的同一掌部位置同时用于轨迹显示和球的平移，抓住瞬间锁定球到掌部的像素偏移，不再随手的角度/尺度放大偏移。轻量 Side 仍使用独立的掌部跟踪和固定像素偏移。两路缺失时均保留抓取状态，位置仅在可用时更新。

画面上的 **Hand (tracked)** 使用上述同一条过滤后掌部轨迹，不再用腕部加固定偏移替代掌心，也不二次滤波。短暂缺失只绘制由当前图像运动验证的点；跟踪失效后断线，重新检测到手时另起一段。`display_palm_*` 保存显示坐标，`wrist_*` 仅保留诊断原值；原始 `palm_*` 仍用于 RMSE 和报告，跟踪补点不会伪装成原始检测。已保存的视频和实验记录不会被重写。两路未标定 RGB 仍有透视影响；不能声称消除了俯仰/远近造成的投影变化。
侧视判定使用拇指尖至食指尖距离 / 掌部尺寸，带短窗口中值滤波、闭合/张开的不同阈值和释放持续时间。指节几何异常时视为未知，不视为张开。
轻量 Side 抓住后的位置向掌心内收 25%（`side_grip_palm_inset=.25`），只在每次抓取时计算并锁定，之后不随手指开合、角度或尺度重算。Top 位置、抓取门槛、球的大小/颜色不因此改变；释放保留最后位置，0 重置回原点击位置。这是视觉放置调整，不是实际球心深度测量。

当前主程序参数见 `astra_demo/config.py`：`pinch_enter_ratio=0.42`、`pinch_ready_ratio=0.60`、`pinch_release_ratio=0.85`、`grab_radius_margin_px=20`、`release_hold_ms=350`、`release_min_frames=3`、`top_region_flow_s=0.5`、`optical_flow_hold_s=0.7`。`trail_*` 调整 Top 掌部的共享位置过滤；`side_tracking_*` 调整轻量 Side 小球。Legacy 参数不驱动当前主程序。
半张开与捏合各需连续 2 个有效检测帧。当前位置无法由当前检测或时限内的图像运动验证时，不允许新抓取；手势缺失最多保留 500 ms 的已确认张开状态。成功抓住后的状态一直保持到明确张开或 R。
这些改进提高对短暂遮挡的容忍，但无法恢复完全遮住的手指；戴执行器后的识别率、阈值和实际延迟仍需实机验证。四区线仅作参考，不参与抓取或释放判定。

自动测试：`python -B -m unittest discover -s astra_demo/tests -v`。离线界面预览：`python -m astra_demo.operator_panel --preview`，其中 G 可预览提示状态，但没有真实手势输入，不会吸附，录像不会写入文件。

## 手动硬件控制（2026-11）

新增全英文 `Haptic Guidance | Operator` 面板：左侧顶视画面，右侧侧视预览与手动指令按钮。
界面以蓝色为基础，HELD 使用绿色；侧视显示 pinch 比值供调试。
工作平面使用宽高比约 3:2 的横向矩形（X 轴长、Y 轴短）与四区虚线，不执行区域命中计算。蓝色小球初始位于左上区，选中状态同时使用文字和边框提示。旧 CSV 的 `target_key` 列保留兼容，
新运行写入 `0`，不再代表目标区域。下文的四宫格实验说明仅适用于旧版。
**主程序的按钮和键盘可发送真实 BLE 指令；成功抓取时额外请求一次确认振动。`--no-ble` 完全关闭硬件输出。**
视觉抓取、小球跟手和抓取 CSV 保留；G 发送振动提示并允许本次附近的张开→捏合，不能强制抓住球。Normal force 仅提示停手，不松开球。

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

界面右侧 +/- 按钮可调节，也可用三组相邻数字键：1/2 调电压，3/4 调时长，5/6 调频率，每组都是前一个减、后一个加。参数作用于下一次指令，不改变正在执行的脉冲。

| 参数 | 初始值 | 范围 / 步长 | 减 / 加 |
| --- | --- | --- | --- |
| Voltage | 0 mV | 0–5000 mV / 25 mV | `1` / `2` |
| Duration | 0.2 s | 0.1–5.0 s / 0.1 s | `3` / `4` |
| Vibration frequency | 100 Hz | 10–200 Hz / 10 Hz | `5` / `6` |

0 mV 时拒绝执行力/振动指令，需由操作者主动提高电压。上限只是软件范围，不代表装置的安全工作范围。
Static 力指令的频率字段固定为 0；频率调节只影响振动。初始值在 `astra_demo/config.py` 的
`manual_voltage_mv`、`manual_duration_s`、`manual_frequency_hz` 中修改；旧 `ble_fixed_*` 和 `ble_pulse_ms` 不再驱动主程序输出。

### BLE 和停止机制

沿用 CEL_GUI 的 9 字节协议、毫伏单位和小端字节序。目标配置仍是 `ble_mac_address`、`ble_uuid`、`ble_enabled`，
默认特征为 `120062c4-b99e-4141-9439-c4f9db977899`，须具有 write 或 write-without-response 属性。
后台线程串行发送，主视觉循环不等待 BLE。一次仅执行一个脉冲，忙时普通手动指令不排队；本次抓取的成功确认可以在界面逻辑中短暂等待提示结束，停止命令优先清除未发送动作。
每次连接先停止输出，再允许新动作；断线重连不重放旧指令。界面区分 Requested、Sent、错误状态，Sent 仅表示写入成功，不证明实际产生了力。

时长由电脑计时，并非写入硬件的数据包：结束时先发 Static Normal 0 mV，约 100 ms 后发 `FF 06 00 00 00 00 00 00 00`。
所以时长是发送零输出前的目标时长，不是已校准的机械力持续时间，通信和释放过程会增加偏差。
退出时也尝试同样的停止流程。**断电、电脑崩溃或 BLE 断线时不能保证停止包送达；硬件是否有超时保护需确认，实测须保留物理断电手段。**
不要同时运行同事的 CEL_GUI 和本程序连接同一设备。

界面和录像叠加文字统一使用 Arial Regular / Bold（Windows 已有系统字体；其他系统依次回退到 Arial、Liberation Sans、DejaVu Sans），遵循 [Nature 图表规范](https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/) 推荐的 Arial / Helvetica 无衬线风格。白底、深蓝灰文字与细边框为主，触觉状态按内容宽度显示为有色小标签；不使用深蓝色整栏、阴影或装饰字体。文字贴图会缓存，避免逐帧重复加载字体。
先更新依赖以安装新增的 Pillow，再预览布局和球体（无需相机、BLE 或 MediaPipe）：

```powershell
python -m pip install -r requirements.txt
python -m astra_demo.operator_panel --preview
```

Windows 双摄像头与真实硬件运行（会尝试连接 BLE，默认电压为 0）：

```powershell
python -m astra_demo.main --subject P01
```

只测试双摄像头识别和小球抓取时，使用 `--no-ble`，不扫描或连接 BLE：

```powershell
python -m astra_demo.main --no-ble --top-camera 0 --side-camera 1 --subject visual_test
```

`0`、`1` 仅为示例，请按实际相机 ID 替换。两个 ID 必须不同，
且应对应两路独立画面；虚拟相机如果转发的是同一个内置摄像头，也不构成双视角。
命令行参数仅影响本次运行，不修改 `config.py` 中的默认配置。纯视觉模式仍记录抓取 CSV。
Windows 下主程序与 `camera_id_probe` 均明确使用 Media Foundation（MSMF），避免不同后端的相机编号不一致。
本次 Camo 与 Action 4 测试中，MSMF 可正常读帧；Camo 的 DirectShow 通道出现过黑帧。
插拔设备或安装虚拟相机后，仍需重新核对相机编号。

默认显示操作员面板、独立顶视窗口和独立侧视窗口。
`show_operator_window` 控制面板显示，`operator_panel_fps=20` 限制面板合成频率，
`show_side_color_window` 控制独立侧视窗口。检测仍在每轮主循环执行。

`--preview` 和 `--snapshot` 始终离线，不创建 BLE 控制器、不连接设备。下文保留的自动振动、zone 配置说明属于历史实验流程，以本节手动控制说明为准。

小球采用预计算的球面光照、柔和高光和局部阴影，逐帧仅做局部透明合成。
小球较原来放大约 10%：Top 默认半径 44→48 px，轻量 Side 在 480 px 高画面中 28→31 px；抓取前后尺寸相同，颜色与材质不变，位置判定使用显示球的实际半径。可独立录制双路视频；离线预览不会写入抓取日志。

## 历史版本记录

以下为旧版实验流程，参数和自动反馈说明不适用于当前入口；请以上文为准。

本项目现在是**纯视觉 / 纯 UVC 相机**方案，不再依赖 Astra 深度流、OpenNI 运行时或 Orbbec 驱动。

当前结构：
- 顶视：手机摄像头或虚拟摄像头，用于 2x2 四宫格定位与小球显示
- 侧视：任意 UVC 彩色相机，用于拇指-食指捏合判定
- 反馈：BLE 单次震动脉冲
- 日志：每次运行自动生成一份 CSV

## 1. 当前抓取逻辑

以本文件开头的交互与 Wearable 说明为准。`IDLE -> ARMED` 必须先收到操作员 G 提示；`ARMED -> GRAB` 必须在球附近完成张开→闭合；`GRAB -> IDLE` 须明确持续张开或手动 R。四宫格不控制吸附或保持。

当前默认参数见 `astra_demo/config.py`：`pinch_enter_ratio=.42`、`pinch_ready_ratio=.60`、`pinch_release_ratio=.85`、`enter_frames=2`、`grasp_gesture_gap_ms=500`、`release_hold_ms=350`、`release_min_frames=3`。Wearable 的 Side 四指整体分数使用相同的 open/close 分界，但不是 fingertip distance。

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

- 默认仍是双 RGB 的 2D 模式；可选的 3D 模式需要 Astra 深度驱动、OpenNI 和现场标定。
- G 提示后，靠近小球并从半张开变为捏合才会抓取；保持至明确张开或 R 释放。
- 抓住后使用掌心跟踪保持相对位置；丢失检测不会自动松手。
- `--no-ble` 不扫描、不连接 BLE，也不向硬件发出指令。
- 3D 模式用一份毫米坐标分别投影到两路视频，蓝色/抓取色保持一致，投影大小随距离变化。

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
