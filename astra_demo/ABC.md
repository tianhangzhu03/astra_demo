# A → B → C：当前 AR / XYZ 版本

更新：2026-11-15。球、障碍和倒扣盒子全部虚拟。原轻量 baseline 入口 `Start_Visual_Demo.cmd` 保留。

## 启动与显示

`Start_AR_Demo.cmd`默认Top/iPhone和Side/Astra RGB，不打开depth、不读取标定。标带记录使用下述全程连续相对位移；`--no-ble`完全禁用BLE。旧 depth、stereo 和 affine 标定代码及入口已移除。

障碍和Top路线XY尺寸缩小20%，保持A→B→C绕行顺序。Top 使用明亮天蓝色圆角障碍和 box，Operator 参考路线为虚线，不显示 A/B/C 节点标注。Side 只显示球和开口朝下的 box，不显示障碍。两路均不画实际手部运动轨迹，但逐帧数据照常保存。Top球直径较此前放大25%；Side球采用75%不透明度，box保留35%不透明度，检测使用未叠加图形的原始帧。球始终橙红色，不因抓取而变色。参与者窗口不显示抓取、触觉或 XYZ 状态；Operator Top 下方显示当前 XYZ。

## Operator dashboard

全英文trial主界面；所有视频仅在 trial 内录制，待机 U/T/Y/B 不会开启录制，ABC 待机不逐帧写 XYZ CSV。Start trial & record [N]建立trial并自动录制Top/Side，End trial & save [E]结束本轮，Restart trial [0]保留旧数据并重开，Exit app [Q]关闭程序。Advanced recording默认收起，包含Top/Side/Both、Operator UI和Raw/Display视频选项；快捷键保留。隐藏按钮不可通过趋势图位置误点。

Side预览区域452×339，Top区域660×340；保持各自宽高比，Side点击映射随新布局更新。状态行分别显示trial内XYZ记录时长、第一条WASD后的任务计时；每路胶带可见性和视频录制状态独立显示，UI video状态位于页脚。未建立trial时明确Preview only，建trial失败原因仍可见。

Live tape motion显示最近30秒Top X/Y和Side Z proxy，零点仍由记录器固定；橙底标记Normal，不平滑、不填补丢失，超过1.5秒的采样间断不连线。E后保留上一轮曲线，下一轮N清空显示历史。图表不改变CSV数据或录制时机，1 a.u.仍为各视角图像高度比例。

## 操作

- P / O：选择球 / box，然后点击 Side；右键直接放 box。独立 Side 和 Operator 缩略图均可操作。trial 中禁止摆放。
- N：建立 trial。默认简单RGB模式先放置Side球/box；任一路识别到黄色胶带或满足原Side手部识别即可启动，不需图纸。手放在 A，第一条成功发送或模拟的 WASD 开始计时。
- W/A/S/D：单击选定平面方向，先发一次 Static 短脉冲，再每秒一次。换向取消旧方向，保持相邻重复脉冲至少间隔 1 秒；不要求按住，松键不停止。
- I / K：同样单击选择 Normal Up / Down，每秒一次。Space 按当前 Lift / Lower 阶段选择 Normal。两者都是 Static direction=0，不声称硬件施力极性不同。
- G：未持球时人工抓取，持球时人工放下，并请求一次 Vibration。没有左右位置、距离或 open-to-pinch 门槛。抓取需要有效位置；已经持球时丢失跟踪不会自动放下。位置丢失时仍能释放，但无法判定放入成功。BLE 失败不撤销人工抓放，事件另记发送结果。
- 路线：A→B 抓取→绕障碍到 C→J 提示 CW 向下转 90°→I 向上入盒→G 放球→K 空手下降→L 提示 CCW 反向转 90°回到起始姿态→E结束本轮录制。当前 RGB 模式目标角度为 90°，以减少侧面胶带遮挡；无需 F 二次确认，J/L 记录人工操作意图并推进对应阶段，未测量手部角度。BLE 仍为 CW=5 / CCW=6 的单次提示，不编码角度，也不保证实际转动量。
- Operator去掉重复的Cancel cue [C] / Release [R]按钮及快捷键；抓放统一用G，停止统一用X。
- X/Esc/Stop：取消持续输出。G/J/L 选择单次 cue 时也取消持续方向。失焦、UI 超过 0.75 s 未更新、trial 切换、断线会清除持续输出；重连不自动续发，迟到节拍不补发。
- L请求CCW后继续记录回转动作，转回后E结束；E提前结束则记为incomplete。0重置，Q保存退出。T/Y/B 控制两路录像，U 控制 Operator 录像，V 在停录时切换 Raw / Display。

G 放下是人工状态变化；成功放入 box 是独立判定，仍检查有效位置、从下方入盒、翻手指令和流程顺序。`success` 不证明实际力、感知、转动方向或实验级定位精度。

## 新 BLE 协议

ABC 使用用户 2026-11-15 提供的 `CEL_GUI (1).py` 对应协议，连接时读取 VERSION_UUID `030062c4-b99e-4141-9439-c4f9db977899`，要求 CEL 名称、版本 `v0.2.1` 和 acknowledged-write characteristic。身份/版本不符时不发送输出；不再靠 `--torsion-firmware-ready` 跳过验证。旧 baseline 继续选择原协议。

START 为九字节 `[FF,10,mode,direction,Vlo,Vhi,Flo,Fhi,start]`。Static=0、Vibration=1；Normal=0、Forward=1、Backward=2、Left=3、Right=4、CW=5、CCW=6。CCW 的通道配方没有在文件里给出，不自行编造。新版 STOP **只有两字节 `FF 06`**，不再先发送零电压 Normal。BLE write 成功只代表发送完成。

默认电压 2500 mV，界面每次加减 500 mV。ABC 脉冲宽度 0.1–0.8 s，默认 0.2 s；固件运行上限按同事 GUI 注释为约 1 s，不把 PC 定时器当成实测输出时长。每秒一次是触发节奏，与 G 的 Vibration 频率分开。

## 默认：连续黄色标带相对位移

不用图纸。Top X向右、Y向上，Side Z为图像向上的代理；XYZ全程连续记录。各视角在trial建立后首次有效识别时设零，整个trial固定，遮挡恢复、WASD/Normal切换都不归零。Shear/Normal仅标记阶段，G/Torsion/Stop保留最近阶段标签，不屏蔽任何轴。各视角独立判断有效性，漏检留空，不插值。检测用原始帧的黄色区域中心；当前MediaPipe 食指 DIP(7) 到 TIP(8) 限定末节区域，HSV 黄色分割、3×3 去噪后取唯一黄色连通区域中心。黑色绑带通常不通过颜色阈值；手背 PCB 位于搜索区之外。无末节关键点、无胶带或多个候选均留空，不回退全画面，不用 landmark 位置替代胶带。佩戴硬件后的遮挡和检出率尚待实测。

schema15：source=`yellow_tape_continuous_rgb`；`*_delta_x/y/z_au`为全程位移，`rgb_marker_phase`为xy/z/idle指令阶段，`rgb_marker_z_segment`仅统计Normal段，不再重设零点，兼容字段overlap固定为0。保留top_valid、side_valid、各视角参考时间及参考值、原始标带像素和各自host采集时间。1 a.u.=对应视角图像高度，Top XY共用Top高度，Side Z用Side高度；不同视角幅值不是同一物理尺度。

报告`trial_*_rgb_marker_displacement.png/.svg/.csv`共用时间轴，Normal段浅橙底，三条曲线全程显示；仅漏检或长时间采样中断留空。相机俯角、透视、指节旋转仍会影响图像位移，不能把曲线当作严格正交的物理XYZ。AR/G/J/L/BLE不依赖这些坐标。旧source的报告仍按原有分阶段语义读取，不修改历史实测数据。


抓取后两路球心跟随各自经过滤波的掌心，松开后保留位置。障碍外观采用圆角与轻微阴影；碰撞仍使用保守的原矩形边界。
