# Windows 顶视 + 侧视抓取 Demo

**当前开发目标为已确认的 A → B → C 虚拟障碍 / 倒扣盒子场景。正式入口 `Start_AR_Demo.cmd`，说明见 [ABC.md](ABC.md)。** 默认沿用Top/iPhone、Side/Astra RGB，无depth、无图纸标定。黄色标带全程记录Top XY / Side上下Z代理，整个trial固定零点；Shear/Normal仅标记阶段。G抓放，J/L请求90°转动，无需F二次确认；AR/G/BLE独立。旧 depth/标定/合成3D测试入口已删除。以下内容描述保留的轻量baseline。

完整 Operator UI 与相机仅在 trial 内录制：N 启动，E/结束/重启/退出停止。待机 U/T/Y/B 不会启动录制，每轮保存独立片段。ABC 待机不逐帧写 XYZ CSV。当前交接见根目录 `HANDOFF.md`。

当前安装、运行、抓取逻辑、BLE 控制与独立录像说明统一维护在[仓库根目录 README](../README.md)。

本机纯视觉测试命令（相机插拔后需核对编号）：

```powershell
python -m astra_demo.main --no-ble --linked-views --top-camera 2 --side-camera 3 --subject visual_linked
```

G 发出单次振动提示后，手在球附近由半张开变为捏合才会抓住。成功确认也为单次振动。抓取保持至明确张开手指或按 R 释放；C 取消提示，H 不再强制抓取。T/Y 分别录顶部/侧面，B 同时录成两个文件，Q 结束录像并退出。默认录制每路完整显示内容。

也可双击根目录 `Start_Visual_Demo.cmd`。轻量模式先在 Side View 点击侧面小球的起始位置；两路随后共用抓取状态，各自随手跟踪，不需要标定或深度。录像和报告保留 Side 像素轨迹，不能视为真实 Z 高度。

**数字 0 / Restart** 一键重来：停止输出、取消抓握提示、两路球回初始位置、清空显示轨迹并启动新 trial 和双路录像。保留相机连接、Side 起始位置和控制参数；Side 手部或相机暂时不可用时自动等待恢复，Esc/X 可取消。新一轮仍须 G 才能抓取、第一条方向指令才开始计时。旧录像保留，未完成的旧 trial 记为 `reset`。

正式记录使用 N 开始新 trial（两路视频正常、Side View 能识别手掌和手指，自动录两路）。START 仅作演示位置提示，不检查手的位置，也不要求 Top View 已识别到手；A 仍为向左引导。第一条已发送的方向指令开始计时；主动放下后自动判定结果。E 提前结束，Q 退出并生成轨迹对比图、timeline 和评价表。数据在 `session_logs/experiment_<subject>_<run_id>/`；完整评价与同步口径见根目录 README。
