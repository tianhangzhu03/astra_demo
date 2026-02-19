from dataclasses import dataclass


@dataclass
class AstraDemoConfig:
    # 顶部摄像头ID（用于九宫格命中判定）
    top_camera_id: int = 2
    # 侧面彩色相机ID（用于手势识别与渲染）
    side_color_camera_id: int = 1

    # 目标采集帧率；调大可降低感知延迟，但CPU占用更高
    camera_fps: int = 30
    depth_unit: str = "mm"

    # 拇指-食指捏合进入阈值（归一化距离）；调大更容易触发
    pinch_enter: float = 0.085
    # 捏合退出阈值（应大于 enter，形成迟滞防抖）；调大更不容易释放
    pinch_exit: float = 0.110

    # 侧面手检测最小置信度；调大可减少误检但可能漏检
    side_min_detection_confidence: float = 0.5
    # 侧面手跟踪最小置信度；调大可减少抖动但可能丢跟踪
    side_min_tracking_confidence: float = 0.65
    # 是否对侧面图做CLAHE增强（弱光/低对比建议开）
    side_use_clahe: bool = True
    # 手势识别输入缩放比（<1可降延迟）；调小更快但精度略降
    hand_process_scale: float = 0.75

    # 深度进入阈值（mm，越小表示要求越近）；当前作为“软门限”
    depth_enter_mm: int = 760
    # 深度退出阈值（mm，应大于 enter）；调大更不容易退出
    depth_exit_mm: int = 860

    # 进入状态所需连续帧数；调大更稳但延迟增加
    enter_frames: int = 2
    # 退出状态所需连续帧数；调大更稳但释放更慢
    exit_frames: int = 2
    # 中点平滑系数（0~1）；调大更跟手，调小更平滑
    smooth_alpha: float = 0.55

    frame_width: int = 640
    frame_height: int = 480

    # 顶部九宫格宽度占比；调大九宫格更大
    grid_w_ratio: float = 0.84
    # 顶部九宫格高度占比；调大九宫格更大
    grid_h_ratio: float = 0.78
    # 顶部九宫格Y起点占比；调大九宫格整体下移
    grid_y_ratio: float = 0.10

    # 深度可视化最小显示距离（mm）；小于该值不高亮
    depth_vis_min_mm: int = 300
    # 深度可视化最大显示距离（mm）；大于该值不高亮
    depth_vis_max_mm: int = 600

    # 深度独立窗口尺寸；调小可降低渲染开销
    depth_view_width: int = 520
    depth_view_height: int = 390

    # 虚拟物体跟随态尺寸；调大球/方块更大
    prop_size_follow: int = 36
    # 虚拟物体抓取态尺寸；调大抓取时更显眼
    prop_size_held: int = 50
    # 虚拟物体初始九宫格位置（1~9）
    prop_init_grid_key: int = 5
    # 松手后是否保持可见（True=保留，False=隐藏）
    prop_idle_visible: bool = True
    # 虚拟物体跟随平滑系数；调大更跟手，调小更丝滑
    prop_follow_alpha: float = 0.32
    # V键切换冷却时间（ms）；防止长按导致连续切换卡顿
    prop_toggle_cooldown_ms: int = 220

    ble_enabled: bool = True
    ble_mac_address: str = "FE:56:9B:F0:CF:0E"
    ble_uuid: str = "120062c4-b99e-4141-9439-c4f9db977899"
    ble_fixed_volts: int = 2000
    ble_fixed_freq: int = 100

    # 抓取释放后回位动画时长（ms）
    release_return_ms: int = 200
