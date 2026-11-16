from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from collections import deque
from contextlib import ExitStack
from pathlib import Path
from sys import platform
import time

import cv2
import numpy as np

from .config import AstraDemoConfig
from .experiment import Experiment, reference_route, draw_trial_overlay
from .grab_logic import GrabContext, update_grab_state, update_manual_grab
from .manual_control import Actuation, ManualBleController, CHANNEL_STATES, TORSION_CHANNELS, TORSION_DIRECTION
from .mp_hands_compat import load_hands_api
from .operator_panel import (OperatorPanel, draw_workspace_boundary, initial_ball_position,
                             enable_dpi_awareness, open_video_window, draw_view_status, show_video_frame,
                             workspace_bounds, text, draw_haptic_status, route_legend_width)
from .session_logger import SessionLogger
from .tracking import TopHandTracker, observe_hand
from .video_recorder import DualRecorder, StreamRecorder
from .virtual_prop import VirtualProp
from .wearable_grip import WearableGrip, wearable_observation, near_palm


def enhance_for_hand_detection(frame_bgr, clahe):
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    light, a, b = cv2.split(lab)
    enhanced = cv2.cvtColor(cv2.merge((clahe.apply(light), a, b)), cv2.COLOR_LAB2BGR)
    return cv2.GaussianBlur(enhanced, (3, 3), 0)


def open_uvc_camera(preferred_id: int, width: int, height: int, fps: int, label: str, strict_id: bool):
    tried = []
    candidate_ids = [preferred_id] if strict_id else [preferred_id, 0, 1, 2, 3, 4]
    for cam_id in candidate_ids:
        if cam_id in tried:
            continue
        tried.append(cam_id)
        # The probe uses the same backend. Camo DirectShow gave black frames on this setup.
        backend = cv2.CAP_MSMF if platform == 'win32' else cv2.CAP_ANY
        cap = cv2.VideoCapture(cam_id, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print(f'[CAM] {label} use id={cam_id} backend={cap.getBackendName()}')
            return cap
        cap.release()
    raise RuntimeError(f'{label} open failed. tried ids={tried}')


def detect_hand(detector, frame, scale, *, allow_edge_on=False):
    if frame is None:
        return None, None
    h, w = frame.shape[:2]
    sample = cv2.resize(frame, (max(1, round(w * scale)), max(1, round(h * scale)))) if scale < .999 else frame
    result = detector.process(cv2.cvtColor(sample, cv2.COLOR_BGR2RGB))
    if not result.multi_hand_landmarks:
        return None, None
    landmarks = result.multi_hand_landmarks[0]
    worlds = getattr(result, 'multi_hand_world_landmarks', None)
    return observe_hand(landmarks, w, h, allow_edge_on=allow_edge_on,
                        world_lms=worlds[0] if worlds else None), landmarks


def main() -> None:
    cfg = AstraDemoConfig()
    parser = argparse.ArgumentParser(description='Dual-camera grasp tracking and independent recording')
    parser.add_argument('--subject', default=cfg.default_subject_id)
    parser.add_argument('--no-ble', action='store_true', help='No BLE scanning, connection or hardware output')
    parser.add_argument('--ble', dest='no_ble', action='store_false', default=argparse.SUPPRESS,
                        help='Enable BLE, overriding the visual launcher default')
    parser.add_argument('--ble-address', type=str.strip, default=cfg.ble_mac_address,
                        help='Use this BLE address for this run without changing the saved default')
    parser.add_argument('--torsion-firmware-ready', action='store_true',
                        help='Legacy mode only. ABC automatically verifies CEL v0.2.1 for CW=5 and CCW=6')
    parser.add_argument('--top-camera', type=int, default=cfg.top_camera_id)
    parser.add_argument('--side-camera', type=int, default=cfg.side_color_camera_id)
    parser.add_argument('--record-content', choices=('display', 'raw'), default='display')
    parser.add_argument('--no-ui-record', action='store_true', help='Disable Operator UI recording at trial start; U toggles it only during a trial')
    parser.add_argument('--reference', type=Path, help='JSON with normalized start/turn/pickup/drop points')
    parser.add_argument('--scenario', choices=('legacy', 'abc'), default='legacy',
                        help='abc: virtual obstacles / inverted box; two RGB cameras and relative tape motion')
    parser.add_argument('--linked-views', action='store_true', help='Lightweight side ball follows the hand; click its initial position, no depth/calibration')
    parser.add_argument('--grasp-mode', choices=('pinch', 'wearable'), default='pinch',
                        help='wearable: Side four-finger grip, Top palm-region proximity; no fingertip distance')
    parser.add_argument('--rgb-only', action='store_true', help='Use linked RGB and independent approximate tape recording; no depth')
    parser.add_argument('--drop-tolerance-px', type=float, default=45., help='Successful placement radius in top-view pixels')
    args = parser.parse_args()
    if not args.ble_address:
        parser.error('BLE address must not be empty.')
    if min(args.top_camera, args.side_camera) < 0 or args.top_camera == args.side_camera:
        parser.error('Top and side views need two different nonnegative camera IDs.')
    if not math.isfinite(args.drop_tolerance_px) or args.drop_tolerance_px <= 0:
        parser.error('Drop tolerance must be positive and finite.')
    scene = None
    abc = args.scenario == 'abc'
    if abc:
        cfg.manual_voltage_mv = 2500  # User-selected ABC starting setpoint; output still requires a cue.
    if abc and args.reference:
        parser.error('ABC uses its own A/B/C route; --reference applies to legacy mode only.')
    if abc:
        from .abc_linked import LinkedAbcScene, LinkedAbcTask, LinkedSceneView
        args.linked_views = True
        scene = LinkedAbcScene()
    custom_reference = None
    if args.reference:
        try:
            custom_reference = json.loads(args.reference.read_text(encoding='utf-8'))
            reference_route(cfg.frame_width, cfg.frame_height, None, custom_reference)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            parser.error(f'Invalid reference: {exc}')
    cfg.ble_enabled = cfg.ble_enabled and not args.no_ble
    cfg.ble_mac_address = args.ble_address
    cfg.record_operator_ui = cfg.record_operator_ui and not args.no_ui_record
    enable_dpi_awareness()
    if not cfg.ble_enabled:
        print('[MODE] Vision only: BLE scanning and hardware output are disabled.')
    else:
        print(f'[BLE] Target {cfg.ble_mac_address}; characteristic {cfg.ble_uuid}')

    def scene_route(frame):
        if abc:
            return scene.reference(frame.shape[1::-1])
        return reference_route(frame.shape[1], frame.shape[0], workspace_bounds(frame.shape[1], frame.shape[0]), custom_reference)

    with ExitStack() as cleanup:
        cleanup.callback(cv2.destroyAllWindows)
        logger = SessionLogger(args.subject, Path(__file__).resolve().parents[1] / cfg.session_log_dir,
                               cfg.pinch_plane_width_cm)
        cleanup.callback(logger.close)
        print(f'[LOG] Writing grab records to {logger.path}')
        ble = ManualBleController(cfg.ble_enabled, cfg.ble_mac_address, cfg.ble_uuid,
                                  torsion_firmware_ready=args.torsion_firmware_ready,
                                  protocol='v0.2.1' if abc else 'legacy')
        experiment = Experiment(logger.output_dir / f'experiment_{logger.subject_id}_{logger.run_id}',
            logger.subject_id, logger.run_id, settings={**asdict(cfg), 'top_camera_id': args.top_camera,
            'side_camera_id': args.side_camera, 'record_content': args.record_content,
            'drop_tolerance_px': args.drop_tolerance_px, 'custom_reference': custom_reference,
            'grasp_mode': args.grasp_mode, 'scenario': args.scenario,
            'torsion_firmware_ready': args.torsion_firmware_ready})
        task = None
        scene_views = {}
        if scene:
            task = LinkedAbcTask(scene, experiment, ble, manual_grasp=True)
            scene_views = {name: LinkedSceneView(scene, name) for name in ('top', 'side')}
            experiment._json('abc_scene.json', scene.snapshot())
            experiment.metadata.update(
                coordinate_system='Independent Top/Side displayed pixels and relative image motion.',
                reference_alignment='Fixed ABC route; no fitting or shifting to the recorded hand path. XYZ trends track the index tape independently.',
                reference_geometry='Frozen approved A-B-C corridors, obstacles and downward-opening box; abc_scene.json.',
                completion_time='First sent/simulated Shear command through capture, lowering and final hand flip.',
                participant_view='Virtual scene with constant sphere appearance. Route, trail, state and cue diagnostics only in operator view.',
                torsion=dict(source='User-supplied CEL_GUI (1).py on 2026-11-15',
                             cw_direction=5, ccw_direction=6, expected_firmware='v0.2.1',
                             behavior='Version checked on connection. J/CW cues flip down; L/CCW cues flip back. Physical torque is not measured.'),
                channel_patterns={cue.value: list(pattern) for cue, pattern in CHANNEL_STATES.items()},
                held_feedback='Tap WASD/I/K or click: finite Static pulses every 1 s. Switch replaces the previous direction; release does not stop. X/Stop, focus loss, stale UI, trial change or disconnect stop. Late slots skipped.',
                normal_directions='I=normal_up, K=normal_down; Space follows task stage. Both use Static direction 0; no physical polarity reversal implied.',
                flip_orientation='J/L record operator intent; rotation angle is not measured.',
                grasp_confirmation='Disabled in ABC: G is one explicit pickup or drop cue; no automatic confirmation pulse.')
            if abc:
                experiment.metadata.update(
                    coordinate_system='Separate displayed Top/Side image pixels; world XYZ and metric heights are unavailable and left blank.',
                    task_tracking='linked_2d', depth_enabled=False, calibrated_3d=False,
                    reference_geometry='Approved A/B/C layout in Top pixels; Side pickup and box center both manually clicked before trial. P selects ball, O selects box; right-click also places box.',
                    flip_orientation='CW down 90 degrees, CCW back 90 degrees to the starting pose; J/L record operator intent without extra confirmation. No measured angle or direction verification.',
                    collision_model='Top pixel palm-center and sphere proxies; no Side obstacle rendering.')
            experiment._save_metadata()
            experiment.metadata.update(
                grasp_control='G manually toggles grab/drop with a valid position for attachment; BLE delivery is recorded independently.',
                ble_protocol='CEL v0.2.1, verified at connection; CW=5, CCW=6, STOP=FF06',
                repeat_control='Single key/click selects a 1 Hz Static pulse direction; X/Stop, focus loss, stale UI, trial changes or disconnect cancel. No replay.',
                participant_view='Constant orange-red ball; no hand trails or state cues. Dashed reference and XYZ diagnostics in operator view only.',
                flip_orientation=('CW down 90 degrees, CCW back 90 degrees to the starting pose; J/L record intent without extra confirmation; angle not measured.'
                                  if abc else 'F confirms flips; no measured orientation. Fixed tabletop XYZ does not rotate with the hand.'))
            experiment._save_metadata()
        if args.grasp_mode == 'wearable':
            experiment.metadata['reference_alignment'] = 'Fixed palm-minus-knuckle-center offset at setup or first valid top observation; no fitting to the path.'
            experiment.metadata['wearable_grip'] = 'Side-only four-finger MCP/PIP/DIP consensus: 3/4, 2/3 or 2/2 usable chains. Top palm hull proximity, no Top finger-pose vote. World landmarks are model estimates, not calibrated XYZ.'
            experiment._save_metadata()
        def finalize_experiment():
            experiment.consume_commands(ble)
            experiment.close()
            try:
                from .trial_report import generate_report
                print(f'[REPORT] {generate_report(experiment.directory)}')
            except Exception as exc:
                print(f'[REPORT] Records saved; report failed: {exc}. Regenerate with python -m astra_demo.trial_report "{experiment.directory}"')
        cleanup.callback(finalize_experiment)
        print(f'[EXPERIMENT] {experiment.directory}')
        # Register each resource as it opens, so partial startup failures also clean up.
        top_cap = open_uvc_camera(args.top_camera, cfg.frame_width, cfg.frame_height,
                                  cfg.camera_fps, 'top', cfg.strict_camera_ids)
        cleanup.callback(top_cap.release)
        side_cap = open_uvc_camera(args.side_camera, cfg.frame_width, cfg.frame_height,
                                   cfg.camera_fps, 'side', cfg.strict_camera_ids)
        cleanup.callback(side_cap.release)
        rgb_marker = None
        if abc:
            from .phase_marker import PhaseMarkerRecorder
            rgb_marker = PhaseMarkerRecorder()
            experiment.command_observer = rgb_marker.command
            experiment.metadata['rgb_marker'] = dict(
                enabled=True, units='a.u.', method='continuous_image_motion',
                independent_of_task=True, setup_status='ready_without_calibration',
                point='Yellow component near index DIP-to-tip segment; no palm substitution.',
                axes='Top X right/Y up; Side Z image-up proxy, not metric XYZ.',
                zero='First valid sample per view after trial setup; fixed across cues and reacquisition.',
                synchronization='Independent host capture times; view age >250ms omitted.',
                limitations='Perspective and finger rotation affect image trends; missing/ambiguous tape is blank.')
            experiment._save_metadata()
        factory, drawing = load_hands_api()
        hands_top = factory.Hands(max_num_hands=1, model_complexity=1,
                                  min_detection_confidence=cfg.top_min_detection_confidence,
                                  min_tracking_confidence=cfg.top_min_tracking_confidence)
        cleanup.callback(hands_top.close)
        hands_side = factory.Hands(max_num_hands=1, model_complexity=1,
                                   min_detection_confidence=cfg.side_min_detection_confidence,
                                   min_tracking_confidence=cfg.side_min_tracking_confidence)
        cleanup.callback(hands_side.close)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        def make_top_tracker():
            return TopHandTracker(cfg.top_region_flow_s, cfg.trail_min_cutoff, cfg.trail_beta, cfg.trail_deadband_px)
        tracker = make_top_tracker()
        prop = VirtualProp(size_follow=cfg.prop_size_follow, size_held=cfg.prop_size_held)
        context = GrabContext()
        wearable = (WearableGrip(cfg.pinch_median_window, gesture_gap_ms=cfg.grasp_gesture_gap_ms,
                    enter_frames=cfg.enter_frames, exit_frames=cfg.release_min_frames,
                    release_hold_ms=cfg.release_hold_ms) if args.grasp_mode == 'wearable' else None)
        if abc:
            print('[GRIP] Operator G toggles grab/drop; no pinch, opening sequence or proximity gate.')
        elif wearable:
            print('[GRIP] Wearable mode: Side four-finger closure; Top palm proximity. G then open-to-close.')
        else:
            print('[GRIP] Pinch baseline: Side thumb/index closure, Top palm region; G is still required.')
        history = deque(maxlen=max(1, cfg.pinch_median_window))
        panel = OperatorPanel(ble, Actuation(cfg.manual_voltage_mv, cfg.manual_duration_s,
                                             cfg.manual_frequency_hz), cfg.grab_confirmation_enabled and not abc)
        panel.configure_task(task)
        if abc:
            panel.linked_views = True  # Enables manual Side placement.
            if rgb_marker:
                panel.xyz_status = rgb_marker.display_status()
        if task:
            panel.trial_status = task.status
        panel.record_mode = args.record_content.title()
        linked = None
        side_window_transform = None
        if args.linked_views:
            from .linked_views import LinkedSideView, camera_point
            palm_inset = 1. if abc else cfg.side_grip_palm_inset
            linked = LinkedSideView(cfg.optical_flow_hold_s, cfg.side_tracking_min_cutoff,
                                    cfg.side_tracking_beta, cfg.side_tracking_deadband_px, palm_inset)
            panel.linked_views = True
            panel.trial_status = task.status if task else 'Place the side ball, then N when Side View sees your hand. START is a guide only.'
            experiment.metadata['linked_views'] = dict(
                coordinate_system='Separate displayed top/side pixel coordinates; one shared grasp state.',
                side_motion='Fixed grasp-time image-pixel offset from palm, per-axis adaptive smoothing and deadband; no per-frame rotation/scale of the offset. Last position retained on release.',
                held_palm_inset=palm_inset,
                held_attachment='palm center in both views' if abc else 'fixed grasp offset',
                side_rise='Initial side ball y minus current side ball y, in pixels; not physical height.',
                depth_enabled=False, calibrated_3d=False)
            experiment._save_metadata()
            print('[MODE] Linked RGB views: no depth; image-space scene. Tape recording is independent. Click Side View to place the ball and box.')
        directory = experiment.directory / 'videos'
        recorders = DualRecorder(directory, cfg.recording_fps, experiment.clock, logger.run_id)
        cleanup.callback(recorders.close)
        ui_recorder = StreamRecorder(directory, 'operator', cfg.recording_fps, experiment.clock, logger.run_id)
        def stop_ui_recording():
            was_active = ui_recorder.active
            ui_recorder.stop()
            panel.ui_record_active = False
            if was_active:
                experiment.event('operator_video_stopped', file=str(ui_recorder.path.relative_to(experiment.directory)))
        cleanup.callback(stop_ui_recording)
        ui_auto_record = False
        experiment.metadata['operator_ui_recording'] = dict(
            automatic=cfg.record_operator_ui, files=[], frame_size=[1180, 900],
            content='Full application canvas with both displayed camera views, settings, buttons and states; excludes desktop, OS cursor and title bar.',
            clock='Same session clock. capture_t_s is UI render completion; top_capture_t_s and side_capture_t_s identify the source camera frames.',
            lifecycle='N starts trial videos; E, trial completion, restart and exit stop all streams. No idle recording; U/T/Y/B apply only during a trial.')
        experiment._save_metadata()
        # Stop hardware before potentially slower video/camera cleanup.
        cleanup.callback(ble.stop)
        print(f'[VIDEO] Separate top/side recordings: {directory}')
        if cfg.show_operator_window:
            panel.open()
        open_video_window('Workspace | Top camera')
        if cfg.show_side_color_window:
            open_video_window('Tracking | Side camera')
            if linked:
                def place_from_side_window(event, x, y, flags, param):
                    from .linked_views import camera_point
                    if event == cv2.EVENT_LBUTTONDOWN or (abc and event == cv2.EVENT_RBUTTONDOWN):
                        point = camera_point(x, y, side_window_transform, linked.size)
                        if point is not None:
                            panel.queue_side_click(point, box=event == cv2.EVENT_RBUTTONDOWN)
                cv2.setMouseCallback('Tracking | Side camera', place_from_side_window)
        ble.start()
        initialized = previous_held = previous_near = False
        route = None
        last_panel = measured_fps = 0.0
        previous_tick = time.monotonic()
        if task:
            print('[ABC] Place Side ball + box, then N at A; G pick/drop and J/L rotation requests need no extra confirmation.')
            print('[ABC] Tap WASD / I (Normal Up) / K (Normal Down): 1 Static pulse/s. X/Stop ends repetition.')
            print('[ABC] J: CW=5 down 90 deg. L: CCW=6 back 90 deg. G: manual grab/drop + one vibration. RGB: no extra flip confirmation; angle not measured.')
            if cfg.ble_enabled:
                print('[ABC] BLE output requires automatic CEL v0.2.1 version verification.')
            print('[ABC] First direction starts the timer. After capture and lowering, L requests the return; finish rotating then E ends recording. 0: reset, Q: save/exit.')
        else:
            print('[INFO] G grasp cue, C cancel cue, R release, T/Y record top/side, B both, Q exit.')
            print('[TRIAL] N: Side View must see palm/fingers. START is a visual guide only. First direction starts timer.')
            print('[TRIAL] 0: reset and restart (keeps Side placement). E: end trial. Q: save reports and exit.')
        ui_start_mode = 'records with each trial' if cfg.record_operator_ui else 'trial UI recording is disabled'
        print(f'[VIDEO] Operator UI {ui_start_mode}. N starts; E/end stops all video. No idle recording.')

        while True:
            ok_top, raw_top = top_cap.read()
            top_mono = time.monotonic()
            top_timestamp = experiment.clock.unix_ms(top_mono)
            ok_side, raw_side = side_cap.read()
            side_mono = time.monotonic()
            side_timestamp = experiment.clock.unix_ms(side_mono)
            now = time.monotonic()
            raw_top = raw_top if ok_top else None
            raw_side = raw_side if ok_side else None
            top = None if raw_top is None else (cv2.flip(raw_top, 1) if cfg.top_mirror_horizontal else raw_top.copy())
            side = None if raw_side is None else (cv2.flip(raw_side, 1) if cfg.side_mirror_horizontal else raw_side.copy())
            if linked and side is not None:
                if linked.set_frame_size(side.shape[1::-1]):
                    experiment.finish('side_resolution_changed')
                    ble.trial_id = ''
                    recorders.close()
                    panel.trial_status = 'Side resolution changed; release and click its ball position again'
                    if abc:
                        scene.side_origin = scene.box_center = None
                        scene.box_manual = False
                        scene.revision += 1
                        task.reset()
                        panel.stop_output()
                if panel.side_place_request is not None:
                    try:
                        linked.place(panel.side_place_request, locked=previous_held or experiment.active is not None)
                        if abc:
                            scene.place_side(linked.origin, linked.size)
                            panel.place_side_mode('box')
                            experiment._json('abc_scene.json', scene.snapshot())
                        experiment.event('side_ball_placed', mono=side_mono,
                                         side_ball_xy=list(linked.origin), side_frame_size=list(linked.size))
                        panel.trial_status = ('Click Side to place the box center, then N. P: ball / O: box.' if abc else
                                              'Side ball ready. Show palm/fingers in Side View, then N. START is a guide only.')
                        panel.message = 'Ball placed. Next Side click sets box center.' if abc else 'Side ball placed / no hardware output'
                    except ValueError as exc:
                        panel.message = str(exc)
                    panel.side_place_request = None
                if abc and panel.side_box_request is not None:
                    try:
                        if previous_held or experiment.active:
                            raise ValueError('End the trial and release the ball before moving the Side box')
                        scene.place_box(panel.side_box_request)
                        experiment.event('side_box_placed', box_center_px=scene.box_center.tolist())
                        experiment._json('abc_scene.json', scene.snapshot())
                        panel.message = 'Side box placed / no hardware output'
                        panel.trial_status = 'Side ball and box ready. N: start at A. Tap WASD / I / K: BLE check.'
                    except ValueError as exc:
                        panel.message = str(exc)
                    panel.side_box_request = None
            if top is not None and not initialized:
                route = scene_route(top)
                prop.initialize_at(route['pickup'])
                initialized = True
            if abc and top is not None:
                route = scene_route(top)
                prop.size_follow = prop.size_held = scene.radius

            top_obs, _ = detect_hand(hands_top, top, cfg.top_hand_process_scale,
                                      allow_edge_on=True)
            side_input = enhance_for_hand_detection(side, clahe) if side is not None and cfg.side_use_clahe else side
            side_obs, side_landmarks = detect_hand(hands_side, side_input, cfg.hand_process_scale,
                                                  allow_edge_on=linked is not None or abc)
            if rgb_marker:
                rgb_marker.observe(top, side, top_mono, side_mono, top_hand=top_obs, side_hand=side_obs)
            if wearable:
                top_obs, side_obs = wearable_observation(top_obs, top=True), wearable_observation(side_obs)
            tracker.observe(top, top_obs, now)
            top_anchor_valid = tracker.valid
            side_anchor_valid = side_obs is not None and (side_obs.grip_xy is not None if wearable else side_obs.pinch_ratio is not None)
            experiment.consume_commands(ble)
            start_trial = False
            if experiment.active and top is not None and list(top.shape[1::-1]) != experiment.active['frame_size']:
                experiment.finish('camera_resolution_changed')
                ble.trial_id = ''
                panel.trial_status = 'Trial ended: top camera resolution changed'
            if panel.trial_request == 'restart':
                # The panel already cancelled queued feedback and requested zero output.
                # End the old attempt before clearing its live state; keep its files.
                experiment.event('trial_reset_requested')
                experiment.finish('reset')
                ble.trial_id = ''
                recorders.close()
                stop_ui_recording()
                ui_auto_record = False
                experiment.clear_display_trace()
                if task:
                    task.reset()
                    for view in scene_views.values():
                        view.reset_trace()
                context, history = GrabContext(), deque(maxlen=max(1, cfg.pinch_median_window))
                if wearable:
                    wearable.reset()
                tracker = make_top_tracker()
                tracker.observe(top, top_obs, now)
                top_anchor_valid = tracker.valid
                previous_held = previous_near = False
                panel.set_grab_state(False, tracker.status)
                panel.grasp_status = 'WAIT CUE [G]'
                if route is not None:
                    prop.initialize_at(route['pickup'])
                if linked:
                    linked.reset()
            if panel.trial_request == 'end':
                panel.stop_output()
                end_result = getattr(task, 'pending_result', None) or 'incomplete'
                experiment.finish(end_result)
                ble.trial_id = ''
                recorders.close()
                panel.trial_status = f'Trial ended {end_result}. N: new trial. 0: restart. Q: reports.'
                if task:
                    task.reset()
                    context, previous_held = GrabContext(), False
                    prop.initialize_at(route['pickup'])
                    linked.reset()
            elif panel.trial_request == 'start' or panel.restart_pending:
                try:
                    if (previous_held or ble.busy or panel.grasp_feedback.output_pending
                            or (ble.enabled and ble.feedback_snapshot().state == 'stopping')):
                        raise ValueError('Release the ball and wait for output to stop before a new trial')
                    if top is None or side is None:
                        raise ValueError('Both camera video feeds are required for dual recording')
                    tape_ready = abc and any(m is not None for m in rgb_marker.markers)
                    if not side_anchor_valid and not tape_ready:
                        raise ValueError('Show your palm and fingers in Side View, then press N')
                    if linked and not linked.ready:
                        raise ValueError('Click the side camera image to place its ball before starting a trial')
                    if task:
                        task.start_ready(None)
                    route = scene_route(top)
                    reference_obs = top_obs if top_obs is not None and np.isfinite(top_obs.pinch_xy).all() else None
                    experiment.begin(route, reference_obs.palm if reference_obs else None,
                                     (reference_obs.palm if abc else reference_obs.pinch_xy) if reference_obs else None, top.shape[1::-1],
                                     tolerance_px=args.drop_tolerance_px)
                    experiment.active['simulated'] = not cfg.ble_enabled
                    ble.trial_id = experiment.trial_id
                    panel.grasp_feedback.cancel()
                    panel.manual_request = None
                    context, history = GrabContext(), deque(maxlen=max(1, cfg.pinch_median_window))
                    if wearable:
                        wearable.reset()
                    tracker = make_top_tracker()
                    tracker.observe(top, top_obs, now)
                    top_anchor_valid = tracker.valid
                    prop.initialize_at(route['pickup'])
                    if task:
                        task.start(None, now)
                        experiment._json('abc_scene.json', scene.snapshot())
                        for view in scene_views.values():
                            view.reset_trace()
                    if linked:
                        linked.reset()
                        experiment.active['side_initial_ball_px'] = list(linked.origin)
                        experiment.active['side_frame_size'] = list(linked.size)
                        experiment._save_trials()
                    start_trial = True
                    panel.restart_pending = False
                    panel.message = 'New trial ready / first direction starts the timer'
                except ValueError as exc:
                    if panel.restart_pending:
                        panel.trial_status = 'Restart waiting: ' + str(exc).replace(', then press N', '')
                        panel.message = 'Restart will continue automatically when ready; Esc/X cancels'
                    else:
                        panel.trial_status = str(exc)
                        experiment.event('trial_start_rejected', reason=str(exc))
            panel.trial_request = None
            prompt_sent, cancel_prompt = panel.grasp_feedback.poll_prompt(observed_at=min(top_mono, side_mono))
            # Current detected/flow-validated palm region, not a fingertip or cached XY.
            current_near = top_anchor_valid and near_palm(tracker, prop.pos_xy,
                                                         prop.size_follow, cfg.grab_radius_margin_px)
            if task:
                current_near = current_near and task.grasp_enabled
                task.poll_output(min(top_mono, side_mono))
            if linked:
                current_near = current_near and linked.ready
            if prompt_sent or cancel_prompt or (not previous_held and (not current_near or not previous_near)):
                history.clear()
            previous_near = current_near
            ratio = None
            if side_obs is not None and side_obs.pinch_ratio is not None:
                history.append(side_obs.pinch_ratio)
                ratio = float(np.median(history))
            else:
                history.clear()
            manual = panel.manual_request
            panel.manual_request = None
            grab_available = (top_anchor_valid and side_obs is not None and linked is not None and linked.ready)
            if abc:
                out = update_manual_grab(context, manual, tracking_valid=grab_available)
            elif wearable:
                out = wearable.update(top_obs, side_obs, prompt_sent=prompt_sent,
                    cancel_prompt=cancel_prompt or manual == 'cancel', manual_release=manual == 'release',
                    hand_near_ball=current_near, top_hand_present=top_anchor_valid,
                    top_anchor_valid=top_anchor_valid, setup_ready=linked is None or linked.ready,
                    now_ms=round(now*1000))
            else:
                out = update_grab_state(
                    context, ratio, prompt_sent=prompt_sent, cancel_prompt=cancel_prompt or manual == 'cancel',
                    hand_near_ball=current_near, top_hand_present=top_anchor_valid,
                    pinch_enter=cfg.pinch_enter_ratio, pinch_ready=cfg.pinch_ready_ratio,
                    pinch_exit=cfg.pinch_release_ratio, enter_frames=cfg.enter_frames,
                    exit_frames=cfg.release_min_frames,
                    now_ms=round(now * 1000), release_hold_ms=cfg.release_hold_ms,
                    gesture_gap_ms=cfg.grasp_gesture_gap_ms, top_anchor_valid=top_anchor_valid,
                    setup_ready=linked is None or linked.ready,
                    manual_release=manual == 'release')
            context, held = out.context, out.trigger_on
            hand_xy = tracker.follow_ball(prop.pos_xy, held, palm_centered=bool(abc and linked))
            tracking_status = tracker.status if top is not None else 'Top camera unavailable'
            prop.update(hand_xy, held, side_timestamp, cfg.prop_idle_visible)
            if linked:
                linked.update(side, side_obs, held, now)
            decision_mono = time.monotonic()
            if task:
                task.update_linked(top_obs, side_obs, prop, linked, held, decision_mono,
                    manual=manual == 'release', position_valid=hand_xy is not None and linked.tracked is not None)
            frame_trial_id = experiment.trial_id
            rgb_marker_record = None
            if rgb_marker:
                rgb_marker_record = rgb_marker.record(experiment.clock, experiment.trial_id,
                    experiment.active['setup_t_s'] if experiment.active else None)
                panel.xyz_status = rgb_marker.display_status()
            panel.update_measurements(experiment.active, rgb_marker_record, experiment.clock.seconds(decision_mono))
            if experiment.active or not abc:
                experiment.sample(mono=decision_mono, top_mono=top_mono, side_mono=side_mono,
                    observation=top_obs, side_valid=side_obs is not None, tracked=hand_xy, ball=prop.pos_xy,
                    held=held, pinch_ratio=ratio, tracking_status=tracking_status,
                    recordings={name: stream.active for name, stream in recorders.streams.items()},
                    linked=linked.record(side_obs) if linked else None,
                    rgb_marker=rgb_marker_record,
                    grasp=dict(grasp_status=out.status, grasp_near=int(current_near),
                               grasp_top_valid=int(top_anchor_valid),
                               side_pinch_raw=side_obs.pinch_ratio if side_obs is not None else None,
                               grip_top_openness=top_obs.grip_openness if top_obs is not None else None,
                               grip_side_openness=side_obs.grip_openness if side_obs is not None else None,
                               **{f'grip_side_{finger}': side_obs.grip_fingers[i] if side_obs is not None else None
                                  for i, finger in enumerate(('index', 'middle', 'ring', 'little'))},
                               grip_source='operator_G' if abc else wearable.source if wearable else 'side_pinch'),
                    anchor_valid=top_obs is not None and np.isfinite(top_obs.pinch_xy).all(),
                    top_tracking=tracker.record(), operator_recording=ui_recorder.active,
                    task=task.record() if task else None)
            if manual:
                experiment.event('manual_' + manual, mono=decision_mono)
                if manual == 'cancel' and experiment.active:
                    experiment.active['manual_interventions'] += 1
            panel.grasp_status = out.status
            panel.grab_available = grab_available
            panel.set_grab_state(held, tracking_status)
            if held and not previous_held:
                logger.log_grab(side_timestamp, side_obs.pinch_norm if side_obs else None, out.target_key)
                experiment.grasp(decision_mono)
                source = 'operator_G' if abc else 'prompt+nearby+' + (wearable.source + '_grip' if wearable else 'open-to-pinch')
                print(f'[GRAB] ts_ms={side_timestamp} source={source}')
            elif previous_held and not held and not task:
                ended = experiment.release(prop.pos_xy, manual=manual == 'release',
                    position_valid=(hand_xy is not None and top_obs is not None), mono=decision_mono)
                if ended:
                    panel.trial_status = f"Trial {ended['trial_id']}: {ended['result'].upper()} | 0: restart, N: next, Q: reports"
                    ble.trial_id = ''
                print(f'[RELEASE] ts_ms={side_timestamp} source={"manual" if manual == "release" else "open fingers"}')
            if task and task.result and experiment.active:
                ended = experiment.finish(task.result, mono=decision_mono)
                panel.stop_output()
                ble.trial_id = ''
                panel.trial_status = f"Trial {ended['trial_id']}: {ended['result'].upper()} | 0: restart, N: next"
            previous_held = held
            if experiment.active:
                phase = 'READY - send first direction' if experiment.active['start_t_s'] is None else (task.status if task else out.status)
                panel.trial_status = f'Trial {experiment.trial_id}: {phase} | 0: restart | E: end trial'

            state_label = 'HELD' if held else 'RELEASED'
            # A single immutable snapshot keeps the two views consistent even if BLE
            # changes state between drawing them. Precise trigger times stay in the logs.
            haptic_display = ble.feedback_snapshot()
            if scene_views:
                for name, frame in (('top', top), ('side', side)):
                    ball = prop.pos_xy if name == 'top' else linked.prop.pos_xy if linked.ready else None
                    point = tracker.record()['point'] if name == 'top' else side_obs.palm if side_obs is not None else None
                    if frame is not None:
                        scene_views[name].draw(frame, ball)
                    if frame_trial_id:
                        scene_views[name].trace(point, now)
                if top is not None:
                    show_video_frame('Workspace | Top camera', top)
                if side is not None and cfg.show_side_color_window:
                    side_window_transform = show_video_frame('Tracking | Side camera', side)
            if top is not None and not scene_views:
                draw_workspace_boundary(top)
                if route is not None:
                    draw_trial_overlay(top, route, experiment, args.drop_tolerance_px)
                prop.draw(top, show_label=False)
                header_right = draw_haptic_status(top, haptic_display, experiment.clock)
                draw_view_status(top, out.status, held, left=header_right+8,
                                 right=top.shape[1]-route_legend_width(top.shape[1])-16)
                text(top, f'{logger.subject_id}  Trial {frame_trial_id or "-"}  t={experiment.clock.seconds(top_mono):.3f}s',
                     (12, top.shape[0]-9), .33)
                show_video_frame('Workspace | Top camera', top)
            if side is not None and not scene_views:
                if side_landmarks is not None:
                    drawing.draw_landmarks(side, side_landmarks, factory.HAND_CONNECTIONS)
                side_state = 'OPENING' if context.release_started_ms is not None else state_label
                if linked:
                    linked.draw(side)
                    if not linked.ready:
                        side_state = 'RELEASE, THEN PLACE BALL' if held else 'CLICK TO PLACE BALL'
                    elif held and linked.tracked is None:
                        side_state = 'HELD | TRACKING LOST'
                header_right = draw_haptic_status(side, haptic_display, experiment.clock)
                draw_view_status(side, side_state, held, left=header_right+8)
                text(side, f'{logger.subject_id}  Trial {frame_trial_id or "-"}  t={experiment.clock.seconds(side_mono):.3f}s',
                     (12, side.shape[0]-9), .33)
                if cfg.show_side_color_window:
                    side_window_transform = show_video_frame('Tracking | Side camera', side)

            frames = {'top': top, 'side': side} if panel.record_mode == 'Display' else {'top': raw_top, 'side': raw_side}
            recording_now = time.monotonic()
            recording_epoch = experiment.clock.unix_ms(recording_now)
            capture_times = {'top': top_mono, 'side': side_mono}
            capture_timestamps = {'top': top_timestamp, 'side': side_timestamp}
            if start_trial:
                recorders.close()
                stop_ui_recording()
                ui_auto_record = False
                # Closing a previous segment can take time: use a fresh, shared recording origin.
                recording_now = time.monotonic()
                recording_epoch = experiment.clock.unix_ms(recording_now)
                started = recorders.start_both(frames, recording_now, recording_epoch,
                    capture_times=capture_times, timestamps=capture_timestamps, trial_id=frame_trial_id)
                experiment.set_videos(recorders, operator=ui_recorder)
                experiment.event('video_started' if started else 'video_start_failed',
                    mono=recording_now, files=experiment.active['videos'])
                if not started:
                    experiment.finish('recording_failed')
                    ble.trial_id = ''
                    panel.trial_status = 'Trial ended: could not record both cameras'
                else:
                    ui_auto_record = cfg.record_operator_ui
            # All recording entry points share the same trial gate, including U/T/Y/B.
            if not experiment.active:
                recorders.close()
                stop_ui_recording()
                ui_auto_record = False
                if panel.record_requests or panel.ui_record_request:
                    panel.message = 'Press N to start a trial before recording'
                panel.record_requests.clear()
                panel.ui_record_request = False
            if panel.restart_pending:
                panel.record_requests.clear()
            for request in panel.record_requests:
                recorders.toggle(request, frames, recording_now, recording_epoch,
                    capture_times=capture_times, timestamps=capture_timestamps, trial_id=frame_trial_id)
                experiment.event('video_toggle', mono=recording_now, view=request,
                    active={name: stream.active for name, stream in recorders.streams.items()},
                    files={name: str(stream.path.relative_to(experiment.directory)) if stream.path else ''
                           for name, stream in recorders.streams.items()})
                experiment.set_videos(recorders, operator=ui_recorder)
            panel.record_requests.clear()
            recorders.submit(frames, capture_timestamps, capture_times, frame_trial_id)
            panel.record_active = {name: stream.active for name, stream in recorders.streams.items()}
            panel.record_status = {name: stream.status for name, stream in recorders.streams.items()}
            tick = time.monotonic()
            fps = 1.0 / max(.001, tick - previous_tick)
            measured_fps = fps if measured_fps == 0 else .9 * measured_fps + .1 * fps
            previous_tick = tick
            if ((cfg.show_operator_window or ui_recorder.active or ui_auto_record or panel.ui_record_request)
                    and tick - last_panel >= 1.0 / max(1, cfg.operator_panel_fps)):
                if panel.ui_record_request:
                    if ui_recorder.active:
                        stop_ui_recording()
                        ui_auto_record = False
                    else:
                        ui_auto_record = True
                    panel.ui_record_request = False
                panel.ui_record_active = ui_recorder.active or ui_auto_record
                operator_top = scene_views['top'].operator_frame(top) if scene_views and top is not None else top
                operator_side = scene_views['side'].operator_frame(side) if scene_views and side is not None else side
                operator_frame = panel.render(operator_top, operator_side, fps=measured_fps, ble_status=ble.status_msg)
                ui_mono = time.monotonic()
                ui_epoch = experiment.clock.unix_ms(ui_mono)
                ui_sources = dict(top=experiment.clock.seconds(top_mono) if top is not None else None,
                                  side=experiment.clock.seconds(side_mono) if side is not None else None)
                if ui_auto_record:
                    ui_auto_record = False
                    started = ui_recorder.start(operator_frame, ui_mono, ui_epoch,
                        capture_mono=ui_mono, capture_epoch_ms=ui_epoch, trial_id=frame_trial_id, source_times=ui_sources)
                    experiment.event('operator_video_started' if started else 'operator_video_failed', mono=ui_mono,
                        file=str(ui_recorder.path.relative_to(experiment.directory)) if ui_recorder.path else '',
                        reason=ui_recorder.error)
                    panel.ui_record_active = started
                    if started:
                        experiment.metadata['operator_ui_recording']['files'].append(str(ui_recorder.path.relative_to(experiment.directory)))
                        experiment._save_metadata()
                        experiment.set_videos(recorders, operator=ui_recorder)
                    else:
                        panel.message = 'UI recording failed: ' + ui_recorder.error
                ui_recorder.submit(operator_frame, ui_epoch, ui_mono, frame_trial_id, source_times=ui_sources)
                if cfg.show_operator_window:
                    panel.show(frame=operator_frame)
                last_panel = tick
            panel.handle_key(cv2.waitKeyEx(1))
            if panel.quit_requested:
                break
            if cfg.show_operator_window and cv2.getWindowProperty(panel.window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
    print('[INFO] Exit. Recordings finalized.')


if __name__ == '__main__':
    main()
