"""Short linked ABC control sequence. No hardware or retained test recordings."""
import json
import tempfile
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import cv2
import numpy as np


def check_linked_sequence(output):
    """Replay the complete RGB task through real controls, without F or hardware."""
    from .abc_linked import LinkedAbcScene, LinkedAbcTask, LinkedSceneView
    from .experiment import Experiment
    from .manual_control import ManualBleController
    from .operator_panel import OperatorPanel
    scene = LinkedAbcScene()
    route = scene.reference((800, 600))
    scene.place_side([240, 310], (640, 480))
    scene.place_box([430, 150])
    controller = ManualBleController(False, '', '', protocol='v0.2.1')
    panel = OperatorPanel(controller, grab_confirmation=False)
    panel._key_down, panel._app_focused = None, lambda: True
    now = [100.]
    with tempfile.TemporaryDirectory() as temp, ExitStack() as cleanup, patch('time.monotonic', side_effect=lambda: now[0]):
        experiment = Experiment(Path(temp)/'session', 'SYNTHETIC', 'controls')
        cleanup.callback(experiment.close)
        trial = experiment.begin(route, scene.paths[0][0], scene.paths[0][0], (800, 600))
        trial['synthetic'] = True
        controller.trial_id = trial['trial_id']
        task = LinkedAbcTask(scene, experiment, controller, manual_grasp=True)
        panel.configure_task(task)
        task.start(None, now[0])
        linked = SimpleNamespace(ready=True, prop=SimpleNamespace(pos_xy=np.array([430., 310.])))
        prop = SimpleNamespace(pos_xy=scene.paths[0][0].copy())
        def observe(point, side_y=310., *, held=False, manual=False):
            now[0] += .4
            prop.pos_xy = np.asarray(point).copy()
            linked.prop.pos_xy = np.array([430., side_y])
            top, side = SimpleNamespace(palm=prop.pos_xy), SimpleNamespace(palm=linked.prop.pos_xy)
            task.update_linked(top, side, prop, linked, held, now[0], manual=manual)
            panel.held, panel.grab_available = held, True
            panel.update_measurements(experiment.active, dict(rgb_marker_top_valid=1, rgb_marker_side_valid=1,
                rgb_marker_delta_x_au=(point[0]-scene.paths[0][0][0])/600,
                rgb_marker_delta_y_au=(scene.paths[0][0][1]-point[1])/600,
                rgb_marker_delta_z_au=(310-side_y)/480,
                rgb_marker_phase='z' if task.phase in ('lift', 'release', 'lower', 'flip_ccw') else 'xy'),
                experiment.clock.seconds(now[0]))
        panel.handle_key(ord('w'))
        experiment.consume_commands(controller)
        for point in scene.paths[0][1:]:
            observe(point)
            observe(point)
        assert task.phase == 'pick'
        panel.handle_key(ord('g'))
        assert panel.manual_request == 'grab'
        trial['grasp_t_s'] = experiment.clock.seconds(now[0])
        observe(scene.paths[0][-1], held=True, manual=True)
        for point in scene.paths[1][1:]:
            observe(point, held=True)
            observe(point, held=True)
        assert task.phase == 'flip_cw'
        panel.handle_key(ord('f'))
        assert task.phase == 'flip_cw'  # Removed shortcut cannot confirm a rotation.
        panel.handle_key(ord('j'))
        assert task.phase == 'lift' and trial['flip_requests'][0]['target_degrees'] == 90
        panel.handle_key(ord('i'))
        observe(scene.paths[1][-1], 215, held=True)
        observe(scene.paths[1][-1], 185, held=True)
        observe(scene.paths[1][-1], 150, held=True)
        observe(scene.paths[1][-1], 150, held=True)
        assert task.phase == 'release' and task.entered_from_below
        panel.handle_key(ord('g'))
        assert panel.manual_request == 'release'
        observe(scene.paths[1][-1], 150, manual=True)
        assert task.captured and task.phase == 'lower'
        panel.handle_key(ord('k'))
        observe(scene.paths[1][-1], 310)
        observe(scene.paths[1][-1], 310)
        assert task.phase == 'flip_ccw'
        panel.handle_key(ord('l'))
        assert task.phase == 'complete' and task.pending_result == 'success', task.pending_result
        assert experiment.active is trial and trial['end_t_s'] is None
        observe(scene.paths[1][-1], 315)  # The final return is still part of the trial.
        panel.handle_key(ord('e'))
        assert panel.trial_request == 'end'
        experiment.finish(task.pending_result)
        assert trial['end_t_s'] > trial['return_requested_t_s']
        assert len(trial['flip_requests']) == 2 and not trial['flip_confirmations']
        experiment.consume_commands(controller)
        top = np.full((600, 800, 3), 205, np.uint8)
        side = np.full((480, 640, 3), 205, np.uint8)
        cv2.circle(side, (430, 150), 12, (0, 255, 255), -1)
        raw = side.copy()
        side_view, top_view = LinkedSceneView(scene, 'side'), LinkedSceneView(scene, 'top')
        side_view.draw(side, (430, 150))
        top_view.draw(top, scene.paths[1][-1])
        assert np.linalg.norm(side[150, 430].astype(float)-raw[150, 430]) < 230
        panel.xyz_status = 'Continuous tape XYZ [a.u.] | fixed trial zero'
        panel.trial_status = 'SYNTHETIC | G / J / L sequence complete without F'
        cv2.imwrite(str(output/'operator.png'), panel.render(top_view.operator_frame(top), side))
        history = list(panel.dashboard.history)
        panel.update_measurements(None, {}, experiment.clock.seconds(now[0]))
        assert list(panel.dashboard.history) == history  # End retains the plot, not idle samples.
        # Hidden recording controls cannot be triggered through the plot.
        panel.record_requests.clear()
        panel.on_mouse(cv2.EVENT_LBUTTONDOWN, 60, 650, None, None)
        assert not panel.record_requests
        panel.on_mouse(cv2.EVENT_LBUTTONDOWN, 510, 110, None, None)
        assert panel.advanced_recording
        panel.on_mouse(cv2.EVENT_LBUTTONDOWN, 60, 650, None, None)
        assert panel.record_requests == ['top']
        cv2.imwrite(str(output/'operator_advanced.png'), panel.render(top_view.operator_frame(top), side))
        panel.on_mouse(cv2.EVENT_LBUTTONDOWN, 510, 110, None, None)
        # A newly started trial clears the previous plot and resets its display timer.
        panel.update_measurements(dict(trial_id='002', setup_t_s=9., start_t_s=None), {}, 9.)
        assert len(panel.dashboard.history) == 1 and panel.dashboard.elapsed == 0
        cv2.imwrite(str(output/'side_transparency.png'), side)
        experiment.close()
    assert controller._thread is None
    return dict(result=trial['result'], no_extra_flip_confirmation=True,
                target_degrees=90, flip_requests=trial['flip_requests'])


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as folder:
        print(json.dumps(check_linked_sequence(Path(folder)), indent=2))
