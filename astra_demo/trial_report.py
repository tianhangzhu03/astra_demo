"""Regenerate evaluation tables and figures from saved records, without cameras or BLE.

Usage: python -m astra_demo.trial_report path/to/experiment_session
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .experiment import point_path_distance


def read_rows(path):
    with Path(path).open(newline='', encoding='utf-8') as file:
        return list(csv.DictReader(file))


def marker_displacement(trial, rows):
    """Keep table axes, subtract the first valid tape sample after trial setup once."""
    origin = trial['setup_t_s'] if trial.get('start_t_s') is None else trial['start_t_s']
    reference = reference_time = None
    result = []
    for row in rows:
        if row['trial_id'] != trial['trial_id'] or row.get('hand_position_source') != 'yellow_tape_registered_depth':
            continue
        t = float(row.get('hand_position_t_s') or row['side_capture_t_s'])
        if t < trial['setup_t_s']:
            continue
        point = np.array([float(row[f'hand_world_{axis}_mm'])
                          if row.get(f'hand_world_{axis}_mm') not in ('', None) else np.nan for axis in 'xyz'])
        valid = str(row.get('spatial_valid')) == '1' and np.isfinite(point).all()
        if valid and reference is None:
            reference, reference_time = point.copy(), t
        sample = dict(t_s=t, time_from_start_s=t-origin, marker_valid=int(valid), reference_t_s=reference_time)
        for i, axis in enumerate('xyz'):
            sample[f'marker_{axis}_mm'] = float(point[i]) if valid else None
            sample[f'delta_{axis}_mm'] = float(point[i]-reference[i]) if valid else None
            sample[f'reference_{axis}_mm'] = float(reference[i]) if reference is not None else None
        result.append(sample)
    return reference, result


def rgb_marker_report(directory, trial, rows, *, figures=True, reprocessed=False):
    """Independent RGB tape trends, retaining the recorded zero and every loss gap."""
    from .phase_marker import FIELDS
    origin = trial['start_t_s'] if trial.get('start_t_s') is not None else trial['setup_t_s']
    selected = []
    for row in rows:
        if row['trial_id'] != trial['trial_id'] or row.get('rgb_marker_source') not in ('yellow_tape_dual_rgb_affine', 'yellow_tape_dual_rgb_stereo', 'yellow_tape_phase_rgb', 'yellow_tape_continuous_rgb'):
            continue
        t = float(row['rgb_marker_t_s'])
        if t < trial['setup_t_s']:
            continue
        selected.append(dict(time_from_start_s=t-origin, **{k: row.get(k) for k in FIELDS}))
    if not selected:
        return {}
    stem = directory / f'trial_{trial["trial_id"]}_rgb_marker_displacement'
    write_table(stem.with_suffix('.csv'), selected, ['time_from_start_s', *FIELDS])
    valid = [r for r in selected if str(r['rgb_marker_valid']) == '1']
    units = selected[0]['rgb_marker_units']
    continuous = selected[0]['rgb_marker_source'] == 'yellow_tape_continuous_rgb'
    phased = continuous or selected[0]['rgb_marker_source'] == 'yellow_tape_phase_rgb'
    suffix = 'mm' if units == 'mm' else 'au'
    warnings = sum(str(r.get('rgb_marker_model_warning')) == '1' for r in valid)
    if figures:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        times, values = [], []
        previous = None
        previous_segment = None
        gaps = np.diff([row['time_from_start_s'] for row in selected])
        positive_gaps = gaps[gaps > 0]
        gap_limit = max(.25, min(1.5, 2.5*float(np.median(positive_gaps)))) if phased and len(positive_gaps) else .25
        for row in selected:
            t = row['time_from_start_s']
            segment = row.get('rgb_marker_z_segment')
            if previous is not None and (t-previous > gap_limit or (phased and not continuous and segment != previous_segment)):
                times.append((t+previous)/2)
                values.append([np.nan]*3)
            times.append(t)
            values.append([float(row[f'rgb_marker_delta_{a}_{suffix}']) if row.get(f'rgb_marker_delta_{a}_{suffix}') not in ('', None) else np.nan for a in 'xyz']
                          if str(row['rgb_marker_valid']) == '1' else [np.nan]*3)
            previous = t
            previous_segment = segment
        values = np.asarray(values)
        fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
        if phased:
            start = None
            for row in selected:
                t = row['time_from_start_s']
                if row.get('rgb_marker_phase') == 'z' and start is None:
                    start = t
                elif row.get('rgb_marker_phase') != 'z' and start is not None:
                    ax.axvspan(start, t, color='#ffe8c1', alpha=.4, zorder=0)
                    start = None
            if start is not None:
                ax.axvspan(start, selected[-1]['time_from_start_s'], color='#ffe8c1', alpha=.4, zorder=0)
        for i, (axis, color) in enumerate(zip('XYZ', ('#555555', '#e24a33', '#348abd'))):
            ax.plot(times, values[:, i], label=('Z proxy (Side)' if continuous and axis == 'Z' else f'Delta {axis}'), color=color, linewidth=1.4)
        if not np.isfinite(values).any():
            ax.text(.5, .5, 'No valid RGB direction samples\nAR / G / BLE operate independently',
                    ha='center', va='center', transform=ax.transAxes)
        ax.axhline(0, color='#aaaaaa', linewidth=.7)
        ax.set(xlabel='Time from first direction command (s)' if trial.get('start_t_s') is not None else 'Time from trial setup (s)',
               ylabel=f'Relative tape displacement ({units})',
               title=('SYNTHETIC | ' if trial.get('synthetic') else 'OFFLINE REPROCESSED | ' if reprocessed else '') + f'Trial {trial["trial_id"]} | yellow index tape | ' + ('continuous image motion / fixed trial zero' if continuous else 'phase-selected image motion' if phased else 'calibrated stereo RGB' if units == 'mm' else 'approximate dual RGB'))
        ax.title.set_fontsize(11)
        if times[-1] > times[0]:
            ax.set_xlim(times[0], times[-1])  # Keep leading/trailing tracking-loss intervals visible.
        ax.grid(alpha=.18)
        ax.legend(ncol=3)
        scale_note = 'Fixed board axes; mm follows printed square size. Live tape accuracy unverified.' if units == 'mm' else 'Each reference move = 1 a.u. per axis, not mm.'
        note = ('Top XY and Side Z proxy: fixed trial zeros. Orange = Normal cue; all axes remain recorded.\n'
                'Each view has its own image-height scale; not world XYZ. Missing samples stay blank.'
                if continuous else 'Top XY: trial zero. Side Z: separate zero per Normal segment; Up/Down share zero.\n'
                '1 a.u. = one image height in its view. Not world XYZ; view amplitudes are not metric-comparable. Gaps stay blank.'
                if phased else 'Zero = first valid sample after setup. ' + scale_note + '\n'
                f'Missing data stay blank. Quality warning on {warnings}/{len(valid)} available samples.')
        fig.supxlabel(note, fontsize=8)
        for suffix in ('png', 'svg'):
            fig.savefig(stem.with_suffix('.'+suffix), dpi=180)
        plt.close(fig)
    return dict(rgb_marker_valid_fraction=len(valid)/len(selected), rgb_marker_units=units,
                rgb_marker_model_warning_fraction=warnings/len(valid) if valid else None,
                rgb_marker_extrapolated_fraction=sum(str(r['rgb_marker_extrapolated']) == '1' for r in valid)/len(valid) if valid else None)


def accuracy_metrics(trial, rows):
    start = trial.get('start_t_s')
    end = trial.get('end_t_s')
    selected = [r for r in rows if r['trial_id'] == trial['trial_id'] and start is not None and
                end is not None and start <= float(r['top_capture_t_s']) <= end
                and (trial.get('task') != 'abc' or r.get('task_phase') in ('reach', 'pick', 'carry'))]
    valid, errors, outbound, returning = [], [], [], []
    path_length, covered_time = 0., 0.
    previous = None
    for row in selected:
        if str(row['top_valid']) != '1':
            previous = None
            continue
        p = np.array([float(row['palm_x_px']), float(row['palm_y_px'])])
        timestamp = float(row['top_capture_t_s'])
        if not np.isfinite(p).all():
            previous = None
            continue
        phase = 'return' if trial.get('grasp_t_s') is not None and timestamp >= trial['grasp_t_s'] else 'outbound'
        reference = trial['palm_return' if phase == 'return' else 'palm_outbound']
        reference_since = trial.get('palm_offset_t_s')
        if reference is not None and (reference_since is None or timestamp >= reference_since):
            error = point_path_distance(p, reference)
            errors.append(error)
            (returning if phase == 'return' else outbound).append(error)
        valid.append(row)
        if previous is not None:
            dt = timestamp - previous[0]
            if 0 < dt <= .25:
                path_length += float(np.linalg.norm(p - previous[1]))
                covered_time += dt
        previous = timestamp, p
    rmse = lambda values: float(np.sqrt(np.mean(np.square(values)))) if values else None
    return {
        'samples': len(selected), 'valid_palm_samples': len(valid),
        'tracking_valid_fraction': len(valid)/len(selected) if selected else None,
        'observed_time_s': covered_time, 'observed_path_length_px': path_length if valid else None,
        'trajectory_rmse_px': rmse(errors),
        'trajectory_mean_error_px': float(np.mean(errors)) if errors else None,
        'trajectory_p95_error_px': float(np.percentile(errors, 95)) if errors else None,
        'outbound_rmse_px': rmse(outbound), 'return_rmse_px': rmse(returning),
        'top_recording_fraction': sum(str(r['top_recording']) == '1' for r in selected)/len(selected) if selected else None,
        'side_recording_fraction': sum(str(r['side_recording']) == '1' for r in selected)/len(selected) if selected else None,
    }


def command_rows(events):
    commands = {}
    for event in sorted(events, key=lambda r: float(r['t_s'])):
        cid = event.get('command_id')
        if not cid:
            continue
        row = commands.setdefault(cid, {k: event.get(k, '') for k in (
            'session_id', 'trial_id', 'command_id', 'cue', 'role', 'commanded_voltage_mv',
            'commanded_duration_s', 'frequency_hz', 'connection_id', 'direction_code', 'commanded_channels_json',
            'repeat_id', 'repeat_index', 'repeat_interval_s')})
        row.setdefault('requested_t_s', '')
        row.setdefault('sent_t_s', '')
        row.setdefault('finished_t_s', '')
        row.setdefault('simulated', False)
        kind = event['event']
        if kind == 'command_requested': row['requested_t_s'] = float(event['t_s'])
        elif kind in ('command_sent', 'command_simulated'):
            row['sent_t_s'] = float(event['t_s'])
            row['simulated'] = kind == 'command_simulated'
        elif kind in ('command_completed', 'command_failed', 'command_expired', 'command_cancelled'):
            row['finished_t_s'] = float(event['t_s'])
        if kind != 'command_cancel_requested':
            row['status'] = kind.removeprefix('command_')
        if event.get('reason'): row['reason'] = event['reason']
    return list(commands.values())


def write_table(path, rows, fallback_fields):
    fields = list(dict.fromkeys(k for row in rows for k in row)) or fallback_fields
    with Path(path).open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fields)
        writer.writeheader()
        writer.writerows(rows)


def _figures(directory, trial, rows, commands, events, metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    identifier = trial['trial_id']
    start = trial.get('start_t_s')
    origin = trial['setup_t_s'] if start is None else start
    end = trial.get('end_t_s') or origin
    samples = [r for r in rows if r['trial_id'] == identifier and
               origin <= float(r['top_capture_t_s']) <= end]
    subject = json.loads((directory.parent/'session.json').read_text(encoding='utf-8'))['subject_id']
    prefix = 'SYNTHETIC | ' if subject.startswith('SYNTHETIC') else ''
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
    for ax, (phase, reference, color) in zip(axes, [('outbound', trial['palm_outbound'], '#2378b5'),
                                     ('return', trial['palm_return'], '#c36c23')]):
        if reference is not None:
            route = np.array(reference)
            ax.plot(route[:, 0], route[:, 1], '--', color='#a0a0a0', linewidth=3.5, label='Reference route')
        else:
            ax.text(.5,.95,'Palm reference unavailable',transform=ax.transAxes,ha='center',fontsize=9)
        x, y, previous = [], [], None
        for row in samples:
            t = float(row['top_capture_t_s'])
            actual_phase = 'return' if trial.get('grasp_t_s') is not None and t >= trial['grasp_t_s'] else 'outbound'
            valid = actual_phase == phase and str(row['top_valid']) == '1'
            if previous is not None and t-previous > .25:
                x.append(np.nan); y.append(np.nan)
            x.append(float(row['palm_x_px']) if valid else np.nan)
            y.append(float(row['palm_y_px']) if valid else np.nan)
            previous = t
        ax.plot(x, y, color=color, linewidth=1.4, label='Measured palm')
        if reference is not None:
            ax.scatter(*reference[0], marker='s', color='#444444', label='Phase start')
        ax.set(xlim=(0, trial['frame_size'][0]), ylim=(trial['frame_size'][1], 0),
               xlabel='Top-view x (px)', ylabel='Top-view y (px)', title=phase.title())
        ax.set_aspect('equal')
        ax.grid(alpha=.15)
        ax.legend(loc='best', fontsize=8)
    fig.suptitle(prefix + f"Trial {identifier} | {trial['result']} | {'VISION ONLY' if trial.get('simulated') else 'hardware session'}")
    error = metrics['trajectory_rmse_px']
    coverage = metrics['tracking_valid_fraction']
    caption = f"Raw palm path; RMSE: {error:.2f} px" if error is not None else 'No evaluable trajectory samples'
    if coverage is not None: caption += f' | valid samples: {coverage:.0%}'
    fig.supxlabel(caption + '\nGaps > 250 ms and missing detections are not interpolated.', fontsize=9)
    for suffix in ('png', 'svg'):
        fig.savefig(directory / f'trial_{identifier}_trajectory.{suffix}', dpi=180)
    plt.close(fig)

    cues = [c for c in commands if c.get('trial_id') == identifier]
    has_spatial = any(r.get('ball_world_z_mm') not in ('', None) for r in samples)
    has_linked = any(r.get('side_ball_y_px') not in ('', None) for r in samples)
    heights = [2, 1.5, max(2, .36*len(cues))] if has_spatial or has_linked else [2, max(2, .36*len(cues))]
    fig, timeline_axes = plt.subplots(len(heights), 1,
        figsize=(11, max(6, 3.5+.36*len(cues))+(2 if has_spatial or has_linked else 0)),
        sharex=True, gridspec_kw={'height_ratios': heights}, constrained_layout=True)
    motion, command_ax = timeline_axes[0], timeline_axes[-1]
    if has_spatial:
        ts, palm_z, ball_z, previous = [], [], [], None
        for row in samples:
            t = float(row['t_s'])
            if previous is not None and t-previous > .25:
                ts.append(t-origin); palm_z.append(np.nan); ball_z.append(np.nan)
            ts.append(t-origin)
            palm_z.append(float(row['hand_world_z_mm']) if str(row.get('spatial_valid')) == '1' else np.nan)
            ball_z.append(float(row['ball_world_z_mm']) if row.get('ball_world_z_mm') not in ('', None) else np.nan)
            previous = t
        timeline_axes[1].plot(ts, palm_z, color='#8B5A86', label='Measured palm surface')
        timeline_axes[1].plot(ts, ball_z, color='#145C8D', label='Virtual ball center (held on data loss)')
        timeline_axes[1].set_ylabel('Table height Z (mm)')
        timeline_axes[1].legend(fontsize=8, loc='upper right')
    elif has_linked:
        ts, palm_y, ball_y, previous = [], [], [], None
        for row in samples:
            t = float(row['t_s'])
            if previous is not None and t-previous > .25:
                ts.append(t-origin); palm_y.append(np.nan); ball_y.append(np.nan)
            ts.append(t-origin)
            palm_y.append(float(row['side_palm_y_px']) if row.get('side_palm_y_px') not in ('', None) else np.nan)
            ball_y.append(float(row['side_ball_y_px']) if row.get('side_ball_y_px') not in ('', None) else np.nan)
            previous = t
        timeline_axes[1].plot(ts, palm_y, color='#8B5A86', label='Detected side palm')
        timeline_axes[1].plot(ts, ball_y, color='#145C8D', label='Side ball display (position retained on loss)')
        timeline_axes[1].set_ylabel('Side image y (px)\nSmaller = higher in image')
        timeline_axes[1].invert_yaxis()
        timeline_axes[1].legend(fontsize=8, loc='upper right')
    times, xs, ys, previous = [], [], [], None
    for row in samples:
        t = float(row['top_capture_t_s'])
        if previous is not None and t-previous > .25:
            times.append(t-origin); xs.append(np.nan); ys.append(np.nan)
        valid = str(row['top_valid']) == '1'
        times.append(t-origin)
        xs.append(float(row['palm_x_px']) if valid else np.nan)
        ys.append(float(row['palm_y_px']) if valid else np.nan)
        previous = t
    motion.plot(times, xs, label='Palm x', color='#2378b5')
    motion.plot(times, ys, label='Palm y', color='#c36c23')
    motion.set(ylabel='Top-view position (px)', title=prefix + f'Trial {identifier} | synchronized host timeline')
    motion.legend(loc='upper right', fontsize=8)
    for label, key, color in [('Grasp', 'grasp_t_s', '#239958'), ('Release', 'release_t_s', '#713d98')]:
        if trial.get(key) is not None:
            x = trial[key]-origin
            for axis in timeline_axes: axis.axvline(x, linestyle=':', color=color, linewidth=1)
            motion.text(x, .04, label, transform=motion.get_xaxis_transform(), ha='center', color=color, fontsize=8,
                        bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .85, 'pad': 2})
    labels = []
    for index, command in enumerate(cues):
        labels.append(f"{command['command_id']} {command['role']} / {command['cue']}\n"
                      f"{command['commanded_voltage_mv']} mV, {command['commanded_duration_s']} s [{command['status']}]")
        sent = command['sent_t_s']
        if sent != '':
            color = '#777777' if command['simulated'] else '#2378b5'
            command_ax.barh(index, float(command['commanded_duration_s']), left=sent-origin, height=.55, color=color)
            command_ax.plot(sent-origin, index, '|', color='black')
            if command['finished_t_s'] != '':
                command_ax.plot(command['finished_t_s']-origin, index, 'x', color='#b34040', markersize=5)
        elif command['requested_t_s'] != '':
            command_ax.plot(command['requested_t_s']-origin, index, 'x', color='#b34040')
    if not cues:
        command_ax.text(.5, .5, 'No haptic commands recorded', transform=command_ax.transAxes, ha='center')
    command_ax.set_yticks(range(len(labels)), labels, fontsize=8)
    command_ax.invert_yaxis()
    command_ax.set_xlabel('Seconds from first directional cue (setup time if no cue was sent)')
    command_ax.set_title('Bars = commanded duration; x = terminal software event. No measured actuator output.', fontsize=9)
    for axis in timeline_axes: axis.grid(axis='x', alpha=.2)
    for suffix in ('png', 'svg'):
        fig.savefig(directory / f'trial_{identifier}_timeline.{suffix}', dpi=180)
    plt.close(fig)
    if has_linked:
        fig, ax = plt.subplots(figsize=(8,6), constrained_layout=True)
        points, previous = [], None
        for row in samples:
            t = float(row['side_capture_t_s'])
            if previous is not None and t-previous > .25:
                points.append([np.nan,np.nan])
            points.append([float(row['side_palm_x_px']),float(row['side_palm_y_px'])]
                          if row.get('side_palm_x_px') not in ('',None) else [np.nan,np.nan])
            previous = t
        points = np.asarray(points)
        ax.plot(*points.T, color='#8B5A86', linewidth=1.5, label='Detected side palm')
        if trial.get('side_initial_ball_px'):
            ax.scatter(*trial['side_initial_ball_px'], color='#145C8D', label='Initial side ball')
        width,height = trial.get('side_frame_size',[640,480])
        ax.set(xlim=(0,width),ylim=(height,0),xlabel='Side image x (px)',ylabel='Side image y (px)',
               title=prefix + f'Trial {identifier} | side-view trajectory')
        ax.set_aspect('equal'); ax.grid(alpha=.15); ax.legend(fontsize=8)
        fig.supxlabel('Image coordinates only; no physical height or 3D calibration. Missing detections are gaps.',fontsize=8)
        for suffix in ('png','svg'):
            fig.savefig(directory/f'trial_{identifier}_side_trajectory.{suffix}',dpi=180)
        plt.close(fig)
    if has_spatial:
        marker_mode = any(r.get('hand_position_source') == 'yellow_tape_registered_depth' for r in samples)
        # Full-rate samples, not the separate 1 Hz feedback summary. NaNs retain loss gaps.
        xyz_times, hand_xyz, ball_xyz, goal_delta = [], [], [], []
        previous = None
        for row in samples:
            t = float(row.get('hand_position_t_s') or row['side_capture_t_s'])-origin
            if previous is not None and t-previous > .25:
                xyz_times.append((t+previous)/2)
                hand_xyz.append([np.nan]*3); ball_xyz.append([np.nan]*3); goal_delta.append([np.nan]*3)
            xyz_times.append(t)
            hand_xyz.append([float(row.get(f'hand_world_{axis}_mm') or 'nan') for axis in 'xyz']
                            if str(row.get('spatial_valid')) == '1' else [np.nan]*3)
            ball_xyz.append([float(row.get(f'ball_world_{axis}_mm') or 'nan') for axis in 'xyz']
                            if str(row.get('ball_position_valid')) != '0' else [np.nan]*3)
            goal_delta.append([float(row.get(f'goal_from_hand_{axis}_mm') or 'nan') for axis in 'xyz'])
            previous = t
        fig, axes = plt.subplots(3 if marker_mode else 2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
        datasets = [('Index tape position' if marker_mode else 'Palm position', hand_xyz),
                    ('Virtual ball position (scene state)', ball_xyz), ('Box center minus index tape', goal_delta)]
        for ax, (label, values) in zip(axes, datasets):
            values = np.asarray(values)
            for column, (axis, color) in enumerate(zip('XYZ', ('#555555', '#e24a33', '#348abd'))):
                ax.plot(xyz_times, values[:, column], color=color, label=axis, linewidth=1.3)
            ax.set(ylabel='Position (mm)', title=label)
            ax.grid(alpha=.18); ax.legend(ncol=3, loc='upper right')
        axes[-1].set_xlabel('Time from first direction command (s)' if start is not None else 'Time from trial setup (s)')
        fig.suptitle(prefix + f'Trial {identifier} | fixed table XYZ; gaps indicate unavailable measurements')
        for suffix in ('png', 'svg'):
            fig.savefig(directory / f'trial_{identifier}_position_xyz.{suffix}', dpi=180)
        plt.close(fig)
        if marker_mode:
            reference, marker_rows = marker_displacement(trial, rows)
            write_table(directory / f'trial_{identifier}_marker_displacement.csv', marker_rows, ['t_s', 'marker_valid'])
            delta = np.asarray(hand_xyz)-(reference if reference is not None else np.full(3, np.nan))
            fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
            for column, (axis, color) in enumerate(zip('XYZ', ('#555555', '#e24a33', '#348abd'))):
                ax.plot(xyz_times, delta[:, column], color=color, label=('Z proxy (Side)' if continuous and axis == 'Z' else f'Delta {axis}'), linewidth=1.5)
            ax.axhline(0, color='#aaaaaa', linewidth=.7)
            ax.set(xlabel='Time from first direction command (s)' if start is not None else 'Time from trial setup (s)',
                   ylabel='Tape displacement (mm)', title=prefix + f'Trial {identifier} | yellow index tape only')
            ax.grid(alpha=.18); ax.legend(ncol=3, loc='best')
            fig.supxlabel('Zero = first valid tape position after trial setup. Fixed table axes; missing observations remain gaps.', fontsize=9)
            for suffix in ('png', 'svg'):
                fig.savefig(directory / f'trial_{identifier}_marker_displacement.{suffix}', dpi=180)
            plt.close(fig)
        fig = plt.figure(figsize=(8, 6), constrained_layout=True)
        ax = fig.add_subplot(projection='3d')
        path, previous = [], None
        for row in samples:
            t = float(row['t_s'])
            if previous is not None and t-previous > .25:
                path.append([np.nan]*3)
            path.append([float(row[f'hand_world_{axis}_mm']) for axis in 'xyz']
                        if str(row.get('spatial_valid')) == '1' else [np.nan]*3)
            previous = t
        points = np.asarray(path)
        ax.plot(*points.T, color='#8B5A86', linewidth=1.6, label='Measured index tape' if marker_mode else 'Measured palm surface')
        if trial.get('task') == 'abc':
            ball = np.array([[float(r[f'ball_world_{axis}_mm']) for axis in 'xyz'] for r in samples])
            ax.plot(*ball.T, color='#2378b5', linewidth=1, label='Virtual ball state')
            scene = trial['scene']
            for path in (scene['reach_world_mm'], scene['carry_world_mm']):
                ax.plot(*np.array(path).T, '--', color='#999999', linewidth=1)
        ax.set(xlabel='World X (mm)', ylabel='World Y (mm)', zlabel='Table height Z (mm)',
               title=prefix + f'Trial {identifier} | calibrated 3D hand trajectory')
        ax.legend(fontsize=8)
        note = ('ABC: measured hand point and virtual ball; dashed lines are the configured reference routes.' if trial.get('task') == 'abc'
                else 'Depth-assisted demo calibration; missing observations are gaps. No 3D reference path specified.')
        fig.supxlabel(note, fontsize=8)
        for suffix in ('png', 'svg'):
            fig.savefig(directory / f'trial_{identifier}_trajectory_3d.{suffix}', dpi=180)
        plt.close(fig)


def generate_report(directory, *, figures=True):
    directory = Path(directory)
    trials = json.loads((directory / 'trials.json').read_text(encoding='utf-8'))
    samples, events = read_rows(directory / 'trajectory.csv'), read_rows(directory / 'events.csv')
    commands = command_rows(events)
    reports = directory / 'reports'
    reports.mkdir(exist_ok=True)
    summaries = []
    for trial in trials:
        metrics = accuracy_metrics(trial, samples)
        summary = {k: trial.get(k) for k in ('trial_id', 'result', 'success', 'simulated',
            'completion_time_s', 'attempt_duration_s', 'placement_error_px', 'manual_interventions',
            'task', 'placement_error_mm', 'collision_seen', 'torsion_available', 'torsion_channels',
            'torsion_packet_status', 'torsion_direction_code', 'torsion_firmware_ready',
            'torsion_output_direction', 'flip_direction_verified', 'task_tracking',
            'flip_confirmation', 'flip_confirmations', 'flip_requests', 'flip_target_degrees', 'flip_angle_measured', 'return_requested_t_s', 'success_scope', 'collision_scope')}
        summary.update(metrics)
        summary.update(rgb_marker_report(reports, trial, samples, figures=figures))
        spatial_rows = [r for r in samples if r['trial_id'] == trial['trial_id'] and
                        r.get('ball_world_z_mm') not in ('', None)]
        if spatial_rows:
            valid = [r for r in spatial_rows if str(r.get('spatial_valid')) == '1']
            summary['spatial_valid_fraction'] = len(valid)/len(spatial_rows)
            summary['peak_hand_height_mm'] = max((float(r['hand_world_z_mm']) for r in valid), default=None)
        summaries.append(summary)
        if figures:
            _figures(reports, trial, samples, commands, events, metrics)
    write_table(reports / 'trial_summary.csv', summaries, ['trial_id', 'result', 'success'])
    write_table(reports / 'commands.csv', commands, ['command_id', 'trial_id', 'cue', 'status'])
    groups = {}
    for mode in ('hardware', 'vision_only'):
        subset = [s for s in summaries if bool(s['simulated']) == (mode == 'vision_only')]
        completed = [s['completion_time_s'] for s in subset if s['success']]
        groups[mode] = {'trials': len(subset), 'completed': len(completed),
                       'completion_rate': len(completed)/len(subset) if subset else None,
                       'mean_successful_completion_time_s': float(np.mean(completed)) if completed else None}
    (reports / 'summary.json').write_text(json.dumps({
        'groups': groups, 'trials': summaries,
        'trajectory_error_definition': 'Sample-weighted nearest-segment Euclidean distance to the predefined phase-specific palm route, in pixels.',
        'limitations': 'Trajectory error uses 2D image pixels. Optional 3D positions use the saved depth-assisted calibration; '
                       'they are estimates, not ground truth or 3D trajectory error. Cameras have no exposure synchronization. '
                       'RGB tape trends use an independent local affine direction map in a.u., not mm or metric 3D. '
                       'Each axis reference has its own arbitrary physical length. Review loss, extrapolation and cross-axis error before comparing trials.'
    }, indent=2, allow_nan=False), encoding='utf-8')
    return reports


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--no-figures', action='store_true')
    args = parser.parse_args()
    print(generate_report(args.directory, figures=not args.no_figures))
