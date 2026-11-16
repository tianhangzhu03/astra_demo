# Astra demo handoff

Updated 2026-11-15. Current task: lightweight Top/iPhone + Side/Astra RGB A-B-C demo. Chinese discussion, English UI. Keep implementation simple; run only checks justified by the current change. No real BLE output without the user's next test request.

## Current behavior

- Entry: Start_AR_Demo.cmd (BLE enabled); visual command: python -m astra_demo.main --scenario abc --rgb-only --no-ble --top-camera 2 --side-camera 3.
- Virtual obstacles, ball and downward-open box. Side P/O places ball/box. N starts trial and camera videos, E ends/saves, 0 restarts, Q saves/exits. Operator UI and Top/Side record only during active trials: N starts, E/completion/restart/Q stop. Idle U/T/Y/B cannot start recording. --no-ui-record disables per-trial UI auto-recording; ABC idle does not write trajectory rows.
- G toggles manual grab/drop with a valid hand position, no pinch/proximity gate. Held ball follows filtered palm centers in both views; release retains position. J/L request CW/CCW 90 degrees, no F confirmation or measured angle. Finish return motion then E.
- WASD/I/K selects finite Static pulses at 1 Hz; direct W to D supported, no intervening X. X stops. 2500 mV initial voltage, 500 mV steps. BLE protocol unchanged.
- Top obstacles now rounded bright sky-blue with subtle edge lighting/shadows; A/B/C labels removed. Original rectangular collision footprints remain conservative near rounded corners. Dashed route is Operator-only; actual hand trails are data-only. Constant orange-red ball (BGR 60/100/245); Top ball +25%, Side ball opacity .75, box .35. Website colors updated too.

## Tape measurement

MediaPipe observe_hand carries index_distal = landmarks 7/8 (DIP/TIP). yellow_marker.detect_marker uses HSV H18..38, S>=90, V>=65, 3x3 opening and connected components. Only a unique yellow component whose centroid is within the distal segment corridor is accepted. Radius = max(7 px, min(.22*hand scale, .65*segment length)); segment length must be 3px..0.8*hand scale. Missing landmarks, missing tape or multiple local candidates -> blank. No whole-frame fallback and no palm substitution. Black straps/magnets usually fail yellow HSV; dorsal PCB is outside the finger region. Actual hardware occlusion/precision still needs live evaluation.

phase_marker records continuous Top XY and Side image-up Z proxy, independent per-view validity and fixed per-trial zero. Shear/Normal only annotate phase. Units are image-height fractions, not metric XYZ. G/AR/BLE remain independent of tape detection.

## Cleanup status

Removed depth_camera, spatial, stereo_rgb, rgb_marker legacy solvers; calibrate_spatial/stereo/rgb, calibration_target; abc_preview, abc_check, rgb_check, stereo_check, test_spatial and Start_AR_Calibration.cmd. main no longer imports/exposes depth or calibration paths; abc_scene retains only shared layout/dashed-path geometry; abc_task obsolete spatial update and F-confirmation code removed. phase_check retains only short current-task sequence. CSV field compatibility moved to phase_marker. Old depth branches removed from test_trial_evaluation. README/ABC instructions updated. Legacy lightweight baseline remains.

Latest cleanup check: user removed the ~30 GiB Operator recording; it is gone. 12 smaller residual videos remain (~840 MiB), chiefly physical_demo/runs (~807 MiB), physical_demo/output (~13 MiB), output/share/demo_dual_view.mp4 (~17 MiB), output/abc/dual_view_preview.mp4 (~4 MiB). No further deletion attempted this turn after earlier policy blocks. Old demo PID30160 is stopped. Generated data, output, calibration, models and video formats are ignored for Git publication; they must not be uploaded.

No live demo currently running. Do not automatically restart recording to inspect UI. Latest visual: bright sky-blue obstacles and constant orange-red ball, no A/B/C labels. Both AR and web animation colors are updated. User requested full source publication to origin/main, integrating the existing 85e0a24 animation update. All current source/launchers/docs are included, physical_demo is retained as the inactive historical experiment; recordings/models/cache/output are excluded. Use git log/status for the final published revision.

Recording change: main starts Operator UI only after a successful trial/video start, stops every stream when no trial is active, and rejects idle U/T/Y/B. E, automatic completion, restart waiting, failed video startup and Q all close recording. ABC trajectory CSV also skips idle frames. Removed startup --record option; --no-ui-record suppresses per-trial UI auto-start. Necessary checks: actual-main ABC N/E/N/Q plus idle U/B, and legacy four-trial restart/release/quit check passed (2 tests, ~24s); each recording CSV has exactly its trial ID. No broad regression. Web built and isolated headless Edge preview captured. Earlier test harness omitted simulated window focus and was terminated, then fixed; no production focus relaxation.

## Necessary checks this turn

- Python source syntax / main --help passed.
- Synthetic marker: distal tape accepted despite distant yellow patch and black strap; no hand or no tape -> blank.
- Short current G/J/I/G/K/L/E sequence passed without hardware.
- Actual main headless RGB smoke: N, G, direct D/W input, G, E, Q, both tape coordinates and held row verified; temporary output only. First smoke had mismatched Side mirror in fake input; corrected synthetic camera config, no production relaxation. Second passed.
- Rounded obstacle preview inspected. No broad regression suite, no new persistent test recordings, no live hardware test.

See astra_demo/ABC.md for operating details. Preserve current user changes; do not reset/clean Git.
