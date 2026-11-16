@echo off
setlocal
cd /d "%~dp0"
echo Starting the independent PHYSICAL OBJECT pilot with an AirPods case.
echo Close the visual demo first so it releases both cameras.
echo BLE is enabled; initial voltage is 0 mV. Add --no-ble for a visual-only run.
echo N: trial. WASD: direction. Space: LOWER/LIFT via the same Normal 0 pulse.
echo Tab: change the next Normal meaning without output. G: grasp cue.
echo Purple trails show 2D hand movement in both views; no 3D reconstruction.
echo F: mark grasp. R: mark release. These two marks do not send vibration.
echo X/Esc: stop output. Q: finalize videos and exit.
echo Files stay in physical_demo\runs. The original demo is unchanged.
python -u -m physical_demo.main --top-camera 2 --side-camera 3 %*
if errorlevel 1 pause
