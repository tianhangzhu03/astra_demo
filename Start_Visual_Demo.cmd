@echo off
setlocal
cd /d "%~dp0"
echo Starting linked top/side views. BLE and depth capture are disabled.
echo Place the SIDE ball, show palm/fingers in Side View, then press N.
echo START is a visual guide only; no Top View hand or start-position check.
echo Pinch baseline: G, open then bring thumb and index together. Top checks the palm region.
echo 0 resets the scene and restarts a trial with both videos; Side placement is kept.
echo Operator UI records continuously from startup. U toggles UI recording independently.
echo Q saves recordings/reports and exits. Keep Camo Camera open on the phone.
python -u -m astra_demo.main --no-ble --linked-views --grasp-mode pinch --top-camera 2 --side-camera 3 --subject visual_linked %*
if errorlevel 1 pause
