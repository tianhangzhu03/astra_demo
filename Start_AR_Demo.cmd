@echo off
setlocal
cd /d "%~dp0"
echo ABC: virtual obstacles, virtual ball, downward-opening virtual box.
echo Default: two RGB cameras, no depth. Yellow tape coordinates are independent of AR and BLE.
echo No board setup needed. Continuous Top XY + Side image-up Z proxy; cues only label phases.
echo Relative image motion only, not millimeters. Each trial uses fixed zeros in both views.
echo Lost tape leaves gaps only in its view; AR and BLE remain independent.
echo Side P/O: position pickup ball / box. In RGB mode place both before N.
echo Use Ball [P] / Box [O] to change placement mode. Right-click Side also places the box.
echo BLE is enabled. Add --no-ble for visuals only. Initial voltage is 2500 mV / 2.5 V; cues trigger output.
echo Tap WASD / I Normal Up / K Normal Down: one Static pulse each second. X stops.
echo J: CW down 90 deg. L: CCW back 90 deg. G: single pick/drop. Idle allows WASD/Normal testing.
echo No extra F confirmation. J/L record rotation intent; G manually grabs/drops. After L, return then E ends recording.
echo BLE checks CEL firmware v0.2.1. CW=5, CCW=6. STOP=FF06.
echo No hand trails. Operator Top shows a dashed route and XYZ status.
echo --no-ble disables all hardware output.
python -u -m astra_demo.main --scenario abc --ble --top-camera 2 --side-camera 3 --subject abc %*
if errorlevel 1 pause
