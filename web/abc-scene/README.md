# A–B–C 3D haptic task preview

Public page: https://tianhangzhu03.github.io/astra_demo/abc-scene/

This independent animation is for reviewing the proposed scenario. It does not
connect to cameras, operate BLE, or collect experiment measurements. The existing
walkthrough at the repository's Pages root and the Python demo are separate.

## Scenario

- A → B: planar movement around the first obstacle, followed by a grasp.
- B → C: transport around the obstacles while avoiding the center area.
- C: flip the hand, lift in +Z, release, withdraw in −Z, and flip back.
- The box has a square top-view footprint and a downward-facing opening (∩ in side view).
- WASD cues are triggered on demand. **1 feedback per 1 second refers to planned
  feedback logging**, not an automatic one-second hardware trigger.
- The ball keeps the same material throughout grasp and release.
- Dimensions and action durations are illustrative. Retaining the released ball
  in the inverted box is a conceptual capture rule, not a gravity simulation.

## Edit and rebuild

Edit `scene.html`, then run from the repository root:

```sh
python web/abc-scene/build.py
```

The build uses only Python's standard library. It embeds the pinned Three.js
0.160.1 distribution and its MIT license into `docs/abc-scene/index.html`.
No CDN, API, server-side runtime, account, or package installation is needed to
view the published animation; the browser must support WebGL.

`standalone-template.html` supplies the standalone styles, sandboxed frame,
responsive sizing, and browser-local view-state persistence. It was exported
from the original visualization; its scene placeholder is replaced by the build.

To preview locally:

```sh
python -m http.server 8765 --directory docs
```

Open `http://localhost:8765/abc-scene/`. The page starts in front-oblique view.
Play advances through the complete scenario, automatically moving to the box
close-up at C. Drag to orbit, scroll to zoom, or use the action selector and
timeline. View state stays in the viewer's own browser and is not uploaded.

## Publish

GitHub Pages already publishes `main:/docs`. Commit the source and generated
page together, then push to `main`; the existing Pages build deploys the new
`/abc-scene/` path. `docs/index.html` remains the original walkthrough.

`docs/abc-scene/preview.png` is the share-card image. Refresh it when scene
geometry changes. Three.js's license is in `vendor/LICENSE.three` and is also
included in the generated HTML.
