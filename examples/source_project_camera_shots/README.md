# Source-authoritative parallel-camera shots

This project keeps `figure.tex` and `camera-shots.json` as authored inputs. To
keep generated output out of the checkout, copy the example to a fresh
temporary directory. From the repository root, run:

```bash
camera_project_directory="$(mktemp -d /tmp/tikz-native-camera-project.XXXXXX)"
cp -R examples/source_project_camera_shots/. "$camera_project_directory/"
(
  cd "$camera_project_directory"
  tikz-native-project build project.json
  tikz-native-project status project.json
  tikz-native-project clean project.json
)
```

The build writes a canonical `camera-shots.json` node next to the disposable
ShapeAsset, compositing plan, and build manifest under `.tikz-native/derived`.
Editing only the authored camera file reuses the ShapeAsset.

This minimal example has no Bridge template, so it does not produce
`generated_scene.py`. When a compatible Bridge template is present, generated
source exposes the immutable `TIKZ_NATIVE_CAMERA_SHOTS` value before user hooks.
It still does not start playback automatically. The host must explicitly use a
world-space `MultiProjectionCamera` binding; the build never applies a second
camera to geometry that an older generated runtime has already projected
locally.

`cameraShots` and `motionJson` are intentionally mutually exclusive in this
first source-project slice. A later coordinated timeline can provide one
well-defined frame order for geometry and camera changes.
