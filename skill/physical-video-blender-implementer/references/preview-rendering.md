# Preview Rendering

## Local Defaults

Use these paths when available:

```powershell
D:\blender\blender.exe
D:\tools\ffmpeg\bin\ffmpeg.exe
D:\vscode_project\PhyGenBench-Blender
```

## Render Command

From the repo root or selected workspace:

```powershell
& "D:\blender\blender.exe" -b --python outputs\scripts\<case_id>.blender.py
```

Successful Blender logs should show frame appends through the final frame.

## Validate MP4

```powershell
& "D:\tools\ffmpeg\bin\ffprobe.exe" -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,nb_frames,duration -of default=noprint_wrappers=1 outputs\previews\<case_id>.preview.mp4
```

For the default preview, expect:

- width=1280
- height=720
- r_frame_rate=24/1
- duration=5.000000
- nb_frames=120

## Extract Preview Sheet

Choose frames aligned to semantic keyframes. For a 120-frame preview with
`K0-K4`, a useful sheet is:

```powershell
& "D:\tools\ffmpeg\bin\ffmpeg.exe" -y -v error -i outputs\previews\<case_id>.preview.mp4 -vf "select='eq(n,0)+eq(n,29)+eq(n,59)+eq(n,89)+eq(n,119)',scale=240:-1,tile=5x1" -frames:v 1 outputs\previews\<case_id>.sheet.png
```

Use 6 tiles when the plan has 6 semantic keyframes.

## Preview Audit

Before final response, inspect the sheet and any suspicious key frames:

- each semantic keyframe is visible at its mapped frame
- caused objects do not appear early
- transition origin, direction, and material continuity are visible
- final state is reached without hiding the middle process
- no `must_avoid` item is obvious in the preview
- output uses white/clay grayscale materials with no semantic colors unless a
  colored diagnostic render was explicitly requested

If the sheet is ambiguous, extract individual frames around the failing
transition.
