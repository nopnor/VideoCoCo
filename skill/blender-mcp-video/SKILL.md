---
name: blender-mcp-video
description: Use for Blender MCP or Blender CLI video-generation work, especially when rendering scripted Blender scenes, debugging MCP port 9876, checking why `blender` is not on PATH, generating direct MP4 output, or fixing Blender 5.x `FFMPEG`/`media_type` render-setting errors. Also use for local PhyGenBench-Blender workflows that produce Blender videos from specs.
---

# Blender MCP Video

## Core Rule

Separate the two Blender access paths:

- **Blender MCP**: controls an already-running GUI Blender instance through port `9876`. Use it for scene inspection, quick code execution, object checks, and viewport/still previews.
- **Blender CLI**: starts Blender from an executable path. Use it for reliable batch rendering and final videos.

Do not treat "MCP is connected" as evidence that the `blender` command is on PowerShell `PATH`.

## Local Defaults

Use these known local paths before broad searching:

```powershell
D:\blender\blender.exe
D:\tools\ffmpeg\bin\ffmpeg.exe
D:\vscode_project\PhyGenBench-Blender
```

If `Get-Command blender` fails, prefer:

```powershell
& "D:\blender\blender.exe" -b --python <script.py>
```

Only search/install Blender or ffmpeg after these paths fail or the user asks.

## Direct MP4 In Blender 5.x

For direct Blender MP4 output, set `media_type` before `file_format`.
This order is required in Blender 5.x because the dynamic `file_format` enum
only exposes movie formats after the media type is switched to video.

```python
scene = bpy.context.scene
scene.render.filepath = str(output_mp4_path)
scene.render.image_settings.media_type = "VIDEO"
scene.render.image_settings.file_format = "FFMPEG"
scene.render.ffmpeg.format = "MPEG4"
scene.render.ffmpeg.codec = "H264"
scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
scene.render.ffmpeg.ffmpeg_preset = "GOOD"
```

If Blender raises an error like:

```text
enum "FFMPEG" not found in ('AVIF', 'JPEG', 'OPEN_EXR', 'PNG', ...)
```

first check whether `media_type` was left as `"IMAGE"`. Do not immediately
switch to external ffmpeg or PNG frame sequences.

## MCP Workflow

When MCP tools are available:

1. Call `get_scene_info` once to confirm the GUI Blender connection.
2. Use `execute_blender_code` for small chunks; avoid sending large unverified scripts all at once.
3. For generated scripts, first run only setup code or a still-frame render.
4. Use `get_viewport_screenshot` only as a quick view; if it returns black, render a still camera frame to disk and inspect that image.

Use MCP for preview/debug. Prefer CLI for final video unless the user specifically wants to render inside the live GUI session.

## CLI Workflow

For `D:\vscode_project\PhyGenBench-Blender`, the normal direct-render command is:

```powershell
cd D:\vscode_project\PhyGenBench-Blender
python scripts\render_spec.py data\specs\output_video_96.spec.json --blender "D:\blender\blender.exe"
```

Expected successful Blender log contains lines like:

```text
Video append frame 1
...
Video append frame 120
```

Validate the output with ffprobe when available:

```powershell
& "D:\tools\ffmpeg\bin\ffprobe.exe" -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,nb_frames,duration -of default=noprint_wrappers=1 outputs\videos\output_video_96.mp4
```

For the current demo, expected metadata is `960x540`, `24/1`, `5.000000`, `120`.

## Fallback Policy

Do not default to external ffmpeg frame-sequence encoding. Use it only when:

- direct Blender MP4 fails after confirming `media_type="VIDEO"`,
- Blender is built without FFmpeg support,
- the user explicitly wants frame-level debugging,
- or reproducibility requires preserved frames.

If frame sequences are used, put them under a temporary or ignored `outputs/frames/...` directory and clean them after successful MP4 encoding unless the user asks to keep them.

## Minimal Diagnostics

Avoid rechecking everything every time. If video output fails, run the smallest relevant check:

```powershell
& "D:\blender\blender.exe" -b --factory-startup --python-expr "import bpy; s=bpy.context.scene; print(s.render.image_settings.media_type, s.render.image_settings.file_format); s.render.image_settings.media_type='VIDEO'; s.render.image_settings.file_format='FFMPEG'; print(s.render.image_settings.media_type, s.render.image_settings.file_format, bpy.app.version_string)"
```

If this succeeds, the bug is in the script settings, not the Blender install.
