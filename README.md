# VideoCoCo

Physics-faithful video generation via **proxy-to-photoreal** composition.

VideoCoCo turns a text prompt into a physically grounded preview and then a
photorealistic video: a text prompt is planned into semantic keyframes, rendered
as a neutral white/clay **proxy** clip that carries the correct motion and
physics, and finally restyled into a **photoreal** clip driven by an edit
instruction.

> **Status.** This repo currently ships a small **toy dataset** (8 triplets) so
> you can inspect the data format. The Agent Skills, inference stack, and tuned
> weights are being prepared and will be added here (weights hosted on the
> Hugging Face Hub). See the roadmap below.

## Toy dataset

`data/toy_cases/` — 8 hand-checked video-to-video (v2v) triplets, one directory
per case:

```
data/toy_cases/
├── manifest.jsonl                 # one JSON line per case (index)
├── 0000_buoyancy/
│   ├── video.mp4                  # source: neutral white/clay physics proxy
│   ├── seedance.mp4               # target: photoreal restyle
│   └── edit_prompt.txt            # instruction used to restyle proxy -> photoreal
├── 0001_stress/
│   └── ...
└── ...
```

Each `manifest.jsonl` line:

```json
{"case_id": "0000_buoyancy", "source": "0000_buoyancy/video.mp4", "target": "0000_buoyancy/seedance.mp4", "instruction": "...", "category": "buoyancy"}
```

- **source** (`video.mp4`) — a grayscale/white-material render. Physical meaning
  is carried by shape, motion, transparency, deformation, and coverage, not color.
- **target** (`seedance.mp4`) — the photorealistic result.
- **instruction** (`edit_prompt.txt`) — the English restyle prompt mapping source
  motion to the photoreal target.

The 8 cases cover: buoyancy, stress/deformation, melting (×2), surface tension,
sublimation, elasticity, and boiling. They are a *toy* sample for format
inspection, not a training-scale corpus.

## Roadmap

- [x] Toy dataset (8 v2v triplets)
- [ ] Agent Skills (prompt → physical plan → Blender proxy → photoreal edit prompt)
- [ ] Inference stack
- [ ] Tuned weights (Hugging Face Hub)

## License

Dataset released for research use. Model code and tuned weights, once added,
build on **Tencent HY-OmniWeaving** and are governed by the Tencent HY Community
License Agreement; those components will ship with the corresponding license and
attribution.
