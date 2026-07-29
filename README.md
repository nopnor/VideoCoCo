# VideoCoCo

Physics-faithful video generation via **proxy-to-photoreal** composition.

VideoCoCo turns a text prompt into a physically grounded preview and then a
photorealistic video: a text prompt is planned into semantic keyframes, rendered
as a neutral white/clay **proxy** clip that carries the correct motion and
physics, and finally restyled into a **photoreal** clip driven by an edit
instruction.

> **Status.** This repo ships the **Agent Skills** (`skill/`), a small **toy
> dataset** (8 triplets), and the **inference** scripts. The tuned transformer is
> being uploaded to the Hugging Face Hub at
> [`mickyhimself/VideoCoCo`](https://huggingface.co/mickyhimself/VideoCoCo).
> See the roadmap below.

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

- [x] Agent Skills (`skill/` — prompt → physical plan → Blender proxy → photoreal edit prompt)
- [x] Toy dataset (8 v2v triplets)
- [x] Inference stack (`inference/` — scripts + upstream patch)
- [ ] Tuned weights (uploading to [Hugging Face Hub](https://huggingface.co/mickyhimself/VideoCoCo))

## 🧠 Our Related Work

- **[DraCo]** [DraCo: Draft as CoT for Text-to-Image Preview and Rare Concept Generation](https://arxiv.org/abs/2512.05112) · [code](https://github.com/CaraJ7/DraCo)
- **[T2I-R1]** [T2I-R1: Reinforcing Image Generation with Collaborative Semantic-level and Token-level CoT](https://arxiv.org/abs/2505.00703) · [code](https://github.com/CaraJ7/T2I-R1)
- **[Image Generation CoT]** [Can We Generate Images with CoT? Let's Verify and Reinforce Image Generation Step by Step](https://arxiv.org/abs/2501.13926) · [code](https://github.com/ZiyuGuo99/Image-Generation-CoT)
- **[NextStep-1]** [NextStep-1: Toward Autoregressive Image Generation with Continuous Tokens at Scale](https://arxiv.org/abs/2508.10711) · [code](https://github.com/stepfun-ai/NextStep-1)
- **[LongCat-Next]** [LongCat-Next](https://github.com/meituan-longcat/LongCat-Next) · [model](https://huggingface.co/meituan-longcat/LongCat-Next)

## License

Dataset released for research use. Model code and tuned weights, once added,
build on **Tencent HY-OmniWeaving** and are governed by the Tencent HY Community
License Agreement; those components will ship with the corresponding license and
attribution.
