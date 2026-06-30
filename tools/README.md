# trellis2.c tools

`trellis_live.c` is the default app:

```sh
../build/trellis-live --display :1
```

It opens raylib, loads an input image, runs stage1 with per-step voxel decode,
then runs stage2 with per-step mesh decode.
It preloads all safetensors before opening the window, so checkpoint I/O does
not block the raylib event loop.

Shared live helpers:

- `trellis_tool_live.h`: in-memory stage1/stage2 live interfaces.
- `trellis_infer.c`: stage1 image/DINO/flow/voxel decode implementation. It also
  builds as the `trellis-infer` debug CLI when `TRELLIS_TOOL_LIBRARY` is not set.
- `trellis_stage2_mesh_live_viewer.c`: stage2 shape flow, per-step mesh decode,
  and raylib mesh upload. It also builds as the standalone
  `trellis-stage2-mesh-live-viewer` debug CLI.

Standalone debug tools:

- `trellis_info.c`: inspect checkpoint manifests.
- `trellis_stage1.c`: stage1 checkpoint and schedule validation.
- `trellis_stage2.c`: stage2 tensor sampler/decode debug CLI.
- `trellis_voxel_viewer.c`: replay written voxel snapshots.
- `trellis_mesh_viewer.c`: replay written mesh snapshots.
- `trellis_verify.c`: small numeric/file verification helper.
