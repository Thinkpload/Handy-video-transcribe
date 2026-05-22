# Implementation Plan — Fork build fixes (Windows, 2026-05-22)

After the Windows toolchain was unblocked (Vulkan SDK, MSVC Dev Shell,
Ninja generator, short `CARGO_TARGET_DIR=D:\t\handy`; see
[scripts/dev-env.ps1](scripts/dev-env.ps1)), `cargo build` for the
`handy` crate fails with **29 rust errors** caused by dependency API drift
in `ort 2.0.0-rc.12` and updated `tauri`.

All errors are confined to two files:

- [src-tauri/src/audio_toolkit/diarization/mod.rs](src-tauri/src/audio_toolkit/diarization/mod.rs) — 28 errors (Meetings feature)
- [src-tauri/src/lib.rs](src-tauri/src/lib.rs) — 1 error

Full log: `build6.log` (kept untracked; regenerate with
`.\scripts\dev-env.ps1 -Run 'cargo build --manifest-path src-tauri/Cargo.toml'`).

---

## 1. `ort` Session run! input macro — 4 errors

Locations:
- [diarization/mod.rs:176](src-tauri/src/audio_toolkit/diarization/mod.rs#L176) — segmentation `run`
- [diarization/mod.rs:239](src-tauri/src/audio_toolkit/diarization/mod.rs#L239) — embedding `run`

Symptoms (E0277 ×2, "?" can only be applied to Try ×2):
```
SessionInputValue<'_>: From<ArrayBase<ViewRepr<&f32>, Dim<[usize; N]>>> is not satisfied
the `?` operator cannot be applied to type `Vec<(Cow<'_, str>, SessionInputValue<'_>)>`
```

**Root cause:** in `ort 2.0.0-rc.12` the `ort::inputs![ ... ]` macro now
returns `Vec<(Cow<str>, SessionInputValue)>` directly (not `Result`), and
`SessionInputValue` no longer implements `From<ArrayView<…>>`. The view
must be wrapped in a `Value` first.

**Fix:** drop the inner `?` and convert the `ArrayView` into a `Value`:
```rust
let input_value = ort::value::Value::from_array(input.view())?;
let outputs = self.segmentation.run(ort::inputs![
    self.seg_input_name.as_str() => input_value
])?;
```
Apply the same shape to both call sites (line 176 segmentation,
line 239 embedding).

---

## 2. `try_extract_tensor` return type changed — 2 errors

Locations:
- [diarization/mod.rs:181](src-tauri/src/audio_toolkit/diarization/mod.rs#L181)
- [diarization/mod.rs:244](src-tauri/src/audio_toolkit/diarization/mod.rs#L244)

Symptom (E0599):
```
no method named `into_dimensionality` found for tuple `(&ort::value::Shape, &[f32])`
```

**Root cause:** `try_extract_tensor::<f32>()` in `ort` 2.0-rc.12 returns
`(&Shape, &[f32])` instead of an `ndarray` view.

**Fix:** rebuild an `ndarray::ArrayView` from the shape + slice, then call
`into_dimensionality`. Helper:
```rust
let (shape, data) = outputs[self.seg_output_name.as_str()]
    .try_extract_tensor::<f32>()?;
let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
let logits = ndarray::ArrayView::from_shape(dims.as_slice(), data)?
    .into_dimensionality::<ndarray::Ix3>()?
    .to_owned();
```
Repeat for the embedding output (Ix2 at line 244).

---

## 3. `Session::inputs` / `outputs` became methods — 4 errors

Locations: [diarization/mod.rs:72-75](src-tauri/src/audio_toolkit/diarization/mod.rs#L72-L75)

Symptom (E0616 ×4): `field 'inputs' of struct 'ort::session::Session' is private`.

**Fix:** mechanical — add `()`:
```rust
let seg_input_name  = segmentation.inputs()[0].name.clone();
let seg_output_name = segmentation.outputs()[0].name.clone();
let emb_input_name  = embedding.inputs()[0].name.clone();
let emb_output_name = embedding.outputs()[0].name.clone();
```

---

## 4. `?` chains lose `Send + Sync` from `ort::Error` — 18 errors

Locations: [diarization/mod.rs:64](src-tauri/src/audio_toolkit/diarization/mod.rs#L64) and `:68` (9 errors per `?` × 2 sites).

Symptom (E0277 ×18): `Send`/`Sync` not satisfied for
`NonNull<OrtSessionOptions>`, `NonNull<OrtMemoryInfo>`,
`NonNull<OrtCustomOpDomain>`, `dyn Any`, `dyn Operator` — all reached
through `anyhow::Error: From<ort::Error<R>>` requiring `R: Send + Sync`.

**Root cause:** in `ort` 2.0-rc.12, `ort::Error<R = ()>` carries a context
parameter `R` (`SessionBuilder` in this case) which contains
non-`Send`/`Sync` raw pointers. The blanket `From` impl for `anyhow::Error`
needs `R: Send + Sync + 'static`, so `?` can't auto-convert
`ort::Error<SessionBuilder>` into `anyhow::Error`.

**Fix options** (pick one, A is easiest):

- **A.** Erase the context with `.map_err(|e| anyhow::anyhow!(e.to_string()))?`
  on the two failing builder chains (lines 64, 68).
- **B.** Switch the function's error type from `anyhow::Error` to
  `Box<dyn Error + Send + Sync>` and add an explicit `From<ort::Error<_>>`
  shim; more invasive.

Go with **A** unless we need the original error chain preserved elsewhere.

---

## 5. `tauri::Manager::manage` now returns `bool` — 1 error

Location: [src/lib.rs:170](src-tauri/src/lib.rs#L170)

Symptom (E0308): `match` arms have incompatible types — `Ok` arm returns
`bool`, `Err` arm returns `()` (from `log::error!`).

**Fix:** discard the bool, restructure the match:
```rust
match managers::meetings_store::MeetingsStore::new(app_handle) {
    Ok(store) => { app_handle.manage(Arc::new(store)); }
    Err(e) => log::error!("Failed to initialise meetings store: {}", e),
}
```

---

## Suggested order

1. §3 (mechanical `inputs()/outputs()`) — eliminates 4 errors and is risk-free.
2. §5 (`lib.rs` manage) — 1 line, unblocks the main crate compile path.
3. §4 (`map_err` on builder chains) — fixes 18 errors with one local change per site.
4. §1 + §2 together — the run/extract pair changes are coupled; touch both
   call sites (segmentation + embedding) in one pass and re-run `cargo build`.

After each step run:
```powershell
. .\scripts\dev-env.ps1
cargo build --manifest-path src-tauri/Cargo.toml 2>&1 | Select-String -Pattern "^error"
```

## Out of scope here

- Why upstream pinned `ort` to rc.12 specifically (check
  `transcribe-rs = "0.3.3"` features — that's the indirect source).
- Whether to bump `ndarray` if the new ort API exposes it directly.
- Audit other diarization helpers for similar API drift once the crate compiles.
