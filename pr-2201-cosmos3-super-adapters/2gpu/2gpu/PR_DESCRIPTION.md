## H100 validation (2 GPUs)

Validated `cosmos3/super-adapters` at Omni commit `5695a9b0ec7c8118949f92c06e22b0c200a52c8e` on one node with 2x NVIDIA H100 80GB HBM3. This is the 2-GPU campaign; 4- and 8-GPU qualification remain separate follow-ups.

Checkpoint: `nvidia/Cosmos3-Super@fe77b66696d645f663b8f27e942b3b43e4629e23` (124 GB local snapshot).

Native runtimes:

- Generation smoke: `sglang==0.5.19`, `flashinfer-python==0.6.18`.
- Reasoner: official SGLang commit `4e9e407d3720045d59cae185c05f33649f4e544e`, `sglang-kernel==0.4.7`, `flashinfer-python==0.6.18`. The released 0.5.19 SRT does not register `Cosmos3ForConditionalGeneration`.

### Feature coverage

| Config | Feature | Result | Validation |
| --- | --- | ---: | --- |
| generation | T2I | PASS | nonconstant 832x480 PNG; fresh-owner restart also passed |
| generation | T2V | PASS | decoded 17/17 frames at 832x480 |
| generation | I2V | PASS | decoded 17/17 frames at 832x480 |
| generation | V2V continuation | PASS | decoded 17/17 frames at 832x480 |
| generation | sound | PASS | 49-frame MP4, stereo AAC present, nonzero audio energy, duration within 0.15s |
| generation | policy | PASS | finite `[16, 9]` action output |
| generation | inverse dynamics | PASS | finite `[16, 9]` action output |
| generation | forward dynamics | PASS | decoded 17/17 frames at 832x480 |
| reasoner | text | PASS | response ends in reference answer `4` |
| reasoner | image | PASS | response contains reference color `red` |
| reasoner | video | PASS | response identifies `red` before `blue` |

Generation: 8/8 passed. Reasoner: 3/3 passed. All cases assert that their owned stage/native process trees are gone after shutdown.

Reasoner reference responses:

| Input | Expected | Actual |
| --- | --- | --- |
| `What is two plus two?` | `4` | `4` after the model's reasoning block |
| solid-red image | `red` | `The dominant color in the image is red.` |
| red frames followed by blue frames | red to blue | `The color transitions from red to blue.` |

### Native lifecycle

The dedicated lifecycle case passed in 212.50s:

1. Started the TP=2 reasoner.
2. Admitted and cancelled a long native request.
3. Sent a follow-up request through the same deployment and received `4`.
4. Terminated the owned stage process and observed the runner's `Dead stage process` failure path within 30s.
5. Asserted process-tree cleanup.
6. Started a fresh native owner and successfully inferred `4` again.

The Python resource tracker still warned about 6 semaphore names and 3 shared-memory names at interpreter exit. No owned worker remained and both GPUs returned to 81,079 MiB free. Treat the tracker warning as an open lifecycle-cleanliness finding.

### Two-GPU memory fit

The unmodified multi-GPU generation setup loaded the 117.85 GB transformer, but the first T2I request OOMed during component transition: GPU 0 had 371 MiB free and failed a 500 MiB allocation; GPU 1 had 46.56 MiB free and failed an 80 MiB allocation.

The complete generation matrix and benchmarks pass with this explicitly labeled 2-GPU fallback:

```json
{
  "generation": {
    "component_residency": {"transformer": "layerwise-offload"},
    "layerwise_resident_layers": {"transformer": 1}
  }
}
```

The effective topology is FSDP/HSDP shard dimension 2 plus native auto-CFG parallel degree 2. Layerwise setup reports 2/128 transformer layers resident and approximately 2.77 GB transformer VRAM, with the rest checkpoint-mapped/host-backed. This result should not be presented as evidence that the stock 4-GPU YAML is wrong; it only establishes that stock residency does not fit 2x80GB.

### Generation examples

These are branch outputs from the checkpoint's official structured prompts and
quality recipe: 1280x720, 189 frames at 24 fps (7.875 seconds), 35 steps, CFG 6,
flow shift 10, and seed 17. Each MP4 was decoded as 189/189 frames and the three
frame contact sheets below were inspected for temporal prompt adherence.

<!--
GitHub PR upload note: drag the three MP4s from generation-quality/ into the
Result cells in the PR editor. GitHub will replace these relative links with
user-attachments URLs and render video players, as in #2107. Do the same for
the contact sheets if inline still previews are wanted. The relative links below
remain directly usable in the saved artifact bundle.
-->

| Task | Prompt | Result |
| --- | --- | --- |
| T2V | [NVIDIA structured prompt](https://huggingface.co/nvidia/Cosmos3-Super/blob/fe77b66696d645f663b8f27e942b3b43e4629e23/assets/example_t2v_prompt.json) | [t2v-output.mp4](generation-quality/t2v-output.mp4)<br><img src="generation-quality/previews/t2v-contact-sheet.jpg" width="720" alt="T2V start, middle, and end frames"> |
| I2V | [NVIDIA structured prompt](https://huggingface.co/nvidia/Cosmos3-Super/blob/fe77b66696d645f663b8f27e942b3b43e4629e23/assets/example_i2v_prompt.json) + [conditioning image](https://huggingface.co/nvidia/Cosmos3-Super/blob/fe77b66696d645f663b8f27e942b3b43e4629e23/assets/example_i2v_input.jpg) | [i2v-output.mp4](generation-quality/i2v-output.mp4)<br><img src="generation-quality/previews/i2v-contact-sheet.jpg" width="720" alt="I2V start, middle, and end frames"> |
| T2V + sound | [NVIDIA audiovisual structured prompt](https://huggingface.co/nvidia/Cosmos3-Super/blob/fe77b66696d645f663b8f27e942b3b43e4629e23/assets/example_t2vs_prompt.json) | [t2vs-output.mp4](generation-quality/t2vs-output.mp4)<br><img src="generation-quality/previews/t2vs-contact-sheet.jpg" width="720" alt="Audiovisual T2V start, middle, and end frames"> |

Observed output and request metrics on the 2-H100 layerwise fallback:

| Task | Media validation | End-to-end request latency | Sampled peak GPU memory (MiB, GPU 0 / 1) |
| --- | --- | ---: | ---: |
| T2V | 1280x720, 189 frames, 7.875s | 462.264s | 26,458 / 27,431 |
| I2V | 1280x720, 189 frames, 7.875s | 463.593s | 27,928 / 28,499 |
| T2V + sound | 1280x720, 189 frames, 7.875s; stereo 48kHz AAC, 7.880s | 464.462s | 27,158 / 28,331 |

The audiovisual output decoded to 370 audio frames / 7.893s with nonzero
energy; measured level was -33.9 dB mean and -4.1 dB maximum. Visual inspection
shows the robot holding the jar, pouring into the cup, and returning upright.

### Reproduction

No Hugging Face token is stored in any script.

```bash
# Resumable, commit-pinned model download; set HF_TOKEN in the environment.
results/cosmos3-super/2gpu/download_model.sh

# Install the exact native reasoner runtime.
results/cosmos3-super/2gpu/install_native_runtime.sh

# Run all smoke/reference/lifecycle cases (or pass generation/reasoner/lifecycle).
results/cosmos3-super/2gpu/run_h100_validation.sh all

# Generate/resume the full-quality PR examples (token-free, checkpoint-pinned).
results/cosmos3-super/2gpu/run_generation_quality.sh
```

### Saved evidence

- `smoke-t2i-layerwise.{log,xml}` and `smoke-generation-layerwise.{log,xml}`: 8 passing generation cases.
- `reasoner-reference.{log,xml}`: 3 passing, reference-asserting reasoner cases.
- `lifecycle-reasoner.{log,xml}`: passing cancellation/failure/cleanup/restart case.
- `generation-quality-{t2v,i2v,t2vs}.log`, `generation-quality/results.jsonl`, and
  `generation-quality/{t2v,i2v,t2vs}-output.mp4`: full structured-prompt quality evidence.
- `expected-failure-stock-2gpu-fsdp-oom.{log,xml}`: preserved stock 2-GPU OOM.
- `expected-failure-sglang-0.5.19-reasoner-registration.{log,xml}`: preserved
  released-runtime incompatibility that motivates the pinned native revision.
- `SHA256SUMS`: artifact integrity manifest.

Expected environment-only warnings: no Mooncake/NIXL relay packages (unused for this local pipeline), EFA unavailable with NCCL falling back to sockets, and unrelated Hunyuan/Pi05 discovery warnings about the duplicate NCCL runtime.
