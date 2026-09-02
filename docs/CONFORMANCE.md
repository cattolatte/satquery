# Conformance to SIH26167

Every mandatory clause of the problem statement, quoted, with where it is
implemented and how it was checked. Clauses not yet met are listed as gaps
rather than omitted.

## Defined input scope

| Required | Status | Where |
|---|---|---|
| Single image — optical/multispectral or SAR | met | `io/inspect.py` → `InputKind.SINGLE` |
| Cross-modal pair — co-registered optical + SAR | met | `InputKind.CROSS_MODAL_PAIR`, `tools/multi_image.py` |
| Bi-temporal pair — two dates, same area | met | `InputKind.BI_TEMPORAL_PAIR`, `ChangeTool` |
| GeoTIFF/TIFF for geospatial imagery | met | `_open()` reads bands 4/3/2 via rasterio |
| PNG/JPEG "only for the prescribed public benchmark datasets" | met | accepted, and flagged in the trace as non-geospatial |

## Mandatory functional scope

> "At least one visual or vision-language component must be fine-tuned or
> otherwise adapted using BigEarthNet.txt or the any open source training data."

Met, and measured. CLIP fine-tuned contrastively on 2,182 BigEarthNet.txt
image–caption pairs, evaluated on 386 held-out patches split by patch so no
scene appears in both halves:

| metric | before | after |
|---|---|---|
| zero-shot land-cover top-1 | 2.1% | **45.8%** |
| image→text R@5 | 1.8% | **28.5%** |
| text→image R@10 | 2.8% | **44.0%** |

> "Visual question answering shall be mandatory. Each solution must
> additionally implement either captioning/scene description or text-guided
> region grounding."

Met, with both rather than either: `VQATool`, `CaptionTool`, `GroundingTool`.

> "Change description or change-based visual question answering from a
> bi-temporal image pair shall be mandatory. A spatial change map may also be
> generated."

Met, with both tasks and the optional change map: `ChangeTool` returns a
`change_map` evidence item alongside boxes and per-class deltas.

> "The system must extract complementary information from a co-registered
> optical/multispectral and SAR image pair."

Met: `CrossModalTool` reads each modality for what it is good at
(`SAR_STRENGTHS` / `OPTICAL_STRENGTHS`) and reports an agreement score, rather
than stacking the two images through one model.

> "The system must automatically select, sequence, and execute the appropriate
> specialist models or tools according to the query and input configuration."

Met: `controller/agent.py`.

## The controller's six enumerated steps

The statement enumerates six; `Controller.run()` implements them in order, and
each is observable in the returned trace.

| # | Required | Implementation |
|---|---|---|
| 1 | interpret the query and classify the requested task | `router.classify()` |
| 2 | check number, modality, format, metadata, compatibility | `inspect_image()`, `classify_inputs()` |
| 3 | select one or more tools from a predefined registry | `Registry.for_task()`, `_plan()` |
| 4 | configure **only permitted** task parameters | `Tool.filter_params()`; dropped names recorded in the trace |
| 5 | combine outputs, estimate confidence, return visual evidence | `_fuse()`, `_confidence()`, `Evidence` |
| 6 | provide an auditable execution summary | `Trace`, returned on every request |

Step 4 is enforced structurally, not by convention: a tool declares `accepts`
in its `ToolSpec` and `invoke()` filters before dispatch, so a tool cannot
receive a parameter it never declared even if the caller sends one.

> "only the observable execution trace … will be evaluated. Internal reasoning
> text is neither required nor evaluated."

The trace carries the task, tool names, permitted parameters, timings, and
outcomes. No internal reasoning text is emitted.

## Representative queries

All five quoted in the statement route correctly, and are conformance tests in
`tests/test_router.py::TestProblemStatementQueries`.

| Query | Routes to |
|---|---|
| "Describe the land-cover and major objects visible in this image." | `caption` |
| "Highlight the water body referred to in the query." | `grounding` |
| "What changed between these two dates, and where did the change occur?" | `change_description` |
| "Use the optical and SAR images together to identify built-up and water-covered regions." | `cross_modal` |
| "Has the built-up area increased, decreased, or remained unchanged?" | `change_vqa` |

The third failed until tested: it contains "did … change" and was read as a
polar question, dispatching change VQA. Paraphrasing the statement's queries
would have hidden that.

## Expected solution

> "an interactive GUI or web application with an agentic remote-sensing AI backend"

Met: `satquery/server.py` + `web/index.html`.

| "The solution should include" | Status |
|---|---|
| Input upload and compatibility checking | met |
| A remote-sensing-adapted vision-language component | met, measured above |
| Specialist tools for VQA, captioning or grounding, change, optical–SAR | met, five tools |
| An agentic controller for routing, execution, integration | met |
| Visual evidence | met — boxes, heat maps, change maps |
| Confidence information | met — per-tool and aggregate |
| Execution summaries | met — full trace on every request |
| Downloadable reports | met — self-contained HTML and JSON |

> "A generic LLM or VLM without remote-sensing adaptation will not satisfy the
> requirements."

Directly evidenced. Stock OpenAI CLIP scores 8.3% zero-shot land-cover on these
patches and RemoteCLIP 2.1%, both far below the 49.0% majority-class baseline.
Adaptation is what makes the system work, not a refinement on top of one that
already did.

## Benchmark results

Measured, not asserted. Full breakdown with baselines in [BENCHMARKS.md](BENCHMARKS.md).

| benchmark | scope | result |
|---|---|---|
| Optical–SAR pairs | cross-modal (mandatory) | fused **37.6%** P@3, **+1.4** over the better single sensor |
| CDVQA | change VQA (mandatory) | **47.3%** over 300 questions |
| RSVQA-LR | single-image VQA | 34.9% over 800; 50.8% scene-level |
| VRSBench VQA | single-image VQA | 7.6% exact / 8.7% lenient over 1,440 |
| VRSBench grounding | text-guided grounding | 0.4% Acc@0.5 IoU |
| VRSBench captioning | scene description | ROUGE-L 0.026 |

Every number comes from the real serving path, the registered tool invoked
through `Tool.invoke`.

Two of these are strong for a system with no task-specific training: change
direction (74.1% increase, 65.7% decrease) and optical–SAR fusion, which is
complementary in the measurable sense that fusing beats either sensor alone.
The rest are weak, and the reasons are structural rather than tuning problems.

## Gaps

Open, and stated rather than glossed:

- **Object-level questions are out of reach.** Eight of VRSBench VQA's twelve
  types, and RSVQA's shape/size questions, ask about an individual object's
  colour, count, position or shape. A scene-level land-cover backbone scores at
  chance on all of them. Closing this needs an open-vocabulary detection head —
  an architectural addition, not a parameter.
- **Counting is not attempted.** 250 of 800 RSVQA questions and 120 of the
  VRSBench sample ask "how many". These score zero by design rather than by
  accident, after an earlier version scored 12.5% by scraping stray digits.
- **Referring grounding fails on small targets.** 0.4% Acc@0.5. The 7×7 patch
  grid cannot express a box smaller than ~73 px; tiling raises resolution
  monotonically but not nearly enough, because CLIP patch tokens were never
  supervised for localisation.
- **Captioning is a vocabulary mismatch.** A 21-class land-cover summary against
  long human descriptions of vehicles and colours. The metrics measure a
  difference in kind.
- **Cross-corpus transfer is weak.** Adaptation on BigEarthNet's CORINE
  vocabulary does not carry to RSVQA's object-centric annotation even on the
  same sensor family; rules that answer "yes" often score *below* chance there.
- **The shipped checkpoint costs RSVQA accuracy.** 34.9% against the
  optical-only checkpoint's 37.1% on the same questions. Taken deliberately,
  because optical–SAR is mandatory and does not work without it.
- **Zero-shot land-cover sits near the majority baseline** on a subset dominated
  by one class. The baseline is a degenerate classifier and ours discriminates
  21, but the gap is real and is not presented as a win.
