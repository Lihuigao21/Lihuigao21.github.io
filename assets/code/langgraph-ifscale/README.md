# LangGraph IFScale serial-context experiment

This bundle contains saved outputs from one actual writing trial per arm,
configured for `gpt-5.6-luna/high`, with 1, 10, 25 and 50 serial writers.
Each writer received only the common writing instructions, its contiguous share
of a fixed 500-term list, its position and soft length target, and all previous
report prose as read-only context. Each started a fresh official CLI conversation.
Prior conversations and prior keyword lists were not supplied to later writers.
Completed fragments were immutable. No keyword feedback, content repair or
final model rewrite was used.

The current single-writer control injects all 500 requirements at once. This is
a large-prompt comparison; it does not separately test native SKILL discovery.
The total target was approximately 2,500 English words for every arm. Splitting
also constrained each word to its assigned fragment, so it reduced freedom to
organize the whole report. This is one sample, not an estimate of stable success
probability or an isolated test of LangGraph versus equivalent Python orchestration.

## Files

- `data.json`: common prompt, ordered 500 terms, four arms and all 86 saved
  fragments, assignments, exact prompt/fragment hashes, visible successful usage,
  call durations, deterministic scores and existing coherence judgments.
- `allocation.csv`: all 86 node assignments and per-node measurements.
- `constraints-001.csv`, `constraints-010.csv`, `constraints-025.csv`,
  `constraints-050.csv`: 500 requirement checks per arm.
- `report-001.txt`, `report-010.txt`, `report-025.txt`, `report-050.txt`:
  verbatim final reports, including the fixed report wrapper.
- `replay.py`: offline LangGraph reconstruction and score verification.
- `requirements.txt`: dependency used by the historical graph and this replay.

The public files omit machine paths, account configuration, request endpoints,
credentials and conversation identifiers. The JSON keeps fragment strings
verbatim, including any newline characters. The common instruction string uses
LF because the historical Python prompt builder used `read_text`, which
normalized its source file's CRLF before model invocation. JSON and CSV files
are UTF-8 with LF file line endings.

## Replay without calling a model

Use Python 3.12 and install the dependency in an environment of your choice:

```sh
python -m pip install -r assets/code/langgraph-ifscale/requirements.txt
python assets/code/langgraph-ifscale/replay.py --output replay-output
```

The command also works from another working directory when invoked by its path.
It finds `data.json` beside the script. It recreates one `StateGraph` per arm,
checks execution order and every saved prompt hash, appends saved fragment bodies
without alteration, verifies final report bytes, and recalculates the metrics.

This mode makes **zero model calls**. It verifies orchestration, saved artifact
identity and deterministic scoring. It does not resample the model, demonstrate
that another run will achieve the same scores, or test native checkpoint recovery.
The original model answers were generated before publication of this bundle.

Expected strict assigned-local scores are `396, 478, 488, 493` of 500; strict
global report scores are `396, 480, 488, 494`. Loose local scores are
`401, 490, 500, 499`; loose global scores are `401, 492, 500, 500`.
Coherence scores `8, 5, 5, 3` are existing single judgments by the same configured
model in new blinded contexts; replay does not regenerate those judgments.
The preselected coherence threshold was 6, and no arm met both complete strict
coverage and that threshold.

## Scoring and attribution

The constraints and matching conventions come from
[IFScale](https://github.com/DistylAI/distylai.github.io/tree/4ee2d961c9847993d8101d32be0d9b3f888ba94c)
and its [paper](https://arxiv.org/abs/2507.11538). The public replay uses a compact,
independently written implementation of the relevant literal-term behavior.
The 500 constraints in this sample are all alphabetic words; wildcard support
from the more general upstream scorer is not needed or implemented here.
No full upstream source file is redistributed in this bundle.

Strict matching ignores case, removes simple Markdown emphasis, splits on
whitespace and strips only the upstream scorer's limited punctuation set.
Thus `solar-powered` does not satisfy `solar`, and curly possessives can also
fail. Loose matching uses case-insensitive regular-expression word boundaries;
it can accept hyphenated and possessive forms, but not `contribute` in place of
`contribution`. Both local and global scores are retained. The exporter checked
both public rules against the frozen original functions across all 500 terms
in every fragment and final report before publication.

Coherence judges saw one complete report and the same writing-quality rubric,
without keyword lists, node counts, coverage scores or the other reports.
These are subjective model judgments rather than human-reviewed ground truth.

## Usage and timing

Visible successful input totals are `20,262`, `167,789`, `440,733`, and `909,306`
tokens. Cached input is a subset of input, not an additional amount to sum.
Three interrupted network requests returned no confirmed answer or usage;
their backend usage is unknown. Saved successful answers were copied without
resampling during recovery and are counted once. Thus the reported visible
successful usage is a lower bound on total usage, not a billing statement.

`successful_writer_seconds` sums the durations of the accepted unique writer
calls, including CLI startup and service wait. The per-arm `wall_seconds`
includes interruptions and recovery gaps. It is not pure model computation time.

## Figures and diagram

`make_figures.py` reads this package's data and exports the two article figures.
Install the optional `requirements-figures.txt` dependencies, then run:

```sh
python assets/code/langgraph-ifscale/make_figures.py
```

It makes no model calls. `writer-stage.elk.json` is the editable semantic source
for the one-writer diagram; its published layout uses elkjs 0.11.1 with layered,
orthogonal routing. The five boxes expand the operations of one writer, rather
than introducing five separate model agents.
