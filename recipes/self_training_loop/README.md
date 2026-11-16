# Self-Training Loop on Judged Agent Traces

Trace a LangGraph agent through Cortex AI Gateway with TruLens, score every run
with LLM judges, and fine-tune Qwen3-8B on the runs the judges approve. The
tuned model then serves the same agent, its runs are judged and kept, and the
next version trains on everything kept so far. No reference answers are used
to select training data or to decide when a version is ready.

This is the code behind the post
[The Self-Training Loop: How Periodic Training on Agent Traces Drives Agent Improvement](https://github.com/Snowflake-Labs/post-train-on-ai-gateway-traces),
ported onto the recipe helpers.

## The agent and its tasks

A ReAct agent (`langgraph.prebuilt.create_react_agent`) answers retail-finance
questions over a synthetic SQLite database with two tools, `sql_query` and
`calculator`. The business rules (only shipped orders count, net value after
discounts, returns valued at the discounted price) exist only in the system
prompt, so a model has to apply them to answer correctly. `tasks.py` builds the
database and 395 tasks deterministically:

| Tasks | Ids | Types | Training | Held out |
|---|---|---|---|---|
| Baseline | `h*` | 7, e.g. month-over-month growth, the second-highest customer | 186 | 49 |
| Drift | `n*` | 4, added after v1, e.g. a category's share of a region's revenue | 130 | 30 |

Each task has a reference answer computed in SQL. It is used only to measure how
well the judges agree with ground truth.

## The loop

1. **Seed.** glm-5.3 on the gateway answers the 186 baseline training tasks 3
   times each (`collect.py`).
2. **Store.** TruLens records each run as OpenTelemetry spans and sends them to
   the gateway, which writes them, its own chat spans, and the judges' eval
   spans to one `AGENT_TRACE_TABLE`. An `agt.trajectory` span carries each
   run's messages, because runs on a sampling job never pass through the
   gateway.
3. **Judge and select.** claude-sonnet-4-6 scores every run with
   `answer_relevance` and four trace-level agent judges (`logical_consistency`,
   `execution_efficiency`, `tool_selection`, `tool_calling`). `select.py` builds
   `TRAJECTORIES` and the view `TRAINING_SET_JUDGED`: runs that finished and
   met every cutoff, the best 2 per task (`sql/trajectories.sql`).
4. **Train.** `train.py` trains a LoRA adapter on Qwen3-8B from the kept traces
   with the shared SFT runner.
5. **Serve and repeat.** `collect.py` serves the checkpoint on a sampling job and
   runs the agent on all 316 training tasks 4 times, standing in for production
   traffic, and on the 79 held-out tasks twice. The next version trains from
   base Qwen3-8B on the seed runs plus every earlier version's training-task
   runs.

A version would be promoted when its judge pass rate on held-out tasks beats
glm-5.3's on both the baseline and the drift tasks.

## Results

Every model answered each held-out task twice. The judge pass rate is the share
of runs that met every selection cutoff, the reference-free signal the loop
uses. Accuracy is against the reference answers:

| Model | Judge pass: baseline | drift | overall | Accuracy: baseline | drift | overall |
|---|---|---|---|---|---|---|
| glm-5.3 | 69.4% | 71.7% | 70.3% | 100.0% | 96.1% | 98.5% |
| Qwen3-8B v1 | 62.2% | 13.3% | 43.7% | 89.6% | 34.5% | 68.8% |
| Qwen3-8B v2 | 72.4% | 45.0% | 62.0% | 88.5% | 76.7% | 84.0% |
| Qwen3-8B v3 | 90.8% | 76.7% | 85.4% | 89.8% | 93.3% | 91.1% |

v1 learned the drift tasks only from its own runs: it had a judge-approved run
on 64 of the 130 drift training tasks, and v2, trained on those, on 116. v3 is
the first version to pass the promotion gate, with a narrow drift margin (46 of
60 passing runs against glm-5.3's 43). Of the runs the judges kept for
training, 97.5 to 99.7% were correct.

| Version | Trained on | Turn examples | Steps |
|---|---|---|---|
| v1 | seed | 1,154 | 289 |
| v2 | seed + v1 | 2,469 | 618 |
| v3 | seed + v1 + v2 | 4,166 | 1,042 |

These results are from the post's runs, which used the same tasks, prompts,
judges, cutoffs, rendering, and training and sampling settings as this recipe.
The port was checked on fresh runs: a seed run (558 runs, 556 correct; 349
traces kept on 180 tasks, giving 1,192 turn examples and 298 steps), a 3-step
training job, and sampling from its checkpoint. A full loop with this code has
not been rerun, so `last_validated` is unset.

## Setup

```bash
uv pip install -e . "tinker-cookbook @ git+https://github.com/thinking-machines-lab/tinker-cookbook.git@nightly" chz datasets
uv pip install trulens-core==2.16.0 trulens-feedback==2.16.0 trulens-apps-langgraph==2.16.0 \
  trulens-apps-langchain==2.16.0 trulens-providers-cortex==2.16.0 trulens-otel-semconv==2.16.0 \
  langgraph==1.2.19 langgraph-prebuilt==1.1.1 langchain-core==1.7.5 langchain-openai==1.7.3 \
  openai==3.45.0 opentelemetry-sdk==1.45.2 opentelemetry-exporter-otlp-proto-http==1.45.2 \
  snowflake-snowpark-python
```

The account needs Cortex AI Gateway, `claude-sonnet-4-6` and `glm-5.3` through
Cortex, and 8 free H200 GPUs (4 to train, 4 to serve).

1. **Gateway logging.** Turn on payload capture and client telemetry with
   `sql/gateway_spec.sql`. It replaces the whole gateway specification, so save
   the current one first and restore it afterwards.
2. **PAT.** The gateway's chat and telemetry endpoints take a programmatic access
   token: `export SNOWFLAKE_PAT=...`.
3. **Schema.** `schema=DATABASE.SCHEMA` (default
   `CORTEX_TRAINING_DB.SELF_TRAINING_LOOP`) holds `TRAJECTORIES` and
   `TRAINING_SET_JUDGED`, and is created if missing. Training jobs go where the
   Cortex Training client points (`CORTEX_TRAINING_DATABASE`,
   `CORTEX_TRAINING_SCHEMA`).

## Run

```bash
C="connection=<profile> schema=<DB>.<SCHEMA>"

# Seed: glm-5.3 on the baseline training tasks, then select
python -m recipes.self_training_loop.collect $C run_tag=seed drift=false rollouts=3 workers=4
python -m recipes.self_training_loop.select  $C run_tags=seed

# v1: train, then collect its training-task and held-out runs
python -m recipes.self_training_loop.train   $C source_run_tags=seed
python -m recipes.self_training_loop.collect $C run_tag=v1-runs rollouts=4 \
    source_job_id=<JOB_ID> checkpoint_id=<CHECKPOINT_ID>
python -m recipes.self_training_loop.collect $C run_tag=v1-eval split=heldout rollouts=2 \
    source_job_id=<JOB_ID> checkpoint_id=<CHECKPOINT_ID>
python -m recipes.self_training_loop.select  $C run_tags=v1-runs,v1-eval

# v2 trains on source_run_tags=seed,v1-runs; v3 on seed,v1-runs,v2-runs.

# Judge pass rate and accuracy per run tag
python -m recipes.self_training_loop.select  $C run_tags=seed-eval,v1-eval,v2-eval,v3-eval report=true
```

For the glm-5.3 held-out row, run `collect.py` without a checkpoint and with
`run_tag=seed-eval split=heldout rollouts=2`.

`train.py` prints the job and checkpoint ids. `collect.py` starts a 4-GPU
sampling job for the checkpoint (`sampling_gpus=`), and cancels it once the
runs finish, before judging. Judging is the slowest step: claude-sonnet-4-6 is
rate limited per account, and a run of 1,264 traces can take several hours.

## How the training data is built

Each assistant turn of a kept trace is one example, with that turn as its only
target. Qwen3's chat template drops the empty `<think></think>` block from
earlier assistant turns but adds it to the generation prompt, so one render of
a whole trace would not match what the model saw at each step. `train.py`
renders turn *i* as the continuation of the generation prompt for the messages
before it, with the pinned Qwen3-8B tokenizer, the tool schemas and the
`/no_think` system prompt that the sampling chat model in `agent.py` uses.
Training runs `ceil(epochs * examples / 8)` steps at a constant learning rate of
1e-4, reshuffling the examples each pass. A trace with a turn over 8,192 tokens
is skipped.

## Files

| File | Purpose |
|---|---|
| `tasks.py` | Database, tasks, reference answers and tools |
| `agent.py` | The agent, the gateway chat model, and the sampling-job chat model |
| `tracing.py` | TruLens exporter to the gateway, the trajectory span, and the judges |
| `collect.py` | Runs the agent on tasks, records and judges each run |
| `select.py` | Builds `TRAJECTORIES` and `TRAINING_SET_JUDGED`, prints results |
| `train.py` | Turn examples from the kept traces, and the training entrypoint |
| `store.py` | Snowpark session and SQL-file helpers |
| `sql/` | Gateway specification, selection SQL and the results query |
| `overrides.json` | Training settings on top of the catalog file |
