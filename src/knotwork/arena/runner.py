"""Resumable Model Arena runner: generate (or reuse) a certified dataset, then ask every model of
every block every item, one flushed line of `results.jsonl` per trial.

    python -m knotwork.arena.runner --results-dir <dir> [--resume] [--retry-failed-only]

Guarantees:

* **Crash-safe and resumable.** Progress lives on disk (the dataset manifest, `results.jsonl`);
  `--resume` skips every trial already recorded, so a restart never duplicates a trial or pays twice.
  `--retry-failed-only` re-asks only the trials whose latest record is an API error.
* **Cooperative control.** The page writes `control.json` (`pause` | `stop`); the runner checks it
  before every step, lets in-flight calls finish, records them and exits cleanly: no orphan workers.
* **Bounded spend.** In live mode the spend cap is checked before every call against the spend so
  far plus a cautious estimate of every call in flight; reaching it ends the run as `budget_stop`.
* **No secrets on disk.** The API key is read from the `OPENROUTER_API_KEY` environment variable
  of this process only; error texts are scrubbed of it before they are recorded.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from knotwork import net
from knotwork.arena import catalog, storage
from knotwork.arena.generation import make_item_job
from knotwork.arena.schemas import ItemRecord, SessionConfig, TrialRecord, trial_key
from knotwork.knotcore.grading import ANSWER_RE, extract, grade_prediction, is_hedged
from knotwork.knotcore.seeds import SeedTree
from knotwork.render.canvas import REPO_ROOT

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
CALL_TIMEOUT_S = 60.0
MAX_RETRIES = 4
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
EST_PROMPT_TOKENS = 1300  # 768-px image + question (the benchmark's preflight figure)
EST_COMPLETION_TOKENS = 300
COST_MARGIN = 1.25
MANIFEST_EVERY = 10  # items between dataset manifest rewrites during generation
REFUSAL_RE = re.compile(r"\b(i can(?:no|')t|i am unable|i'm unable|sorry|cannot determine)\b", re.I)
PRICES_FILE = "prices.json"  # per-model USD/token prices the page resolved (no secrets)


# ------------------------------------------------------------------ process helpers
def pid_alive(pid: int | None) -> bool:
    """Whether a process id is running. Never signals the process (on Windows `os.kill` would
    terminate it), so it is safe to call on any pid."""
    if not pid or pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, int(pid))  # QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel32.CloseHandle(handle)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def launch(
    results_dir: Path, live: bool, api_key: str | None = None, retry_failed: bool = False
) -> subprocess.Popen[bytes]:
    """Start the runner in the background, detached from the page's process group. The key, if
    any, travels only in the child's environment."""
    cmd = [sys.executable, "-m", "knotwork.arena.runner", "--results-dir", str(results_dir),
           "--resume"]  # fmt: skip
    if retry_failed:
        cmd.append("--retry-failed-only")
    env = {k: v for k, v in os.environ.items() if k != "OPENROUTER_API_KEY"}
    env["KNOTWORK_NETWORK"] = "openrouter" if live else "deny"
    if live and api_key:
        env["OPENROUTER_API_KEY"] = api_key
    src = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = os.pathsep.join(p for p in (src, env.get("PYTHONPATH", "")) if p)
    results_dir.mkdir(parents=True, exist_ok=True)
    flags: dict[str, Any] = {}
    if sys.platform == "win32":
        flags["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    else:
        flags["start_new_session"] = True
    with (results_dir / "runner.log").open("ab") as log:
        process = subprocess.Popen(
            cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(REPO_ROOT), **flags
        )
    return cast("subprocess.Popen[bytes]", process)


# ------------------------------------------------------------------ grading
def normalise_reply(reply: str, item: ItemRecord) -> str:
    """Rack port answers are graded on the port number: `PP-07`, `port 7` and `07` all mean 07."""
    if item.domain != "rack_patch" or item.answer.get("type") != "label":
        return reply
    digits = re.findall(r"\d+", extract(reply))
    return f"ANSWER: {int(digits[-1]):02d}" if digits else reply


def categorise(reply: str, item: ItemRecord, correct: bool) -> str | None:
    """The deterministic error taxonomy of a graded reply (None when it is correct)."""
    if correct:
        return None
    text = reply.strip()
    if not text or (REFUSAL_RE.search(text) and not ANSWER_RE.search(text)):
        return "refusal_or_empty"
    if not ANSWER_RE.search(text) or is_hedged(text):
        return "format_violation"
    kind = str(item.answer.get("type", ""))
    if kind == "premise":
        return "false_premise_error"
    if kind == "float":
        return "numeric_tolerance_miss"
    return "wrong_port_or_strand"


def grade(reply: str, item: ItemRecord) -> tuple[bool, str, str | None]:
    """(correct, extracted answer, error category) for a model reply."""
    if is_hedged(reply):
        return False, extract(reply).strip()[:80], "format_violation"
    norm = normalise_reply(reply, item)
    correct = grade_prediction(norm, item.answer) == 1.0
    extracted = extract(norm).strip()[:80] if ANSWER_RE.search(norm) else ""
    return correct, extracted, categorise(reply, item, correct)


# ------------------------------------------------------------------ model calls
@dataclass
class Reply:
    text: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: float
    attempts: int
    simulated: bool = False


class ApiError(RuntimeError):
    def __init__(self, message: str, attempts: int) -> None:
        super().__init__(message)
        self.attempts = attempts


def mock_reply(
    model: str, item: ItemRecord, prices: dict[str, tuple[float, float]], repeat_idx: int = 0
) -> Reply:
    """A deterministic simulated model: skill from the model id, harder presets lower it. Each
    repeat draws afresh (repeat 0 keeps the draw it had before repeats existed)."""
    path = ("arena-mock", model, item.bundle_id) + ((f"r{repeat_idx}",) if repeat_idx else ())
    rng = SeedTree(0).child(*path).rng()
    skill = 0.30 + 0.5 * SeedTree(0).child("arena-skill", model).rng().random()
    skill += {"Clean": 0.12, "Hard logic": -0.15, "High stress": -0.10}.get(item.preset, 0.0)
    skill -= 0.05 if item.domain == "rack_patch" else 0.0
    roll = float(rng.random())
    truth = item.truth
    wrong = truth
    if item.options:
        others = [o for o in item.options if o != truth]
        wrong = others[int(rng.integers(len(others)))] if others else truth
    elif item.answer.get("type") in ("int", "float"):
        try:
            wrong = str(round(float(truth) + float(rng.choice([-2, -1, 1, 2])), 2)).removesuffix(
                ".0"
            )
        except ValueError:
            wrong = "0"
    if roll < 0.03:
        text = "Sorry, I cannot determine this from the image."
    elif roll < 0.08:
        text = f"The cable seems to end near {wrong}."
    elif roll < 0.08 + max(0.05, min(0.95, skill)) * 0.92:
        text = f"Following the path carefully.\nANSWER: {truth}"
    else:
        text = f"Following the path carefully.\nANSWER: {wrong}"
    out_tokens = int(rng.integers(60, 340))
    p, c = prices.get(model) or catalog.prices(model)
    return Reply(text, EST_PROMPT_TOKENS, out_tokens, EST_PROMPT_TOKENS * p + out_tokens * c,
                 float(rng.integers(400, 3200)), 1, simulated=True)  # fmt: skip


class Runner:
    """One run over one results folder. `transport` and `sleep` are injectable for tests."""

    def __init__(
        self,
        cfg: SessionConfig,
        results_dir: Path,
        retry_failed: bool = False,
        transport: Any = None,
        sleep: Callable[[float], None] = time.sleep,
        gen_workers: int = 1,
    ) -> None:
        self.cfg, self.results, self.retry_failed = cfg, results_dir, retry_failed
        self.data = storage.resolve(cfg.data_dir)
        self.transport, self.sleep, self.gen_workers = transport, sleep, max(1, gen_workers)
        self.key = os.environ.get("OPENROUTER_API_KEY", "") if cfg.mode == "live" else ""
        raw = storage.read_json(results_dir / PRICES_FILE, {}) or {}
        self.prices = {str(m): (float(v[0]), float(v[1])) for m, v in raw.items()}
        self.status = storage.read_status(results_dir)
        self._last_call = 0.0

    # -------------------------------------------------------------- bookkeeping
    def _save(self, **changes: Any) -> None:
        self.status = self.status.model_copy(update={**changes, "updated_utc": storage.utc_now()})
        storage.write_status(self.results, self.status)

    def _scrub(self, text: str) -> str:
        return text.replace(self.key, "***") if self.key else text

    def _halt(self) -> str | None:
        action = storage.read_control(self.results).action
        return None if action == "run" else action

    def _halt_state(self, action: str) -> None:
        state = "paused" if action == "pause" else "stopped"
        self._save(state=state, message=f"{state.title()} on request; resume any time.",
                   finished_utc=storage.utc_now())  # fmt: skip

    def run(self) -> int:
        if self.cfg.mode == "live" and not self.key:
            self._save(
                state="failed", pid=os.getpid(), message="Live mode needs OPENROUTER_API_KEY."
            )
            return 2
        self._save(state="running", pid=os.getpid(), started_utc=storage.utc_now(),
                   finished_utc="", message="", spend_cap_usd=self.cfg.spend_cap_usd,
                   total_items=self.cfg.total_items, total_trials=self.cfg.total_trials)  # fmt: skip
        try:
            items = self.generate()
            if items is None:
                return 0
            return self.evaluate(items)
        except Exception as exc:  # the process boundary: every failure is recorded, then raised
            self._save(state="failed", message=self._scrub(f"{type(exc).__name__}: {exc}")[:400],
                       finished_utc=storage.utc_now())  # fmt: skip
            raise

    # -------------------------------------------------------------- phase 1: dataset
    def generate(self) -> list[ItemRecord] | None:
        """Every item of every block, generated once (reused on resume). None if halted."""
        self._save(phase="generate")
        if self.cfg.data_mode == "reuse":
            wanted = {b.block_id for b in self.cfg.blocks}
            reused = [i for i in storage.dataset_items(self.data) if i.block_id in wanted]
            self._ready(reused)
            return reused
        have = {i.key: i for i in storage.load_manifest(self.data)}
        jobs = []
        for block in self.cfg.blocks:
            for k in range(block.n):
                seed = block.seed + k
                old = have.get(f"{block.block_id}:{k}")
                if old and (old.seed, old.domain, old.preset) == (seed, block.domain, block.preset):
                    continue
                jobs.append({"block_id": block.block_id, "domain": block.domain,
                             "preset": block.preset, "index": k, "seed": seed,
                             "data_dir": str(self.data)})  # fmt: skip
        wanted_keys = {f"{b.block_id}:{k}" for b in self.cfg.blocks for k in range(b.n)}
        items = {k: v for k, v in have.items() if k in wanted_keys}
        self._ready(list(items.values()))
        halted = self._run_jobs(jobs, items)
        self._write_manifest(list(items.values()))
        if halted:
            self._halt_state(halted)
            return None
        return list(items.values())

    def _run_jobs(self, jobs: list[dict[str, Any]], items: dict[str, ItemRecord]) -> str | None:
        def done(record: dict[str, Any]) -> None:
            item = ItemRecord.model_validate(record)
            items[item.key] = item
            self._ready(list(items.values()))
            if len(items) % MANIFEST_EVERY == 0:
                self._write_manifest(list(items.values()))

        if self.gen_workers <= 1:
            for job in jobs:
                halted = self._halt()
                if halted:
                    return halted
                done(make_item_job(job))
            return None
        with ProcessPoolExecutor(self.gen_workers) as pool:
            pending: set[Future[dict[str, Any]]] = set()
            queue = list(reversed(jobs))
            halted = None
            while queue or pending:
                halted = halted or self._halt()
                while queue and not halted and len(pending) < 2 * self.gen_workers:
                    pending.add(pool.submit(make_item_job, queue.pop()))
                if not pending:
                    break
                finished, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
                for fut in finished:
                    done(fut.result())
            return halted

    def _ready(self, items: list[ItemRecord]) -> None:
        self._save(items_ready=len(items), items_certified=sum(i.certified for i in items))

    def _write_manifest(self, items: list[ItemRecord]) -> None:
        meta = {
            "session": self.cfg.slug,
            "generator": "knotwork.arena",
            "updated_utc": storage.utc_now(),
        }
        storage.write_manifest(self.data, items, meta)

    # -------------------------------------------------------------- phase 2: trials
    def plan(self, items: list[ItemRecord]) -> list[tuple[str, str, ItemRecord, int]]:
        by_block: dict[str, list[ItemRecord]] = {}
        for item in items:
            if item.certified:
                by_block.setdefault(item.block_id, []).append(item)
        out = []
        for block in self.cfg.blocks:
            for item in sorted(by_block.get(block.block_id, []), key=lambda i: i.item_index):
                for model in self.cfg.models_for(block):
                    for r in range(self.cfg.repeats):
                        tid = trial_key(block.block_id, item.item_index, model, r)
                        out.append((tid, model, item, r))
        return out

    def evaluate(self, items: list[ItemRecord]) -> int:
        planned = self.plan(items)
        done = storage.load_trials(self.results)
        ids = {t[0] for t in planned}
        if self.retry_failed:
            todo = [t for t in planned if t[0] in done and done[t[0]].status == "api_error"]
        else:
            todo = [t for t in planned if t[0] not in done]
        mine = [done[i] for i in ids if i in done]
        self._latest: dict[str, str] = {t.trial_id: t.status for t in mine}
        self._save(phase="evaluate", total_trials=len(planned),
                   completed_trials=sum(t.status == "ok" for t in mine),
                   failed_trials=sum(t.status == "api_error" for t in mine),
                   spend_usd=round(sum(t.cost_usd for t in done.values()), 6))  # fmt: skip
        halted = self._dispatch(todo)
        if halted in ("pause", "stop"):
            self._halt_state(halted)
            return 0
        if halted == "budget":
            self._save(state="budget_stop", phase="finished", finished_utc=storage.utc_now(),
                       message=f"Spend cap ${self.cfg.spend_cap_usd:.2f} reached.")  # fmt: skip
            return 0
        failed = self.status.failed_trials
        note = (
            f"{failed} API call(s) failed: Retry failed calls re-asks only those." if failed else ""
        )
        self._save(state="done", phase="finished", finished_utc=storage.utc_now(), message=note)
        return 0

    def _estimate(self, model: str) -> float:
        p, c = self.prices.get(model) or catalog.prices(model)
        out = min(EST_COMPLETION_TOKENS, self.cfg.max_tokens)
        return (EST_PROMPT_TOKENS * p + out * c) * COST_MARGIN

    def _dispatch(self, todo: list[tuple[str, str, ItemRecord, int]]) -> str | None:
        live = self.cfg.mode == "live"
        gap = 60.0 / float(self.cfg.rpm)
        queue = list(reversed(todo))
        pending: dict[Future[Reply], tuple[str, str, ItemRecord, int, float]] = {}
        halted: str | None = None
        with ThreadPoolExecutor(self.cfg.max_concurrency) as pool:
            while queue or pending:
                halted = halted or self._halt()
                while queue and not halted and len(pending) < self.cfg.max_concurrency:
                    tid, model, item, rep = queue[-1]
                    reserve = self._estimate(model) if live else 0.0
                    in_flight = sum(r[4] for r in pending.values())
                    if (
                        live
                        and self.status.spend_usd + in_flight + reserve > self.cfg.spend_cap_usd
                    ):
                        halted = "budget"
                        break
                    queue.pop()
                    wait_s = self._last_call + gap - time.monotonic()
                    if wait_s > 0:
                        self.sleep(wait_s)
                    self._last_call = time.monotonic()
                    fut = pool.submit(self._ask, model, item, rep)
                    pending[fut] = (tid, model, item, rep, reserve)
                if not pending:
                    break
                finished, _ = wait(list(pending), timeout=1.0, return_when=FIRST_COMPLETED)
                for fut in finished:
                    tid, model, item, rep, _r = pending.pop(fut)
                    self._record(tid, model, item, fut, rep)
        return halted

    def _record(
        self, tid: str, model: str, item: ItemRecord, fut: Future[Reply], repeat_idx: int = 0
    ) -> None:
        base: dict[str, Any] = {"trial_id": tid, "block_id": item.block_id, "item_index": item.item_index,
                "bundle_id": item.bundle_id, "model": model, "repeat_idx": repeat_idx, "truth": item.truth,
                "answer_type": str(item.answer.get("type", "")), "ts_utc": storage.utc_now()}  # fmt: skip
        try:
            reply = fut.result()
        except ApiError as exc:
            trial = TrialRecord(**base, status="api_error", error_category="api_error",
                                error=self._scrub(str(exc))[:300], attempts=exc.attempts)  # fmt: skip
        else:
            correct, extracted, category = grade(reply.text, item)
            trial = TrialRecord(
                **base, status="ok", correct=correct, extracted=extracted,
                reply=self._scrub(reply.text)[:4000], error_category=cast(Any, category),
                latency_ms=round(reply.latency_ms, 1), prompt_tokens=reply.prompt_tokens,
                completion_tokens=reply.completion_tokens, cost_usd=round(reply.cost_usd, 8),
                attempts=reply.attempts, simulated=reply.simulated,
            )  # fmt: skip
        storage.append_trial(self.results, trial)
        previous = self._latest.get(tid)
        self._latest[tid] = trial.status
        statuses = [previous, trial.status]
        completed = self.status.completed_trials - (statuses[0] == "ok") + (statuses[1] == "ok")
        failed = (
            self.status.failed_trials - (statuses[0] == "api_error") + (statuses[1] == "api_error")
        )
        self._save(completed_trials=completed, failed_trials=failed,
                   spend_usd=round(self.status.spend_usd + trial.cost_usd, 6))  # fmt: skip

    def _ask(self, model: str, item: ItemRecord, repeat_idx: int = 0) -> Reply:
        if self.cfg.mode == "mock":
            if self.cfg.mock_delay_s:
                self.sleep(self.cfg.mock_delay_s)
            return mock_reply(model, item, self.prices, repeat_idx)
        return self._ask_live(model, item)

    def _ask_live(self, model: str, item: ItemRecord) -> Reply:
        import httpx

        from knotwork.bench.prompts import image_png, messages

        png = image_png(self.data / item.dir / "beauty.png")
        body = {"model": model, "messages": messages(png, item.question), "temperature": self.cfg.temperature,
                "max_tokens": self.cfg.max_tokens, "usage": {"include": True},
                "provider": {"data_collection": "deny", "allow_fallbacks": True}}  # fmt: skip
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json",
                   "X-Title": "knotwork-model-arena"}  # fmt: skip
        net.check(ENDPOINT)
        last = "no attempt"
        for attempt in range(1, MAX_RETRIES + 2):
            t0 = time.perf_counter()
            try:
                with httpx.Client(timeout=CALL_TIMEOUT_S, transport=self.transport) as client:
                    resp = client.post(ENDPOINT, json=body, headers=headers)
                status = resp.status_code
                data = resp.json() if resp.content else None
            except (httpx.TimeoutException, httpx.TransportError, ValueError) as exc:
                status, data = 599, None
                last = type(exc).__name__
            latency = (time.perf_counter() - t0) * 1000
            if status == 200 and isinstance(data, dict):
                choice = (data.get("choices") or [{}])[0]
                text = str((choice.get("message") or {}).get("content") or "")
                usage = data.get("usage") or {}
                pt, ct = (
                    int(usage.get("prompt_tokens") or 0),
                    int(usage.get("completion_tokens") or 0),
                )
                cost = usage.get("cost")
                p, c = self.prices.get(model) or catalog.prices(model)
                usd = float(cost) if isinstance(cost, (int, float)) else pt * p + ct * c
                return Reply(text, pt, ct, usd, latency, attempt)
            if status != 599:
                last = f"HTTP {status}: {str(data)[:160]}"
            if status not in RETRY_STATUS and status != 599 or attempt > MAX_RETRIES:
                raise ApiError(last, attempt)
            jitter = float(
                SeedTree(0).child("jitter", model, item.bundle_id, attempt).rng().random()
            )
            self.sleep(min(30.0, 2.0 ** (attempt - 1)) * (0.5 + jitter))
        raise ApiError(last, MAX_RETRIES + 1)


# ------------------------------------------------------------------ entry point
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m knotwork.arena.runner", description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--resume", action="store_true", help="continue an existing session")
    parser.add_argument("--retry-failed-only", action="store_true")
    parser.add_argument("--gen-workers", type=int, default=0, help="0 = min(4, cpu_count - 1)")
    args = parser.parse_args(argv)
    results = storage.resolve(args.results_dir)
    cfg = storage.load_config(results)
    if cfg is None:
        print(f"arena: no valid config.json in {results}", file=sys.stderr)
        return 2
    status = storage.read_status(results)
    if status.state == "running" and status.pid != os.getpid() and pid_alive(status.pid):
        print(
            f"arena: a runner (pid {status.pid}) is already working on {results}", file=sys.stderr
        )
        return 3
    if (results / storage.RESULTS).is_file() and not (args.resume or args.retry_failed_only):
        print("arena: results exist; pass --resume to continue them", file=sys.stderr)
        return 2
    storage.send_control(results, "run")  # a fresh launch clears an earlier pause or stop
    workers = args.gen_workers or max(1, min(4, (os.cpu_count() or 2) - 1))
    return Runner(cfg, results, args.retry_failed_only, gen_workers=workers).run()


if __name__ == "__main__":
    raise SystemExit(main())
