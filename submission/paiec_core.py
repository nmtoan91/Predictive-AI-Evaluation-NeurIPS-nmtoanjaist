"""TagCover: the predictor of our PAIEC 2026 submission.

Numpy at prediction time; torch and transformers only for the optional LLM rater.
Everything here is a pure function of (input, labeled): the caches hold
deterministic per-item and per-evidence results and never change an answer.

    logit P(correct) = logit(pair rate) + tag offsets + rating slopes + item offset

PAIR RATE: a three-level logistic-normal model, solved exactly on a grid of logits.
    draw        mu_D  ~ N(prior_mean, draw_sd^2)    level of the hidden draw
    benchmark   mu_b  ~ N(mu_D, bench_sd^2)         one per benchmark_id
    pair        theta ~ N(mu_b, pair_sd^2)          one per subject-benchmark pair
    label       y     ~ Bernoulli(sigmoid(theta))
  Every label in `labeled` (other subjects on the benchmark, other benchmarks of
  the draw) moves the target's prior through mu_b and mu_D.

ITEM STAGES (each off at sd = 0), fitted one after another on the residuals of
the labelled observations of the target's benchmark, taken against leave-one-out
pair rates, one Newton step each:
  tag_sd    an offset per value of an item_features key (`lang=go`)
  feat_sd   a slope per LLM rating (log10 minutes, difficulty), z-scored inside the
            benchmark, around the prior slope `feat_prior` ("rated harder = solved less")
  item_sd   an offset for the target item from other subjects' labels on the same item

OUTPUT: the item shift is clipped to +-8 logits and mixed with the pair rate,
    p = (1 - shrink) * sigmoid(logit(rate) + shift) + shrink * rate,  clipped to [eps, 1 - eps].

LLM RATER (features ["llm"]): an instruction-tuned LLM rates each item text once
(greedy decoding). During local scoring the ratings of the public items come
from a cache (PAIEC_FEATURE_DIR, written by scripts/compute_ratings.py with the
same class), so local scores and the platform see the same numbers. With
llm_safe = 1 the stage is optional: without a GPU, short of host memory, after
llm_budget seconds of rating in one worker, or when the model fails, items get
no rating and the predictor answers as the same model without the stage.
llm_call_s bounds the rating time inside one predict() call, because the
platform stops a run when one call exceeds its per-call timeout.

ACQUISITION ("tagcover", make_acquire): buy the current candidate when its tag
group holds the fewest of the pair's labels so far; always buy once the
candidates left are no more than the labels left.
"""

import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import time

import numpy as np

SUBJECT_FIELDS = ("normalized_name", "provider", "release_date", "access_date", "harness",
                  "harness_version", "reasoning_effort", "subject_features_extra")

DEFAULTS = {
    "family": "hier",
    # pair rate (logit scale)
    "grid_limit": 8.0, "grid_step": 0.2,
    "prior_mean": 0.0, "draw_sd": 0.0, "bench_sd": 1.5, "pair_sd": 1.0,
    # item stages
    "tag_sd": 0.0, "tag_min_count": 2,
    "item_sd": 0.0,
    "features": [], "feat_sd": 0.0, "feat_min_items": 6.0, "feat_prior": 0.0,
    # LLM rater: llm_fields picks the rating columns (0 = log10 minutes, 1 = difficulty, then
    # steps, knowledge, precision); llm_tokens new tokens are generated (24 cover the first two)
    "text_chars": 6000.0,
    "llm_model": "", "llm_cache": "", "llm_mode": "rating", "llm_fields": [],
    "llm_tokens": 64.0, "llm_safe": 0.0, "llm_budget": 0.0, "llm_host_gb": 0.0, "llm_call_s": 0.0,
    # output
    "shrink": 0.0, "eps": 1e-4,
    # acquisition policy used by make_acquire: "random" means ship no labeling.py
    "acquire": "random",
}


# --------------------------------------------------------------------------- small helpers

def load_config(model_file):
    """DEFAULTS updated by the config.json beside model.py; unknown keys are an error."""
    directory = os.path.dirname(os.path.abspath(model_file))
    config = dict(DEFAULTS)
    path = os.path.join(directory, "config.json")
    if os.path.exists(path):
        with open(path) as handle:
            given = json.load(handle)
        unknown = sorted(set(given) - set(DEFAULTS))
        if unknown:
            raise KeyError(f"unknown config keys {unknown}; known: {sorted(DEFAULTS)}")
        config.update(given)
    if config["family"] != "hier" or config["llm_mode"] != "rating":
        raise ValueError("this core implements family 'hier' with llm_mode 'rating' only")
    return config, directory


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p, eps=1e-6):
    p = min(1.0 - eps, max(eps, float(p)))
    return math.log(p / (1.0 - p))


def clip_text(text, max_chars):
    """Head and tail of a long text: task prompts put boilerplate first and the
    actual question last, so the tail gets two thirds of the budget."""
    text = text or ""
    max_chars = int(max_chars)
    if len(text) <= max_chars:
        return text
    head = max_chars // 3
    return text[:head] + "\n...\n" + text[-(max_chars - head):]


def text_key(text):
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()


def parse_tags(text):
    """`a=1;b=x` -> [("a", "1"), ("b", "x")]."""
    out = []
    for part in (text or "").split(";"):
        if "=" in part:
            key, value = part.split("=", 1)
            out.append((key.strip(), value.strip()))
    return out


# --------------------------------------------------------------------------- evidence

class _State:
    """`labeled` grouped by benchmark and pair. Built once per distinct evidence."""

    def __init__(self, labeled):
        self.pairs = {}      # (bench, subject key) -> {"n", "s", "subject"}
        self.benches = {}    # bench -> [pair keys], in first-seen order
        self.obs = {}        # bench -> [(pair key, item dict, y)]
        self.fits = {}
        self.stages = {}
        for (subject, item), label in labeled or ():
            bench = item.get("benchmark_id") or ""
            pair_key = (bench, tuple(subject.get(field) or "" for field in SUBJECT_FIELDS))
            pair = self.pairs.get(pair_key)
            if pair is None:
                pair = self.pairs[pair_key] = {"n": 0, "s": 0, "subject": subject}
                self.benches.setdefault(bench, []).append(pair_key)
            y = 1 if label else 0
            pair["n"] += 1
            pair["s"] += y
            self.obs.setdefault(bench, []).append((pair_key, item, y))


_STATES = []


def _state_of(labeled):
    """The state for this evidence, from a three-entry cache.

    The fingerprint is the labels plus the first and last item, which is enough
    to tell apart the evidence sets one process ever sees: a worker gets one
    frozen `labeled` per checkpoint, the acquisition process a growing prefix.
    """
    labeled = labeled or ()
    if labeled:
        first, last = labeled[0][0][1], labeled[-1][0][1]
        mark = (len(labeled), tuple(1 if entry[1] else 0 for entry in labeled),
                first.get("item_content"), last.get("item_content"),
                labeled[-1][0][0].get("normalized_name"), last.get("benchmark_id"))
    else:
        mark = (0,)
    for known, state in _STATES:
        if known == mark:
            return state
    state = _State(labeled)
    _STATES.append((mark, state))
    if len(_STATES) > 3:
        del _STATES[0]
    return state


# --------------------------------------------------------------------------- pair rate

class _Grid:
    def __init__(self, limit, step):
        count = int(round(2.0 * limit / step)) + 1
        self.x = np.linspace(-limit, limit, count)
        self.step = float(step)
        self.p = _sigmoid(self.x)
        self.log_p = np.log(self.p)
        self.log_q = np.log1p(-self.p)
        self._kernels = {}

    def kernel(self, sd):
        """K[i, j] = P(child = x_i | parent = x_j) for child ~ N(parent, sd^2)."""
        key = round(float(sd), 4)
        kernel = self._kernels.get(key)
        if kernel is None:
            z = (self.x[:, None] - self.x[None, :]) / max(key, 0.5 * self.step)
            matrix = np.exp(-0.5 * z * z)
            kernel = self._kernels[key] = matrix / matrix.sum(axis=0, keepdims=True)
        return kernel

    def likelihood(self, s, n):
        return np.exp(s * self.log_p + (n - s) * self.log_q)


_GRIDS = {}


def _grid(config):
    key = (float(config["grid_limit"]), float(config["grid_step"]))
    if key not in _GRIDS:
        _GRIDS[key] = _Grid(*key)
    return _GRIDS[key]


class _HierFit:
    """Exact messages of the three-level model on the grid."""

    def __init__(self, state, config):
        self.state = state
        self.grid = grid = _grid(config)
        width = max(float(config["draw_sd"]), 0.5 * grid.step)
        prior = np.exp(-0.5 * ((grid.x - float(config["prior_mean"])) / width) ** 2)
        self.draw_prior = prior / prior.sum()
        self.sigma, self.tau = float(config["bench_sd"]), float(config["pair_sd"])
        self.u, self.v, self.cavity, self._rates = {}, {}, {}, {}
        for bench, pair_keys in state.benches.items():
            product = np.ones_like(grid.x)
            for pair_key in pair_keys:
                pair = state.pairs[pair_key]
                message = grid.kernel(self.tau).T @ grid.likelihood(pair["s"], pair["n"])
                self.u[pair_key] = message   # pair -> benchmark
                product = product * message
                product = product / product.sum()
            self.v[bench] = grid.kernel(self.sigma).T @ product   # benchmark -> draw

    def _cavity(self, bench, pair_key):
        """Prior over the pair's theta from everything except the pair's own labels."""
        outside = self.cavity.get(bench)
        if outside is None:
            belief = self.draw_prior.copy()
            for other, message in self.v.items():
                if other != bench:
                    belief = belief * message
                    belief = belief / belief.sum()
            outside = self.cavity[bench] = self.grid.kernel(self.sigma) @ belief
        inside = outside.copy()
        for other in self.state.benches.get(bench, ()):
            if other != pair_key:
                inside = inside * self.u[other]
                inside = inside / inside.sum()
        return self.grid.kernel(self.tau) @ inside

    def rates(self, pair_key):
        """(posterior mean rate, the same without one of the pair's failures, without
        one of its successes). The last two are the leave-one-out rates the item
        stages take residuals against."""
        cached = self._rates.get(pair_key)
        if cached is None:
            pair = self.state.pairs.get(pair_key) or {"n": 0, "s": 0}
            n, s = pair["n"], pair["s"]
            prior, grid = self._cavity(pair_key[0], pair_key), self.grid
            out = [0.0, 0.0, 0.0]
            for index, (s_used, n_used) in enumerate(((s, n), (s, n - 1), (s - 1, n - 1))):
                if n_used < 0 or s_used < 0 or s_used > n_used:
                    continue
                posterior = prior * grid.likelihood(s_used, n_used)
                out[index] = float(posterior @ grid.p) / float(posterior.sum())
            cached = self._rates[pair_key] = tuple(out)
        return cached


# --------------------------------------------------------------------------- LLM rater

_CALL = [0.0]     # when the predict() call under way started (Core.predict)
_DEFERRED = [0]   # ratings put off to a later call so far (Rater._live)


class _PerItem:
    """Memo of a per-item computation, keyed by the item text. A value made while a
    rating was put off to a later call is not kept: the next call asks again."""

    def __init__(self, function):
        self.function, self.memo = function, {}

    def __call__(self, item):
        content = item.get("item_content") or ""
        value = self.memo.get(content)
        if value is None:
            if len(self.memo) > 6000:
                self.memo.clear()
            before = _DEFERRED[0]
            value = self.function(item)
            if _DEFERRED[0] == before:
                self.memo[content] = value
        return value


# The organizers' archive check and smoke test import model.py with this set and
# then call predict() on toy inputs: no place to load a model. Their tools set
# it only around the import, so it is read here, at import time, as well.
_SMOKE_AT_IMPORT = os.environ.get("PREDICTIVE_EVAL_LOCAL_SMOKE_TEST") == "1"


def _smoke():
    return _SMOKE_AT_IMPORT or os.environ.get("PREDICTIVE_EVAL_LOCAL_SMOKE_TEST") == "1"


def _hf_cache_dir():
    for candidate in (os.environ.get("HF_HOME", "").strip(), "/app/hf_cache"):
        if candidate and os.path.isdir(candidate):
            return candidate
    return None


class FeatureCache:
    """<PAIEC_FEATURE_DIR>/<name>/: ratings computed offline for the public items.

    vectors.npy (N x 5) + keys.json (N text keys). A missing directory or key
    means "rate it now", unless PAIEC_REQUIRE_CACHE=1 (set by scripts/evaluate.py),
    where a miss is an error instead of a silent model load.
    """

    def __init__(self, name):
        self.rows, self.vectors = {}, None
        root = os.environ.get("PAIEC_FEATURE_DIR", "")
        directory = os.path.join(root, name) if root and name else ""
        if directory and os.path.exists(os.path.join(directory, "vectors.npy")):
            self.vectors = np.load(os.path.join(directory, "vectors.npy"), mmap_mode="r")
            with open(os.path.join(directory, "keys.json")) as handle:
                self.rows = {key: index for index, key in enumerate(json.load(handle))}
        self.required = os.environ.get("PAIEC_REQUIRE_CACHE") == "1"

    def get(self, key):
        row = self.rows.get(key)
        if row is not None:
            return np.asarray(self.vectors[row], dtype=np.float32)
        if self.required and not _smoke():
            raise KeyError(f"rating cache miss for {key}: run scripts/compute_ratings.py")
        return None


RATER_PROMPT = (
    "You are an expert at judging how hard a task is.\n"
    "Read the task below, which was given to an AI system as part of a benchmark, and rate it.\n"
    "Answer with one JSON object and nothing else, with these integer fields:\n"
    '  "minutes": how many minutes a skilled human professional would need to complete it correctly,\n'
    '  "difficulty": overall difficulty from 1 (trivial) to 10 (research-level),\n'
    '  "steps": how many distinct reasoning or action steps a correct solution needs, from 1 to 10,\n'
    '  "knowledge": how much specialised knowledge it needs, from 1 to 10,\n'
    '  "precision": how unforgiving the grading is likely to be, from 1 (lenient) to 10 (one mistake fails).\n\n'
    "TASK:\n{task}\n\nJSON:"
)
RATER_FIELDS = ("minutes", "difficulty", "steps", "knowledge", "precision")
_NUMBER = re.compile(r'"(\w+)"\s*:\s*"?(-?\d+(?:\.\d+)?)')


def parse_rating(reply):
    """Rater reply -> 5 numbers (log10 minutes, then the four 1-10 scales); NaN where missing."""
    found = {name: float(value) for name, value in _NUMBER.findall(reply or "")}
    values = [found.get(name, float("nan")) for name in RATER_FIELDS]
    minutes = values[0]
    values[0] = math.log10(min(max(minutes, 0.1), 1e5)) if minutes == minutes else float("nan")
    for index in range(1, 5):
        if values[index] == values[index]:
            values[index] = min(max(values[index], 1.0), 10.0)
    return np.array(values, dtype=np.float32)


def _note(message):
    """One line in the scoring log (the worker's stderr): what the optional stage did."""
    try:
        print(f"[paiec pid {os.getpid()}] {message}", file=sys.stderr, flush=True)
    except Exception:
        pass


def _free_memory_gb():
    """Host memory a load can count on: what the machine has available, capped by the
    container's limit; None where neither can be read."""
    values = []
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    values.append(float(line.split()[1]) / 2 ** 20)
    except (OSError, ValueError, IndexError):
        pass
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            with open(path) as handle:
                limit = handle.read().strip()
            if limit.isdigit() and int(limit) < 2 ** 50:
                values.append(int(limit) / 2 ** 30)
        except OSError:
            pass
    return min(values) if values else None


class Rater:
    """An instruction-tuned LLM rating a task from its text: five scales generated
    as one JSON object, greedy decoding.

    `safe`: the stage is optional. Without a GPU, or when the model cannot be
    loaded or run, items get no rating (NaN), which takes them out of the slope
    fit downstream: the submission answers as the same model without the LLM
    instead of failing or running for days on a CPU. More guards in this mode:
    `budget` seconds of rating per worker process (workers restart at each of
    the six checkpoints), `host_gb` of host memory that must be free before the
    model is loaded, and `call_s` seconds of rating inside one predict() call
    (the rest of a benchmark's labelled items is rated in the next calls). In a
    platform worker the model is loaded at import time, which is under the
    8-hour deadline only. Each worker writes to stderr when the stage starts,
    how fast it goes and why it stops (`[paiec pid ...]` lines).
    PAIEC_LLM_ON_CPU=1 lets the stage run without a GPU.
    """

    NEUTRAL = np.array([1.0, 5.0, 5.0, 5.0, 5.0], dtype=np.float32)

    def __init__(self, model_id, cache_name, max_chars, tokens=64, safe=False, budget=0.0, host_gb=0.0,
                 call_s=0.0):
        self.model_id, self.max_chars = model_id, int(max_chars)
        self.tokens, self.safe, self.off = int(tokens), bool(safe), False
        self.budget, self.host_gb, self.call_s = float(budget), float(host_gb), float(call_s)
        self.spent, self.rated, self.failed = 0.0, 0, 0
        self.MISSING = np.full(len(self.NEUTRAL), np.nan, dtype=np.float32)
        self.cache = FeatureCache(cache_name)
        self.model = self.tokenizer = None

    def prompts(self, texts):
        return [RATER_PROMPT.format(task=clip_text(text, self.max_chars)) for text in texts]

    def _load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cache_dir = _hf_cache_dir()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, cache_dir=cache_dir)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        on_gpu = torch.cuda.is_available()
        dtype = torch.float16 if on_gpu else torch.float32
        try:  # straight onto the device: half the host memory of load-then-move
            placement = {"device_map": {"": 0}} if on_gpu else {}
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, torch_dtype=dtype, cache_dir=cache_dir, low_cpu_mem_usage=True, **placement)
        except (ImportError, ValueError, TypeError):  # no accelerate in the container
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=dtype, cache_dir=cache_dir)
            self.model.to("cuda" if on_gpu else "cpu")
        self.model.eval()

    def rate(self, texts, batch_size=4):
        import torch

        if self.model is None:
            self._load()
        prompts, out = self.prompts(texts), []
        for start in range(0, len(prompts), batch_size):
            chats = [self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                                        add_generation_prompt=True, enable_thinking=False)
                     for prompt in prompts[start:start + batch_size]]
            batch = self.tokenizer(chats, return_tensors="pt", padding=True).to(self.model.device)
            with torch.no_grad():
                generated = self.model.generate(**batch, max_new_tokens=self.tokens, do_sample=False)
            for row in generated[:, batch["input_ids"].shape[1]:]:
                out.append(parse_rating(self.tokenizer.decode(row, skip_special_tokens=True)))
        return np.stack(out) if out else np.zeros((0, len(RATER_FIELDS)), dtype=np.float32)

    def _stop(self, reason):
        self.off = True
        _note(f"llm stage off after {self.rated} ratings in {self.spent:.0f} s: {reason}")

    def _host_need(self):
        """Through accelerate the weights go to the GPU shard by shard; without it they
        pass through host memory (measured: 15.6 GB for a 4B model)."""
        try:
            direct = importlib.util.find_spec("accelerate") is not None
        except Exception:
            direct = False
        return min(self.host_gb, 6.0) if direct else self.host_gb

    def _ready(self):
        """True with the model loaded (loading it now if the guards allow); otherwise the stage is off."""
        if self.model is not None:
            return True
        try:
            import torch

            if not torch.cuda.is_available() and os.environ.get("PAIEC_LLM_ON_CPU") != "1":
                self._stop("no GPU")
                return False
            free, need = _free_memory_gb(), self._host_need()
            if free is not None and free < need:
                self._stop(f"{free:.0f} GB of host memory free, loading needs {need:.0f}")
                return False
            self._load()
            return True
        except Exception as error:  # it does not load here: it never will
            self.model = None
            self._stop(f"{type(error).__name__}: {' '.join(str(error).split())[:160]}")
            return False

    def warm(self):
        """Load at import time in a platform worker: the start-up of a worker is not under
        the per-call timeout, its first predict() call is."""
        if (self.safe and not self.off and self.model is None and self.cache.vectors is None
                and os.environ.get("PAIEC_PREDICTION_WORKER_ID") is not None and not _smoke()):
            started = time.time()
            if self._ready():
                _note(f"llm stage ready: {self.model_id} on {self.model.device}, loaded in {time.time() - started:.0f} s")

    def _live(self, text):
        """The rating of one text, or MISSING (stage off, put off to a later call, or failed)."""
        if not self.safe:
            return self.rate([text])[0]
        if self.off:
            return self.MISSING
        if self.call_s and _CALL[0] and time.time() - _CALL[0] > self.call_s:
            _DEFERRED[0] += 1  # this call has rated long enough; the next one goes on
            return self.MISSING
        started = time.time()
        if not self._ready():
            return self.MISSING
        try:
            vector = self.rate([text])[0]
        except Exception as error:
            self.failed += 1
            if self.failed >= 3:  # it keeps failing
                self._stop(f"{type(error).__name__}: {' '.join(str(error).split())[:160]}")
            return self.MISSING
        self.failed = 0
        self.spent += time.time() - started
        self.rated += 1
        if self.rated in (1, 10, 100) or self.rated % 500 == 0:
            _note(f"llm stage on: {self.model_id} on {self.model.device}, {self.rated} ratings in {self.spent:.0f} s, "
                  f"{_DEFERRED[0]} put off to a later call")
        if self.budget and self.spent > self.budget:
            self._stop(f"time budget of {self.budget:.0f} s used")
        return vector

    def __call__(self, item):
        text = item.get("item_content") or ""
        vector = self.cache.get(text_key(clip_text(text, self.max_chars)))
        if vector is None:
            if _smoke():
                return self.NEUTRAL
            vector = self._live(text)
            if vector is self.MISSING:
                return vector
        vector = np.asarray(vector, dtype=np.float32)
        return np.where(np.isfinite(vector), vector, self.NEUTRAL)


# --------------------------------------------------------------------------- item stages

_Z_LIMIT = 4.0       # standardized rating values beyond this are treated as this
_SHIFT_LIMIT = 8.0   # largest total move of an item's logit away from its pair's rate


class _BenchStage:
    """Item-level terms for one benchmark, fitted on its labelled observations."""

    def __init__(self, core, state, fit, bench):
        config = core.config
        self.core, self.config = core, config
        observations = state.obs.get(bench, [])
        self.items = [item for _, item, _ in observations]
        self.y = np.array([y for _, _, y in observations], dtype=np.float64)
        base = []
        for pair_key, _, y in observations:
            rates = fit.rates(pair_key)
            base.append(_logit(rates[2] if y else rates[1]))
        eta = np.array(base, dtype=np.float64)
        count = len(observations)

        # tags: one offset per value, keys in sorted order, each on the residuals so far
        self.tag_offsets = {}
        tag_sd = float(config["tag_sd"])
        if tag_sd > 0 and count:
            tags = [dict(parse_tags(item.get("item_features"))) for item in self.items]
            for key in sorted({key for row in tags for key in row}):
                groups = {}
                for index, row in enumerate(tags):
                    if key in row:
                        groups.setdefault(row[key], []).append(index)
                labelled = sum(len(rows) for rows in groups.values())
                if len(groups) > max(2.0, 0.5 * labelled):
                    continue  # nearly one value per item (an index): no group to pool over
                p = _sigmoid(eta)
                for value, rows in groups.items():
                    if len(rows) < int(config["tag_min_count"]):
                        continue
                    rows = np.asarray(rows)
                    delta = float((self.y[rows] - p[rows]).sum()
                                  / ((p[rows] * (1 - p[rows])).sum() + 1.0 / tag_sd ** 2))
                    self.tag_offsets[(key, value)] = delta
                    eta[rows] += delta

        # ratings: z-scored within the benchmark, one slope each around a prior slope. A value
        # that is missing (NaN: a rating not made, or not made yet) takes no part.
        self.feature_fit = None
        self.complete = True  # False while ratings of labelled items are put off to a later call
        feat_sd = float(config["feat_sd"])
        if feat_sd > 0 and core.feature_functions and count:
            before = _DEFERRED[0]
            matrix = np.stack([core.features(item) for item in self.items])
            self.complete = _DEFERRED[0] == before
            distinct = {}
            for index, item in enumerate(self.items):
                distinct.setdefault(item.get("item_content") or "", index)
            rows = np.asarray(sorted(distinct.values()))
            if len(rows) >= int(config["feat_min_items"]):
                known = np.isfinite(matrix)
                if known.all():
                    centre = matrix[rows].mean(axis=0)
                    spread = matrix[rows].std(axis=0)
                    usable = spread > 1e-6
                else:
                    held = known[rows]
                    counts = np.maximum(held.sum(axis=0), 1)
                    centre = np.where(held, matrix[rows], 0.0).sum(axis=0) / counts
                    spread = np.sqrt(np.where(held, (np.where(held, matrix[rows], 0.0) - centre) ** 2, 0.0).sum(axis=0)
                                     / counts)
                    usable = (spread > 1e-6) & (held.sum(axis=0) >= 2)
                    matrix = np.where(known, matrix, centre)  # z = 0 where the value is missing
                spread = np.where(usable, spread, 1.0)
                # clipped: a column that is almost constant has a tiny spread, and one
                # odd item would otherwise get a z-score in the thousands
                z = np.clip((matrix - centre) / spread * usable, -_Z_LIMIT, _Z_LIMIT)
                slopes = np.full(matrix.shape[1], float(config["feat_prior"])) * core.feature_signs
                trial = eta + z @ slopes
                p = _sigmoid(trial)
                slopes = slopes + (z.T @ (self.y - p)) / ((z ** 2).T @ (p * (1 - p)) + 1.0 / feat_sd ** 2)
                eta = eta + z @ slopes
                self.feature_fit = (centre, spread, usable, slopes)

        p = _sigmoid(eta)
        self.residual = self.y - p
        self.variance = p * (1 - p)

        # item: residuals per labelled item
        self.by_item = {}
        if float(config["item_sd"]) > 0:
            for index, item in enumerate(self.items):
                entry = self.by_item.setdefault(item.get("item_content") or "", [0.0, 0.0])
                entry[0] += self.residual[index]
                entry[1] += self.variance[index]

    def shift(self, item):
        config, total = self.config, 0.0
        content = item.get("item_content") or ""
        if self.tag_offsets:
            for tag in parse_tags(item.get("item_features")):
                total += self.tag_offsets.get(tag, 0.0)
        if self.feature_fit is not None:
            centre, spread, usable, slopes = self.feature_fit
            values = self.core.features(item)
            values = np.where(np.isfinite(values), values, centre)  # a missing value moves nothing
            z = np.clip((values - centre) / spread * usable, -_Z_LIMIT, _Z_LIMIT)
            total += float(z @ slopes)
        if self.by_item:
            entry = self.by_item.get(content)
            if entry is not None:
                total += entry[0] / (entry[1] + 1.0 / float(config["item_sd"]) ** 2)
        return total


# --------------------------------------------------------------------------- the predictor

class Core:
    def __init__(self, config):
        self.config = config
        self.identity = id(self)
        self.raters, self.feature_functions, signs = [], [], []
        for name in config["features"]:
            if name != "llm":
                raise KeyError(f"unknown feature {name!r}")
            rater = Rater(config["llm_model"], config["llm_cache"], config["text_chars"], config["llm_tokens"],
                          config["llm_safe"], config["llm_budget"], config["llm_host_gb"], config["llm_call_s"])
            self.raters.append(rater)
            fields = [int(index) for index in config["llm_fields"]] or list(range(len(rater.NEUTRAL)))
            self.feature_functions.append(lambda item, rater=rater, fields=fields: rater(item)[fields])
            signs.extend([-1.0] * len(fields))  # a harder rating means fewer successes
        self.feature_signs = np.asarray(signs, dtype=np.float64)
        self._features = _PerItem(lambda item: np.concatenate(
            [np.asarray(function(item), dtype=np.float64) for function in self.feature_functions]))
        self.staged = any(float(config[key]) > 0 for key in ("tag_sd", "item_sd", "feat_sd"))
        for rater in self.raters:  # in a platform worker: load now, before the first timed call
            rater.warm()

    def features(self, item):
        return self._features(item)

    def predict(self, input, labeled=None):
        _CALL[0] = time.time()
        subject, item = input
        state = _state_of(labeled)
        fit = state.fits.get(self.identity)
        if fit is None:
            fit = state.fits[self.identity] = _HierFit(state, self.config)
        bench = item.get("benchmark_id") or ""
        pair_key = (bench, tuple(subject.get(field) or "" for field in SUBJECT_FIELDS))
        rate = fit.rates(pair_key)[0]
        eps = float(self.config["eps"])
        if self.staged and bench in state.obs:
            stage = state.stages.get((self.identity, bench))
            if stage is None:
                stage = _BenchStage(self, state, fit, bench)
                if stage.complete:  # otherwise the next call builds it again, with more ratings made
                    state.stages[(self.identity, bench)] = stage
            # bounded: no item-level evidence is worth more than this, and exp() must not overflow
            moved = max(-_SHIFT_LIMIT, min(_SHIFT_LIMIT, stage.shift(item)))
            shifted = 1.0 / (1.0 + math.exp(-(_logit(rate) + moved)))
            shrink = float(self.config["shrink"])
            rate = (1.0 - shrink) * shifted + shrink * rate
        return float(min(1.0 - eps, max(eps, rate)))


def make_predict(model_file):
    """The predict(input, labeled) of the model whose config.json sits next to `model_file`."""
    config, _ = load_config(model_file)
    return Core(config).predict


# --------------------------------------------------------------------------- acquisition

def make_acquire(model_file):
    """The acquisition_function of the `tagcover` policy.

    A yes/no rule on the current candidate (the stream is one pass) that never
    ends a pair short of its budget: once the candidates left are only as many
    as the labels left, it buys. Before that it buys a candidate whose
    item_features group has the fewest of this pair's labels so far, so the
    pair's labels are spread evenly over its tag groups.
    """
    config, _ = load_config(model_file)
    if config["acquire"] != "tagcover":
        raise KeyError(f"unknown acquire policy {config['acquire']!r} (a 'random' model ships no labeling.py)")

    def own_items(input, labeled):
        subject, item = input
        bench = item.get("benchmark_id")
        return [entry[0][1] for entry in labeled or ()
                if entry[0][0] == subject and entry[0][1].get("benchmark_id") == bench]

    def acquisition_function(input, labeled=None, context=None):
        if context["labels_remaining"] <= 0:
            return False
        if context["items_remaining"] <= context["labels_remaining"]:
            return True
        counts = {}
        for item in own_items(input, labeled):
            for tag in parse_tags(item.get("item_features")):
                counts[tag] = counts.get(tag, 0) + 1
        tags = parse_tags(input[1].get("item_features"))
        if not tags or not counts:
            return True
        keys = {key for key, _ in counts}
        mine = [counts.get(tag, 0) for tag in tags if tag[0] in keys]
        return bool(not mine or min(mine) <= min(counts.values()))

    return acquisition_function
