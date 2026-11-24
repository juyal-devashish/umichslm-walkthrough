"""Per-token annotations derived from Harmony structure, at load time.

Three training-side features read the same message structure, so it is parsed
once here and emitted as per-token tensors next to ``input_ids``:

* ``token_role`` (int8): 0 = other, 1 = assistant, 2 = tool result. Every token
  of a message -- from its ``<|start|>`` through its end marker -- carries the
  message's role, the same span convention ``scripts/sft_prep/record.py`` uses
  for its loss segments. Drives the assistant-only loss metric and P6.
* ``loss_weights`` (float32, only when ``w_tool != 1``): ``w_tool`` on tool-result
  tokens, 1 elsewhere (P6, role-weighted pretraining loss).
* ``errhead_labels`` (int8): at the ``<|call|>`` token that closes an assistant
  tool call, 1 if any tool result answering it is an error, 0 if none is; -1
  everywhere else and at calls with no result in the document (P3a).
* ``unlik_targets`` (int64, label-aligned like ``labels``): at position ``t`` of a
  CORRECTED call, the token id of the FAILING call at the same offset, where the
  two differ; -100 elsewhere (P3c). See :func:`annotate_messages` for the exact
  rule.

Alignment. Text is parsed into messages by splitting on the literal
``<|start|>``; tokens are split on the ``<|start|>`` token id. The tokenizer maps
every literal occurrence of a special token to its id (verified for the
production SuperBPE tokenizer: ``<|start|>`` -> 4), so the two counts agree and
message ``k`` in text is message ``k`` in tokens. A document where they do not
agree is left unannotated and counted (``stats["misaligned"]``), never guessed.

Error labels come from the tool result's TEXT (``classify_tool_result``):
structured status fields first (mini-coder ``<returncode>``, JSON ``error`` /
``status`` / ``success`` / ``isError`` / HTTP status), then regexes. Calibrated on
``prepared_v2/{agent_traj,tool_calling}``; see ``scripts/survey_tool_errors.py``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

IGNORE_INDEX = -100

ROLE_OTHER = 0
ROLE_ASSISTANT = 1
ROLE_TOOL = 2

START = "<|start|>"
MESSAGE = "<|message|>"
END_MARKERS = ("<|end|>", "<|call|>", "<|return|>")

# Harmony control tokens the annotator needs. Resolved from the tokenizer at
# construction time and checked, never assumed.
_REQUIRED = ("<|start|>", "<|end|>", "<|message|>", "<|channel|>", "<|call|>", "<|return|>")
_OPTIONAL = ("<|tool_call|>",)


@dataclass
class AnnotateOptions:
    """What the loader should emit. ``enabled`` False -> nothing changes."""

    enabled: bool = False
    w_tool: float = 1.0
    errhead: bool = False
    unlik: bool = False
    # "aligned": every offset of the corrected call where the failing call's
    # token differs. "first_diverge": only the first such offset, the one
    # position where the two calls share an identical prefix.
    unlik_mode: str = "aligned"

    @property
    def emit_weights(self) -> bool:
        return self.enabled and float(self.w_tool) != 1.0


@dataclass
class TextMsg:
    role: str
    channel: Optional[str]
    content: str
    marker: str  # one of END_MARKERS, or "" when the message is unterminated


@dataclass
class TokMsg(TextMsg):
    tok_start: int = 0  # [tok_start, tok_end) in document token coordinates
    tok_end: int = 0


@dataclass
class Specials:
    start: int
    end: int
    message: int
    channel: int
    call: int
    ret: int
    tool_call: Optional[int]
    all_ids: frozenset = field(default_factory=frozenset)

    @classmethod
    def from_tokenizer(cls, tokenizer) -> "Specials":
        ids: Dict[str, Optional[int]] = {}
        unk = getattr(tokenizer, "unk_token_id", None)
        for s in _REQUIRED + _OPTIONAL:
            enc = tokenizer.encode(s, add_special_tokens=False)
            tid = enc[0] if len(enc) == 1 else None
            if tid is None or tid == unk:
                if s in _REQUIRED:
                    raise ValueError(
                        f"tokenizer does not map {s!r} to a single special token (got {enc}); "
                        "Harmony annotations need the production tokenizer"
                    )
                tid = None
            ids[s] = tid
        # Every added/special token: never used as an unlikelihood target.
        special = set(getattr(tokenizer, "all_special_ids", []) or [])
        added = getattr(tokenizer, "get_added_vocab", None)
        if callable(added):
            special |= set(added().values())
        special |= {v for v in ids.values() if v is not None}
        return cls(
            start=ids["<|start|>"],
            end=ids["<|end|>"],
            message=ids["<|message|>"],
            channel=ids["<|channel|>"],
            call=ids["<|call|>"],
            ret=ids["<|return|>"],
            tool_call=ids["<|tool_call|>"],
            all_ids=frozenset(special),
        )


# --------------------------------------------------------------------------
# Text parsing
# --------------------------------------------------------------------------

_ROLE_RE = re.compile(r"[^\s<]*")
_CHANNEL_RE = re.compile(r"<\|channel\|>([^\s<]+)")


def parse_harmony(text: str) -> List[TextMsg]:
    """Split ``text`` into messages at every literal ``<|start|>``.

    Anything before the first ``<|start|>`` is not a message (and is not
    returned). Message ``k`` spans from the k-th ``<|start|>`` to the next one.
    """
    out: List[TextMsg] = []
    pos = text.find(START)
    while pos >= 0:
        nxt = text.find(START, pos + len(START))
        seg = text[pos + len(START) : (nxt if nxt >= 0 else len(text))]
        mi = seg.find(MESSAGE)
        header = seg if mi < 0 else seg[:mi]
        role = _ROLE_RE.match(header).group(0)
        ch = _CHANNEL_RE.search(header)
        body = "" if mi < 0 else seg[mi + len(MESSAGE) :]
        cut, marker = len(body), ""
        for m in END_MARKERS:
            j = body.find(m)
            if 0 <= j < cut:
                cut, marker = j, m
        out.append(TextMsg(role, ch.group(1) if ch else None, body[:cut], marker))
        pos = nxt
    return out


# --------------------------------------------------------------------------
# Error classification
# --------------------------------------------------------------------------

_RETURNCODE_RE = re.compile(r"^\s*<returncode>\s*(-?\d+)\s*</returncode>")
# OpenHands-style trailing status lines (nemotron_sft_swe_v3): a structured
# exit code written into the text, so it is trusted like <returncode>.
_EXIT_MARKER_RE = re.compile(
    r"\[(?:The command completed|Command finished) with exit code (-?\d+)\.?\]"
)
# OpenHands prefixes every observation; strip it before start-anchored checks.
_OBS_PREFIX_RE = re.compile(r"^\s*OBSERVATION:\s*")
_TRACEBACK_RE = re.compile(r"Traceback \(most recent call last\):")
# Error words only at the START of the result: deep inside an output they are
# usually data (a grep hit, a log line, source code), not the call's status.
_LEADING_RE = re.compile(
    r"^\s*(?:\[?(?:(?:an )?error|fatal|exception|failed|failure|invalid|unable to|could not|"
    r"couldn't|cannot|can't|not found|unknown (?:command|tool|function)|denied|forbidden|"
    r"unauthorized|timeout|timed out)\b|[A-Za-z_.]*(?:Error|Exception|Fault)\b\s*[:(])",
    re.IGNORECASE,
)
_SHELL_RE = re.compile(
    r"command not found|No such file or directory|Permission denied|\[Errno \d+\]|"
    r"returned non-zero exit status|exit(?:ed with)? (?:code|status)[:= ]+[1-9]\d*\b|"
    r"Segmentation fault|Killed\s*$|npm ERR!|^fatal: |^error: |^E: ",
    re.MULTILINE,
)
_HTTP_RE = re.compile(
    r"\b[45]\d\d (?:Client|Server) Error\b|\bHTTP(?:/\d(?:\.\d)?)?[ :]+[45]\d\d\b|"
    r"\b(?:400 Bad Request|401 Unauthorized|403 Forbidden|404 Not Found|405 Method Not Allowed|"
    r"409 Conflict|422 Unprocessable Entity|429 Too Many Requests|500 Internal Server Error|"
    r"502 Bad Gateway|503 Service Unavailable|504 Gateway Timeout)\b|"
    r"\bstatus(?:[ _]?code)?[\"']?\s*[:=]\s*[45]\d\d\b",
    re.IGNORECASE,
)
# "not found" / "does not exist" near the start (SWE-agent: "Directory src not
# found", "File x not found"). "No matches found" is a valid empty result and is
# deliberately NOT matched.
_NOT_FOUND_RE = re.compile(r"\b(?:not found|does not exist|doesn't exist|no such)\b", re.I)
_EDIT_REJECT_RE = re.compile(
    r"Your proposed edit has introduced new syntax error|\bMcpError\b|"
    r"Error (?:executing|calling) tool|An error occurred (?:when|while) calling tool"
)

_JSON_ERR_STATUS = {"error", "failed", "failure", "fail", "err"}
_JSON_MSG_KEYS = ("message", "msg", "detail", "result", "output")


def _json_status(obj) -> Optional[str]:
    """Structured verdict from a JSON tool result: reason, "" (success), or None."""
    if not isinstance(obj, dict):
        return None
    # A process exit status is the strongest field there is (open_swe_traces:
    # {"returncode": N, "output": ...}); trust it over anything in the output.
    for k in ("returncode", "return_code", "exit_code", "exitCode", "exit_status"):
        v = obj.get(k)
        if isinstance(v, int) and not isinstance(v, bool):
            return "json_returncode" if v != 0 else ""
    for k in ("isError", "is_error"):
        if obj.get(k) is True:
            return "json_is_error"
    err = obj.get("error", obj.get("errors"))
    if err not in (None, "", False, [], {}, "null", "None"):
        return "json_error"
    st = obj.get("status")
    if isinstance(st, str) and st.strip().lower() in _JSON_ERR_STATUS:
        return "json_status"
    for k in ("success", "ok"):
        if obj.get(k) is False:
            return "json_success_false"
    for k in ("status_code", "statusCode", "code", "status", "http_status"):
        v = obj.get(k)
        if isinstance(v, int) and 400 <= v <= 599:
            return "json_http"
    for k in _JSON_MSG_KEYS:
        v = obj.get(k)
        if isinstance(v, str) and _LEADING_RE.match(v):
            return "json_message"
    return ""


def classify_tool_result(content: str) -> Optional[str]:
    """Return the reason a tool result is an ERROR, or None for a success.

    Order matters: a structured status field is trusted over any text pattern
    (mini-coder's ``<returncode>0</returncode>`` wraps outputs that print
    "Error:" as data; a JSON body with ``"status": {...}`` is an order record).
    """
    if not content:
        return None
    m = _RETURNCODE_RE.match(content)
    if m:
        return "returncode" if int(m.group(1)) != 0 else None
    tail = _EXIT_MARKER_RE.findall(content[-400:])
    if tail:
        return "exit_marker" if int(tail[-1]) != 0 else None
    content = _OBS_PREFIX_RE.sub("", content, count=1)
    s = content.strip()
    if s[:1] == "{" and len(s) <= 200_000:
        try:
            verdict = _json_status(json.loads(s))
        except (ValueError, RecursionError):
            verdict = None
        if verdict is not None:
            return verdict or None
    head = content[:400]
    if _LEADING_RE.match(content):
        return "leading_error"
    # Long outputs: tracebacks and shell errors sit at the end, after prints.
    scan = content if len(content) <= 24_000 else content[:12_000] + content[-12_000:]
    if _TRACEBACK_RE.search(scan):
        return "traceback"
    if _EDIT_REJECT_RE.search(head):
        return "tool_rejected"
    if _SHELL_RE.search(scan):
        return "shell"
    if _HTTP_RE.search(scan):
        return "http"
    if _NOT_FOUND_RE.search(content[:200]):
        return "not_found"
    return None


# --------------------------------------------------------------------------
# Token alignment and annotation
# --------------------------------------------------------------------------


def align_messages(
    ids: Sequence[int], text_msgs: List[TextMsg], sp: Specials
) -> Optional[List[TokMsg]]:
    """Attach token ranges to parsed messages; None if the counts disagree."""
    arr = np.asarray(ids)
    starts = np.flatnonzero(arr == sp.start).tolist()
    if len(starts) != len(text_msgs):
        return None
    out: List[TokMsg] = []
    for k, m in enumerate(text_msgs):
        te = starts[k + 1] if k + 1 < len(starts) else len(ids)
        out.append(TokMsg(m.role, m.channel, m.content, m.marker, starts[k], te))
    return out


def _call_span(ids: Sequence[int], m: TokMsg, sp: Specials) -> Tuple[int, int]:
    """Token span [s, e) of the call itself inside an assistant call message.

    From the first ``<|tool_call|>`` (skipping any preamble prose before it)
    through ``<|call|>``; when the call is written without ``<|tool_call|>``
    (JSON envelope / raw content), from the token after ``<|message|>``.
    """
    seg = ids[m.tok_start : m.tok_end]
    s = None
    if sp.tool_call is not None:
        try:
            s = m.tok_start + list(seg).index(sp.tool_call)
        except ValueError:
            s = None
    if s is None:
        try:
            s = m.tok_start + list(seg).index(sp.message) + 1
        except ValueError:
            s = m.tok_start
    e = m.tok_end
    # Trim to just past the closing <|call|>.
    for j in range(m.tok_end - 1, s - 1, -1):
        if ids[j] == sp.call:
            e = j + 1
            break
    return s, e


_BREAKS_ERROR_CHAIN = {"user", "system", "developer"}


def annotate_messages(
    ids: Sequence[int],
    msgs: Optional[List[TokMsg]],
    sp: Specials,
    opts: AnnotateOptions,
    stats: Optional[Dict[str, int]] = None,
) -> Dict[str, np.ndarray]:
    """Per-token arrays for one document (see the module docstring).

    Unlikelihood rule (P3c), exactly: for each assistant call ``F`` whose tool
    results contain an error, find the NEXT assistant call ``C`` in the same
    document with no user/system/developer message in between (the agent's own
    reasoning may intervene). Take both calls' spans (``_call_span``). For each
    offset ``k < min(len F, len C)`` with ``F[k] != C[k]`` and neither token a
    special/control token, set ``unlik_targets[C_start + k] = F[k]``: when the
    model predicts the corrected call's k-th token, push DOWN the probability
    of the failing call's k-th token. Identical re-issues produce no targets.
    ``unlik_mode="first_diverge"`` keeps only the first differing offset -- the
    one place where both calls share the same prefix, so ``p(F[k])`` is exactly
    "the model repeats the failing call".

    ``unlik_need`` (internal, not emitted to the model) holds ``F``'s start, so
    a packer that splits a document can drop targets whose failing call (and
    hence its error) fell outside the window.
    """
    n = len(ids)
    role = np.zeros(n, dtype=np.int8)
    errlab = np.full(n, -1, dtype=np.int8)
    unlik = np.full(n, IGNORE_INDEX, dtype=np.int64)
    need = np.full(n, -1, dtype=np.int64)
    out = {"token_role": role, "errhead_labels": errlab, "unlik_targets": unlik, "unlik_need": need}
    if not msgs:
        return out

    for m in msgs:
        if m.role == "assistant":
            role[m.tok_start : m.tok_end] = ROLE_ASSISTANT
        elif m.role == "tool":
            role[m.tok_start : m.tok_end] = ROLE_TOOL

    if not (opts.errhead or opts.unlik or stats is not None):
        return out

    calls: List[Tuple[int, Optional[bool]]] = []  # (msg index, is_error or None)
    for i, m in enumerate(msgs):
        if m.role != "assistant" or m.marker != "<|call|>":
            continue
        j = i + 1
        is_err: Optional[bool] = None
        while j < len(msgs) and msgs[j].role == "tool":
            reason = classify_tool_result(msgs[j].content)
            is_err = bool(is_err) or reason is not None
            if stats is not None and reason is not None:
                stats[f"reason:{reason}"] = stats.get(f"reason:{reason}", 0) + 1
            j += 1
        calls.append((i, is_err))
        if is_err is None:
            if stats is not None:
                stats["calls_no_result"] = stats.get("calls_no_result", 0) + 1
            continue
        if stats is not None:
            stats["calls_labelled"] = stats.get("calls_labelled", 0) + 1
            stats["calls_error"] = stats.get("calls_error", 0) + int(is_err)
        # The <|call|> token: the last one inside the message.
        for p in range(m.tok_end - 1, m.tok_start - 1, -1):
            if ids[p] == sp.call:
                errlab[p] = 1 if is_err else 0
                break

    for ci, (i, is_err) in enumerate(calls):
        if not is_err or ci + 1 >= len(calls):
            continue
        j = calls[ci + 1][0]
        if any(msgs[x].role in _BREAKS_ERROR_CHAIN for x in range(i + 1, j)):
            continue
        fs, fe = _call_span(ids, msgs[i], sp)
        cs, ce = _call_span(ids, msgs[j], sp)
        f_ids, c_ids = ids[fs:fe], ids[cs:ce]
        if list(f_ids) == list(c_ids):
            if stats is not None:
                stats["repeat_after_error"] = stats.get("repeat_after_error", 0) + 1
            continue
        if stats is not None:
            stats["corrected_after_error"] = stats.get("corrected_after_error", 0) + 1
        if not opts.unlik:
            continue
        for k in range(min(len(f_ids), len(c_ids))):
            a, b = int(f_ids[k]), int(c_ids[k])
            if a == b or a in sp.all_ids or b in sp.all_ids:
                continue
            unlik[cs + k] = a
            need[cs + k] = fs
            if stats is not None:
                stats["unlik_targets"] = stats.get("unlik_targets", 0) + 1
            if opts.unlik_mode == "first_diverge":
                break
    return out


class HarmonyAnnotator:
    """Tokenizer-bound front end: text or SFT segments -> per-token arrays."""

    def __init__(self, tokenizer, opts: AnnotateOptions):
        if opts.unlik_mode not in ("aligned", "first_diverge"):
            raise ValueError(
                f"unlik_mode must be 'aligned' or 'first_diverge', got {opts.unlik_mode!r}"
            )
        self.sp = Specials.from_tokenizer(tokenizer)
        self.opts = opts
        self.stats: Dict[str, int] = {}

    def _bump(self, k: str, n: int = 1) -> None:
        self.stats[k] = self.stats.get(k, 0) + n

    def from_text(self, ids: Sequence[int], text: str) -> Dict[str, np.ndarray]:
        msgs = None
        if START in text:
            msgs = align_messages(ids, parse_harmony(text), self.sp)
            if msgs is None:
                self._bump("misaligned")
            else:
                self._bump("harmony_docs")
        return annotate_messages(ids, msgs, self.sp, self.opts, self._call_stats())

    def _call_stats(self) -> Optional[Dict[str, int]]:
        # Classifying every tool result costs a json.loads + regexes per
        # result; skip it when only the role weights / metric are wanted.
        return self.stats if (self.opts.errhead or self.opts.unlik) else None

    def from_segments(
        self,
        ids: Sequence[int],
        text: str,
        segments: Sequence[dict],
        seg_tok_bounds: Sequence[Tuple[int, int]],
    ) -> Dict[str, np.ndarray]:
        """SFT records: roles come from the record's own segments, no re-parse.

        ``seg_tok_bounds[k]`` is segment k's token range (the SFT packer
        tokenizes segment by segment, so these are exact).
        """
        msgs: List[TokMsg] = []
        for seg, (ts, te) in zip(segments, seg_tok_bounds):
            if te <= ts:
                continue
            chunk = text[seg["start"] : seg["end"]]
            parsed = parse_harmony(chunk)
            if parsed:
                p = parsed[0]
                msgs.append(
                    TokMsg(
                        seg.get("role") or p.role, seg.get("channel"), p.content, p.marker, ts, te
                    )
                )
            else:
                msgs.append(TokMsg(seg.get("role", ""), seg.get("channel"), chunk, "", ts, te))
        self._bump("harmony_docs")
        return annotate_messages(ids, msgs, self.sp, self.opts, self._call_stats())

    def model_arrays(self, ann: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """The subset of ``ann`` that goes to the model, plus ``loss_weights``."""
        out = {"token_role": ann["token_role"]}
        if self.opts.errhead:
            out["errhead_labels"] = ann["errhead_labels"]
        if self.opts.unlik:
            out["unlik_targets"] = ann["unlik_targets"]
        if self.opts.emit_weights:
            out["loss_weights"] = np.where(
                ann["token_role"] == ROLE_TOOL, np.float32(self.opts.w_tool), np.float32(1.0)
            ).astype(np.float32)
        return out


# Tensor dtypes for the collate step.
ANNOTATION_KEYS = ("token_role", "errhead_labels", "unlik_targets", "loss_weights")
PAD_VALUES = {
    "token_role": ROLE_OTHER,
    "errhead_labels": -1,
    "unlik_targets": IGNORE_INDEX,
    "loss_weights": 0.0,
}
