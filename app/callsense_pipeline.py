"""
callsense_pipeline.py
=====================
NABRA AI - end-to-end call analysis pipeline (Steps 12.1 -> Final).

    audio -> Whisper (ASR) -> Pyannote (who spoke when) -> alignment
          -> DistilBERT (agent / customer) -> Qwen (summary + resolution)

Used by both the Colab notebook and the Streamlit dashboard (app.py).
Heavy libraries (torch, whisper, pyannote, transformers) are imported lazily,
so the pure-logic helpers can be imported and unit-tested without a GPU.
"""
from __future__ import annotations

import datetime as _dt
import difflib
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
PROJECT_DIR = Path(os.environ.get("CALLSENSE_PROJECT_DIR", "/content/CallSense_AI"))
ROLE_MODEL_DIR = PROJECT_DIR / "models/speaker_role/distilbert_final"
OUTPUT_DIR = PROJECT_DIR / "outputs/new_calls"

WHISPER_MODEL = "small.en"
DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
LLM_ID = "Qwen/Qwen2.5-3B-Instruct"

SUPPORTED_EXT = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma", ".mp4"}
VALID_STATUS = ("resolved", "unresolved", "unclear")


class PipelineError(Exception):
    """A failure that belongs to one named stage of the pipeline."""

    def __init__(self, stage: str, message: str, original: Exception | None = None):
        super().__init__(f"[{stage}] {message}")
        self.stage = stage
        self.message = message
        self.original = original


# ==========================================================================
# 12.1  Audio preparation and validation
# ==========================================================================
def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def prepare_audio(src, work_dir, min_seconds: float = 1.0, long_call_minutes: float = 20.0) -> dict:
    """Validate an uploaded file and convert it to 16 kHz mono WAV.

    Raises PipelineError("audio", ...) for missing / empty / corrupt / silent /
    unsupported files, so the caller never sees a raw ffmpeg traceback.
    """
    src = Path(src)
    work_dir = Path(work_dir)
    if not src.exists():
        raise PipelineError("audio", f"File not found: {src.name}")
    if src.stat().st_size == 0:
        raise PipelineError("audio", "The file is empty (0 bytes).")
    if src.suffix.lower() not in SUPPORTED_EXT:
        raise PipelineError(
            "audio",
            f"Unsupported file type '{src.suffix}'. Use one of: {', '.join(sorted(SUPPORTED_EXT))}",
        )
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise PipelineError("audio", "ffmpeg/ffprobe is not installed on this machine.")

    work_dir.mkdir(parents=True, exist_ok=True)
    out = work_dir / f"{src.stem}_16k.wav"
    proc = _run(["ffmpeg", "-y", "-v", "error", "-i", str(src),
                 "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(out)])
    if proc.returncode != 0 or not out.exists() or out.stat().st_size < 1000:
        tail = (proc.stderr or "").strip().splitlines()[-1:] or ["unknown decoder error"]
        raise PipelineError("audio", f"Could not decode the audio (corrupt file?): {tail[0][:160]}")

    probe = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                  "-of", "default=nw=1:nk=1", str(out)])
    try:
        duration = float(probe.stdout.strip())
    except ValueError:
        raise PipelineError("audio", "Could not read the audio duration.")
    if duration < min_seconds:
        raise PipelineError("audio", f"Audio is too short ({duration:.1f}s).")

    # Silence check (ffmpeg volumedetect prints max_volume in dBFS)
    vol = _run(["ffmpeg", "-v", "info", "-i", str(out), "-af", "volumedetect", "-f", "null", "-"])
    m = re.search(r"max_volume:\s*(-?[\d.]+|-inf)\s*dB", vol.stderr or "")
    if m and (m.group(1) == "-inf" or float(m.group(1)) < -60.0):
        raise PipelineError("audio", "The audio appears to be silent (no speech detected).")

    warnings = []
    if duration > long_call_minutes * 60:
        warnings.append(f"Long call ({duration / 60:.0f} min): processing will take longer.")
    return {
        "wav_path": out,
        "original_name": src.name,
        "duration_sec": round(duration, 2),
        "size_mb": round(src.stat().st_size / 1024 ** 2, 2),
        "warnings": warnings,
    }


def concat_audio(paths, out_path) -> Path:
    """Concatenate several audio files into one 16 kHz mono WAV (long-call stress test)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    list_file = out_path.with_suffix(".txt")
    list_file.write_text("".join(f"file '{Path(p).resolve()}'\n" for p in paths))
    proc = _run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(list_file),
                 "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(out_path)])
    list_file.unlink(missing_ok=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-300:])
    return out_path


# ==========================================================================
# 12.2  Whisper (ASR)
# ==========================================================================
def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(text).lower())).strip()


def clean_segments(df: pd.DataFrame, max_repeats: int = 3) -> pd.DataFrame:
    """Drop empty / invalid segments and Whisper repetition loops."""
    if df.empty:
        return df.reset_index(drop=True)
    df = df.copy()
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df = df[(df["text"] != "") & (df["end"] > df["start"])].reset_index(drop=True)
    if df.empty:
        return df
    keys = df["text"].map(_norm)
    run_id = (keys != keys.shift()).cumsum()
    run_pos = df.groupby(run_id).cumcount()
    run_len = df.groupby(run_id)["text"].transform("size")
    # a run longer than `max_repeats` is a hallucination loop -> keep only its first segment
    drop = (run_len > max_repeats) & (run_pos > 0)
    return df[~drop].reset_index(drop=True)


def load_whisper(name: str = WHISPER_MODEL, device: str | None = None):
    import torch
    import whisper
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return whisper.load_model(name, device=device)


def whisper_transcribe(model, wav_path, fp16: bool = False) -> pd.DataFrame:
    result = model.transcribe(
        str(wav_path), language="en", task="transcribe", fp16=fp16,
        condition_on_previous_text=False,  # avoids repetition loops on long calls
        verbose=False,
    )
    rows = [{"start": float(s["start"]), "end": float(s["end"]), "text": s["text"].strip()}
            for s in result["segments"]]
    return clean_segments(pd.DataFrame(rows, columns=["start", "end", "text"]))


# ==========================================================================
# 12.3  Pyannote (diarization)
# ==========================================================================
def load_diarizer(name: str = DIARIZATION_MODEL, token: str | None = None, device: str | None = None):
    import torch
    from pyannote.audio import Pipeline
    token = token or os.environ.get("HF_TOKEN")
    if not token:
        raise PipelineError("setup", "Hugging Face token is missing (set HF_TOKEN).")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    pipe = Pipeline.from_pretrained(name, token=token)
    pipe.to(torch.device(device))
    return pipe


def run_diarization(pipeline, wav_path, num_speakers: int | None = 2) -> pd.DataFrame:
    try:
        out = pipeline(str(wav_path), num_speakers=num_speakers) if num_speakers else pipeline(str(wav_path))
    except Exception:
        # e.g. a call where only one person speaks: let pyannote choose 1..2
        out = pipeline(str(wav_path), min_speakers=1, max_speakers=2)
    annotation = getattr(out, "speaker_diarization", out)
    rows = [{"start": float(t.start), "end": float(t.end), "speaker": s}
            for t, _, s in annotation.itertracks(yield_label=True)]
    return pd.DataFrame(rows, columns=["start", "end", "speaker"])


# ==========================================================================
# 12.4  Align transcript segments with speakers
# ==========================================================================
def align_speakers(whisper_df: pd.DataFrame, diar_df: pd.DataFrame, max_gap: float = 1.0) -> pd.DataFrame:
    """Give every Whisper segment the speaker with the largest time overlap.

    Segments that fall in a gap between diarization turns take the nearest turn
    if it is within `max_gap` seconds, otherwise they are labelled UNKNOWN.
    """
    out = whisper_df.copy().reset_index(drop=True)
    if out.empty or diar_df.empty:
        out["speaker"] = "UNKNOWN"
        return out
    ds = diar_df["start"].to_numpy(float)
    de = diar_df["end"].to_numpy(float)
    dspk = diar_df["speaker"].to_numpy()
    speakers = sorted(set(dspk))
    labels = []
    for s, e in zip(out["start"].to_numpy(float), out["end"].to_numpy(float)):
        overlap = np.clip(np.minimum(e, de) - np.maximum(s, ds), 0.0, None)
        if overlap.sum() > 0:
            totals = {spk: overlap[dspk == spk].sum() for spk in speakers}
            labels.append(max(totals, key=totals.get))
            continue
        mid = (s + e) / 2.0
        dist = np.where(ds > mid, ds - mid, np.where(de < mid, mid - de, 0.0))
        j = int(dist.argmin())
        labels.append(dspk[j] if dist[j] <= max_gap else "UNKNOWN")
    out["speaker"] = labels
    return out


# ==========================================================================
# 12.5  DistilBERT -> agent / customer
# ==========================================================================
def load_role_model(model_dir=ROLE_MODEL_DIR, device: str | None = None):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise PipelineError("setup", f"DistilBERT model not found: {model_dir}")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device).eval()
    return tok, model


def predict_agent_probs(tokenizer, model, texts, device: str, batch_size: int = 32) -> list[float]:
    import torch
    label2id = {str(k).lower(): int(v) for k, v in model.config.label2id.items()}
    if "agent" not in label2id:
        raise PipelineError("roles", f"'agent' label missing in model config: {label2id}")
    agent_id = label2id["agent"]
    probs: list[float] = []
    texts = [t if isinstance(t, str) and t.strip() else "." for t in texts]
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i:i + batch_size], padding=True, truncation=True,
                        max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            p = torch.softmax(model(**{k: v for k, v in enc.items() if k != "token_type_ids"}).logits, dim=-1)[:, agent_id]
        probs.extend(p.cpu().tolist())
    return probs


def assign_roles(df: pd.DataFrame, agent_probs, min_margin: float = 0.05):
    """Map diarization speakers to agent / customer from mean agent probability.

    Returns (df_with_roles, info). `info` explains how confident the mapping is.
    """
    df = df.copy().reset_index(drop=True)
    df["agent_probability"] = list(agent_probs)
    info = {"role_mapping": {}, "speaker_scores": {}, "margin": None, "warnings": []}
    known = df[df["speaker"] != "UNKNOWN"]
    if known.empty:
        df["role"] = "unknown"
        info["warnings"].append("No speaker could be assigned to any segment.")
        return df, info

    scores = known.groupby("speaker")["agent_probability"].mean().sort_values(ascending=False)
    talk = (known["end"] - known["start"]).groupby(known["speaker"]).sum()
    info["speaker_scores"] = {k: round(float(v), 4) for k, v in scores.items()}
    mapping: dict[str, str] = {}

    if len(scores) == 1:
        spk = scores.index[0]
        mapping[spk] = "agent" if scores.iloc[0] >= 0.5 else "customer"
        info["warnings"].append(
            "Only one speaker was detected; the role was guessed from the text alone.")
    else:
        main = list(talk.sort_values(ascending=False).index[:2])   # two people who talk most
        main_scores = scores.loc[main].sort_values(ascending=False)
        mapping[main_scores.index[0]] = "agent"
        mapping[main_scores.index[1]] = "customer"
        info["margin"] = round(float(main_scores.iloc[0] - main_scores.iloc[1]), 4)
        if info["margin"] < min_margin:
            info["warnings"].append(
                f"Low confidence in role assignment (score margin {info['margin']:.3f}). "
                "Both speakers sound alike to the classifier - review manually.")
        for spk in scores.index:                                    # extra speakers (rare)
            if spk not in mapping:
                closest = (main_scores - scores[spk]).abs().idxmin()
                mapping[spk] = mapping[closest]
        if len(scores) > 2:
            info["warnings"].append(
                f"{len(scores)} speakers were detected; extra speakers were merged into the closest role.")

    info["role_mapping"] = mapping
    df["role"] = df["speaker"].map(mapping).fillna("unknown")
    return df, info


def merge_turns(df: pd.DataFrame, max_gap: float = 2.0) -> pd.DataFrame:
    """Merge consecutive segments of the same role into conversation turns."""
    cols = ["start", "end", "role", "text"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    d = df.sort_values(["start", "end"]).reset_index(drop=True)
    turns: list[dict] = []
    for r in d.itertuples():
        text = str(r.text).strip()
        if not text:
            continue
        if turns and turns[-1]["role"] == r.role and r.start - turns[-1]["end"] <= max_gap:
            turns[-1]["end"] = max(turns[-1]["end"], float(r.end))
            turns[-1]["text"] += " " + text
        else:
            turns.append({"start": float(r.start), "end": float(r.end), "role": r.role, "text": text})
    return pd.DataFrame(turns, columns=cols)


def build_dialogue(turns: pd.DataFrame) -> str:
    return "\n".join(f"{str(r.role).upper()}: {r.text}" for r in turns.itertuples())


# ==========================================================================
# 12.6  LLM (Qwen) - summary and resolution detection
# ==========================================================================
SYSTEM_PROMPT = """
You are a customer service call analyst.

Analyze the provided conversation.

Return ONLY a valid JSON object with these keys:
- summary
- customer_problem
- agent_actions
- resolution_status
- resolution_reason
- evidence

Rules:
1. Summarize only information present in the transcript.
2. resolution_status must be one of:
   resolved, unresolved, unclear.
3. Mark resolved ONLY when there is explicit evidence
   that the issue was fixed or the requested action
   was successfully completed.
4. A promise to follow up is NOT proof of resolution.
5. If the conversation is incomplete or ambiguous,
   choose unclear.
6. Evidence must be a short exact quote from the
   transcript, or an empty string if none exists.
7. Do not invent information.
""".strip()

REDUCE_NOTE = """
8. The input may start with NOTES summarising the earlier parts of a long call,
   followed by the verbatim END of the call. Use both. The evidence quote may come
   from either (the notes contain exact quotes).
""".strip()

CHUNK_PROMPT = """
You are taking notes on one part of a customer service call.
Write at most 120 words covering: what the customer wants, what the agent did,
and any explicit statement that something was fixed, refused, or promised later.
Include up to 2 short exact quotes in double quotes for the most important
statements. Do not invent anything. Output plain text only.
""".strip()

JSON_KEYS = ["summary", "customer_problem", "agent_actions",
             "resolution_status", "resolution_reason", "evidence"]


def load_llm(model_id: str = LLM_ID):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    if not torch.cuda.is_available():
        raise PipelineError("setup", "A GPU is required for the LLM (Runtime > Change runtime type > T4 GPU).")
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(model_id)
    llm = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto",
                                               quantization_config=quant, torch_dtype=torch.float16)
    llm.eval()
    return tok, llm


def make_llm_fn(tokenizer, llm, max_prompt_tokens: int = 7000) -> Callable[[str, str, int], str]:
    """Return llm(system, user, max_new_tokens) -> text.

    Unlike truncating the prompt (which would cut off the assistant tag at the end
    and break the answer), an over-long prompt raises, and the caller chunks instead.
    """
    import torch

    def _llm(system: str, user: str, max_new_tokens: int = 500) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(llm.device)
        if inputs["input_ids"].shape[1] > max_prompt_tokens:
            raise ValueError(f"Prompt too long ({inputs['input_ids'].shape[1]} tokens).")
        with torch.inference_mode():
            out = llm.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                               pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    return _llm


def split_dialogue(text: str, count_tokens: Callable[[str], int], max_tokens: int) -> list[str]:
    """Split a transcript into chunks on line boundaries (never mid-sentence if avoidable)."""
    chunks, cur, cur_tok = [], [], 0
    pieces: list[str] = []
    for line in text.splitlines():
        if count_tokens(line) <= max_tokens:
            pieces.append(line)
        else:                                   # a single enormous turn: split by words
            words, buf = line.split(), []
            for w in words:
                buf.append(w)
                if count_tokens(" ".join(buf)) >= max_tokens:
                    pieces.append(" ".join(buf))
                    buf = []
            if buf:
                pieces.append(" ".join(buf))
    for p in pieces:
        t = count_tokens(p) + 1
        if cur and cur_tok + t > max_tokens:
            chunks.append("\n".join(cur))
            cur, cur_tok = [], 0
        cur.append(p)
        cur_tok += t
    if cur:
        chunks.append("\n".join(cur))
    if len(chunks) > 1 and count_tokens(chunks[-1]) < max_tokens * 0.25:   # avoid a tiny last chunk
        chunks[-2] += "\n" + chunks.pop()
    return chunks


def _balanced_json(text: str) -> Optional[str]:
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return text[start:]          # truncated output: caller tries to close it


def extract_json(text: str) -> Optional[dict]:
    """Parse the first JSON object in an LLM reply; tolerate fences, trailing commas, truncation."""
    if not text:
        return None
    cand = _balanced_json(re.sub(r"```(?:json)?", "", text))
    if not cand:
        return None
    cand = re.sub(r",\s*([}\]])", r"\1", cand)
    for suffix in ("", '"}', "}", '"]}', ']}', '"]}'):
        try:
            obj = json.loads(cand + suffix)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def normalize_result(d: dict) -> dict:
    out: dict = {}
    for k in JSON_KEYS:
        v = d.get(k, "")
        if k == "agent_actions":
            if isinstance(v, str):
                v = [v] if v.strip() else []
            elif isinstance(v, list):
                v = [str(x).strip() for x in v if str(x).strip()]
            else:
                v = []
            out[k] = v
        else:
            out[k] = "" if v is None else str(v).strip()
    status = out["resolution_status"].lower().strip()
    out["resolution_status"] = status if status in VALID_STATUS else "unclear"
    return out


def check_evidence(evidence: str, transcript: str) -> str:
    """Return 'exact', 'fuzzy', 'empty' or 'none' - is the quote really in the transcript?"""
    ev, tr = _norm(evidence), _norm(transcript)
    if not ev:
        return "empty"
    if ev in tr:
        return "exact"
    ev_w, tr_w = ev.split(), tr.split()
    if len(ev_w) >= 3:
        # fuzzy: >=80% of the quote's words appear, in order, inside one short window of the transcript
        ev_set, width = set(ev_w), int(len(ev_w) * 1.5) + 1
        for i, word in enumerate(tr_w):
            if word not in ev_set:
                continue
            blocks = difflib.SequenceMatcher(None, ev_w, tr_w[i:i + width], autojunk=False).get_matching_blocks()
            if sum(b.size for b in blocks) >= 0.8 * len(ev_w):
                return "fuzzy"
    return "none"


def analyze_transcript(
    transcript: str,
    llm: Callable[[str, str, int], str],
    count_tokens: Callable[[str], int],
    single_pass_tokens: int = 4500,
    chunk_tokens: int = 2500,
    strict_evidence: bool = True,
    max_retries: int = 1,
):
    """Summarise a call and decide resolved / unresolved / unclear.

    Short calls: one LLM pass. Long calls: notes per chunk (map) + final decision
    from the notes and the verbatim end of the call (reduce).
    `strict_evidence`: a 'resolved' verdict without a verifiable quote is downgraded
    to 'unclear' (rule 3 of the prompt, enforced in code).
    """
    meta = {"mode": "single", "chunks": 1, "tokens": 0, "parse_ok": True,
            "retries": 0, "evidence_match": "empty", "downgraded": False, "raw_output": ""}
    empty = {"summary": "", "customer_problem": "", "agent_actions": [],
             "resolution_status": "unclear", "resolution_reason": "", "evidence": ""}
    if not transcript.strip():
        empty["resolution_reason"] = "No speech was transcribed."
        meta["parse_ok"] = False
        return empty, meta

    n_tokens = count_tokens(transcript)
    meta["tokens"] = n_tokens
    system, user = SYSTEM_PROMPT, transcript

    if n_tokens > single_pass_tokens:
        chunks = split_dialogue(transcript, count_tokens, chunk_tokens)
        meta.update(mode="map_reduce", chunks=len(chunks))
        notes = []
        for i, ch in enumerate(chunks[:-1], 1):
            notes.append(f"[Part {i}/{len(chunks)}] {llm(CHUNK_PROMPT, ch, 220).strip()}")
        system = SYSTEM_PROMPT + "\n" + REDUCE_NOTE
        user = ("NOTES ON EARLIER PARTS:\n" + "\n".join(notes) +
                f"\n\nEND OF THE CALL (verbatim, part {len(chunks)}/{len(chunks)}):\n" + chunks[-1])

    parsed, raw = None, ""
    for attempt in range(max_retries + 1):
        prompt_user = user if attempt == 0 else (
            user + "\n\n(Your previous reply was not valid JSON. Reply with ONLY the JSON object.)")
        raw = llm(system, prompt_user, 500)
        parsed = extract_json(raw)
        meta["retries"] = attempt
        if parsed is not None:
            break

    meta["raw_output"] = raw
    if parsed is None:
        meta["parse_ok"] = False
        empty["resolution_reason"] = "Model output could not be parsed."
        return empty, meta

    result = normalize_result(parsed)
    meta["evidence_match"] = check_evidence(result["evidence"], transcript)
    if strict_evidence and result["resolution_status"] == "resolved" and meta["evidence_match"] in ("none", "empty"):
        result["resolution_status"] = "unclear"
        result["resolution_reason"] = (result["resolution_reason"] +
                                       " [Downgraded from 'resolved': no verifiable evidence quote.]").strip()
        meta["downgraded"] = True
    return result, meta


# ==========================================================================
# Model container + end-to-end call processing (12.7)
# ==========================================================================
@dataclass
class PipelineModels:
    """Holds the four models. Tests can replace it with any object exposing the same methods."""
    device: str
    whisper_model: object = None
    diarizer: object = None
    role_tokenizer: object = None
    role_model: object = None
    llm_tokenizer: object = None
    llm_model: object = None

    @property
    def has_llm(self) -> bool:
        return self.llm_model is not None

    def transcribe(self, wav_path) -> pd.DataFrame:
        return whisper_transcribe(self.whisper_model, wav_path, fp16=(self.device == "cuda"))

    def diarize(self, wav_path, num_speakers: int | None = 2) -> pd.DataFrame:
        return run_diarization(self.diarizer, wav_path, num_speakers)

    def agent_probs(self, texts) -> list[float]:
        return predict_agent_probs(self.role_tokenizer, self.role_model, texts, self.device)

    def llm(self, system: str, user: str, max_new_tokens: int = 500) -> str:
        if not hasattr(self, "_llm_fn"):
            self._llm_fn = make_llm_fn(self.llm_tokenizer, self.llm_model)
        return self._llm_fn(system, user, max_new_tokens)

    def count_tokens(self, text: str) -> int:
        if self.llm_tokenizer is not None:
            return len(self.llm_tokenizer.encode(text))
        return int(len(text.split()) * 1.4)


def build_models(project_dir=PROJECT_DIR, hf_token=None, load_llm_model: bool = True,
                 device: str | None = None) -> PipelineModels:
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    m = PipelineModels(device=device)
    m.whisper_model = load_whisper(device=device)
    m.diarizer = load_diarizer(token=hf_token, device=device)
    m.role_tokenizer, m.role_model = load_role_model(Path(project_dir) / "models/speaker_role/distilbert_final", device)
    if load_llm_model:
        m.llm_tokenizer, m.llm_model = load_llm()
    return m


def _free_gpu():
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def process_call(audio_path, models, out_dir=None, num_speakers: int | None = 2,
                 progress: Callable[[str, str], None] | None = None, save: bool = True,
                 work_dir=None) -> dict:
    """Run the full pipeline on one audio file. Never raises: failures are returned in the result.

    result["status"]: "ok" | "partial" (transcript fine, LLM failed) | "failed".
    """
    audio_path = Path(audio_path)
    out_dir = Path(out_dir or OUTPUT_DIR)
    work_dir = Path(work_dir or (PROJECT_DIR / "data/uploads_16k"))
    t_start = time.perf_counter()
    res: dict = {"call_id": audio_path.stem, "file": audio_path.name, "status": "ok", "error": None,
                 "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
                 "warnings": [], "timings": {}, "audio": {}, "roles": {}, "stats": {},
                 "analysis": None, "analysis_meta": None, "domain": None,
                 "segments": pd.DataFrame(), "turns": pd.DataFrame(), "dialogue": ""}

    def step(stage: str, message: str, fn):
        if progress:
            progress(stage, message)
        t0 = time.perf_counter()
        try:
            value = fn()
        except PipelineError:
            raise
        except Exception as exc:                                   # noqa: BLE001
            text = f"{type(exc).__name__}: {exc}"
            if "out of memory" in text.lower():
                _free_gpu()
                text += "  (GPU out of memory - try a shorter file or restart the runtime)"
            raise PipelineError(stage, text[:300], exc)
        res["timings"][stage] = round(time.perf_counter() - t0, 2)
        return value

    try:
        info = step("audio", "Validating and converting audio...", lambda: prepare_audio(audio_path, work_dir))
        res["audio"] = {k: (str(v) if isinstance(v, Path) else v) for k, v in info.items() if k != "warnings"}
        res["warnings"] += info["warnings"]
        wav = info["wav_path"]

        whisper_df = step("asr", "Transcribing speech (Whisper)...", lambda: models.transcribe(wav))
        if whisper_df.empty:
            raise PipelineError("asr", "No speech could be transcribed from this audio.")

        diar_df = step("diarization", "Detecting who is speaking (Pyannote)...",
                       lambda: models.diarize(wav, num_speakers))
        aligned = step("alignment", "Aligning text with speakers...", lambda: align_speakers(whisper_df, diar_df))
        n_unknown = int((aligned["speaker"] == "UNKNOWN").sum())
        if n_unknown:
            res["warnings"].append(f"{n_unknown} segment(s) could not be matched to a speaker.")

        def _roles():
            probs = models.agent_probs(aligned["text"].tolist())
            return assign_roles(aligned, probs)

        roles_df, role_info = step("roles", "Classifying agent / customer (DistilBERT)...", _roles)
        res["roles"] = role_info
        res["warnings"] += role_info["warnings"]
        res["segments"] = roles_df
        turns = merge_turns(roles_df)
        res["turns"] = turns
        res["dialogue"] = build_dialogue(turns)

        dur = (turns["end"] - turns["start"]).groupby(turns["role"]).sum() if not turns.empty else {}
        a, c = float(dur.get("agent", 0.0)), float(dur.get("customer", 0.0))
        res["stats"] = {
            "turns": int(len(turns)), "segments": int(len(roles_df)),
            "agent_talk_sec": round(a, 1), "customer_talk_sec": round(c, 1),
            "agent_talk_pct": round(100 * a / (a + c), 1) if (a + c) else None,
            "words": int(roles_df["text"].str.split().str.len().sum()),
        }

        try:
            res["domain"] = step("domain", "Detecting the call domain...", lambda: predict_domain(res["dialogue"], models))
        except PipelineError as exc:
            res["domain"] = None
            res["warnings"].append(f"Domain detection failed ({exc.message}).")

        if not models.has_llm:
            res["warnings"].append("LLM not loaded: summary and resolution were skipped.")
            res["status"] = "partial"
        else:
            try:
                analysis, meta = step(
                    "llm", "Summarising and detecting resolution (Qwen)...",
                    lambda: analyze_transcript(res["dialogue"], models.llm, models.count_tokens))
                res["analysis"], res["analysis_meta"] = analysis, meta
                if not meta["parse_ok"]:
                    res["warnings"].append("The LLM answer could not be parsed - resolution is 'unclear'.")
                if meta["downgraded"]:
                    res["warnings"].append("Resolution downgraded to 'unclear': the quoted evidence was not found in the transcript.")
                if meta["mode"] == "map_reduce":
                    res["warnings"].append(f"Long call analysed in {meta['chunks']} parts.")
            except PipelineError as exc:
                res["status"] = "partial"
                res["error"] = {"stage": exc.stage, "message": exc.message}
                res["warnings"].append(f"Analysis failed ({exc.message}). The transcript is still available.")
    except PipelineError as exc:
        res["status"] = "failed"
        res["error"] = {"stage": exc.stage, "message": exc.message}

    res["timings"]["total"] = round(time.perf_counter() - t_start, 2)
    if save and res["status"] != "failed":
        try:
            save_result(res, out_dir)
        except Exception as exc:                                    # noqa: BLE001
            res["warnings"].append(f"Could not save results: {exc}")
    return res


# ==========================================================================
# Saving / loading / summarising results
# ==========================================================================
def result_to_json(res: dict) -> dict:
    d = {k: v for k, v in res.items() if k not in ("segments", "turns")}
    d["turns"] = res["turns"].to_dict("records") if isinstance(res["turns"], pd.DataFrame) else res["turns"]
    return d


def save_result(res: dict, out_dir=None) -> dict:
    out_dir = Path(out_dir or OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    cid = res["call_id"]
    paths = {"json": out_dir / f"{cid}_analysis.json",
             "segments": out_dir / f"{cid}_segments.csv",
             "dialogue": out_dir / f"{cid}_dialogue.txt"}
    paths["json"].write_text(json.dumps(result_to_json(res), indent=2, ensure_ascii=False, default=str),
                             encoding="utf-8")
    res["segments"].to_csv(paths["segments"], index=False)
    paths["dialogue"].write_text(res["dialogue"], encoding="utf-8")
    return paths


def load_saved_results(out_dir=None) -> list[dict]:
    out_dir = Path(out_dir or OUTPUT_DIR)
    items = []
    for p in sorted(out_dir.glob("*_analysis.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            items.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:                                           # noqa: BLE001
            continue
    return items


def summarize_results(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        a = r.get("analysis") or {}
        rows.append({
            "call": r["file"], "status": r["status"],
            "failed_stage": (r.get("error") or {}).get("stage", ""),
            "duration_s": (r.get("audio") or {}).get("duration_sec"),
            "processing_s": r["timings"].get("total"),
            "domain": (r.get("domain") or {}).get("label", ""),
            "resolution": a.get("resolution_status", ""),
            "llm_mode": (r.get("analysis_meta") or {}).get("mode", ""),
            "warnings": len(r["warnings"]),
            "error": (r.get("error") or {}).get("message", "")[:90],
        })
    return pd.DataFrame(rows)


# ==========================================================================
# Domain detection + project results (used by the dashboard)
# ==========================================================================
DOMAIN_DIR = PROJECT_DIR / "models/domain_classifier"
RESULTS_DIR = PROJECT_DIR / "evaluation/final"
DEFAULT_DOMAINS = ["agriculture", "aviation", "banking", "deliveryservice", "energy", "entertainment",
                   "finance", "food", "health", "hospitality", "insurance", "realestate", "retail",
                   "technology", "telecom", "travel"]
DOMAIN_NAMES = {"deliveryservice": "Delivery Service", "realestate": "Real Estate"}


def domain_name(label) -> str:
    label = str(label or "").lower()
    return DOMAIN_NAMES.get(label, label.title()) if label else "Unknown"


def load_domains() -> list[str]:
    p = DOMAIN_DIR / "domains.json"
    try:
        return [str(d).lower() for d in json.loads(p.read_text())] if p.exists() else DEFAULT_DOMAINS
    except Exception:                                                # noqa: BLE001
        return DEFAULT_DOMAINS


def classify_domain_llm(transcript: str, llm, domains=None) -> str:
    """Zero-shot: ask Qwen to pick the industry from a fixed list ('other' if none fits)."""
    domains = list(domains or load_domains())
    system = ("You classify customer service calls by industry. Reply with exactly one word from this list: "
              + ", ".join(domains) + ", other. No explanation.")
    raw = re.sub(r"[^a-z]", "", llm(system, transcript[:6000], 8).lower())
    for d in sorted(domains, key=len, reverse=True):
        if d in raw:
            return d
    return "other"


def classify_domain_tfidf(transcript: str) -> Optional[dict]:
    path = DOMAIN_DIR / "tfidf_logistic_regression.joblib"
    if not path.exists():
        return None
    import joblib
    pipe = joblib.load(path)
    proba = pipe.predict_proba([transcript])[0]
    j = int(np.argmax(proba))
    return {"label": str(pipe.classes_[j]), "confidence": round(float(proba[j]), 3), "method": "tfidf"}


def predict_domain(transcript: str, models) -> Optional[dict]:
    """Qwen zero-shot when the LLM is loaded, otherwise the TF-IDF baseline (weak: see project results)."""
    if not transcript.strip():
        return None
    if getattr(models, "has_llm", False):
        return {"label": classify_domain_llm(transcript, models.llm), "confidence": None, "method": "llm"}
    return classify_domain_tfidf(transcript)


DEFAULT_RESULTS = {   # numbers from the project's final evaluation run; evaluation/final/project_metrics.json overrides them
    "asr": {"model": "Whisper small.en", "wer": 0.1363, "cer": 0.0933, "calls": 7},
    "roles": {"baseline_acc": 0.8201, "baseline_f1": 0.8085, "distilbert_acc": 0.8681, "distilbert_f1": 0.8591,
              "e2e_acc": 0.9836, "e2e_f1": 0.9826, "e2e_calls": 7},
    "domain": {"tfidf_test_acc": 0.1429, "tfidf_test_f1": 0.0741, "llm_acc": None, "n": None},
    "resolution": {"n": 14, "accuracy": 0.3571, "macro_f1": 0.1852, "resolved_precision": 1.0,
                   "resolved_recall": 0.3846, "false_resolved_rate": 0.0, "pipe_n": 7, "pipe_accuracy": 0.5714,
                   "human_resolved": 13, "human_unresolved": 1, "human_unclear": 0, "json_parse": 1.0},
    "summary": {"mean_score": 4.38, "n": 13},
}


def load_results() -> dict:
    out = json.loads(json.dumps(DEFAULT_RESULTS))
    p = RESULTS_DIR / "project_metrics.json"
    try:
        if p.exists():
            for k, v in json.loads(p.read_text()).items():
                out.setdefault(k, {}).update(v)
    except Exception:                                                # noqa: BLE001
        pass
    return out


def load_examples() -> Optional[dict]:
    p = RESULTS_DIR / "examples.json"
    try:
        return json.loads(p.read_text()) if p.exists() else None
    except Exception:                                                # noqa: BLE001
        return None


# ==========================================================================
# Final evaluation helpers (LLM resolution accuracy)
# ==========================================================================
def reference_transcripts(split_df: pd.DataFrame) -> dict[str, str]:
    """Ground-truth dialogue per call, from the labelled split CSV."""
    out = {}
    df = split_df.copy()
    df["role"] = df["role"].astype(str).str.lower().str.strip()
    for fname, g in df.groupby("file_name"):
        turns = merge_turns(g[["start", "end", "role", "text"]].dropna(subset=["text"]))
        out[Path(str(fname)).stem] = build_dialogue(turns)
    return out


def pipeline_transcripts(pred_dir) -> dict[str, str]:
    """Dialogue per call from the Step 9.14 pipeline outputs (Whisper + Pyannote + DistilBERT)."""
    out = {}
    for p in sorted(Path(pred_dir).glob("*_speaker_roles.csv")):
        df = pd.read_csv(p)
        if {"start", "end", "role", "text"} <= set(df.columns):
            out[p.name.replace("_speaker_roles.csv", "")] = build_dialogue(
                merge_turns(df[["start", "end", "role", "text"]].dropna(subset=["text"])))
    return out


def run_llm_review(ref: dict, pipe: dict, split_of: dict, models, progress=print) -> pd.DataFrame:
    """Run the LLM on every reference transcript (and on the pipeline transcript when available)."""
    rows = []
    for i, (cid, transcript) in enumerate(ref.items(), 1):
        if progress:
            progress(f"[{i}/{len(ref)}] {cid}")
        row = {"call_id": cid, "split": split_of.get(cid, "")}
        sources = [("ref", transcript)] + ([("pipe", pipe[cid])] if cid in pipe else [])
        for tag, text in sources:
            res, meta = analyze_transcript(text, models.llm, models.count_tokens)
            row.update({f"llm_status_{tag}": res["resolution_status"],
                        f"summary_{tag}": res["summary"],
                        f"reason_{tag}": res["resolution_reason"],
                        f"evidence_{tag}": res["evidence"],
                        f"evidence_match_{tag}": meta["evidence_match"],
                        f"parse_ok_{tag}": meta["parse_ok"],
                        f"mode_{tag}": meta["mode"]})
        row["human_status"] = ""
        row["summary_score"] = np.nan
        row["human_notes"] = ""
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_resolution(df: pd.DataFrame, human_col: str, pred_col: str) -> dict:
    """Accuracy / macro-F1 / confusion matrix of the LLM against human labels.

    `false_resolved_rate` = share of calls the LLM called 'resolved' that humans did not.
    It is the most important number: a wrongly 'resolved' call hides a problem.
    """
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    d = df[df[human_col].isin(VALID_STATUS) & df[pred_col].isin(VALID_STATUS)]
    if d.empty:
        raise ValueError("No rows with both a human label and an LLM prediction.")
    y, p = d[human_col], d[pred_col]
    labels = list(VALID_STATUS)
    pred_resolved = (p == "resolved").sum()
    true_resolved = (y == "resolved").sum()
    both = ((p == "resolved") & (y == "resolved")).sum()
    return {
        "n": int(len(d)),
        "accuracy": round(float(accuracy_score(y, p)), 4),
        "macro_f1": round(float(f1_score(y, p, labels=labels, average="macro", zero_division=0)), 4),
        "resolved_precision": round(float(both / pred_resolved), 4) if pred_resolved else None,
        "resolved_recall": round(float(both / true_resolved), 4) if true_resolved else None,
        "false_resolved_rate": round(float(1 - both / pred_resolved), 4) if pred_resolved else None,
        "confusion": pd.DataFrame(confusion_matrix(y, p, labels=labels),
                                  index=[f"human_{l}" for l in labels],
                                  columns=[f"llm_{l}" for l in labels]),
        "report": classification_report(y, p, labels=labels, zero_division=0),
    }