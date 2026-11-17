#!/usr/bin/env python
"""
batch_infer.py — batch voice conversion for EZ-VC.

The repo ships inference as a notebook (src/f5_tts/infer/infer.ipynb), which
reloads XEUS, the k-means, BigVGAN and the DiT every time you want to convert a
file. This script loads them ONCE and then converts a whole directory.

Every step is taken from that notebook unchanged — same unit extraction, same
chunking, same infer_process arguments, same 16 kHz output. If the notebook and
this script disagree on a file, that is a bug here, not a design choice.

    # one target voice, a directory of sources
    python batch_infer.py --ref target.wav --src_dir sources/ --out converted/

    # many pairs, from a manifest
    python batch_infer.py --manifest pairs.tsv --out converted/

    # the same on several GPUs: one worker per card, each on its share of the
    # manifest, one progress line here (done/total, rate, time left); Ctrl-C
    # stops every worker, and the same line resumes (files made are kept)
    python batch_infer.py --manifest pairs.tsv --out converted/ --cards 0,1,3

manifest.tsv — one pair per line, tab separated, optional third column for the
output name:

    source.wav<TAB>reference.wav
    source.wav<TAB>reference.wav<TAB>custom_name.wav
    source.wav<TAB>reference.wav<TAB>subfolder/custom_name.wav   (subfolders are created)

A NOTE ON "ref_text" AND "src_text"
  These are NOT transcripts. EZ-VC reuses F5-TTS's text arguments to carry
  DISCRETE UNITS extracted from audio — XEUS layer 14, quantised by a 500-cluster
  k-means, mapped into private-use unicode characters and de-duplicated. Both the
  reference and the source go through the same extract_units() call. You never
  supply text to this model, and the commented-out string in the notebook is an
  example of what the units look like, not something you should edit.
"""

import argparse
import os
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")   # hub downloads over plain HTTP (the xet transfer can stall)
import re
import subprocess
import sys
import time
import traceback

import numpy as np
import soundfile as sf
from cached_path import cached_path
from hydra.utils import get_class
from omegaconf import OmegaConf

from f5_tts.infer.utils_infer import (
    cfg_strength,
    cross_fade_duration,
    fix_duration,
    infer_process,
    load_model,
    load_vocoder,
    sway_sampling_coef,
    target_rms,
)
from f5_tts.infer.utils_xeus import ApplyKmeans, extract_units, load_xeus_model

AUDIO_EXT = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus")
OUT_SR = 16000                   # the notebook writes at 16 kHz

# defaults as invoked in the notebook's final cell, not the library defaults
NB_NFE = 12
NB_SPEED = 1.0


def default_config_path():
    """configs/ lives next to infer/ inside the installed f5_tts package."""
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, "..", "configs", "F5TTS_Base_EZ-VC.yaml"),
                 os.path.join(here, "configs", "F5TTS_Base_EZ-VC.yaml")):
        if os.path.exists(cand):
            return os.path.normpath(cand)
    try:                                   # installed as a package
        import f5_tts
        p = os.path.join(os.path.dirname(f5_tts.__file__), "configs",
                         "F5TTS_Base_EZ-VC.yaml")
        if os.path.exists(p):
            return p
    except Exception:
        pass
    return None


def build_jobs(a):
    """Return [(src_path, ref_path, out_name), ...]."""
    jobs = []
    if a.manifest:
        with open(a.manifest, encoding="utf-8") as fh:
            for ln, line in enumerate(fh, 1):
                line = line.rstrip("\n")
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                parts = line.split("\t") if "\t" in line else line.split(",")
                parts = [p.strip() for p in parts if p.strip()]
                if len(parts) < 2:
                    print(f"  manifest line {ln}: need source and reference, skipped",
                          file=sys.stderr)
                    continue
                src, ref = parts[0], parts[1]
                name = parts[2] if len(parts) > 2 else os.path.basename(src)
                jobs.append((src, ref, name))
    else:
        srcs = sorted(os.path.join(a.src_dir, f) for f in os.listdir(a.src_dir)
                      if f.lower().endswith(AUDIO_EXT))
        jobs = [(s, a.ref, os.path.basename(s)) for s in srcs]
    return jobs


def run_workers(a, jobs):
    """--cards: one copy of this script per card (CUDA_VISIBLE_DEVICES=<card>, --shard i/n = every n-th manifest
    line from the i-th), its output in <out>/ezvc_card<card>.log; here one progress line counted from the files that
    appear. Returns the exit code: 0 when every job's file exists and every worker ended well."""
    cards = [c for c in a.cards.split(",") if c]
    n = len(cards)
    base = [sys.executable, os.path.abspath(__file__), "--out", a.out, "--nfe", str(a.nfe), "--speed", str(a.speed),
            "--device", a.device, "--vocoder_name", a.vocoder_name, "--ckpt_file", a.ckpt_file, "--vocab_file", a.vocab_file]
    base += ["--manifest", a.manifest] if a.manifest else ["--src_dir", a.src_dir, "--ref", a.ref]
    if a.config:
        base += ["--config", a.config]
    if a.overwrite:
        base.append("--overwrite")
    if a.limit:
        base += ["--limit", str(a.limit)]
    os.makedirs(a.out, exist_ok=True)
    procs, logs = [], []
    for i, c in enumerate(cards):
        log = os.path.join(a.out, f"ezvc_card{c}.log")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": c}
        env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        with open(log, "w") as fh:
            procs.append(subprocess.Popen(base + ["--shard", f"{i}/{n}"], stdout=fh, stderr=subprocess.STDOUT, env=env))
        logs.append(log)
    outs = [os.path.join(a.out, name) for _, _, name in jobs]
    total = len(outs)
    print(f"{total} file(s) on {n} card(s) ({a.cards}); each worker's log: {a.out}/ezvc_card<N>.log; "
          f"the first minute loads the models; Ctrl-C stops every worker", flush=True)
    t0 = time.time()
    try:
        while True:
            done = sum(os.path.exists(p) for p in outs)
            el = time.time() - t0
            rate = done / el if done else 0.0
            left = f"{(total - done) / rate / 60:.0f} min left" if rate else "..."
            print(f"\r  {done}/{total}  {el / 60:.0f} min  {rate * 60:.0f}/min  {left}      ", end="", flush=True)
            if all(p.poll() is not None for p in procs):
                break
            time.sleep(10)
    except KeyboardInterrupt:
        for p in procs:
            p.terminate()
        print("\nstopped; the files made are kept, the same line resumes")
        return 130
    print()
    done = sum(os.path.exists(p) for p in outs)
    print(f"{done} of {total} converted  ->  {a.out}")
    bad = 0
    for c, p, log in zip(cards, procs, logs):
        if p.returncode:
            bad += 1
            with open(log, errors="replace") as fh:
                tail = fh.read().splitlines()[-8:]
            print(f"card {c} ended with code {p.returncode}; the end of its log {log}:\n  " + "\n  ".join(tail))
    return 1 if bad or done < total else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", help="directory of source clips (the speech to convert)")
    ap.add_argument("--ref", help="reference clip (the target voice)")
    ap.add_argument("--manifest", help="TSV: source<TAB>reference[<TAB>output_name]")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--vocoder_name", default="bigvgan")
    ap.add_argument("--config", default=None, help="default: configs/F5TTS_Base_EZ-VC.yaml")
    ap.add_argument("--ckpt_file", default="hf://SPRINGLab/EZ-VC/model_2700000.safetensors")
    ap.add_argument("--vocab_file", default="hf://SPRINGLab/EZ-VC/vocab.txt")
    ap.add_argument("--nfe", type=int, default=NB_NFE, help=f"denoising steps (notebook uses {NB_NFE})")
    ap.add_argument("--speed", type=float, default=NB_SPEED)
    ap.add_argument("--overwrite", action="store_true", help="redo files already in --out")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry_run", action="store_true", help="list the jobs and exit")
    ap.add_argument("--cards", help="several GPUs, e.g. 0,1,3: one worker per card on its share of the jobs, one progress line")
    ap.add_argument("--shard", help="internal (a --cards worker): i/n, every n-th job from the i-th")
    a = ap.parse_args()

    if not a.manifest and not (a.src_dir and a.ref):
        ap.error("give either --manifest, or both --src_dir and --ref")

    jobs = build_jobs(a)
    if a.limit:
        jobs = jobs[: a.limit]
    if a.shard:                                        # a --cards worker: its share, cut before the skip below so the
        i, n = map(int, a.shard.split("/"))            # shares never shift as files appear
        jobs = jobs[i::n]
    if not jobs and not a.shard:
        sys.exit("nothing to do — no source clips found")

    os.makedirs(a.out, exist_ok=True)
    if not a.overwrite:
        before = len(jobs)
        jobs = [j for j in jobs if not os.path.exists(os.path.join(a.out, j[2]))]
        if before != len(jobs):
            print(f"skipping {before - len(jobs)} already in {a.out} (use --overwrite to redo)")

    print(f"{len(jobs)} file(s) to convert, {len({r for _, r, _ in jobs})} distinct reference voice(s)")
    if a.dry_run:
        for s, r, n in jobs[:20]:
            print(f"  {os.path.basename(s)}  +  {os.path.basename(r)}  ->  {n}")
        if len(jobs) > 20:
            print(f"  ... and {len(jobs)-20} more")
        return
    if a.cards:
        sys.exit(run_workers(a, jobs))
    if not jobs:                                       # a worker whose share is already done
        return

    cfg = a.config or default_config_path()
    if not cfg or not os.path.exists(cfg):
        sys.exit("could not find F5TTS_Base_EZ-VC.yaml — pass --config explicitly")

    # ---- load everything ONCE (this is the whole point of the script) -------
    t0 = time.time()
    print(f"loading XEUS + k-means on {a.device} ...", flush=True)
    xeus_model = load_xeus_model(a.device).eval()
    apply_kmeans = ApplyKmeans(a.device)

    print(f"loading vocoder ({a.vocoder_name}) ...", flush=True)
    vocoder = load_vocoder(vocoder_name=a.vocoder_name, device=a.device)

    print("loading EZ-VC checkpoint ...", flush=True)
    ckpt = str(cached_path(a.ckpt_file)) if a.ckpt_file.startswith("hf://") else a.ckpt_file
    vocab = str(cached_path(a.vocab_file)) if a.vocab_file.startswith("hf://") else a.vocab_file
    model_cfg = OmegaConf.load(cfg)
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    ema_model = load_model(model_cls, model_cfg.model.arch, ckpt,
                           mel_spec_type=a.vocoder_name, vocab_file=vocab, device=a.device)
    print(f"models ready in {time.time()-t0:.0f}s\n")

    # ---- the notebook's infer(), unchanged ----------------------------------
    def infer(ref_audio, ref_text, gen_text, spd, nfe):
        segments = []
        for chunk in re.split(r"(?=\[\w+\])", gen_text):
            gen_text_ = re.sub(r"\[(\w+)\]", "", chunk).strip()
            audio_segment, _, _ = infer_process(
                ref_audio, ref_text, gen_text_, ema_model, vocoder,
                mel_spec_type=a.vocoder_name, target_rms=target_rms,
                cross_fade_duration=cross_fade_duration, nfe_step=nfe,
                cfg_strength=cfg_strength, sway_sampling_coef=sway_sampling_coef,
                speed=spd, fix_duration=fix_duration, device=a.device,
            )
            segments.append(audio_segment)
        return np.concatenate(segments) if segments else None

    # reference units are reused across every source that shares a reference
    ref_cache = {}
    done = failed = 0
    t0 = time.time()
    for i, (src, ref, name) in enumerate(jobs, 1):
        try:
            if not os.path.exists(src):
                raise FileNotFoundError(src)
            if ref not in ref_cache:
                if not os.path.exists(ref):
                    raise FileNotFoundError(ref)
                ref_cache[ref] = extract_units(ref, xeus_model, apply_kmeans, a.device)
            ref_text = ref_cache[ref]
            src_text = extract_units(src, xeus_model, apply_kmeans, a.device)

            gen = infer(ref, ref_text, src_text, spd=a.speed, nfe=a.nfe)
            if gen is None:
                raise RuntimeError("no audio segments generated")
            out_path = os.path.join(a.out, name)
            os.makedirs(os.path.dirname(out_path) or a.out, exist_ok=True)   # names may carry subfolders (speaker/clip.wav)
            sf.write(out_path, gen, OUT_SR)
            done += 1
            if i % 10 == 0 or i == len(jobs):
                el = time.time() - t0
                print(f"  {i}/{len(jobs)}  ok={done} failed={failed}  "
                      f"{el/max(i,1):.1f}s/file  eta {(len(jobs)-i)*el/max(i,1)/60:.1f} min",
                      flush=True)
        except Exception as e:
            failed += 1
            print(f"  FAILED {os.path.basename(src)}: {type(e).__name__}: {e}", file=sys.stderr)
            if os.environ.get("EZVC_TRACE"):
                traceback.print_exc()

    print(f"\n{done} converted, {failed} failed  ->  {a.out}")
    if failed:
        print("set EZVC_TRACE=1 for full tracebacks")
        sys.exit(1)


if __name__ == "__main__":
    main()
