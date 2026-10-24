#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Καταγραφή περιβάλλοντος -> env.json  (γενικός κανόνας #5 του πρωτοκόλλου).

Καταγράφει: git commit, pip freeze, έκδοση Ollama, GPU & driver, sha256 του GGUF,
του adapter και του αρχείου prompts (main.py), num_ctx/num_predict/decoding.

ΕΚΤΕΛΕΣΗ:
  python capture_env.py
Συμπληρώστε τα paths παρακάτω ώστε να δείχνουν στα δικά σας αρχεία.
"""
import json, os, subprocess, hashlib, sys

# Paths — προσαρμόστε
GGUF_PATH    = os.getenv("GGUF_PATH", "../../data/techqa-lora-f16.gguf")   # host: από data/revision_experiment1
ADAPTER_DIR  = os.getenv("ADAPTER_DIR", "../../adapter")   # ο φάκελος του QLoRA adapter
PROMPTS_FILE = os.getenv("PROMPTS_FILE", "../../app/main.py")  # εκεί ζουν RAG_SYSTEM_PROMPT/ABSTAIN_PHRASE
NUM_CTX      = int(os.getenv("NUM_CTX", "8192"))
NUM_PREDICT  = int(os.getenv("NUM_PREDICT", "1024"))  # == main.py LLM_NUM_PREDICT
DECODING     = {"temperature": 0}


def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception as e:
        return f"(n/a: {e})"


def sha256_file(path):
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# prompt template hash — ΙΔΙΟΣ υπολογισμός με το exp1d_verify.py (stage_bc_hooks).
os.environ["MAIN_PY_PATH"] = PROMPTS_FILE   # ώστε το stage_bc_hooks να διαβάσει ΑΥΤΟ το main.py
def prompt_template_sha():
    try:
        import stage_bc_hooks as _H
        if not _H.RAG_SYSTEM_PROMPT:
            print("[WARN] RAG_SYSTEM_PROMPT κενό — ελέγξτε PROMPTS_FILE/MAIN_PY_PATH.")
            return None
        return _H.prompt_template_sha256()
    except Exception as e:
        print(f"[WARN] prompt_template hash: {e}")
        return None


SKIP_DIRS = {"venv", ".venv", "env", "__pycache__", ".git", "node_modules", ".ipynb_checkpoints"}

def sha256_dir(path):
    """sha256 των αρχείων του adapter (ταξινομημένα). Παραλείπει venv/junk & symlinks, και
    δεν σκάει σε μη-αναγνώσιμα (π.χ. Windows symlinks του venv)."""
    if not path or not os.path.isdir(path):
        return None
    h = hashlib.sha256(); skipped = []
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]      # ΜΗΝ μπεις σε venv/junk
        for name in sorted(files):
            fp = os.path.join(root, name)
            if os.path.islink(fp):
                skipped.append(fp); continue
            rel = os.path.relpath(fp, path).replace(os.sep, "/")
            try:
                with open(fp, "rb") as f:
                    h.update(rel.encode())
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        h.update(chunk)
            except OSError:
                skipped.append(fp)
    if skipped:
        print(f"[ΣΗΜΕΙΩΣΗ] adapter sha256: παραλείφθηκαν {len(skipped)} αρχεία "
              f"(symlinks/venv/μη-αναγνώσιμα), π.χ. {skipped[0]}")
    return h.hexdigest()


env = {
    "git_commit": sh("git rev-parse HEAD"),
    "git_status_dirty": sh("git status --porcelain") != "",
    "ollama_version": sh("ollama --version"),
    "ollama_models": sh("ollama list"),
    "gpu": sh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader"),
    "python": sys.version,
    "pip_freeze": sh(f"{sys.executable} -m pip freeze").splitlines(),
    "sha256": {
        "gguf": sha256_file(GGUF_PATH),
        "adapter_dir": sha256_dir(ADAPTER_DIR),
        "prompts_file": sha256_file(PROMPTS_FILE),
        "prompt_template": prompt_template_sha(),   # αποτύπωμα build_messages (για 1Δ έλεγχο 4)
    },
    "num_ctx": NUM_CTX,
    "num_predict": NUM_PREDICT,
    "decoding": DECODING,
}

with open("env.json", "w", encoding="utf-8") as f:
    json.dump(env, f, indent=2, ensure_ascii=False)
print("[OK] env.json γράφτηκε.")
for k in ("git_commit", "ollama_version", "gpu"):
    print(f"  {k}: {env[k]}")
print(f"  sha256.prompts_file:     {env['sha256']['prompts_file']}")
print(f"  sha256.prompt_template:  {env['sha256']['prompt_template']}")
print(f"  sha256.gguf:         {env['sha256']['gguf']}")
print(f"  sha256.adapter_dir:  {env['sha256']['adapter_dir']}")
if not GGUF_PATH or not ADAPTER_DIR:
    print("\n[ΣΗΜΕΙΩΣΗ] Ορίστε GGUF_PATH και ADAPTER_DIR για πλήρη sha256 (env vars ή στο αρχείο).")
