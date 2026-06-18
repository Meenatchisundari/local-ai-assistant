import argparse, json, sys, time
import httpx

FASTAPI_BASE = "http://127.0.0.1:8000"
OLLAMA_BASE  = "http://127.0.0.1:11434"
TEST_PROMPT  = "Reply with exactly one word: hello"

def check(label, ok, detail=""):
    mark = "OK" if ok else "FAIL"
    line = f"  [{mark}] {label}"
    if detail:
        line += f"  ->  {detail}"
    print(line)
    return ok

def run_checks(model):
    all_ok = True
    try:
        r = httpx.get(f"{OLLAMA_BASE}/api/tags", timeout=5.0)
        tags_ok = r.status_code == 200
        available = [m["name"] for m in r.json().get("models", [])] if tags_ok else []
        all_ok &= check("Ollama /api/tags", tags_ok, f"{len(available)} models found")
    except Exception as exc:
        check("Ollama /api/tags", False, str(exc))
        print("\n  Ollama is not running. Start it with:  ollama serve")
        return False

    model_present = any(model in m for m in available)
    all_ok &= check(f"Model '{model}' available", model_present,
                    "run: ollama pull " + model if not model_present else "")

    try:
        r = httpx.get(f"{FASTAPI_BASE}/health", timeout=5.0)
        health = r.json()
        fa_ok = r.status_code == 200 and health.get("ollama_reachable")
        all_ok &= check("FastAPI /health", fa_ok, str(health))
    except Exception as exc:
        check("FastAPI /health", False, str(exc))
        print("\n  FastAPI is not running. Start it with:")
        print("  uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload")
        return False

    print(f"\n  [..] Streaming smoke test  (model={model}) ...")
    t_start = time.perf_counter()
    t_first = None
    tokens = []
    error = None
    try:
        with httpx.stream("POST", f"{FASTAPI_BASE}/chat",
                json={"model": model, "prompt": TEST_PROMPT, "stream": True,
                      "options": {"temperature": 0.0}}, timeout=60.0) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                line = raw_line.removeprefix("data: ").strip()
                if not line or line == "[DONE]":
                    continue
                try:
                    chunk = json.loads(line)
                except:
                    continue
                tok = chunk.get("response", "")
                if tok and t_first is None:
                    t_first = time.perf_counter()
                if tok:
                    tokens.append(tok)
                if chunk.get("done"):
                    break
    except Exception as exc:
        error = str(exc)

    t_end = time.perf_counter()
    if error or t_first is None:
        all_ok &= check("Streaming smoke test", False, error or "no tokens received")
    else:
        ttft  = round(t_first - t_start, 3)
        total = round(t_end - t_start, 3)
        text  = "".join(tokens).strip()
        tok_s = round(len(text.split()) / max(total - ttft, 0.001), 1)
        all_ok &= check("Streaming smoke test", True,
                        f"TTFT={ttft}s  tok/s={tok_s}  total={total}s  response='{text[:60]}'")
    return all_ok

p = argparse.ArgumentParser()
p.add_argument("--model", default="tinyllama")
args = p.parse_args()
print("\n" + "="*50)
print("  Phase 1 Pre-flight Validation")
print("="*50)
ok = run_checks(args.model)
print()
if ok:
    print("  All checks passed - ready to benchmark\n")
    sys.exit(0)
else:
    print("  One or more checks failed - fix issues above\n")
    sys.exit(1)
