import argparse, csv, json, platform, sys, time
from datetime import datetime
from pathlib import Path
import httpx, psutil

FASTAPI_BASE   = "http://127.0.0.1:8000"
OLLAMA_BASE    = "http://127.0.0.1:11434"
DEFAULT_MODELS = ["tinyllama", "qwen2:1.5b"]
DEFAULT_REPS   = 3
DEFAULT_OUT    = Path("data/results")

PROMPTS = [
    {"id": "fqa_01", "category": "factual_qa",
     "text": "What is the capital of France? Answer in one sentence."},
    {"id": "fqa_02", "category": "factual_qa",
     "text": "Name the three primary colours used in light RGB. List only the colours."},
    {"id": "fqa_03", "category": "factual_qa",
     "text": "What does CPU stand for? Answer in one sentence."},
    {"id": "fqa_04", "category": "factual_qa",
     "text": "In what year did the first Moon landing occur? One sentence."},
    {"id": "fqa_05", "category": "factual_qa",
     "text": "What is the boiling point of water at sea level in Celsius?"},
    {"id": "sum_01", "category": "summarisation",
     "text": "Summarise in exactly two sentences: The Python programming language was created by Guido van Rossum and first released in 1991. It emphasises code readability and uses significant indentation. Python supports multiple programming paradigms including procedural, object-oriented, and functional programming."},
    {"id": "sum_02", "category": "summarisation",
     "text": "Give a one-sentence summary: Machine learning is a subset of artificial intelligence that enables systems to learn from data and improve their performance on tasks without being explicitly programmed."},
    {"id": "inst_01", "category": "instruction_following",
     "text": "List exactly five programming languages, one per line, no numbering, no extra text."},
    {"id": "inst_02", "category": "instruction_following",
     "text": "Respond with only the word YES if 2 plus 2 equals 4, or only the word NO if it does not."},
    {"id": "inst_03", "category": "instruction_following",
     "text": "Write a haiku about winter. Output only the haiku, no title or explanation."},
    {"id": "rsn_01", "category": "reasoning",
     "text": "If a train travels at 60 mph for 2.5 hours, how far does it travel? Show your working briefly."},
    {"id": "rsn_02", "category": "reasoning",
     "text": "Alice is taller than Bob. Bob is taller than Carol. Who is the shortest? Explain in one sentence."},
    {"id": "rsn_03", "category": "reasoning",
     "text": "What is 17 times 13? Think step by step, then give the final answer on its own line."},
    {"id": "json_01", "category": "json_generation",
     "text": "Return ONLY a valid JSON object with keys name, language, and year_created for the Python programming language. No markdown, no explanation."},
    {"id": "json_02", "category": "json_generation",
     "text": "Return ONLY a valid JSON array of three objects, each with keys city and country, for any three European capital cities. No markdown, no explanation."},
    {"id": "edge_01", "category": "edge_case",
     "text": "Respond with an empty JSON object: {}"},
    {"id": "edge_02", "category": "edge_case",
     "text": "What is the meaning of life? Answer in exactly four words."},
    {"id": "edge_03", "category": "edge_case",
     "text": "Translate Hello world into three languages. Format: Language: Translation, one per line."},
    {"id": "lng_01", "category": "longer_generation",
     "text": "Write a short paragraph of 4 to 6 sentences explaining why local AI inference is useful for privacy-sensitive applications."},
    {"id": "lng_02", "category": "longer_generation",
     "text": "Describe the difference between quantization levels Q4 and Q8 in the context of large language models in 3 to 5 sentences."},
]

CSV_FIELDNAMES = [
    "run_id","session_id","timestamp_utc","model","prompt_id","prompt_category",
    "repetition","temperature","ttft_s","tokens_per_sec","total_latency_s",
    "output_tokens","prompt_tokens","first_token_text","error",
    "model_parameter_size","model_quantization","model_family",
    "hw_os","hw_cpu_physical","hw_cpu_logical","hw_cpu_freq_max_mhz",
    "hw_ram_total_gb","hw_python_version",
]

def capture_hardware():
    freq = psutil.cpu_freq()
    return {
        "python_version": platform.python_version(),
        "os": f"{platform.system()} {platform.release()}",
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "cpu_freq_max_mhz": round(freq.max, 1) if freq else None,
        "ram_total_gb": round(psutil.virtual_memory().total / 1024**3, 2),
    }

def get_model_info(model):
    try:
        r = httpx.post(f"{OLLAMA_BASE}/api/show", json={"name": model}, timeout=10)
        d = r.json().get("details", {})
        return {"parameter_size": d.get("parameter_size","unknown"),
                "quantization_level": d.get("quantization_level","unknown"),
                "family": d.get("family","unknown")}
    except:
        return {"parameter_size":"unknown","quantization_level":"unknown","family":"unknown"}

def measure(model, prompt, temperature=0.0):
    result = {"ttft_s":None,"tokens_per_sec":None,"total_latency_s":None,
              "output_tokens":None,"prompt_tokens":None,"first_token_text":None,"error":None}
    payload = {"model":model,"prompt":prompt,"stream":True,"options":{"temperature":temperature}}
    t_start = time.perf_counter()
    t_first = None
    tokens = []
    out_count = None
    prompt_count = None
    try:
        with httpx.stream("POST", f"{FASTAPI_BASE}/chat", json=payload, timeout=120.0) as resp:
            resp.raise_for_status()
            for raw in resp.iter_lines():
                line = raw.removeprefix("data: ").strip()
                if not line or line == "[DONE]":
                    continue
                try:
                    chunk = json.loads(line)
                except:
                    continue
                tok = chunk.get("response","")
                if tok and t_first is None:
                    t_first = time.perf_counter()
                    result["first_token_text"] = repr(tok)
                if tok:
                    tokens.append(tok)
                if chunk.get("done"):
                    out_count = chunk.get("eval_count")
                    prompt_count = chunk.get("prompt_eval_count")
                    break
    except Exception as exc:
        result["error"] = str(exc)
        return result

    t_end = time.perf_counter()
    if t_first is None:
        result["error"] = "no tokens received"
        return result

    ttft = t_first - t_start
    total = t_end - t_start
    gen_window = total - ttft
    output_tokens = out_count if (out_count and out_count > 0) else len("".join(tokens).split())
    tok_s = output_tokens / gen_window if gen_window > 0 else 0.0

    result.update({
        "ttft_s": round(ttft, 4),
        "tokens_per_sec": round(tok_s, 2),
        "total_latency_s": round(total, 4),
        "output_tokens": output_tokens,
        "prompt_tokens": prompt_count,
    })
    return result

def run_benchmark(models, prompts, reps, out_dir, temperature=0.0, quick=False):
    session_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"benchmark_{session_id}.csv"
    hw = capture_hardware()
    suite = prompts[:10] if quick else prompts

    print(f"\n{'='*55}")
    print(f"  Phase 1 Benchmark  —  session {session_id}")
    print(f"{'='*55}")
    print(f"  Models  : {', '.join(models)}")
    print(f"  Prompts : {len(suite)}  Reps: {reps}  Temp: {temperature}")
    print(f"  Output  : {csv_path}")
    print(f"  RAM     : {hw['ram_total_gb']} GB  CPUs: {hw['cpu_physical_cores']}p/{hw['cpu_logical_cores']}l")
    print(f"{'='*55}\n")

    model_meta = {}
    for m in models:
        print(f"  [meta] {m} ... ", end="", flush=True)
        model_meta[m] = get_model_info(m)
        print(model_meta[m]["parameter_size"])

    total_runs = len(models) * len(suite) * reps
    counter = 0

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for model in models:
            meta = model_meta[model]
            print(f"\n--- {model} ({meta['parameter_size']} / {meta['quantization_level']}) ---")
            for prompt in suite:
                for rep in range(1, reps+1):
                    counter += 1
                    run_id = f"{session_id}__{model.replace(':','_')}__{prompt['id']}__rep{rep:02d}"
                    ts = datetime.utcnow().isoformat(timespec="milliseconds")
                    print(f"  [{counter:>3}/{total_runs}] {prompt['id']} rep{rep}/{reps}  ", end="", flush=True)
                    m = measure(model, prompt["text"], temperature)
                    if m["error"]:
                        print(f"ERROR: {m['error']}")
                    else:
                        print(f"TTFT={m['ttft_s']:.3f}s  tok/s={m['tokens_per_sec']:.1f}  total={m['total_latency_s']:.3f}s  tokens={m['output_tokens']}")
                    row = {
                        "run_id":run_id,"session_id":session_id,"timestamp_utc":ts,
                        "model":model,"prompt_id":prompt["id"],"prompt_category":prompt["category"],
                        "repetition":rep,"temperature":temperature,
                        **{k:m[k] for k in ("ttft_s","tokens_per_sec","total_latency_s","output_tokens","prompt_tokens","first_token_text","error")},
                        "model_parameter_size":meta["parameter_size"],
                        "model_quantization":meta["quantization_level"],
                        "model_family":meta["family"],
                        "hw_os":hw["os"],"hw_cpu_physical":hw["cpu_physical_cores"],
                        "hw_cpu_logical":hw["cpu_logical_cores"],
                        "hw_cpu_freq_max_mhz":hw["cpu_freq_max_mhz"],
                        "hw_ram_total_gb":hw["ram_total_gb"],
                        "hw_python_version":hw["python_version"],
                    }
                    writer.writerow(row)
                    fh.flush()

    print(f"\n  Done. {counter} runs written to:\n  {csv_path}\n")
    return csv_path

def print_summary(csv_path):
    try:
        import pandas as pd
    except ImportError:
        print("pandas not available - pip install pandas")
        return
    df = pd.read_csv(csv_path)
    ok = df[df["error"].isna()]
    if ok.empty:
        print("  All runs errored - check FastAPI/Ollama")
        return
    print("\n" + "="*55)
    print("  Per-model summary")
    print("="*55)
    s = ok.groupby("model").agg(
        runs=("run_id","count"),
        ttft_mean=("ttft_s","mean"),
        ttft_p95=("ttft_s", lambda x: x.quantile(0.95)),
        tok_s_mean=("tokens_per_sec","mean"),
        latency_mean=("total_latency_s","mean"),
    ).round(3)
    print(s.to_string())
    errs = df[df["error"].notna()]
    if not errs.empty:
        print(f"\n  {len(errs)} errored runs:")
        print(errs[["model","prompt_id","repetition","error"]].to_string(index=False))
    print()

p = argparse.ArgumentParser()
p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
p.add_argument("--reps", type=int, default=DEFAULT_REPS)
p.add_argument("--temperature", type=float, default=0.0)
p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
p.add_argument("--quick", action="store_true")
p.add_argument("--no-summary", action="store_true")
args = p.parse_args()
csv_path = run_benchmark(args.models, PROMPTS, args.reps, args.out_dir, args.temperature, args.quick)
if not args.no_summary:
    print_summary(csv_path)
