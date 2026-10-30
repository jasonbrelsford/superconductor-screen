"""One parallel worker: stability checks for its slice of the queue the last run left behind.

    python check_shard.py INDEX COUNT        # needs MP_API_KEY

The GitHub workflow runs COUNT of these at once (a public repo's Actions minutes are free), each
taking every COUNT-th entry of results/queue.json. Each writes shard_out/checks_INDEX.json and
shard_out/cache_INDEX.json; the main job merges them into the published results.
"""
import datetime as dt
import json
import os
import sys

from run import METHOD
from sc import structure

ROOT = os.path.dirname(os.path.abspath(__file__))
BUDGET_S = int(os.environ.get("SHARD_BUDGET_S", 95 * 60))


def main():
    i, n = int(sys.argv[1]), int(sys.argv[2])
    out = os.path.join(ROOT, "shard_out")
    os.makedirs(out, exist_ok=True)
    qpath = os.path.join(ROOT, "results", "queue.json")
    queue = json.load(open(qpath)) if os.path.exists(qpath) else []
    mine = queue[i::n]
    key = os.environ.get("MP_API_KEY", "").strip()
    checks = {}
    cache_out = os.path.join(out, f"cache_{i}.json")
    if key and mine:
        src = os.path.join(ROOT, "results", "chgnet_cache.json")
        if os.path.exists(src):   # start from the shared cache, write to this shard's own file
            json.dump(json.load(open(src)), open(cache_out, "w"))
        chk = structure.Checker(key, cache_out, BUDGET_S)
        today = dt.date.today().isoformat()
        try:
            for item in mine:
                if chk.time_left() < 300:
                    break
                try:
                    r = chk.check(item["formula"], item["parent"], [tuple(s) for s in item["swaps"]])
                except TimeoutError:
                    break
                except Exception as ex:
                    r = {"status": f"calculation failed ({type(ex).__name__})"}
                r["checked"], r["method"], r["by"] = today, METHOD, f"shard {i}"
                checks[item["formula"]] = r
                print("structure check:", item["formula"], r, flush=True)
        finally:
            chk.close()
    json.dump(checks, open(os.path.join(out, f"checks_{i}.json"), "w"), indent=1, sort_keys=True)
    print(f"shard {i}/{n}: {len(checks)} of {len(mine)} checked")


if __name__ == "__main__":
    main()
