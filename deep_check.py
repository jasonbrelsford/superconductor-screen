"""Deep stability checks on a bigger machine (TOWER): the same CHGNet calculation as the daily
GitHub run, but more candidates per run, larger doped cells and more dopant arrangements.

    python deep_check.py [hours]      # default 5 hours; needs MP_API_KEY in the environment

It works through the strongest leaderboard candidates that neither machine has checked yet,
records every result in a local SQLite database (data/screen.db, never committed), writes
results/structure_checks_tower.json and pushes it, so the next GitHub run publishes it.
"""
import datetime as dt
import json
import os
import sqlite3
import subprocess
import sys
import time

import pandas as pd

from sc import data, hits, structure

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results", "structure_checks_tower.json")
DB = os.path.join(ROOT, "data", "screen.db")


def git(*args):
    pre = []
    token = os.environ.get("GITHUB_ADMIN_TOKEN", "").strip()
    if token and args and args[0] in ("pull", "push", "fetch"):   # authenticate without a stored credential
        import base64
        b = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        pre = ["-c", "credential.helper=", "-c", f"http.extraheader=AUTHORIZATION: basic {b}"]
    return subprocess.run(["git", *pre, *args], cwd=ROOT, capture_output=True, text=True)


def main():
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    key = os.environ.get("MP_API_KEY", "").strip()
    if not key:
        sys.exit("MP_API_KEY is not set on this machine")
    git("pull", "-q", "--rebase", "origin", "main")
    structure.MAX_DOPED_SITES = 320   # TOWER can afford bigger cells
    structure.CONFIGS = 5

    board = pd.read_csv(os.path.join(ROOT, "results", "leaderboard.csv"))
    done = {}
    for name in ("structure_checks.json", "structure_checks_tower.json"):
        p = os.path.join(ROOT, "results", name)
        if os.path.exists(p):
            done.update(json.load(open(p)))
    mine = json.load(open(OUT)) if os.path.exists(OUT) else {}
    queue = board[~board["formula"].isin(set(done))].sort_values("score", ascending=False)

    os.makedirs(os.path.dirname(DB), exist_ok=True)
    db = sqlite3.connect(DB)
    db.execute("create table if not exists checks (formula text primary key, parent text, score real, "
               "pred_tc real, status text, e_above_hull real, computed_as text, cell_atoms int, host text, checked text)")

    chk = structure.Checker(key, os.path.join(ROOT, "results", "chgnet_cache.json"), hours * 3600)
    today = dt.date.today().isoformat()
    n = 0
    try:
        for _, row in queue.iterrows():
            if chk.time_left() < 600:
                break
            swaps = row.get("swaps")
            if not isinstance(swaps, str):
                continue
            try:
                r = chk.check(row["formula"], row["parent"])
            except TimeoutError:
                break
            except Exception as ex:
                r = {"status": f"calculation failed ({type(ex).__name__})"}
            r["checked"] = today
            r["by"] = "tower"
            r["method"] = 2
            mine[row["formula"]] = r
            db.execute("insert or replace into checks values (?,?,?,?,?,?,?,?,?,?)",
                       (row["formula"], row["parent"], float(row["score"]), float(row["pred_Tc_K"]), r["status"],
                        r.get("e_above_hull_eV"), r.get("computed_as"), r.get("cell_atoms"), r.get("host"), today))
            db.commit()
            n += 1
            print(f"[{n}] {row['formula']}: {r['status']} {r.get('e_above_hull_eV', '')}", flush=True)
            if n % 10 == 0:   # save progress as we go
                json.dump(mine, open(OUT, "w"), indent=1, sort_keys=True)
    finally:
        chk.close()
        json.dump(mine, open(OUT, "w"), indent=1, sort_keys=True)
        db.close()

    new_hits = hits.find(board, mine, hits.records(data.load()))
    hits.notify(new_hits)
    git("add", "results/structure_checks_tower.json", "results/chgnet_cache.json", "results/hits.json")
    if git("diff", "--cached", "--quiet").returncode:
        git("-c", "user.name=TOWER deep check", "-c", "user.email=jasonbrelsford@users.noreply.github.com",
            "commit", "-q", "-m", f"Deep checks from TOWER: {n} candidates")
        git("pull", "-q", "--rebase", "origin", "main")
        print(git("push", "-q", "origin", "main").stderr.strip() or "pushed")
    print(f"done: {n} checked, {len(new_hits)} new hits")


if __name__ == "__main__":
    main()
