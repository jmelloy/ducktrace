#!/usr/bin/env python3
"""Print a summary report from the sessions/events DuckDB store.

Usage: python report.py [--db data/sessions.duckdb]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


def _rule(title: str, width: int = 78) -> None:
    print("\n" + "─" * width)
    print(title)
    print("─" * width)


def _table(con, sql, headers, fmts=None, params=None):
    rows = con.execute(sql, params or []).fetchall()
    if not rows:
        print("  (no data)")
        return
    fmts = fmts or ["{}"] * len(headers)
    cells = [[f.format(v) if v is not None else "" for f, v in zip(fmts, r)] for r in rows]
    widths = [max(len(h), *(len(c[i]) for c in cells)) for i, h in enumerate(headers)]
    # right-align numeric-looking columns (those whose fmt has alignment digits or commas)
    numeric = [any(tok in fmts[i] for tok in (",", ".", ">")) for i in range(len(headers))]
    def fmt_row(vals):
        return "  ".join(
            (v.rjust(widths[i]) if numeric[i] else v.ljust(widths[i]))
            for i, v in enumerate(vals)
        )
    print("  " + fmt_row(headers))
    print("  " + "  ".join("-" * w for w in widths))
    for c in cells:
        print("  " + fmt_row(c))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="data/sessions.duckdb")
    args = ap.parse_args()
    con = duckdb.connect(str(Path(args.db).expanduser()), read_only=True)

    # ---- overview ----
    n_sess, n_ev, cost, toks, t0, t1 = con.execute("""
        SELECT (SELECT count(*) FROM sessions),
               (SELECT count(*) FROM events),
               (SELECT sum(cost_usd) FROM sessions),
               (SELECT sum(total_tokens) FROM sessions),
               (SELECT min(started_at) FROM sessions),
               (SELECT max(ended_at) FROM sessions)
    """).fetchone()
    _rule("OVERVIEW")
    print(f"  Sessions   : {n_sess:,}")
    print(f"  Events     : {n_ev:,}")
    print(f"  Date range : {str(t0)[:10]} → {str(t1)[:10]}")
    print(f"  Tokens     : {toks:,}")
    print(f"  Cost (USD) : ${cost:,.2f}")

    _rule("BY SOURCE")
    _table(con,
        """SELECT source, count(*), sum(event_count), sum(total_tokens), sum(cost_usd)
           FROM sessions GROUP BY 1 ORDER BY 5 DESC""",
        ["source", "sessions", "events", "tokens", "cost $"],
        ["{}", "{:,}", "{:,}", "{:,}", "{:,.2f}"])

    _rule("TOP REPOSITORIES (by cost)")
    _table(con,
        """SELECT coalesce(repository,'(unknown)'), count(*), sum(total_tokens), sum(cost_usd)
           FROM sessions GROUP BY 1 ORDER BY 4 DESC LIMIT 15""",
        ["repository", "sessions", "tokens", "cost $"],
        ["{}", "{:,}", "{:,}", "{:,.2f}"])

    _rule("BY MODEL")
    _table(con,
        """SELECT coalesce(model,'(unknown)'), count(*), sum(total_tokens), sum(cost_usd)
           FROM sessions WHERE model IS NOT NULL AND model <> '' GROUP BY 1 ORDER BY 4 DESC""",
        ["model", "sessions", "tokens", "cost $"],
        ["{}", "{:,}", "{:,}", "{:,.2f}"])

    _rule("COST BY DAY (last 14 active days)")
    _table(con,
        """SELECT cast(started_at AS DATE) d, count(*), sum(total_tokens), sum(cost_usd)
           FROM sessions WHERE started_at IS NOT NULL
           GROUP BY 1 ORDER BY 1 DESC LIMIT 14""",
        ["day", "sessions", "tokens", "cost $"],
        ["{}", "{:,}", "{:,}", "{:,.2f}"])

    _rule("FILE EDITS BY TYPE")
    _table(con,
        """SELECT file_ext, count(*), sum(lines_added), sum(lines_removed)
           FROM events WHERE role='tool_use' AND file_path IS NOT NULL
           GROUP BY 1 ORDER BY 2 DESC LIMIT 12""",
        ["ext", "edits", "+lines", "-lines"],
        ["{}", "{:,}", "{:,}", "{:,}"])

    _rule("TOP TOOLS")
    _table(con,
        """SELECT source, tool_name, count(*)
           FROM events WHERE role='tool_use' AND tool_name IS NOT NULL
           GROUP BY 1,2 ORDER BY 3 DESC LIMIT 12""",
        ["source", "tool", "calls"],
        ["{}", "{}", "{:,}"])

    _rule("PR ACTIVITY")
    _table(con,
        """SELECT source, pr_action, count(*)
           FROM events WHERE pr_action IS NOT NULL
           GROUP BY 1,2 ORDER BY 3 DESC LIMIT 12""",
        ["source", "pr action", "count"],
        ["{}", "{}", "{:,}"])
    npr = con.execute("SELECT count(DISTINCT pr_number) FROM events WHERE pr_number IS NOT NULL").fetchone()[0]
    print(f"\n  Distinct PRs referenced: {npr:,}")

    _rule("TOP SESSIONS (by cost)")
    _table(con,
        """SELECT coalesce(title,'(untitled)'), source, coalesce(repository,''),
                  message_count, round(cost_usd,2)
           FROM sessions ORDER BY cost_usd DESC LIMIT 15""",
        ["title", "src", "repository", "msgs", "cost $"],
        ["{:.40}", "{}", "{:.28}", "{:,}", "{:,.2f}"])

    print()
    con.close()


if __name__ == "__main__":
    main()
