#!/usr/bin/env python3
"""
Evaluates the memory pipeline (summarize_session + update_user_profile)
against fixture conversations.

Usage: python tests/eval_memory.py
"""

import sys
import json
import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import main as miru

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SCORES_DIR = Path(__file__).parent / "scores"

JUDGE_SYSTEM = (
    'Du bewertest eine KI-Ausgabe anhand eines Kriteriums. '
    'Antworte NUR mit validem JSON: {"result": "ja", "reason": "kurze Begründung"} '
    'oder {"result": "nein", "reason": "kurze Begründung"}'
)


def judge(criterion: str, output: str) -> tuple[bool, str]:
    try:
        raw = miru._call_llm(
            [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": f'Kriterium: "{criterion}"\nAusgabe: "{output[:600]}"'},
            ],
            max_tokens=80,
            temperature=0.1,
        )
        parsed = miru._parse_json_response(raw)
        if parsed:
            return parsed.get("result") == "ja", parsed.get("reason", "")
    except Exception:
        pass
    return False, "judge failed"


def run_fixture(fixture: dict) -> dict:
    name = fixture["name"]
    session_messages = fixture.get("session_messages", [])
    data = fixture["history"].copy()
    checks = fixture.get("memory_checks", {})
    results: dict = {"fixture": name, "summary": None, "profile": None, "checks": [], "passed": 0, "total": 0}

    if not session_messages:
        print("  [skip] no session_messages – nothing to summarize")
        return results

    print("  Running summarize_session()...")
    summary = miru.summarize_session(session_messages)
    if not summary:
        print("  [FAIL] summarize_session returned None")
        return results

    results["summary"] = summary
    print(f"  summary   : {summary.get('summary', '')[:120]}")
    print(f"  themes    : {summary.get('themes', [])}")
    print(f"  key_facts : {summary.get('key_facts', [])}")
    print(f"  mood      : {summary.get('mood_observed')}")

    print("  Running update_user_profile()...")
    profile = miru.update_user_profile(data, summary)
    results["profile"] = profile
    print(f"  profile   : {profile[:200]}")

    summary_text = json.dumps(summary, ensure_ascii=False).lower()

    # Keyword presence checks
    for term in checks.get("summary_must_contain_any", []):
        passed = term.lower() in summary_text
        results["checks"].append({"check": f"summary enthält '{term}'", "passed": passed})
        results["total"] += 1
        if passed:
            results["passed"] += 1

    for term in checks.get("profile_must_contain_any", []):
        passed = term.lower() in profile.lower()
        results["checks"].append({"check": f"profil enthält '{term}'", "passed": passed})
        results["total"] += 1
        if passed:
            results["passed"] += 1

    # LLM quality checks
    quality_criteria = [
        "Enthält die Zusammenfassung die emotionale Kernaussage des Gesprächs?",
        "Sind die key_facts stabile, sitzungsunabhängige Fakten über den Nutzer (keine reinen Gesprächsinhalte)?",
    ]
    for criterion in quality_criteria:
        passed, reason = judge(criterion, json.dumps(summary, ensure_ascii=False))
        results["checks"].append({"check": criterion, "passed": passed, "reason": reason})
        results["total"] += 1
        if passed:
            results["passed"] += 1

    return results


def main():
    SCORES_DIR.mkdir(exist_ok=True)
    fixtures = sorted(FIXTURES_DIR.glob("*.json"))

    if not fixtures:
        print("No fixtures found in tests/fixtures/")
        return

    all_results = []

    for fixture_path in fixtures:
        with open(fixture_path, encoding="utf-8") as f:
            fixture = json.load(f)

        if not fixture.get("session_messages"):
            continue

        print(f"\n{'='*60}")
        print(f"Fixture : {fixture['name']}")
        print(f"Desc    : {fixture.get('description', '')}")
        print("=" * 60)

        result = run_fixture(fixture)
        all_results.append(result)

        score = result["passed"]
        total = result["total"]
        pct = 100 * score // total if total else 0
        print(f"\n  Score: {score}/{total} ({pct}%)")
        for check in result["checks"]:
            mark = "✓" if check["passed"] else "✗"
            reason = f" – {check.get('reason', '')}" if check.get("reason") else ""
            print(f"    {mark} {check['check']}{reason}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    score_file = SCORES_DIR / f"memory_{timestamp}.json"
    with open(score_file, "w", encoding="utf-8") as f:
        json.dump({"timestamp": timestamp, "results": all_results}, f, ensure_ascii=False, indent=2)
    print(f"\nScore log saved → {score_file.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
