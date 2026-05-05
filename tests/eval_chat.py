#!/usr/bin/env python3
"""
Evaluates chat response quality against fixture scenarios using LLM-as-judge.

Usage: python tests/eval_chat.py
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
    'Du bewertest eine KI-Antwort anhand eines Kriteriums. '
    'Antworte NUR mit validem JSON: {"result": "ja", "reason": "kurze Begründung"} '
    'oder {"result": "nein", "reason": "kurze Begründung"}'
)


def judge(criterion: str, response: str) -> tuple[bool, str]:
    try:
        raw = miru._call_llm(
            [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": f'Kriterium: "{criterion}"\nAntwort: "{response[:800]}"'},
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
    history = fixture["history"].copy()
    test_message = fixture.get("test_message", "")
    rubric = fixture.get("chat_rubric", [])
    results: dict = {"fixture": name, "response": "", "checks": [], "passed": 0, "total": len(rubric)}

    if not test_message:
        print("  [skip] no test_message defined")
        return results

    system_prompt = miru.build_system_prompt(history)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": test_message},
    ]

    print(f"  Input    : \"{test_message[:100]}\"")
    response = miru.get_chat_response(messages)
    results["response"] = response
    print(f"  Response : {response[:250]}{'...' if len(response) > 250 else ''}")

    for criterion in rubric:
        passed, reason = judge(criterion, response)
        results["checks"].append({"check": criterion, "passed": passed, "reason": reason})
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
    score_file = SCORES_DIR / f"chat_{timestamp}.json"
    with open(score_file, "w", encoding="utf-8") as f:
        json.dump({"timestamp": timestamp, "results": all_results}, f, ensure_ascii=False, indent=2)
    print(f"\nScore log saved → {score_file.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
