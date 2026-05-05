"""
Miru Mind – Local Mental Health Chatbot
========================================
Phase 1: Groq API (powerful model, fast iteration)
Phase 2: Ollama local (privacy-first, on-device)

Setup:
  1. pip install -r requirements.txt
  2. Create a .env file with GROQ_API_KEY=your_key
     → Free account: https://console.groq.com
  3. python main.py
"""

import json
import re
import datetime
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.markdown import Markdown
except ImportError:
    print("Please install dependencies first: pip install -r requirements.txt")
    exit(1)

# ─── Configuration ────────────────────────────────────────────────────────────

# Switch between modes here:
# "groq"  → Groq API (Phase 1, powerful model, requires internet)
# "local" → Ollama local (Phase 2, privacy-first, no internet)
MODE = "groq"

# Groq settings
GROQ_MODEL = "llama-3.3-70b-versatile"

# Ollama settings
OLLAMA_MODEL = "llama3.2:1b"
OLLAMA_HOST  = "http://localhost:11434"   # or Mac IP: "http://192.168.1.X:11434"

# Memory settings
RECENT_SESSIONS_COUNT = 3   # How many past session summaries to inject
MOOD_TREND_DAYS       = 7   # Mood trend over the last N days

DATA_FILE = Path(__file__).parent / "data" / "history.json"
console = Console()

# ─── Client initialisation ────────────────────────────────────────────────────

if MODE == "groq":
    try:
        from groq import Groq
        groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    except ImportError:
        print("Groq not installed: pip install groq")
        exit(1)
    except Exception as e:
        print(f"Groq error: {e}")
        exit(1)
elif MODE == "local":
    try:
        import ollama
        ollama_client = ollama.Client(host=OLLAMA_HOST)
    except ImportError:
        print("Ollama not installed: pip install ollama")
        exit(1)

# ─── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT_BASE = """Du bist Miru – ein ruhiger, warmherziger Begleiter für mentale Gesundheit.

# DEINE PERSÖNLICHKEIT
Du klingst wie ein guter Freund mit therapeutischer Ausbildung: geerdet, geduldig, nie wertend.
Einfache, ehrliche Sprache – keine Fachbegriffe, keine Worthülsen.
Du bist neugierig auf den Menschen, nicht auf das Problem.

# SPRACHE & STIL
- Kurze Sätze bevorzugen. Manchmal ein Halbsatz oder Fragment – das ist okay.
- Natürlicher Gesprächsfluss, kein Therapieschema
- Variiere deinen Einstieg: nicht jede Antwort mit "Das klingt..." beginnen
- Deutsch wie man es wirklich spricht, nicht wie aus einem Lehrbuch
- Emotionen direkt benennen, ohne sie zu dramatisieren
- Gelegentlich umgangssprachlich, aber nie flach

# GESPRÄCHSFÜHRUNG
Höre zu, zeige dass du es verstanden hast, und stelle eine einzige, echte Frage –
nicht nach Schema, sondern so wie es sich natürlich ergibt.

Antworte kurz (3–5 Sätze).

EINE FRAGE – STRIKTE REGEL:
Deine Antwort enthält exakt ein Fragezeichen. Nie zwei, nie null.
"Ist es X, Y oder Z?" sind drei Fragen in einer – verboten. Wähle die eine wichtigste Frage.

# BEISPIELE – SO KLINGT MIRU

Nutzer: "Ich bin so gestresst von der Arbeit."
Miru: "Arbeitsstress kann sich richtig einfressen – besonders wenn er einfach nicht aufhört.
Was macht dir gerade den größten Kopf?"

Nutzer: "Ich weiß nicht, ich fühle mich einfach leer."
Miru: "Leer – das ist ein schweres Gefühl, auch wenn es von außen unsichtbar ist.
Seit wann ist das so bei dir?"

Nutzer: "Heute war mal wieder alles zu viel."
Miru: "Klingt wie ein richtig zäher Tag. Was hat dich am meisten mitgenommen?"

Nutzer: "Ich glaube, ich bin einfach zu schwach für alles."
Miru: "Das sagst du über dich – ich höre, wie überwältigt du gerade bist.
Was bringt dich zu dem Gefühl, nicht mithalten zu können?"

# KENNENLERNEN – ERSTE SITZUNGEN
Wenn du den Namen des Nutzers noch nicht kennst: Frage in deiner Begrüßung vorsichtig danach nachdem du dich selbst vorgestellt hast – natürlich, nicht wie ein Formular.
In den ersten Sitzungen nach und nach ermitteln (nie als Liste abfragen):
- Lebensphase / Alter (ergibt sich oft von selbst aus dem Kontext)
- Wohnsituation (allein, mit Partner, Familie)
- Arbeit oder Hauptbeschäftigung
Wenn der Name bekannt ist: benutze ihn gelegentlich, aber natürlich – nicht bei jedem Satz.

# TECHNIKEN (nur wenn passend, nie aufdrängen)
- Gedankenmuster benennen: "Ich höre, dass du dich selbst sehr hart beurteilst..."
- Atemübung anbieten: "Magst du kurz innehalten? Drei tiefe Atemzüge helfen manchmal."
- Reframing: "Was würdest du einem Freund sagen, der genau das erlebt?"

# SICHERHEIT – HÖCHSTE PRIORITÄT
Bei Krisenzeichen (Suizidgedanken, Selbstverletzung):
→ Ruhig bleiben, ernst nehmen, nicht lösen wollen
→ IMMER sagen: "Bitte ruf jetzt die Telefonseelsorge an: 0800 111 0 111 – kostenlos, 24/7, anonym"
→ Danach fragen: "Bist du gerade in Sicherheit?"

# GRENZEN
Du bist kein Therapeut. Wenn jemand eine Diagnose, Medikamentenberatung oder
professionelle Behandlung braucht, sagst du klar: "Dafür bin ich nicht die richtige
Anlaufstelle – aber ich kann dir helfen, den ersten Schritt zu einem Fachmann zu machen."

Sprache: Immer Deutsch. Nie Englisch, auch wenn der Nutzer Englisch schreibt."""

SUMMARIZATION_PROMPT = """Du analysierst eine abgeschlossene Gesprächssitzung und erstellst eine kompakte Zusammenfassung.

Antworte NUR mit validem JSON in exakt diesem Format (keine weiteren Texte):
{
  "summary": "1-2 Sätze über das Gespräch und die emotionale Lage des Nutzers",
  "themes": ["Thema1", "Thema2"],
  "key_facts": ["Wichtiger Fakt über den Nutzer", "Weiterer Fakt"],
  "mood_observed": 5
}

mood_observed: geschätzte Stimmung des Nutzers am Ende (1-10).
key_facts: stabile Fakten über den Nutzer (Name, Alter, Lebenssituation, Arbeit, wiederkehrende Themen), keine Gesprächsinhalte."""

USER_PROFILE_UPDATE_PROMPT = """Du pflegst das Kurzprofil eines Nutzers für einen Mental-Health-Begleiter.

Dir werden das bisherige Profil und Infos aus einer neuen Sitzung gegeben.
Schreibe ein aktualisiertes Profil als kurzen Absatz (max. 80 Wörter) auf Deutsch.
Behalte wichtige stabile Fakten, lass Veraltetes weg. Keine Gesprächsinhalte, nur Persönlichkeit und Lebenssituation.
Antworte NUR mit dem Profiltext, ohne Formatierung oder Erklärungen."""

# ─── LLM core ────────────────────────────────────────────────────────────────

def _call_llm(messages: list[dict], max_tokens: int = 300, temperature: float = 0.7) -> str:
    if MODE == "groq":
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content
    elif MODE == "local":
        response = ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            options={"temperature": temperature, "num_predict": max_tokens}
        )
        return response["message"]["content"]
    return ""

def get_chat_response(messages: list[dict]) -> str:
    try:
        return _call_llm(messages, max_tokens=300, temperature=0.7)
    except Exception as e:
        return f"[Fehler: {e}]"

# ─── Memory ───────────────────────────────────────────────────────────────────

def _parse_json_response(raw: str) -> dict | None:
    """Extracts and parses a JSON object from an LLM response."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return None
        return None

def summarize_session(session_messages: list[dict]) -> dict | None:
    """Calls the LLM to produce a compact structured summary of the session."""
    if not session_messages:
        return None
    transcript = "\n".join(
        f"{'Nutzer' if m['role'] == 'user' else 'Miru'}: {m['content']}"
        for m in session_messages
    )
    try:
        raw = _call_llm(
            [
                {"role": "system", "content": SUMMARIZATION_PROMPT},
                {"role": "user", "content": f"Sitzungsinhalt:\n{transcript}"}
            ],
            max_tokens=300,
            temperature=0.2
        )
        return _parse_json_response(raw)
    except Exception:
        return None

def update_user_profile(data: dict, summary: dict) -> str:
    """Rewrites the cumulative user profile incorporating the latest session facts."""
    current_profile = data.get("user_profile", "")
    new_facts = "; ".join(summary.get("key_facts", []))
    update_text = f"Neue Sitzung: {summary.get('summary', '')} Neue Fakten: {new_facts}"
    try:
        return _call_llm(
            [
                {"role": "system", "content": USER_PROFILE_UPDATE_PROMPT},
                {"role": "user", "content": f"Bisheriges Profil:\n{current_profile or '(noch keins)'}\n\n{update_text}"}
            ],
            max_tokens=150,
            temperature=0.2
        ).strip()
    except Exception:
        return current_profile

def finalize_session(data: dict, session_messages: list[dict]) -> None:
    """Summarizes the session, updates the user profile, and persists everything."""
    if not session_messages:
        return
    console.print("[dim]Sitzung wird zusammengefasst...[/dim]", end="\r")
    summary = summarize_session(session_messages)
    session_entry: dict = {
        "date": datetime.datetime.now().isoformat(),
        "messages": session_messages,
    }
    if summary:
        session_entry["summary"] = summary.get("summary", "")
        session_entry["themes"] = summary.get("themes", [])
        session_entry["key_facts"] = summary.get("key_facts", [])
        if summary.get("mood_observed") is not None:
            session_entry["mood_observed"] = summary["mood_observed"]
        data["user_profile"] = update_user_profile(data, summary)
    data["sessions"].append(session_entry)
    save_history(data)

def build_memory_context(data: dict) -> str:
    parts = []

    # ── Cumulative user profile ──
    user_profile = data.get("user_profile", "")
    if user_profile:
        parts.append(f"NUTZERPROFIL:\n{user_profile}")

    # ── Mood trend over the last N days ──
    mood_entries = data.get("moods", [])
    if mood_entries:
        cutoff = datetime.datetime.now() - datetime.timedelta(days=MOOD_TREND_DAYS)
        recent = [
            entry for entry in mood_entries
            if datetime.datetime.fromisoformat(entry["date"]) > cutoff
        ]
        if recent:
            values = [entry["value"] for entry in recent]
            average = sum(values) / len(values)
            trend = "steigend" if values[-1] > values[0] else "fallend" if values[-1] < values[0] else "stabil"
            latest_notes = [entry["note"] for entry in recent[-3:] if entry.get("note")]
            mood_text = (
                f"STIMMUNGSTREND (letzte {MOOD_TREND_DAYS} Tage):\n"
                f"- Durchschnitt: {average:.1f}/10 (Trend: {trend})\n"
                f"- Letzte Einträge: {', '.join([str(v) for v in values[-5:]])}/10"
            )
            if latest_notes:
                mood_text += f"\n- Notizen: {'; '.join(latest_notes)}"
            parts.append(mood_text)

    # ── Session summaries (compact, no raw transcripts) ──
    sessions = data.get("sessions", [])
    if sessions:
        recent_sessions = sessions[-RECENT_SESSIONS_COUNT:]
        session_lines = []
        for session in recent_sessions:
            date = session["date"][:10]
            if session.get("summary"):
                line = f"- {date}: {session['summary']}"
                themes = ", ".join(session.get("themes", []))
                if themes:
                    line += f" [Themen: {themes}]"
                session_lines.append(line)
            else:
                # Fallback for sessions saved before summarization was added
                messages = session.get("messages", [])
                user_msgs = [m["content"] for m in messages if m["role"] == "user"]
                if user_msgs:
                    session_lines.append(f"- {date}: \"{user_msgs[0][:120]}...\"")
        if session_lines:
            parts.append(
                f"VERGANGENE GESPRÄCHE (letzte {len(recent_sessions)} Sitzungen):\n"
                + "\n".join(session_lines)
            )

    if not parts:
        return ""

    return (
        "\n\nGEDÄCHTNIS – Was du über den Nutzer weißt:\n"
        + "\n\n".join(parts)
        + "\n\nNutze dieses Wissen um Kontinuität zu zeigen. "
        "Beziehe dich natürlich darauf, ohne es aufzulisten. "
        "Wenn der Name des Nutzers bekannt ist, sprich ihn gelegentlich damit an. "
        "Beginne das Gespräch warm und frage wie es dem Nutzer heute geht."
    )

def build_system_prompt(data: dict) -> str:
    context = build_memory_context(data)
    if context:
        return SYSTEM_PROMPT_BASE + context
    return SYSTEM_PROMPT_BASE + "\n\nErste Sitzung – du kennst den Nutzer noch nicht. Frage in deiner Begrüßung warm nach seinem Namen, dann wie es ihm geht."

# ─── Data persistence (local, JSON) ───────────────────────────────────────────

def load_history() -> dict:
    if DATA_FILE.exists():
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"sessions": [], "moods": []}

def save_history(data: dict):
    DATA_FILE.parent.mkdir(exist_ok=True)
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def save_mood(data: dict, value: int, note: str = ""):
    entry = {
        "date": datetime.datetime.now().isoformat(),
        "value": value,
        "note": note
    }
    data["moods"].append(entry)
    save_history(data)

# ─── UI helpers ───────────────────────────────────────────────────────────────

def show_welcome(has_memory: bool):
    mode_label = (
        f"[yellow]Groq API[/yellow] – Modell: {GROQ_MODEL}"
        if MODE == "groq"
        else f"[green]Lokal (Ollama)[/green] – Modell: {OLLAMA_MODEL}"
    )
    memory_label = "[green]✓ Gedächtnis aktiv[/green]" if has_memory else "[dim]Erste Sitzung[/dim]"
    console.print(Panel.fit(
        "[bold cyan]🌿 Miru Mind[/bold cyan]\n"
        "[dim]Dein persönlicher Gesprächspartner für mentale Gesundheit[/dim]\n\n"
        f"Modus: {mode_label}\n"
        f"Gedächtnis: {memory_label}\n\n"
        "Befehle:\n"
        "  [yellow]/stimmung[/yellow]  – Stimmung eintragen (1–10)\n"
        "  [yellow]/verlauf[/yellow]   – Stimmungsverlauf anzeigen\n"
        "  [yellow]/neu[/yellow]       – Neue Sitzung starten\n"
        "  [yellow]/beenden[/yellow]   – Beenden",
        border_style="cyan"
    ))

def show_mood_history(data: dict):
    entries = data.get("moods", [])
    if not entries:
        console.print("[dim]Noch keine Stimmungseinträge.[/dim]")
        return
    console.print("\n[bold]📊 Stimmungsverlauf:[/bold]")
    for entry in entries[-10:]:
        date = entry["date"][:10]
        bar = "█" * entry["value"] + "░" * (10 - entry["value"])
        note = f"  [dim]{entry['note']}[/dim]" if entry.get("note") else ""
        color = "green" if entry["value"] >= 7 else "yellow" if entry["value"] >= 4 else "red"
        console.print(f"  {date}  [{color}]{bar}[/{color}] {entry['value']}/10{note}")
    console.print()

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    data = load_history()

    has_memory = bool(data.get("sessions") or data.get("moods") or data.get("user_profile"))
    system_prompt = build_system_prompt(data)

    show_welcome(has_memory)

    messages = [{"role": "system", "content": system_prompt}]

    console.print("\n[dim]Miru denkt...[/dim]", end="\r")
    greeting = get_chat_response(messages)
    messages.append({"role": "assistant", "content": greeting})
    console.print(Panel(Markdown(greeting), title="🌿 Miru", border_style="cyan"))

    session_messages = []

    while True:
        try:
            user_input = Prompt.ask("\n[bold green]Du[/bold green]").strip()
        except (KeyboardInterrupt, EOFError):
            user_input = "/beenden"

        if not user_input:
            continue

        if user_input == "/beenden":
            finalize_session(data, session_messages)
            console.print("\n[cyan]Sitzung gespeichert. Auf Wiedersehen! 🌿[/cyan]")
            break

        elif user_input == "/neu":
            finalize_session(data, session_messages)
            data = load_history()
            system_prompt = build_system_prompt(data)
            messages = [{"role": "system", "content": system_prompt}]
            session_messages = []
            console.print("[dim]Neue Sitzung gestartet.[/dim]")
            continue

        elif user_input == "/verlauf":
            show_mood_history(data)
            continue

        elif user_input == "/stimmung":
            try:
                value = int(Prompt.ask("Wie fühlst du dich? [bold](1–10)[/bold]"))
                value = max(1, min(10, value))
                note = Prompt.ask("Kurze Notiz (optional)", default="")
                save_mood(data, value, note)
                console.print(f"[green]✓ Stimmung ({value}/10) gespeichert.[/green]")
            except ValueError:
                console.print("[red]Bitte eine Zahl zwischen 1 und 10 eingeben.[/red]")
            continue

        messages.append({"role": "user", "content": user_input})
        session_messages.append({"role": "user", "content": user_input})

        console.print("[dim]Miru denkt...[/dim]", end="\r")
        response = get_chat_response(messages)

        messages.append({"role": "assistant", "content": response})
        session_messages.append({"role": "assistant", "content": response})

        console.print(Panel(Markdown(response), title="🌿 Miru", border_style="cyan"))


if __name__ == "__main__":
    main()
