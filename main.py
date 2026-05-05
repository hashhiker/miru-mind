"""
Gspänli – Dein privater Begleiter
====================================
Phase 1: Groq API (powerful model, fast iteration)
Phase 2: Ollama local (privacy-first, on-device, kein Dritter)

Setup:
  1. pip install -r requirements.txt
  2. .env Datei erstellen mit GROQ_API_KEY=dein_key
     → Kostenloser Account: https://console.groq.com
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
    print("Bitte zuerst Abhängigkeiten installieren: pip install -r requirements.txt")
    exit(1)

# ─── Konfiguration ────────────────────────────────────────────────────────────

# Modus wechseln:
# "groq"  → Groq API (Phase 1, starkes Modell, braucht Internet)
# "local" → Ollama lokal (Phase 2, alles bleibt auf dem Gerät)
MODE = "groq"

# Groq Modell wählen:
#
#   "llama-3.3-70b-versatile"  → Beste Qualität, Produktion       (Standard)
#   "llama-3.1-8b-instant"     → Kleiner, schneller – guter Mittelweg
#   "llama-3.2-3b-preview"     → Nächste Approximation an Apple on-device (~3B)
#   "llama-3.2-1b-preview"     → Kleinst möglich – zeigt harte Grenzen
#
# Tipp: Qualität messen nach Wechsel:
#   python tests/eval_chat.py
GROQ_MODEL = "llama-3.3-70b-versatile"

# Ollama
OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_HOST  = "http://localhost:11434"

# Gedächtnis
RECENT_SESSIONS_COUNT = 3
CHECKIN_TREND_DAYS    = 14   # Für Mustererkennung: 2 Wochen
MIN_CHECKINS_FOR_PATTERNS = 5  # Erst ab 5 Check-ins werden Muster analysiert

DATA_FILE = Path(__file__).parent / "data" / "history.json"
console = Console()

# ─── Client Initialisierung ───────────────────────────────────────────────────

if MODE == "groq":
    try:
        from groq import Groq
        groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    except ImportError:
        print("Groq nicht installiert: pip install groq")
        exit(1)
    except Exception as e:
        print(f"Groq Fehler: {e}")
        exit(1)
elif MODE == "local":
    try:
        import ollama
        ollama_client = ollama.Client(host=OLLAMA_HOST)
    except ImportError:
        print("Ollama nicht installiert: pip install ollama")
        exit(1)

# ─── System Prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT_BASE = """Du bist Gspänli – ein privater Begleiter für die Momente wo man einfach reden muss.

# WAS GSPÄNLI IST – UND WAS NICHT
Du bist der Raum zwischen "mir geht's okay" und "ich brauche Hilfe".
Viele Menschen funktionieren gut nach aussen, aber innen stimmt etwas nicht – und sie wissen noch nicht mal genau was.
Du hilfst ihnen das zu verstehen. Nicht durch Ratschläge. Durch Zuhören, Nachfragen, und indem du Dinge zurückspiegelst die sie selbst nicht sehen.

Du bist kein Therapeut, kein Coach, kein Problemlöser.
Du bist das Gspänli das zuhört bevor man weiss was man eigentlich sagen will.

# DEINE HALTUNG
Geerdet, geduldig, direkt – nie wertend, nie aufgeregt, nie übertrieben empathisch.
Du reagierst nicht auf das Problem, sondern auf den Menschen dahinter.
Du gibst keine Ratschläge ausser du wirst explizit darum gebeten.
Du erzeugst keinen Leistungsdruck – kein "du machst Fortschritte", keine Vergleiche mit gestern,
keine Bewertung ob jemand "richtig" oder "falsch" fühlt.
Manchmal reicht es einfach da zu sein.

# SPRACHE
Kurze Sätze. Kein Therapieschema, keine Fachbegriffe, keine Worthülsen.
Deutsch wie man es wirklich spricht – direkt, manchmal ein Fragment, manchmal nur ein Wort.
Variiere den Einstieg. Nicht jede Antwort gleich beginnen.
Emotionen direkt benennen ohne sie zu dramatisieren.

VERBOTEN:
- "Das klingt wirklich herausfordernd" und ähnliche Therapie-Floskeln
- Mehrere Fragen hintereinander
- Lange Erklärungen oder Zusammenfassungen was der Nutzer gerade gesagt hat
- Übertriebene Empathie die unecht klingt ("Oh, das tut mir so leid")

# EINE FRAGE – STRIKTE REGEL
Deine Antwort enthält exakt ein Fragezeichen. Nie zwei, nie null.
Wähle die eine Frage die wirklich wichtig ist – nicht die sichere, die naheliegende.
"Ist es X oder Y?" sind zwei Fragen – verboten.

Antworte kurz: 2–4 Sätze. Weniger ist oft mehr.

# SO KLINGT GSPÄNLI – BEISPIELE

Nutzer: "Ich bin so gestresst von der Arbeit."
Gspänli: "Arbeitsstress der einfach nicht aufhört – das zehrt.
Was macht dir gerade am meisten zu schaffen?"

Nutzer: "Ich weiss nicht, ich fühle mich irgendwie leer."
Gspänli: "Leer ist schwer zu greifen – von aussen sieht man's nicht mal.
Seit wann ist das so?"

Nutzer: "Heute war wieder alles zu viel."
Gspänli: "Klingt wie ein zäher Tag. Was hat am meisten Kraft gekostet?"

Nutzer: "Ich glaube ich bin einfach zu schwach für alles."
Gspänli: "Das sagst du über dich. Ich höre wie viel du gerade trägst.
Was bringt dich dazu, so über dich zu denken?"

Nutzer: "Bin einfach müde."
Gspänli: "Nur körperlich – oder tiefer?"

Nutzer: "Weiss nicht mal warum ich hier schreibe."
Gspänli: "Manchmal reicht das schon. Was ist gerade los bei dir?"

# DER SPIEGEL – KERN VON GSPÄNLI
Du erinnerst dich. Über Wochen. Das ist dein eigentlicher Wert.
Wenn du aus dem Gedächtnis Muster erkennst, sprich sie im richtigen Moment an –
beiläufig, nie als Report, nie als Liste. Nur wenn es sich natürlich ergibt.

Beispiele wie das klingt:
- "Letzte Woche nach dem Sport hast du besser geschlafen – ist das noch so?"
- "Du hast schon ein paarmal erwähnt dass Montage schwer sind. Heute auch?"
- "Wenn ich das so höre – du trägst das schon eine Weile mit dir rum."

Muster benennen ohne zu diagnostizieren. Beobachten ohne zu urteilen.

# CHECK-IN – BEILÄUFIG, NIE ALS FORMULAR
Gegen Ende eines Gesprächs oder wenn jemand nur kurz vorbeischaut,
frag nach einem einzigen Faktor – nie mehrere auf einmal:
- "Wie hast du geschlafen?"
- "Warst du heute draussen?"
Kein Pflichtprogramm. Nur wenn es passt.

# BEGRÜSSUNG
VERBOTEN: "Hallo wieder" / "Willkommen zurück" / "Schön dich zu sehen" /
          "Wie war dein Tag?" als Einstieg bei bekannten Personen /
          jede wörtliche Übersetzung englischer Phrasen

ERSTE SITZUNG:
→ Kurz vorstellen, nach dem Namen fragen – natürlich, nicht wie ein Formular.
   "Hoi. Ich bin Gspänli – ein ruhiger Ort zum Reden, ganz für dich. Wie heisst du?"

BEKANNTE PERSON:
→ Direkt einsteigen. Warm, kurz, echt.
   "Hoi [Name]. Wie geht's dir gerade?"
   "Hey – was bringt dich heute her?"
   "Hoi. Was ist los?"

HEUTE BEREITS DAGEWESEN:
→ Anknüpfen, nicht neu anfangen.
   "Hoi nochmal. Was hat sich seither getan?"
   "Du bist nochmal da – was liegt dir noch auf dem Herzen?"

In frühen Gesprächen nach und nach kennenlernen (nie als Liste abfragen):
Name, Lebensphase, Arbeit, Wohnsituation.
Namen gelegentlich benutzen – aber natürlich, nicht bei jedem Satz.

# TECHNIKEN – NUR WENN PASSEND, NIE AUFDRÄNGEN
- Inneren Kritiker benennen: "Ich höre wie hart du mit dir bist."
- Perspektivwechsel: "Was würdest du jemandem sagen, dem genau das passiert?"
- Innehalten: "Magst du kurz durchatmen? Manchmal hilft das."

# SICHERHEIT – HÖCHSTE PRIORITÄT
Bei Krisenzeichen (Suizidgedanken, Selbstverletzung, Hoffnungslosigkeit):
→ Ruhig bleiben. Ernst nehmen. Nicht lösen wollen.
→ IMMER sofort nennen:
   143 (CH – Die Dargebotene Hand, kostenlos, 24/7, anonym)
   0800 111 0 111 (DE/AT – kostenlos, 24/7, anonym)
→ Danach fragen: "Bist du gerade in Sicherheit?"
→ Kein Themenwechsel bis du eine Antwort hast.

# GRENZEN
Kein Therapeut. Bei Diagnosen, Medikamenten, professioneller Hilfe:
"Dafür bin ich nicht die richtige Anlaufstelle – aber ich kann dir helfen den ersten Schritt zu machen."

Sprache: Immer Deutsch."""

SUMMARIZATION_PROMPT = """Du analysierst eine abgeschlossene Gesprächssitzung und erstellst eine kompakte Zusammenfassung.

Antworte NUR mit validem JSON (keine weiteren Texte):
{
  "summary": "1-2 Sätze über das Gespräch und die emotionale Lage",
  "themes": ["Thema1", "Thema2"],
  "key_facts": ["Stabiler Fakt über den Nutzer", "Weiterer Fakt"],
  "mood_observed": 5,
  "lifestyle_signals": {
    "sleep": "gut|mittel|schlecht|null",
    "exercise": true
  }
}

mood_observed: geschätzte Stimmung am Ende (1-10).
key_facts: stabile Fakten (Name, Alter, Lebenssituation, Arbeit, wiederkehrende Themen) – keine Gesprächsinhalte.
lifestyle_signals.sleep: nur setzen wenn Schlaf explizit im Gespräch erwähnt wurde, sonst null.
lifestyle_signals.exercise: nur setzen wenn Sport/Bewegung erwähnt wurde, sonst null."""

USER_PROFILE_UPDATE_PROMPT = """Du pflegst das Kurzprofil eines Nutzers für Gspänli, einen privaten Begleiter.

Bisheriges Profil und neue Sitzungsinfos werden dir gegeben.
Schreibe ein aktualisiertes Profil als kurzen Absatz (max. 80 Wörter) auf Deutsch.
Behalte wichtige stabile Fakten, lass Veraltetes weg.
Keine Gesprächsinhalte – nur Persönlichkeit und Lebenssituation.
Antworte NUR mit dem Profiltext, ohne Formatierung."""

# ─── LLM Core ────────────────────────────────────────────────────────────────

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

# ─── Gedächtnis & Mustererkennung ────────────────────────────────────────────

def _parse_json_response(raw: str) -> dict | None:
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

def analyze_patterns(data: dict) -> str:
    """
    Analysiert Check-in Daten der letzten CHECKIN_TREND_DAYS Tage
    und gibt Muster als Text zurück der in den System-Prompt injiziert wird.
    Mindestens MIN_CHECKINS_FOR_PATTERNS Check-ins nötig.
    """
    checkins = data.get("checkins", [])
    if len(checkins) < MIN_CHECKINS_FOR_PATTERNS:
        return ""

    cutoff = datetime.datetime.now() - datetime.timedelta(days=CHECKIN_TREND_DAYS)
    recent = [
        c for c in checkins
        if datetime.datetime.fromisoformat(c["date"]) > cutoff
    ]
    if len(recent) < MIN_CHECKINS_FOR_PATTERNS:
        return ""

    insights = []

    # ── Schlaf → Stimmung ──
    sleep_mood: dict[str, list[int]] = {"gut": [], "mittel": [], "schlecht": []}
    for c in recent:
        s = c.get("sleep")
        m = c.get("mood")
        if s in sleep_mood and m:
            sleep_mood[s].append(m)

    if sleep_mood["gut"] and sleep_mood["schlecht"]:
        avg_good = sum(sleep_mood["gut"]) / len(sleep_mood["gut"])
        avg_bad  = sum(sleep_mood["schlecht"]) / len(sleep_mood["schlecht"])
        if avg_good - avg_bad >= 1.5:
            insights.append(
                f"Schlaf hängt stark mit Stimmung zusammen: "
                f"nach guten Nächten Stimmung Ø {avg_good:.1f}/10, "
                f"nach schlechten Ø {avg_bad:.1f}/10."
            )

    # ── Sport → Stimmung am Folgetag ──
    after_exercise = []
    after_rest     = []
    for i in range(1, len(recent)):
        mood_next = recent[i].get("mood")
        if mood_next is None:
            continue
        if recent[i - 1].get("exercise"):
            after_exercise.append(mood_next)
        else:
            after_rest.append(mood_next)

    if len(after_exercise) >= 3 and after_rest:
        avg_ex   = sum(after_exercise) / len(after_exercise)
        avg_rest = sum(after_rest) / len(after_rest)
        if avg_ex - avg_rest >= 1.0:
            insights.append(
                f"Nach Sporttagen ist die Stimmung am Folgetag "
                f"durchschnittlich {avg_ex - avg_rest:.1f} Punkte höher."
            )

    # ── Wochentag-Muster ──
    weekday_moods: dict[int, list[int]] = {}
    for c in recent:
        wd = datetime.datetime.fromisoformat(c["date"]).weekday()
        m  = c.get("mood")
        if m:
            weekday_moods.setdefault(wd, []).append(m)

    if len(weekday_moods) >= 5:
        avgs = {wd: sum(ms) / len(ms) for wd, ms in weekday_moods.items() if len(ms) >= 2}
        if avgs:
            worst_wd = min(avgs, key=avgs.get)
            best_wd  = max(avgs, key=avgs.get)
            days_de  = ["Montage", "Dienstage", "Mittwoche", "Donnerstage",
                         "Freitage", "Samstage", "Sonntage"]
            if avgs[best_wd] - avgs[worst_wd] >= 2.0:
                insights.append(
                    f"Wochenmuster: {days_de[best_wd]} laufen tendenziell besser "
                    f"(Ø {avgs[best_wd]:.1f}/10), {days_de[worst_wd]} schwerer "
                    f"(Ø {avgs[worst_wd]:.1f}/10)."
                )

    if not insights:
        return ""

    return (
        "ERKANNTE MUSTER (aus Check-in Daten der letzten 2 Wochen):\n"
        + "\n".join(f"- {i}" for i in insights)
        + "\nSprich diese Muster beiläufig im Gespräch an wenn es sich natürlich ergibt – "
          "nie als Liste oder Report."
    )

def build_memory_context(data: dict) -> str:
    parts = []

    # ── Nutzerprofil ──
    user_profile = data.get("user_profile", "")
    if user_profile:
        parts.append(f"NUTZERPROFIL:\n{user_profile}")

    # ── Check-in Trend ──
    checkins = data.get("checkins", [])
    if checkins:
        cutoff = datetime.datetime.now() - datetime.timedelta(days=CHECKIN_TREND_DAYS)
        recent = [
            c for c in checkins
            if datetime.datetime.fromisoformat(c["date"]) > cutoff
        ]
        if recent:
            moods    = [c["mood"] for c in recent if c.get("mood")]
            avg_mood = sum(moods) / len(moods) if moods else None
            trend    = ""
            if len(moods) >= 2:
                trend = (
                    " (steigend)" if moods[-1] > moods[0]
                    else " (fallend)" if moods[-1] < moods[0]
                    else " (stabil)"
                )
            sleep_counts = {"gut": 0, "mittel": 0, "schlecht": 0}
            for c in recent:
                if c.get("sleep") in sleep_counts:
                    sleep_counts[c["sleep"]] += 1
            sport_days = sum(1 for c in recent if c.get("exercise"))

            lines = []
            if avg_mood:
                lines.append(f"- Stimmung Ø: {avg_mood:.1f}/10{trend}")
            if any(sleep_counts.values()):
                lines.append(
                    f"- Schlaf: {sleep_counts['gut']}× gut / "
                    f"{sleep_counts['mittel']}× mittel / "
                    f"{sleep_counts['schlecht']}× schlecht"
                )
            if sport_days:
                lines.append(f"- Sport: {sport_days} von {len(recent)} Tagen")

            if lines:
                parts.append(
                    f"CHECK-IN TREND (letzte {CHECKIN_TREND_DAYS} Tage):\n"
                    + "\n".join(lines)
                )

    # ── Erkannte Muster ──
    patterns = analyze_patterns(data)
    if patterns:
        parts.append(patterns)

    # ── Vergangene Gespräche ──
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
        "\n\nGEDÄCHTNIS – Was du über diese Person weißt:\n"
        + "\n\n".join(parts)
        + "\n\nNutze dieses Wissen für Kontinuität. Beziehe dich natürlich darauf, "
          "ohne es aufzulisten. Namen gelegentlich benutzen. Beginne das Gespräch warm."
    )

def build_system_prompt(data: dict) -> str:
    context = build_memory_context(data)
    if context:
        return SYSTEM_PROMPT_BASE + context
    return (
        SYSTEM_PROMPT_BASE
        + "\n\nErste Sitzung – du kennst diese Person noch nicht. "
          "Stelle dich kurz als Gspänli vor und frage warm nach dem Namen."
    )

# ─── Session Zusammenfassung ─────────────────────────────────────────────────

def summarize_session(session_messages: list[dict]) -> dict | None:
    if not session_messages:
        return None
    transcript = "\n".join(
        f"{'Nutzer' if m['role'] == 'user' else 'Gspänli'}: {m['content']}"
        for m in session_messages
    )
    try:
        raw = _call_llm(
            [
                {"role": "system", "content": SUMMARIZATION_PROMPT},
                {"role": "user", "content": f"Sitzungsinhalt:\n{transcript}"}
            ],
            max_tokens=400,
            temperature=0.2
        )
        return _parse_json_response(raw)
    except Exception:
        return None

def update_user_profile(data: dict, summary: dict) -> str:
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
    if not session_messages:
        return
    console.print("[dim]Gespräch wird zusammengefasst...[/dim]", end="\r")
    summary = summarize_session(session_messages)

    session_entry: dict = {
        "date": datetime.datetime.now().isoformat(),
        "messages": session_messages,
    }
    if summary:
        session_entry["summary"]  = summary.get("summary", "")
        session_entry["themes"]   = summary.get("themes", [])
        session_entry["key_facts"] = summary.get("key_facts", [])
        if summary.get("mood_observed") is not None:
            session_entry["mood_observed"] = summary["mood_observed"]

        # Lifestyle-Signale aus dem Gespräch als Check-in speichern
        signals = summary.get("lifestyle_signals", {})
        if signals.get("sleep") or signals.get("exercise") is not None:
            implicit_checkin = {
                "date": datetime.datetime.now().isoformat(),
                "mood": summary.get("mood_observed"),
                "sleep": signals.get("sleep"),
                "exercise": signals.get("exercise"),
                "source": "conversation"   # Unterscheidet von manuellem Check-in
            }
            data.setdefault("checkins", []).append(implicit_checkin)

        data["user_profile"] = update_user_profile(data, summary)

    data.setdefault("sessions", []).append(session_entry)
    save_history(data)

# ─── Datenpersistenz ──────────────────────────────────────────────────────────

def load_history() -> dict:
    if DATA_FILE.exists():
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Migration: altes Format unterstützen
        data.setdefault("sessions", [])
        data.setdefault("moods", [])
        data.setdefault("checkins", [])
        return data
    return {"sessions": [], "moods": [], "checkins": [], "user_profile": ""}

def save_history(data: dict):
    DATA_FILE.parent.mkdir(exist_ok=True)
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def save_checkin(data: dict, mood: int, sleep: str, exercise: bool, note: str = ""):
    """Speichert einen manuellen Check-in (Stimmung + Schlaf + Sport)."""
    entry = {
        "date": datetime.datetime.now().isoformat(),
        "mood": mood,
        "sleep": sleep,        # "gut" / "mittel" / "schlecht"
        "exercise": exercise,  # True / False
        "note": note,
        "source": "manual"
    }
    data.setdefault("checkins", []).append(entry)
    # Rückwärtskompatibel: auch in moods schreiben
    data.setdefault("moods", []).append({
        "date": entry["date"],
        "value": mood,
        "note": note
    })
    save_history(data)

# ─── Terminal UI ──────────────────────────────────────────────────────────────

def show_welcome(has_memory: bool):
    mode_label = (
        f"[yellow]Groq API[/yellow] – {GROQ_MODEL}"
        if MODE == "groq"
        else f"[green]Lokal (Ollama)[/green] – {OLLAMA_MODEL}"
    )
    memory_label = "[green]✓ Gedächtnis aktiv[/green]" if has_memory else "[dim]Erstes Gespräch[/dim]"
    console.print(Panel.fit(
        "[bold green]🌿 Gspänli[/bold green]\n"
        "[dim]Dein privater Begleiter – alles bleibt bei dir[/dim]\n\n"
        f"Modus: {mode_label}\n"
        f"Gedächtnis: {memory_label}\n\n"
        "Befehle:\n"
        "  [yellow]/checkin[/yellow]   – Stimmung, Schlaf & Sport eintragen\n"
        "  [yellow]/verlauf[/yellow]   – Check-in Verlauf anzeigen\n"
        "  [yellow]/neu[/yellow]       – Neues Gespräch starten\n"
        "  [yellow]/beenden[/yellow]   – Beenden & speichern",
        border_style="green"
    ))

def show_checkin_history(data: dict):
    checkins = data.get("checkins", [])
    if not checkins:
        console.print("[dim]Noch keine Check-ins.[/dim]")
        return
    console.print("\n[bold]📊 Check-in Verlauf:[/bold]")
    sleep_icons = {"gut": "🌙✓", "mittel": "🌙~", "schlecht": "🌙✗", None: "  "}
    for c in checkins[-10:]:
        date     = c["date"][:10]
        mood     = c.get("mood", 0)
        bar      = "█" * mood + "░" * (10 - mood)
        color    = "green" if mood >= 7 else "yellow" if mood >= 4 else "red"
        sleep_ic = sleep_icons.get(c.get("sleep"), "  ")
        sport_ic = "🏃" if c.get("exercise") else "  "
        source   = "[dim](aus Gespräch)[/dim]" if c.get("source") == "conversation" else ""
        console.print(
            f"  {date}  [{color}]{bar}[/{color}] {mood}/10  "
            f"{sleep_ic} {sport_ic} {source}"
        )
    console.print()

def do_checkin(data: dict):
    """Interaktiver Check-in im Terminal."""
    console.print("\n[bold green]── Check-in ──[/bold green]")
    try:
        mood_raw = Prompt.ask("Stimmung [bold](1–10)[/bold]")
        mood = max(1, min(10, int(mood_raw)))

        sleep_raw = Prompt.ask("Schlaf [bold](gut / mittel / schlecht)[/bold]", default="mittel").lower()
        sleep = sleep_raw if sleep_raw in ("gut", "mittel", "schlecht") else "mittel"

        exercise_raw = Prompt.ask("Sport heute [bold](j/n)[/bold]", default="n").lower()
        exercise = exercise_raw in ("j", "ja", "y", "yes")

        note = Prompt.ask("Kurze Notiz (optional)", default="")

        save_checkin(data, mood, sleep, exercise, note)
        sport_txt = "🏃 Sport" if exercise else ""
        console.print(f"[green]✓ Check-in gespeichert: {mood}/10 · Schlaf {sleep} {sport_txt}[/green]")
    except ValueError:
        console.print("[red]Ungültige Eingabe.[/red]")

# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    data = load_history()
    has_memory = bool(
        data.get("sessions") or data.get("checkins") or data.get("user_profile")
    )
    system_prompt = build_system_prompt(data)
    show_welcome(has_memory)

    messages = [{"role": "system", "content": system_prompt}]

    console.print("\n[dim]Gspänli denkt...[/dim]", end="\r")
    greeting = get_chat_response(messages)
    messages.append({"role": "assistant", "content": greeting})
    console.print(Panel(Markdown(greeting), title="🌿 Gspänli", border_style="green"))

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
            console.print("\n[green]Gespräch gespeichert. Tschüss! 🌿[/green]")
            break

        elif user_input == "/neu":
            finalize_session(data, session_messages)
            data = load_history()
            system_prompt = build_system_prompt(data)
            messages = [{"role": "system", "content": system_prompt}]
            session_messages = []
            console.print("[dim]Neues Gespräch gestartet.[/dim]")
            continue

        elif user_input == "/verlauf":
            show_checkin_history(data)
            continue

        elif user_input == "/checkin":
            do_checkin(data)
            continue

        messages.append({"role": "user", "content": user_input})
        session_messages.append({"role": "user", "content": user_input})

        console.print("[dim]Gspänli denkt...[/dim]", end="\r")
        response = get_chat_response(messages)

        messages.append({"role": "assistant", "content": response})
        session_messages.append({"role": "assistant", "content": response})

        console.print(Panel(Markdown(response), title="🌿 Gspänli", border_style="green"))


if __name__ == "__main__":
    main()