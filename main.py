"""
Miru Mind – Lokaler Therapie-Chatbot
======================================
Phase 1: Groq API (starkes Modell, schnelles Iterieren)
Phase 2: Ollama lokal (Privacy-first, on-device)

Setup:
  1. pip install -r requirements.txt
  2. .env Datei erstellen mit GROQ_API_KEY=dein_key
     → Kostenloser Account: https://console.groq.com
  3. python main.py
"""

import json
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
    print("Bitte zuerst installieren: pip install -r requirements.txt")
    exit(1)

# ─── Konfiguration ────────────────────────────────────────────────────────────

# Wechsle hier zwischen den Modi:
# "groq"  → Groq API (Phase 1, starkes Modell, braucht Internet)
# "local" → Ollama lokal (Phase 2, Privacy-first, kein Internet)
MODE = "groq"

# Groq Einstellungen
GROQ_MODEL = "llama-3.3-70b-versatile"

# Ollama Einstellungen
OLLAMA_MODEL = "llama3.2:1b"
OLLAMA_HOST  = "http://localhost:11434"   # oder Mac-IP: "http://192.168.1.X:11434"

# Gedächtnis Einstellungen
LETZTE_SITZUNGEN = 3   # Wie viele vergangene Sitzungen Miru kennt
STIMMUNG_TAGE    = 7   # Stimmungstrend der letzten N Tage

DATA_FILE = Path.home() / ".miruMind" / "verlauf.json"
console = Console()

# ─── Client initialisieren ────────────────────────────────────────────────────

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

# ─── System-Prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT_BASIS = """Du bist Miru – ein ruhiger, warmherziger Begleiter für mentale Gesundheit.

# DEINE PERSÖNLICHKEIT
Du sprichst wie ein guter Freund mit therapeutischer Ausbildung: geerdet, geduldig,
nie wertend. Du benutzt einfache, warme Sprache – keine Fachbegriffe, keine Floskeln.
Du bist neugierig auf den Menschen, nicht auf das Problem.

# GESPRÄCHSFÜHRUNG
Jede Antwort folgt diesem Muster:
1. ANERKENNEN – zeige dass du gehört hast ("Das klingt wirklich erschöpfend...")
2. VERTIEFEN – eine einzige, offene Frage ("Was davon belastet dich am meisten?")
3. RAUM LASSEN – keine Ratschläge bevor du das Bild vollständig verstehst

Antworte kurz (3-5 Sätze). Stelle immer nur eine Frage pro Antwort.

# BEISPIELE – SO KLINGT MIRU

Nutzer: "Ich bin so gestresst von der Arbeit."
Miru: "Das höre ich dir an – Arbeitsstress kann wirklich zermürbend sein, besonders
wenn er sich aufstaut. Was ist es gerade, das dir am meisten zusetzt – die Menge,
bestimmte Situationen, oder etwas anderes?"

Nutzer: "Ich weiß nicht, ich fühle mich einfach leer."
Miru: "Dieses Gefühl der Leere ist schwer zu beschreiben, aber du hast es gerade
getan – und das zählt. Seit wann merkst du das bei dir?"

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

# ─── Gedächtnis ───────────────────────────────────────────────────────────────

def baue_gedaechtnis_kontext(daten: dict) -> str:
    """
    Erstellt einen Kontext-Block aus vergangenen Sitzungen und Stimmungsdaten
    der dann in den System-Prompt eingefügt wird.
    """
    teile = []

    # ── Stimmungstrend der letzten N Tage ──
    stimmungen = daten.get("stimmungen", [])
    if stimmungen:
        grenze = datetime.datetime.now() - datetime.timedelta(days=STIMMUNG_TAGE)
        recent = [
            s for s in stimmungen
            if datetime.datetime.fromisoformat(s["datum"]) > grenze
        ]
        if recent:
            werte = [s["wert"] for s in recent]
            durchschnitt = sum(werte) / len(werte)
            trend = "steigend" if werte[-1] > werte[0] else "fallend" if werte[-1] < werte[0] else "stabil"
            letzte_notizen = [s["notiz"] for s in recent[-3:] if s.get("notiz")]

            stimmung_text = (
                f"STIMMUNGSTREND (letzte {STIMMUNG_TAGE} Tage):\n"
                f"- Durchschnitt: {durchschnitt:.1f}/10 (Trend: {trend})\n"
                f"- Letzte Einträge: {', '.join([str(w) for w in werte[-5:]])}/10"
            )
            if letzte_notizen:
                stimmung_text += f"\n- Notizen: {'; '.join(letzte_notizen)}"
            teile.append(stimmung_text)

    # ── Letzte N Sitzungen zusammenfassen ──
    sitzungen = daten.get("sitzungen", [])
    if sitzungen:
        letzte = sitzungen[-LETZTE_SITZUNGEN:]
        sitzungs_texte = []

        for s in letzte:
            datum = s["datum"][:10]
            nachrichten = s.get("nachrichten", [])
            # Nur User-Nachrichten extrahieren für kompakten Überblick
            user_msgs = [
                m["content"] for m in nachrichten
                if m["role"] == "user"
            ]
            if user_msgs:
                # Erste und letzte Nachricht als Zusammenfassung
                vorschau = user_msgs[0][:120]
                sitzungs_texte.append(f"- {datum}: \"{vorschau}...\"")

        if sitzungs_texte:
            teile.append(
                f"VERGANGENE GESPRÄCHE (letzte {len(letzte)} Sitzungen):\n"
                + "\n".join(sitzungs_texte)
            )

    if not teile:
        return ""

    return (
        "\n\nGEDÄCHTNIS – Was du über den Nutzer weißt:\n"
        + "\n\n".join(teile)
        + "\n\nNutze dieses Wissen um Kontinuität zu zeigen. "
        "Beziehe dich natürlich darauf, ohne es aufzulisten. "
        "Beginne das Gespräch warm und frage wie es dem Nutzer heute geht."
    )

def baue_system_prompt(daten: dict) -> str:
    """Kombiniert Basis-Prompt mit Gedächtnis-Kontext."""
    kontext = baue_gedaechtnis_kontext(daten)
    if kontext:
        return SYSTEM_PROMPT_BASIS + kontext
    else:
        return SYSTEM_PROMPT_BASIS + "\n\nBeginne das Gespräch warm und frage wie es dem Nutzer heute geht."

# ─── Datenspeicherung (lokal, JSON) ───────────────────────────────────────────

def lade_verlauf() -> dict:
    if DATA_FILE.exists():
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"sitzungen": [], "stimmungen": []}

def speichere_verlauf(daten: dict):
    DATA_FILE.parent.mkdir(exist_ok=True)
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(daten, f, ensure_ascii=False, indent=2)

def stimmung_speichern(daten: dict, wert: int, notiz: str = ""):
    eintrag = {
        "datum": datetime.datetime.now().isoformat(),
        "wert": wert,
        "notiz": notiz
    }
    daten["stimmungen"].append(eintrag)
    speichere_verlauf(daten)

# ─── Chat-Logik ───────────────────────────────────────────────────────────────

def chat_antwort(nachrichten: list[dict]) -> str:
    try:
        if MODE == "groq":
            antwort = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=nachrichten,
                max_tokens=300,
                temperature=0.7
            )
            return antwort.choices[0].message.content

        elif MODE == "local":
            antwort = ollama_client.chat(
                model=OLLAMA_MODEL,
                messages=nachrichten,
                options={"temperature": 0.7, "num_predict": 300}
            )
            return antwort["message"]["content"]

    except Exception as e:
        return f"[Fehler: {e}]"

# ─── UI Hilfsfunktionen ───────────────────────────────────────────────────────

def zeige_willkommen(hat_gedaechtnis: bool):
    modus_label = (
        f"[yellow]Groq API[/yellow] – Modell: {GROQ_MODEL}"
        if MODE == "groq"
        else f"[green]Lokal (Ollama)[/green] – Modell: {OLLAMA_MODEL}"
    )
    gedaechtnis_label = "[green]✓ Gedächtnis aktiv[/green]" if hat_gedaechtnis else "[dim]Erste Sitzung[/dim]"
    console.print(Panel.fit(
        "[bold cyan]🌿 Miru Mind[/bold cyan]\n"
        "[dim]Dein persönlicher Gesprächspartner für mentale Gesundheit[/dim]\n\n"
        f"Modus: {modus_label}\n"
        f"Gedächtnis: {gedaechtnis_label}\n\n"
        "Befehle:\n"
        "  [yellow]/stimmung[/yellow]  – Stimmung eintragen (1–10)\n"
        "  [yellow]/verlauf[/yellow]   – Stimmungsverlauf anzeigen\n"
        "  [yellow]/neu[/yellow]       – Neue Sitzung starten\n"
        "  [yellow]/beenden[/yellow]   – Beenden",
        border_style="cyan"
    ))

def zeige_stimmungsverlauf(daten: dict):
    eintraege = daten.get("stimmungen", [])
    if not eintraege:
        console.print("[dim]Noch keine Stimmungseinträge.[/dim]")
        return
    console.print("\n[bold]📊 Stimmungsverlauf:[/bold]")
    for e in eintraege[-10:]:
        datum = e["datum"][:10]
        balken = "█" * e["wert"] + "░" * (10 - e["wert"])
        notiz = f"  [dim]{e['notiz']}[/dim]" if e.get("notiz") else ""
        farbe = "green" if e["wert"] >= 7 else "yellow" if e["wert"] >= 4 else "red"
        console.print(f"  {datum}  [{farbe}]{balken}[/{farbe}] {e['wert']}/10{notiz}")
    console.print()

# ─── Hauptprogramm ────────────────────────────────────────────────────────────

def main():
    daten = lade_verlauf()

    # Gedächtnis aufbauen
    hat_gedaechtnis = bool(daten.get("sitzungen") or daten.get("stimmungen"))
    system_prompt = baue_system_prompt(daten)

    zeige_willkommen(hat_gedaechtnis)

    nachrichten = [{"role": "system", "content": system_prompt}]

    console.print("\n[dim]Miru denkt...[/dim]", end="\r")
    begruessung = chat_antwort(nachrichten)
    nachrichten.append({"role": "assistant", "content": begruessung})
    console.print(Panel(Markdown(begruessung), title="🌿 Miru", border_style="cyan"))

    sitzung_nachrichten = []

    while True:
        try:
            eingabe = Prompt.ask("\n[bold green]Du[/bold green]").strip()
        except (KeyboardInterrupt, EOFError):
            eingabe = "/beenden"

        if not eingabe:
            continue

        if eingabe == "/beenden":
            if sitzung_nachrichten:
                daten["sitzungen"].append({
                    "datum": datetime.datetime.now().isoformat(),
                    "nachrichten": sitzung_nachrichten
                })
                speichere_verlauf(daten)
            console.print("\n[cyan]Sitzung gespeichert. Auf Wiedersehen! 🌿[/cyan]")
            break

        elif eingabe == "/neu":
            # Gedächtnis beim Neustart aktualisieren
            daten = lade_verlauf()
            system_prompt = baue_system_prompt(daten)
            nachrichten = [{"role": "system", "content": system_prompt}]
            sitzung_nachrichten = []
            console.print("[dim]Neue Sitzung gestartet.[/dim]")
            continue

        elif eingabe == "/verlauf":
            zeige_stimmungsverlauf(daten)
            continue

        elif eingabe == "/stimmung":
            try:
                wert = int(Prompt.ask("Wie fühlst du dich? [bold](1–10)[/bold]"))
                wert = max(1, min(10, wert))
                notiz = Prompt.ask("Kurze Notiz (optional)", default="")
                stimmung_speichern(daten, wert, notiz)
                console.print(f"[green]✓ Stimmung ({wert}/10) gespeichert.[/green]")
            except ValueError:
                console.print("[red]Bitte eine Zahl zwischen 1 und 10 eingeben.[/red]")
            continue

        nachrichten.append({"role": "user", "content": eingabe})
        sitzung_nachrichten.append({"role": "user", "content": eingabe})

        console.print("[dim]Miru denkt...[/dim]", end="\r")
        antwort = chat_antwort(nachrichten)

        nachrichten.append({"role": "assistant", "content": antwort})
        sitzung_nachrichten.append({"role": "assistant", "content": antwort})

        console.print(Panel(Markdown(antwort), title="🌿 Miru", border_style="cyan"))


if __name__ == "__main__":
    main()