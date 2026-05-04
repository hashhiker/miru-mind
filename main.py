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
GROQ_MODEL = "llama-3.1-70b-versatile"

# Ollama Einstellungen
OLLAMA_MODEL = "llama3.2:1b"
OLLAMA_HOST  = "http://localhost:11434"   # oder Mac-IP: "http://192.168.1.X:11434"

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

SYSTEM_PROMPT = """Du bist ein einfühlsamer, unterstützender Gesprächspartner für mentale Gesundheit.

WICHTIGE REGELN:
- Du bist KEIN Ersatz für professionelle Therapie. Weise bei Bedarf darauf hin.
- Stelle keine Diagnosen.
- Antworte immer auf Deutsch, ruhig und empathisch.
- Frage nach, wenn du etwas nicht verstehst.
- Nutze aktives Zuhören: fasse zusammen, was der Nutzer gesagt hat.
- Halte Antworten kurz (3-5 Sätze), außer der Nutzer fragt nach mehr.

BEI KRISENZEICHEN (Suizidgedanken, Selbstverletzung):
- Nimm es ernst und bleib ruhig.
- Sage IMMER: "Bitte wende dich jetzt an die Telefonseelsorge: 0800 111 0 111 (kostenlos, 24/7)"
- Versuche nicht, die Krise selbst zu lösen.

DEINE ROLLE:
- Aktives Zuhören und Reflexion
- Sanfte Fragen stellen, um Gedanken zu ordnen
- Ermutigung und Validierung von Gefühlen
- Bei Bedarf: einfache CBT-Techniken vorschlagen (Atemübungen, Gedankenmuster erkennen)

Beginne das Gespräch warm und frage wie es dem Nutzer heute geht."""

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

def zeige_willkommen():
    modus_label = (
        f"[yellow]Groq API[/yellow] – Modell: {GROQ_MODEL}"
        if MODE == "groq"
        else f"[green]Lokal (Ollama)[/green] – Modell: {OLLAMA_MODEL}"
    )
    console.print(Panel.fit(
        "[bold cyan]🌿 Miru Mind[/bold cyan]\n"
        "[dim]Dein persönlicher Gesprächspartner für mentale Gesundheit[/dim]\n\n"
        f"Modus: {modus_label}\n\n"
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
    zeige_willkommen()
    daten = lade_verlauf()

    nachrichten = [{"role": "system", "content": SYSTEM_PROMPT}]

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
            nachrichten = [{"role": "system", "content": SYSTEM_PROMPT}]
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