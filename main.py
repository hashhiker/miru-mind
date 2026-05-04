"""
TherapieBot – Lokaler Prototyp
===============================
Läuft komplett lokal via Ollama. Keine Daten verlassen dein Gerät.

Setup:
  1. Ollama installieren: https://ollama.com
  2. Modell laden:        ollama pull llama3.2
  3. Dependencies:        pip install ollama rich
  4. Starten:             python therapiebot.py
"""

import json
import datetime
import os
from pathlib import Path

try:
    import ollama
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.markdown import Markdown
except ImportError:
    print("Bitte zuerst installieren: pip install ollama rich")
    exit(1)

# ─── Konfiguration ────────────────────────────────────────────────────────────

MODEL = "llama3.2"          # Alternativ: "phi3" oder "gemma3:1b"
DATA_FILE = Path.home() / ".therapiebot" / "verlauf.json"
console = Console()

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
    """Lädt den gespeicherten Verlauf vom lokalen Gerät."""
    if DATA_FILE.exists():
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"sitzungen": [], "stimmungen": []}

def speichere_verlauf(daten: dict):
    """Speichert den Verlauf lokal."""
    DATA_FILE.parent.mkdir(exist_ok=True)
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(daten, f, ensure_ascii=False, indent=2)

def stimmung_speichern(daten: dict, wert: int, notiz: str = ""):
    """Speichert einen Stimmungseintrag."""
    eintrag = {
        "datum": datetime.datetime.now().isoformat(),
        "wert": wert,        # 1–10
        "notiz": notiz
    }
    daten["stimmungen"].append(eintrag)
    speichere_verlauf(daten)

# ─── Chat-Logik ───────────────────────────────────────────────────────────────

def chat_antwort(nachrichten: list[dict]) -> str:
    """Sendet Nachrichten an das lokale Ollama-Modell und gibt die Antwort zurück."""
    try:
        antwort = ollama.chat(
            model=MODEL,
            messages=nachrichten,
            options={"temperature": 0.7, "num_predict": 300}
        )
        return antwort["message"]["content"]
    except Exception as e:
        return f"[Fehler: Ollama nicht erreichbar – läuft es? Starte mit: ollama serve]\n{e}"

# ─── UI Hilfsfunktionen ───────────────────────────────────────────────────────

def zeige_willkommen():
    console.print(Panel.fit(
        "[bold cyan]🌿 TherapieBot[/bold cyan]\n"
        "[dim]Läuft lokal – keine Daten verlassen dein Gerät[/dim]\n\n"
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
    for e in eintraege[-10:]:  # Letzte 10 anzeigen
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

    # Sitzung vorbereiten
    nachrichten = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Begrüßung vom Bot
    console.print("\n[dim]Bot denkt...[/dim]", end="\r")
    begruessung = chat_antwort(nachrichten)
    nachrichten.append({"role": "assistant", "content": begruessung})
    console.print(Panel(Markdown(begruessung), title="🌿 Bot", border_style="cyan"))

    sitzung_nachrichten = []

    while True:
        try:
            eingabe = Prompt.ask("\n[bold green]Du[/bold green]").strip()
        except (KeyboardInterrupt, EOFError):
            eingabe = "/beenden"

        if not eingabe:
            continue

        # ── Befehle ──
        if eingabe == "/beenden":
            # Sitzung speichern
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

        # ── Normale Nachricht ──
        nachrichten.append({"role": "user", "content": eingabe})
        sitzung_nachrichten.append({"role": "user", "content": eingabe})

        console.print("[dim]Bot denkt...[/dim]", end="\r")
        antwort = chat_antwort(nachrichten)

        nachrichten.append({"role": "assistant", "content": antwort})
        sitzung_nachrichten.append({"role": "assistant", "content": antwort})

        console.print(Panel(Markdown(antwort), title="🌿 Bot", border_style="cyan"))


if __name__ == "__main__":
    main()