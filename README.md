# ParkiScan Neuroverse

**A gamified, multimodal movement-screening and rehabilitation-support platform for people with Parkinson's, built to be accessible to people with disabilities.**

> **Not a medical device.** ParkiScan Neuroverse is a research prototype and screening aid. It does not diagnose, predict or treat any condition. Every number it produces describes how a game task was performed. Nothing here is clinically validated.

## Problem statement

Parkinson's disease is usually detected late and monitored only in occasional clinic visits, while exercise-based rehabilitation is hard to keep up at home. Existing tests are clinic-bound, repetitive and not designed for people with tremor, limited mobility or low confidence with technology.

ParkiScan Neuroverse turns assessment into an accessible game. A participant plays short tasks (eye tracking, voice, memory, spiral drawing, reaction, gait), seated or standing, while a webcam, microphone and two low-cost wearable motion sensors record movement features. A local service turns these into a per-session summary that a clinician or caregiver can review and compare over time.

## Architecture
![Uploading image.png…]()

![ParkiScan Neuroverse architecture](docs/architecture.png)

| Layer | What it is in this repo |
|---|---|
| 1. Sensing | Webcam (owned by Unity), microphone, mouse/tablet, and ESP32 + MPU6050 wrist and ankle nodes (100 Hz JSON over USB serial, 921600 baud) |
| 2. Game client | Unity project `ParkiScan_Neuroverse`: game levels, BlazePose (runs inside Unity), NIA guide, local telemetry CSV |
| 3. Local service | `ParkiScanML/`: FastAPI on `127.0.0.1:8765`, feature extractors, session store, optional research models (off by default) |
| 4. Output | Per-session JSON summary with features, quality warnings and notices. No diagnosis, no risk label |
| Planned | Multimodal fusion, clinician/caregiver report, adaptive rehab missions. **Not built yet** |

Everything runs on the participant's own computer. The service binds to localhost and uploads nothing. Source for the diagram: [`docs/architecture.mmd`](docs/architecture.mmd) (SVG also in `docs/`).

## Repository layout

```
ParkiScanML/        Local analysis service (Python): API, features, training, firmware, tests. See ParkiScanML/README.md
ProjectSettings/    Unity project settings
Assets/             Unity game (scripts, scenes, resources) - delivered separately, see note below
SourceAssets/       Source 3D assets (characters, environment)
docs/               Architecture diagram
```

## Quick start

Analysis service (macOS):

```bash
cd ParkiScanML
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
pytest              # 28 tests
./run_local.sh      # http://127.0.0.1:8765, API docs at /docs
```

Game: open the project in Unity, press Play. If the service is not running the game still works and saves results locally.

Full details (API table, datasets, training, voice generation): [`ParkiScanML/README.md`](ParkiScanML/README.md).

## Hardware: ESP32 + MPU6050 motion nodes

Two nodes (wrist for tremor, ankle for gait), each an ESP32 dev board with an MPU6050.

| MPU6050 | ESP32 |
|---|---|
| VCC | 3V3 |
| GND | GND |
| SDA | GPIO21 |
| SCL | GPIO22 |
| AD0 | GND (address 0x68) |

1. Open `ParkiScanML/firmware/esp32_mpu6050/esp32_mpu6050.ino`, board "ESP32 Dev Module", set `PLACEMENT` to `wrist` or `ankle`, flash.
2. Start the service, then bridge the node: `python -m parkiscan.sensors.serial_bridge --port /dev/cu.usbserial-XXXX --session <id>`
3. No hardware? `python -m parkiscan.sensors.simulator --seconds 10 --walking` (output is labelled **SIMULATED DATA**).

Power is 5 V over USB only.

## Research status (honest summary)

Models are trained with participant-grouped cross-validation on small public datasets. Only three are flagged usable for research, and research scores are **off by default** (`PARKISCAN_ENABLE_RESEARCH_SCORES=false`).

| Dataset | Participants | ROC AUC | Usable for research |
|---|---|---|---|
| Daphnet freezing-of-gait (ankle IMU) | 10 | 0.88 | Yes (event detection, not diagnosis) |
| PhysioNet gait | 31 | 0.90 | Yes, small sample |
| Spiral drawings | 40 | 0.83 | Yes, small sample |
| Oxford voice, multi-sound speech, telemonitoring, ET_PD_VIS | 32-42 | - | No (CI includes chance, no better than baseline, or suspected confound) |

See `ParkiScanML/reports/training_report.md`.

## Accessibility, standards and privacy

* Designed **in alignment with** Indian standards relevant to accessible ICT and electronic equipment safety, including IS 17802 (ICT accessibility) and the IS 13450 series (IEC 60601-based safety). It is **not certified** to any BIS standard.
* Seated and standing task pathways, voice guidance, large targets.
* Local-only processing, no cloud upload, session IDs restricted to a safe character set, questionnaire answers stored but never scored.
* Participant recordings are never sent to third parties. ElevenLabs (optional) is used only to synthesise NIA's scripted lines.

## Secrets

Keys go only in `ParkiScanML/.env` (git-ignored). An ElevenLabs key was shared earlier in a conversation; revoke it and create a new one before publishing.

## Team

Built by college students: Arya and Kajjol.
