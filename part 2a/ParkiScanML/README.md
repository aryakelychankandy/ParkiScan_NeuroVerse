# ParkiScanML

Local analysis service for **ParkiScan Neuroverse**. Unity sends each completed task record to it, it
extracts movement, voice, drawing, gaze and sensor features, and it writes a per-session summary. It runs
**only on this computer** (binds to `127.0.0.1`) and uploads nothing.

> **Not a medical device.** Every number this service produces describes how a game task was performed.
> Nothing here is a diagnosis, nothing is clinically validated, and the dataset-trained models are
> research benchmarks only. Questionnaire responses are stored, never scored or interpreted.

## 1 · Setup (macOS)

```bash
cd ~/ParkiScan_Neuroverse/ParkiScanML
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then edit .env (see §2)
pytest                        # 28 tests, ~10 s
./run_local.sh                # http://127.0.0.1:8765  · API docs at /docs
```

`run_local.sh` creates `.venv` and `.env` automatically on first run. Unity finds the service through
`Assets/ParkiScan/Resources/ParkiScanServiceConfig.json` (`http://127.0.0.1:8765`). If the service is not
running, the game still works: results are saved locally and a one-time "ANALYSIS SERVICE OFFLINE" toast appears.

## 2 · Secrets and ElevenLabs (NIA's voice)

* An ElevenLabs key was pasted into a conversation earlier. **Treat it as compromised: revoke it in the
  ElevenLabs dashboard and create a new one.**
* Put the new key only in `ParkiScanML/.env` → `ELEVENLABS_API_KEY=...`. `.env` is git-ignored; the key is
  never written to Unity code, assets, logs, README files or responses (`/capabilities` only says
  `configured: true/false`).
* `ELEVENLABS_VOICE_ID=hpp4J3VqNfWAUOO0d1Us` is pre-filled in `.env.example`.
* ElevenLabs is used **only** to synthesise NIA's scripted lines. Participant recordings are never sent to it
  or to any other third party.

Generate NIA voice clips (written to `Assets/ParkiScan/Resources/NiaVoice/<lang>/<id>.mp3`, where `<id>` is the
same SHA-1 prefix Unity's `LocalizationManager.VoiceId` computes, so the game picks them up automatically):

```bash
python tools/generate_nia_voice.py --lang en --dry-run   # list lines, no API calls
python tools/generate_nia_voice.py --lang en             # English lines found in Say("...") calls
python tools/generate_nia_voice.py --lang ta hi ml te    # translated, non-clinical lines only
```

Translations are drafts awaiting native-speaker review; clinical items are skipped until marked `"reviewed": true`.
Lines built at runtime (string concatenation) are not discovered automatically and stay "TEXT ONLY".

## 3 · Datasets

```bash
python tools/prepare_datasets.py --zips ~/Downloads     # unpacks into data/raw/<name>/ + MANIFEST.json
```

The speech dataset contains a `.rar`; install `unar` first (`brew install unar`). Recognised archives:
gait-in-neurodegenerative-disease-database, parkinsonstelemonitoring, parkinsons (Oxford voice),
parkinsonspeechdatasetwithmultipletypesofsoundrecordings, parkinsondiseasespiraldrawings…, daphnetfreezingofgait,
ET_PD_VIS. Citations are written into `data/raw/MANIFEST.json`. Raw data is git-ignored.

## 4 · Training (participant-grouped)

```bash
python -m training.train_all                # all datasets found; ~7 min on a laptop
python -m training.train_all --only spiral  # one dataset
```

Splits are grouped by participant (StratifiedGroupKFold ×5 repeats, GroupKFold for regression,
leave-one-subject-out for Daphnet). Every model is compared with a dummy baseline. Results:
`reports/training_report.md` / `.json`; models: `models/<dataset>.joblib` + `.json` metadata including
automatic quality flags.

Results from the run delivered with this workspace (best non-baseline model, pooled out-of-fold):

| dataset | participants | best model | metric | 95% CI (participant bootstrap) | usable for research scores |
|---|---|---|---|---|---|
| Daphnet FoG (ankle acc, window detection) | 10 | hist. gradient boosting | ROC AUC 0.88 | 0.85–0.92 | yes (detection in people with PD, not diagnosis) |
| PhysioNet gait (PD vs control) | 31 | random forest | ROC AUC 0.90 | 0.75–1.00 | yes, small sample |
| Spiral tablet, static test (PD vs control) | 40 | random forest | ROC AUC 0.83 | 0.68–0.95 | yes, small sample |
| Oxford voice (PD vs healthy) | 32 | logistic regression | ROC AUC 0.72 | 0.47–0.94 | **no** — CI includes chance |
| Multiple sound recordings (PD vs healthy) | 40 | logistic regression | ROC AUC 0.67 | 0.49–0.82 | **no** — CI includes chance |
| Telemonitoring (motor UPDRS regression) | 42 | ridge | MAE 7.36 vs baseline 7.15 | — | **no** — worse than predicting the mean for unseen people |
| ET_PD_VIS (rendered gaze images) | 40 | logistic regression | ROC AUC 1.00 | 1.00–1.00 | **no** — implausibly perfect; likely confound (task duration / rendering) |

Read these honestly: samples are tiny, recordings come from lab hardware unlike a webcam, mouse or
microphone at home, and the voice datasets ship pre-computed Praat/MDVP features that this service does not
reproduce. That is why research scores are **off by default** (`PARKISCAN_ENABLE_RESEARCH_SCORES=false`) and,
when switched on, are applied only to spiral traces and ankle-IMU windows (the two cases where the same
extractor produced the training features) and are labelled experimental.

## 5 · API

| method | path | purpose |
|---|---|---|
| GET | `/health` | liveness |
| GET | `/capabilities` | features, datasets on disk, models loaded, TTS configured (bool), privacy |
| POST | `/session/start` · `/session/stop` | session lifecycle (`sessions/<id>/session.json`) |
| POST | `/tasks` | Unity task record + raw CSV path → stored, analysed where an extractor matches |
| POST | `/features/voice` | multipart audio (`mode=speech` or `mode=clap`, optional `target_bpm`); audio is not stored |
| POST | `/features/spiral` · `/cognition` · `/gait` · `/pose` · `/gaze` | JSON feature extraction |
| POST | `/sensors/imu` | batch of IMU packets → `sessions/<id>/imu.csv` |
| GET | `/session/{id}/summary` | everything for a session, with notices and quality warnings |
| POST | `/tts` | optional, disabled unless `PARKISCAN_ENABLE_TTS=true` |

Raw CSV paths from Unity are read only if they sit under `PARKISCAN_ALLOWED_RAW_ROOTS`
(default: the game's telemetry folder in `~/Library/Application Support/DefaultCompany/ParkiScan_Neuroverse`).
Session ids are restricted to `[A-Za-z0-9._-]`.

## 6 · ESP32 + MPU-6050 sensor

* Firmware: `firmware/esp32_mpu6050/esp32_mpu6050.ino` (no extra libraries). Wiring: VCC→3V3, GND→GND,
  SDA→GPIO21, SCL→GPIO22. Set `PLACEMENT` (wrist / ankle / waist) before flashing. 100 Hz JSON lines at 921600 baud.
* Bridge to the service: `python -m parkiscan.sensors.serial_bridge --port /dev/cu.usbserial-XXXX --session <id>`
  (`python -m serial.tools.list_ports` lists ports).
* No hardware? `python -m parkiscan.sensors.simulator --seconds 10 --walking` prints packets marked
  `"sim": true` with device ids starting `SIM-`; summaries label them **SIMULATED DATA**.

## 7 · Optional models

`./tools/download_models.sh` fetches MediaPipe face/pose `.task` files; `--asr` also installs faster-whisper
(`small`, runs offline). Neither is needed by the service; Unity already runs BlazePose itself and owns the webcam.

## 8 · Layout

```
app/            FastAPI service (config, schemas, session store, TTS)
parkiscan/      features/ (voice, spiral, gait, cognition, pose, gaze), datasets/ loaders, sensors/ (IMU, simulator, serial bridge)
training/       train_all.py — participant-grouped evaluation and model export
tools/          prepare_datasets.py, generate_nia_voice.py, download_models.sh, compose_music.py (game music/SFX)
firmware/       ESP32 sketch
models/ reports/  trained models and evaluation reports
tests/          pytest suite (features, API, sensors, secret hygiene)
```
