# DriveSense 🚗👁️
**Real-time Driver Drowsiness Detection** · Python · OpenCV · Dlib · NumPy

---

## What it detects

| Signal | Method | Threshold |
|---|---|---|
| Eye closure | Eye Aspect Ratio (EAR) | EAR < 0.25 for 20+ frames |
| Yawning | Mouth Aspect Ratio (MAR) | MAR > 0.65 for 15+ frames |
| Low blink rate | Blink timestamp tracking | < 8 blinks/min over 60s window |

Alarm latency is **under 150 ms** from detection to alert.

---

## Quick Start

### 1. Install & download model
```bash
python setup.py
```

This installs all dependencies and downloads the dlib 68-point predictor (~100 MB).

### 2. Run
```bash
python drivesense.py
```

### Options
```
--predictor  PATH    Path to shape_predictor_68_face_landmarks.dat (default: current dir)
--camera     INT     Camera device index (default: 0)
```

Example with external USB camera:
```bash
python drivesense.py --camera 1
```

---

## HUD Layout

```
┌──────────────────────────────────────────────────┐
│ DRIVESENSE   Session 00:00         30 FPS  8.3ms │ ← Banner
├───────────┬──────────────────────────────────────┤
│ EAR       │                                      │
│ MAR       │        Live webcam feed              │
│ Blink/min │        with face landmarks           │
│ Blinks    │                                      │
│ Yawns     │                                      │
│ [gauges]  │                                      │
├───────────┴──────────────────────────────────────┤
│ [EAR graph]        [ status: NORMAL / ALERT ]    │ ← Footer
└──────────────────────────────────────────────────┘
```

---

## Keyboard Controls

| Key | Action |
|-----|--------|
| `Q` | Quit   |
| `R` | Reset session stats |
| `S` | Save screenshot |

---

## How It Works

1. **Face detection** — Dlib's HOG-based frontal face detector.
2. **Landmark extraction** — 68-point predictor maps eyes and mouth contours.
3. **EAR calculation** — Ratio of vertical to horizontal eye distances.  
   When EAR < threshold for N consecutive frames → drowsy.
4. **MAR calculation** — Same principle for mouth openness → yawning.
5. **Blink rate** — Timestamps of completed blinks in a sliding window.  
   Very low rate → possible microsleep.
6. **Alarm** — Cross-platform audio alert fires in a background thread,  
   keeping main loop latency < 150 ms.

---

## Dependencies

- `opencv-python` — Camera capture and frame rendering
- `dlib` — Face detection + facial landmark prediction
- `numpy` — Numerical operations
- `scipy` — Euclidean distance for EAR/MAR
- `imutils` — Landmark index helpers

---

## Tuning

Edit the constants at the top of `drivesense.py`:

```python
EAR_THRESHOLD     = 0.25   # Lower = less sensitive to eye closure
EAR_CONSEC_FRAMES = 20     # More frames = slower but fewer false alarms
MAR_THRESHOLD     = 0.65   # Higher = less sensitive to yawning
LOW_BLINK_RATE    = 8      # blinks/min below which alert fires
```
