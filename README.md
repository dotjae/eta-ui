# Live ETA Prediction & Validation UI

This repository is a **demonstration and validation frontend** built on top of an existing ETA prediction suite.

The goal is to provide a **simple, well-designed web UI** that allows a user to:
- Select a public transit provider (initially **:contentReference[oaicite:0]{index=0}**),
- Select an **active trip** and a **future stop**,
- Observe **live ETA predictions** generated from real-time vehicle positions,
- Detect the **actual arrival time** of the vehicle at the stop using GTFS-Realtime data,
- Compare predictions against reality and visualize **prediction errors**.

This project is **not a passenger-facing product**. It is an evaluation and research tool intended to:
- Validate ETA models under real-world conditions,
- Inspect prediction behavior over time,
- Produce interpretable error statistics from live data.

---

## Conceptual Overview

The system operates as a closed loop:

1. **Live data ingestion**
   - Poll GTFS-Realtime feeds at a fixed interval:
     - VehiclePositions (for prediction)
     - TripUpdates (for arrival confirmation)

2. **Prediction**
   - Each new vehicle position is fed into the existing ETA prediction pipeline.
   - The pipeline is treated as a black box and reused as-is.
   - Predictions are generated repeatedly for the same vehicle–stop pair as time advances.

3. **Arrival detection**
   - The system determines when the vehicle arrives at the selected stop
     using GTFS-Realtime data (preferably TripUpdates).
   - The actual arrival timestamp is recorded once.

4. **Evaluation**
   - All predictions made before arrival are compared against the actual arrival time.
   - Errors are computed and summarized.
   - Results are displayed visually in the UI.

---

## User Interaction Flow

1. **Provider selection**
   - Currently fixed to MBTA (future extensibility is allowed but not required).

2. **Trip selection**
   - Display a list of *currently active trips*.
   - Each trip corresponds to a specific vehicle.

3. **Stop selection**
   - Display only stops that the vehicle has **not yet passed**.
   - Stops are ordered by stop sequence.

4. **Live prediction phase**
   - Vehicle positions are polled at a fixed cadence.
   - ETA predictions are updated and displayed continuously.

5. **Arrival & evaluation**
   - Once arrival is detected:
     - Show the actual arrival time.
     - Show prediction errors.
     - Plot useful statistics (e.g., error vs time-to-arrival).

---

## Design Principles

- Reuse the existing ETA prediction codebase without modification where possible.
- Keep responsibilities clearly separated:
  - data ingestion
  - prediction
  - arrival detection
  - evaluation
  - UI
- Prefer clarity and interpretability over feature completeness.
- Make assumptions explicit and easy to change.

---

## Scope (Intentional Limitations)

- Single provider (MBTA only).
- One trip and one stop at a time.
- No long-term analytics dashboard (session-level evaluation only).
- No model training or retraining in this repository.

This repo exists to **observe models in the wild**, not to optimize them.
