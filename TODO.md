# TODO — Agent Execution Plan

This file lists the **high-level, ordered steps** the agent should follow.
The agent has full access to the existing ETA prediction codebase and must reuse it.

---

## 1. Repository Setup
- Copy the existing ETA prediction suite into this repository.
- Identify the public interfaces needed to:
  - run predictions given a vehicle position and timestamp,
  - obtain ordered upcoming stops for a trip.

---

## 2. GTFS-Realtime Integration
- Implement polling for:
  - VehiclePositions (live vehicle state),
  - TripUpdates (arrival confirmation).
- Normalize incoming data into a consistent internal format.
- Maintain a list of **currently active trips**.

---

## 3. Trip & Stop Resolution
- For a selected trip:
  - Identify the assigned vehicle.
  - Determine which stops are still in the future.
- Expose:
  - active trips,
  - future stops per trip,
  for consumption by the UI.

---

## 4. Live Prediction Loop
- On a fixed interval:
  - Fetch the latest vehicle position.
  - Invoke the existing ETA prediction pipeline.
  - Store each prediction with:
    - timestamp of prediction,
    - predicted arrival time,
    - vehicle, trip, and stop identifiers,
    - model metadata (if available).

---

## 5. Arrival Detection
- Monitor TripUpdates for the selected trip and stop.
- Detect when the vehicle arrives at the stop.
- Record the **actual arrival timestamp** exactly once.

---

## 6. Evaluation Logic
- After arrival:
  - Match all prior predictions to the actual arrival time.
  - Compute prediction errors (e.g., seconds early/late).
- Derive simple summary statistics:
  - final error,
  - MAE over the session,
  - error vs time-to-arrival.

---

## 7. UI Implementation
- Build a simple UI that allows:
  - provider selection (MBTA only),
  - trip selection,
  - stop selection.
- Display:
  - live predictions as they are generated,
  - arrival confirmation,
  - evaluation plots and statistics after arrival.

---

## 8. Validation & Polish
- Verify:
  - arrival detection correctness,
  - consistent time handling (timezones, timestamps),
  - stability under polling jitter.
- Ensure the UI clearly communicates:
  - what is predicted,
  - when it was predicted,
  - what actually happened.

---

## Done Criteria
- A user can watch a bus approach a stop in real time.
- The system predicts arrival repeatedly until arrival occurs.
- Actual arrival is detected from GTFS-Realtime.
- Prediction errors are computed and visualized clearly.
