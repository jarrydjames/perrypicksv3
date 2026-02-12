# Q3 Model Redesign: 5:00 Remaining in Q3 → Rest-of-Game Forecast

## Objective
Use the Q3 model at **5:00 remaining in Q3** to predict the **balance of game**:
- remaining total points (rest of Q3 + Q4)
- remaining margin (rest of Q3 + Q4)
- converted final total/final margin by adding snapshot state

## Data Gathering Updates
- Build each training row from the first Q3 play-by-play event with `clock <= 5:00`.
- Persist snapshot score features:
  - `q3_5m_home`, `q3_5m_away`, `q3_5m_total`, `q3_5m_margin`
- Keep existing game-context features (rates, behavior counts, boxscore team totals).
- Restrict Q3 action counts to information available at inference time (Q1/Q2 and Q3 events with `clock >= 5:00`).

## Training Target Updates
- Replace target labels from cumulative-Q3 state to **remaining-game targets**:
  - `remaining_total = final_total - q3_5m_total`
  - `remaining_margin = final_margin - q3_5m_margin`
- Continue training two-head models on `(remaining_total, remaining_margin)`.

## Calibration Updates
- Calibrate intervals using residuals against `remaining_total` and `remaining_margin`.
- At runtime, transform calibrated bands back to final-game scale by adding snapshot state.

## Runtime / Model Selection Updates
- Q3 trigger remains at game monitor rule: period 3 with <=5:00 remaining.
- Q3 runtime prediction flow:
  1. infer current snapshot score
  2. predict remaining total/margin
  3. derive final total/margin by adding snapshot baseline
  4. convert interval outputs to final-game intervals the same way

## Model Selection Guidance
- Keep current multi-model two-head training stack, but evaluate/select using:
  - MAE on `remaining_total`, `remaining_margin`
  - calibration coverage on transformed final-game intervals
  - edge performance at Q3 trigger in backtests
- Select champion model based on weighted blend:
  - 50% remaining-target MAE
  - 30% interval coverage calibration error
  - 20% betting utility metric (CLV / realized edge proxy)

## Validation Additions
- Add unit tests for:
  - Q3 snapshot extraction at <=5:00
  - runtime conversion from remaining predictions → final predictions
