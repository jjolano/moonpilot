"""A radar lead's sensor latency, learned online from near-rest leads.

`radard` composes `vLead = vRel + v_ego_hist[0]`, where the ego speed is taken `CP.radarDelay`
older than "now" so it lines up with when the radar actually measured the range rate. On a car
whose `radarDelay` is 0 — Toyota, and every car without an override — that composition uses the
*current* `vEgo` against a measurement that is `τ` seconds stale, so a stationary target reads
``vLead ≈ aEgo · τ``. Over the 429 near-rest radar samples on route 000003d2 where the ego was
braking or accelerating past 0.6 m/s², the ratio was 0.136 s either way; aligning the ego speed
by 0.16 s cut the stationary target's median |vLead| from 0.20 to 0.07 m/s. The gap (`dRel`) is
stale by the same `τ`, which is the closing the stopping floor then owed at the end — so the
planner adds `applied()` to a *radar* lead's age in `_lead_age`.

On a car whose `radarDelay` already matches the sensor (`0.1` Honda, `0.06` Ford), the same
computation reads `τ ≈ 0`: the residual is what is left after radard's own alignment, so a
learned ~0 is correct and there is no double-count. That is why the old fingerprint dict
(`MOONPILOT_RADAR_LATENCY = {"TOYOTA_RAV4_TSS2": 0.15}`) is unnecessary: the capability is
observable on every radar car, not keyed by name.

Four gates decide whether a frame teaches:

- the lead is present and radar (a vision lead has no sensor latency of this kind);
- ``|aEgo| ≥ 0.6`` — the excitation floor the 429-sample measurement used;
- ``0 < vLead / aEgo ≤ MAX`` — signs agree (a positive latency), and the ratio is inside the
  one-sided ceiling. This is also the stationarity gate: a lead moving at absolute speed ``v``
  reads ``τ_est = v/aEgo + τ``, which leaves the window unless ``v`` is small, and during
  braking it goes negative and is dropped.

`applied()` is 0.0 until `MIN_SAMPLES` clear those gates — pass-through, i.e. the planner this
fork had before — and clamped to ``[0, MAX]`` after. It persists in `MoonpilotRadarLatency`
with its evidence count on the fork planner's own cadence, seeded at construction the same way
`moonpilot/pitch.py` and `moonpilot/latency.py` seed. A learned value is not a behavior the
driver flips, so there is no row in the registry.
"""

import math
from typing import Any


MOONPILOT_RADAR_LATENCY_KEY = "MoonpilotRadarLatency"  # the param the learned value persists in, as FLOAT
MOONPILOT_RADAR_LATENCY_SAMPLES_KEY = "MoonpilotRadarLatencySamples"  # its evidence, as INT
MOONPILOT_RADAR_LATENCY_MAX = 0.3  # s; one-sided ceiling — past this the projection is longer than
# the window the stopping floor makes decisions over, and a
# wrong high value is the direction that shortens the gap
# at the end of a stop
MOONPILOT_RADAR_LATENCY_MIN_SAMPLES = 50  # gated frames before the mean is applied; the RAV4 route
# alone carried 429, so one drive of stop-and-go clears it
MOONPILOT_RADAR_LATENCY_MIN_AEGO = 0.6  # m/s^2; excitation floor, the same one the 429-sample
# measurement used

def persisted_seed(params: Any) -> tuple[float, int] | None:
  """The persisted estimate as a seed: `(latency, evidence samples)` when usable, else None."""
  seeded = params.get(MOONPILOT_RADAR_LATENCY_KEY, return_default=True)
  if not isinstance(seeded, (int, float)) or not math.isfinite(seeded) or seeded <= 0.0:
    return None
  seeded_samples = params.get(MOONPILOT_RADAR_LATENCY_SAMPLES_KEY, return_default=True)
  samples = seeded_samples if isinstance(seeded_samples, int) else 0
  return min(seeded, MOONPILOT_RADAR_LATENCY_MAX), max(samples, MOONPILOT_RADAR_LATENCY_MIN_SAMPLES)


class RadarLatencyEstimator:
  """`τ = vLead / aEgo` on near-rest radar frames, as a running mean over the gated samples."""

  def __init__(self):
    self.estimate = 0.0
    self.samples = 0

  def seed(self, latency: float, samples: int) -> None:
    """Resume a persisted estimate, clamped, so the next boot plans with it from the first frame."""
    self.estimate = min(max(float(latency), 0.0), MOONPILOT_RADAR_LATENCY_MAX)
    self.samples = max(int(samples), 0)

  def restore(self, params: Any) -> None:
    seed = persisted_seed(params)
    if seed is not None:
      self.seed(*seed)

  def persist(self, params: Any) -> None:
    """The value *and its evidence*: a drive that only earned a few samples hands them on, and it is the count that
    decides whether the next boot applies them (`persisted_seed`), so a partial drive is not thrown away."""
    if self.samples > 0:
      params.put(MOONPILOT_RADAR_LATENCY_KEY, round(self.estimate, 4))
      params.put(MOONPILOT_RADAR_LATENCY_SAMPLES_KEY, self.samples)

  @property
  def status(self) -> str:
    return "estimated" if self.samples >= MOONPILOT_RADAR_LATENCY_MIN_SAMPLES else "measuring"

  def applied(self) -> float:
    """What `_lead_age` adds to a radar lead's age. Zero until trusted, and bounded after."""
    if self.status != "estimated":
      return 0.0
    return min(max(self.estimate, 0.0), MOONPILOT_RADAR_LATENCY_MAX)

  def update(self, lead, a_ego: float) -> None:
    """One frame of one lead. An absent, vision, unexcited or implausible frame teaches nothing."""
    if not (lead.present and lead.radar):
      return
    v_lead = float(lead.vLead)
    a_ego = float(a_ego)
    if not (math.isfinite(v_lead) and math.isfinite(a_ego)):
      return
    if abs(a_ego) < MOONPILOT_RADAR_LATENCY_MIN_AEGO:
      return
    tau = v_lead / a_ego
    if not (0.0 < tau <= MOONPILOT_RADAR_LATENCY_MAX):
      return
    self.samples += 1
    self.estimate += (tau - self.estimate) / self.samples
