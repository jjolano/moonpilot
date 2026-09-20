"""The domain rules the model store is: protocols, structural verdicts, codecs, boot commit and
model jobs.

Every test here is a rule the rest of the feature reads rather than a restatement of the code:
`admit` (moonpilot/modelcatalog.py) is `member_reason`/`set_reason`, the worker and both panels
speak the request/status codecs, and the model processes read what `commit_boot_selection` wrote.
So a rule that moves here moves for all of them at once, which is the point of the module.
"""

import json
import os
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest import mock
from pathlib import Path

from moonpilot import models

ROOT = Path(__file__).resolve().parents[2]

RECIPE = "a" * 64

# A member `set_reason` accepts: the inputs the supercombo protocol feeds it, and a 512-wide
# `hidden_state` against a 512-deep `features_buffer`, which is the one structural property the
# recurrence needs.
SUPERCOMBO_INPUTS = {
  "img": [1, 12, 128, 256],
  "big_img": [1, 12, 128, 256],
  "features_buffer": [1, 24, 512],
  "desire_pulse": [1, 25, 8],
  "traffic_convention": [1, 2],
  "action_t": [1, 2],
}
SUPERCOMBO_SLICES = {name: [0, 512] for name in models.DRIVING_SLICES}


class FakeParams:
  """`Params` as the fork reads it: a `get` that returns the declared default when asked and `None`
  otherwise, a `put` that stores, and a `remove`. Deliberately not `get_bool` -- the fork's own rule
  is that it ignores the declared default."""

  DEFAULTS = {models.DRIVING_KEY: "", models.MONITORING_KEY: ""}

  def __init__(self, values: dict | None = None):
    self.values = dict(values or {})

  def get(self, key, block=False, return_default=False):
    if key in self.values:
      return self.values[key]
    return self.DEFAULTS.get(key) if return_default else None

  def put(self, key, value, block=False):
    self.values[key] = value

  def remove(self, key):
    self.values.pop(key, None)


def profile_document(name: str) -> dict:
  return {
    "schema": 1,
    "type": "profile",
    "name": name,
    "slot_sets": [["supercombo"]],
    "connections": [],
    "required_configuration": list(models.CONFIGURATION_KEYS),
    "inputs": {},
    "outputs": {},
    "state": {},
    "implementation_source": {},
  }


class StoreCase(unittest.TestCase):
  """A temp `models_root()` and the helpers to populate it."""

  def setUp(self):
    self._tmp = tempfile.TemporaryDirectory()
    self._patches = [
      mock.patch.object(models.paths, "data_dir", lambda feature: os.path.join(self._tmp.name, feature)),
      mock.patch.object(models, "_STATUS_CACHE", None),
    ]
    for patch in self._patches:
      patch.start()
    os.makedirs(models.packages_dir(), exist_ok=True)

  def tearDown(self):
    for patch in reversed(self._patches):
      patch.stop()
    self._tmp.cleanup()

  # --- building a store --------------------------------------------------------------------------
  def _write_json(self, path: str, value: dict) -> None:
    models.write_json(path, value)

  def add_package(self, role: str, *, input_shapes: dict | None, slices: dict | None, configuration: dict | None = None, size: int = 4) -> str:
    """A stored package: `recipe.json` with one member, `profile.json`, and an artifact whose bytes
    are `size` long. Returns the *digest of the bytes it wrote*, which is the selection id the rest
    of the store is keyed on -- writing them through the canonical serialization `Manifest.create`
    uses, so `load_package` accepts the package it just built."""
    # Real bytes, hashed the way an artifact is: the boot commit re-hashes them, so a placeholder
    # digest would test the wrong branch.
    artifact_bytes = bytes(size)
    artifact_sha256 = models.hashlib.sha256(artifact_bytes).hexdigest()
    agreed = configuration or {"frame_skip": 4, "LAT_SMOOTH_SECONDS": 0.0, "LONG_SMOOTH_SECONDS": 0.3}
    document = {
      "schema": 1,
      "type": "recipe",
      "profile": "c" * 64,
      "members": {
        role: {
          "artifact": {"sha256": artifact_sha256, "size": size, "format": "onnx"},
          "source": {},
          # Every member carries the recorded configuration as well as the recipe.
          "configuration": dict(agreed),
          "missing": [],
          "inputs": {},
          "outputs": {},
          "targets": ["QCOM"],
          "metadata": {"input_shapes": input_shapes or {}, "output_slices": slices or {}},
        }
      },
      "configuration": dict(agreed),
    }
    raw = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = models.hashlib.sha256(raw.encode()).hexdigest()
    # The exact bytes a `ModelStore` install writes (`Manifest.raw`), because the digest above is a
    # hash of *those* and `load_package` re-serializes to check it.
    # The digest *is* the package directory in a real store, so the fixture writes where a
    # `ModelStore` would: `packages/<digest>/`.
    os.makedirs(models.package_dir(digest), exist_ok=True)
    with open(os.path.join(models.package_dir(digest), "recipe.json"), "w", encoding="utf-8") as handle:
      handle.write(raw)
    self._write_json(os.path.join(models.package_dir(digest), "profile.json"), profile_document("comma/test/v1"))
    with open(os.path.join(models.package_dir(digest), artifact_sha256), "wb") as handle:
      handle.write(artifact_bytes)
    return digest

  @staticmethod
  def artifact_sha256(digest: str) -> str:
    """The stored member artifact's name, read back through `load_package` -- which is also the
    check that the fixture's package is readable at all."""
    package = models.load_package(digest)
    assert package is not None
    artifact = package["recipe"]["members"]["supercombo"]["artifact"]["sha256"]
    return str(artifact)

  def add_build(self, selection: str, protocol: str, *, state: str = "built", error: str | None = None) -> str:
    key = models.build_key(selection, protocol)
    os.makedirs(os.path.dirname(models.build_model(key)), exist_ok=True)
    if state == "built":
      with open(models.build_model(key), "wb") as handle:
        handle.write(b"pickle")
    self._write_json(models.build_record(key), {"schema": 1, "state": state, "error": error, "protocol": protocol, "selection": selection})
    return key


class TestProtocols(unittest.TestCase):
  def test_the_role_sets_are_what_the_catalog_publishes(self):
    # The four profiles the live catalog carries, by their member role sets. A family that exists in
    # the catalog and no protocol here is a family the driver cannot select, which is a decision --
    # so this pins the decision rather than letting a rename move it.
    self.assertEqual(models.protocol_for_roles(["supercombo"]), models.SUPERCOMBO)
    self.assertEqual(frozenset(models.protocol_for_roles(["vision", "on_policy"]).roles), frozenset({"vision", "on_policy"}))  # type: ignore[union-attr]
    self.assertEqual(models.protocol_for_roles(["vision", "off_policy", "on_policy"]).id, "comma.split-vision-off-on.v1")  # type: ignore[union-attr]
    self.assertEqual(models.protocol_for_roles(["dmonitoring"]).kind, models.MONITORING)  # type: ignore[union-attr]
    self.assertIsNone(models.protocol_for_roles(["supercombo", "on_policy"]))

  def test_the_required_outputs_are_what_the_consumers_read(self):
    # `fill_model_msg`/`fill_pose_msg`/`parse_model_outputs` read every one of these names off the
    # model-output dict; dropping one from a protocol would let admission accept a model that dies
    # inside modeld.
    self.assertLessEqual(
      {
        "plan",
        "lane_lines",
        "lane_lines_prob",
        "road_edges",
        "lead",
        "lead_prob",
        "meta",
        "desire_state",
        "desire_pred",
        "pose",
        "wide_from_device_euler",
        "road_transform",
        "hidden_state",
      },
      models.DRIVING_SLICES,
    )
    self.assertLessEqual({"wheel_on_right", "face_descs_lhd", "face_descs_rhd", "sleep_prob_rhd"} - {"sleep_prob_rhd"}, models.DM_SLICES)
    # `sleep_prob_*` is deliberately absent: a 551-slice legacy DM model lacks it and nothing here
    # reads `driverStateV2.sleepProb`.
    self.assertTrue("sleep_prob_lhd" not in models.DM_SLICES)
    self.assertTrue("sleep_prob_lhd" in models.DMONITORING.optional_slices)


class TestMemberReason(unittest.TestCase):
  def member(self, **overrides) -> dict:
    base = {
      "input_shapes": {
        "img": [1, 12, 128, 256],
        "big_img": [1, 12, 128, 256],
        "features_buffer": [1, 24, 512],
        "desire_pulse": [1, 25, 8],
        "traffic_convention": [1, 2],
        "action_t": [1, 2],
      },
      "slices": {name: [0, 8] for name in models.DRIVING_SLICES},
      "format": "onnx",
      "targets": ["QCOM"],
    }
    base.update(overrides)
    return base

  def test_a_supercombo_member_that_declares_what_the_runtime_feeds_is_accepted(self):
    self.assertIsNone(models.member_reason("supercombo", models.SUPERCOMBO, self.member()))

  def test_an_input_the_runtime_does_not_supply_refuses(self):
    # The pre-`action_t` generation: `lateral_control_params`, `prev_desired_curv`, `desire`.
    member = self.member()
    member["input_shapes"]["lateral_control_params"] = [1, 2]
    self.assertEqual(models.member_reason("supercombo", models.SUPERCOMBO, member), models.UNSUPPORTED_INPUT)

  def test_a_missing_required_output_refuses_the_set(self):
    member = self.member()
    member["slices"] = {name: [0, 8] for name in models.DRIVING_SLICES - {"lane_lines"}}
    self.assertEqual(models.set_reason(models.SUPERCOMBO, {"supercombo": member}), f"{models.MISSING_OUTPUTS}: lane_lines")

  def test_a_non_qcom_target_refuses(self):
    member = self.member(targets=["AMD"])
    self.assertEqual(models.member_reason("supercombo", models.SUPERCOMBO, member), models.NOT_QCOM)

  def test_the_hidden_state_width_must_match_the_policy_buffer(self):
    # The one structural property the recurrence depends on: 512 features in, 512 features out. A
    # mismatch is a reshape error inside the compiled JIT, which is a crash on the road.
    member = self.member()
    member["slices"] = dict(member["slices"], hidden_state=[0, 256])
    self.assertEqual(models.set_reason(models.SUPERCOMBO, {"supercombo": member}), models.HIDDEN_STATE_MISMATCH)

  def test_the_action_era_decides_which_files_a_member_may_be_in(self):
    old_head = self.member(slices={**{name: [0, 8] for name in models.DRIVING_SLICES}, "action": [0, 2]}, published_at="2026-05-05T18:16:17-07:00")
    new_head = self.member(slices={**{name: [0, 8] for name in models.DRIVING_SLICES}, "action": [0, 2]}, published_at="2026-06-02T18:14:34-07:00")
    # The current convention decodes `action[0,0] / max(1, v_ego)**2`; a pre-change head there would
    # be a wrong steering request, and the old convention is the split family's.
    self.assertEqual(models.member_reason("supercombo", models.SUPERCOMBO, old_head), models.ACTION_ERA)
    self.assertIsNone(models.member_reason("supercombo", models.SUPERCOMBO, new_head))
    self.assertIsNone(
      models.member_reason("vision", models.SPLIT_VISION_OFF_ON, {**old_head, "input_shapes": {"img": [1, 12, 128, 256], "big_img": [1, 12, 128, 256]}})
    )
    old_split = {**old_head, "input_shapes": {"desire_pulse": [1, 25, 8], "features_buffer": [1, 25, 512], "traffic_convention": [1, 2]}}
    new_split = {**old_split, "published_at": "2026-06-02T18:14:34-07:00"}
    self.assertIsNone(models.member_reason("on_policy", models.SPLIT_VISION_OFF_ON, old_split))
    self.assertEqual(models.member_reason("on_policy", models.SPLIT_VISION_OFF_ON, new_split), models.ACTION_ERA)

  def test_a_member_with_no_recorded_interface_refuses(self):
    self.assertEqual(models.member_reason("supercombo", models.SUPERCOMBO, {"input_shapes": {}, "slices": {}}), models.NO_METADATA)


class TestCodecs(unittest.TestCase):
  def test_a_request_round_trips_and_a_malformed_one_is_dropped(self):
    params = FakeParams()
    request_id = models.request(params, "install", RECIPE)
    self.assertEqual(models.read_request(params), {"schema": 1, "id": request_id, "op": "install", "recipe": RECIPE})
    # A malformed request is `None`, never a retry: a loop on something unparsable is a stuck worker.
    for bad in (
      "{",
      "[]",
      json.dumps({"schema": 1, "id": "", "op": "install", "recipe": RECIPE}),
      json.dumps({"schema": 1, "id": "1", "op": "explode", "recipe": RECIPE}),
      json.dumps({"schema": 1, "id": "1", "op": "install", "recipe": "../../etc/passwd"}),
    ):
      params.put(models.REQUEST_KEY, bad)
      self.assertIsNone(models.read_request(params), bad)

  def test_the_status_shape_is_fixed(self):
    params = FakeParams()
    empty = models.status(params)
    self.assertEqual(empty["job"], None)
    self.assertEqual(empty["installed"], [])
    self.assertEqual(list(empty), ["schema", "catalog", "job", "installed", "storage"])
    models.publish_status(params, job=models.Job(id="1", op="install", selection=RECIPE, phase="downloading", received=5, total=10))
    status = models.status(params)
    self.assertEqual(status["job"]["phase"], "downloading")
    self.assertEqual(status["job"]["received"], 5)
    # An unknown phase is a bug in the writer, not a rendering case: the entry is dropped.
    params.put(models.STATUS_KEY, json.dumps({"schema": 1, "job": {"phase": "teleporting"}}))
    self.assertIsNone(models.status(params)["job"])
    params.put(models.STATUS_KEY, "not json")
    self.assertEqual(models.status(params), empty)

  def test_status_ignores_legacy_selection_entries(self):
    params = FakeParams()
    models.publish_status(params, installed=[{"recipe": RECIPE, "name": "a"}, {"composition": "c-" + "e" * 32, "name": "legacy"}])
    installed = models.status(params)["installed"]
    self.assertEqual(len(installed), 1)
    self.assertEqual(models.selection_of(installed[0]), RECIPE)


class TestChooserEntries(unittest.TestCase):
  def test_stock_built_and_catalog_entries_are_ordered_and_deduplicated(self):
    params = FakeParams()
    built_recipe = "b" * 64
    packaged_recipe = "c" * 64
    monitoring_recipe = "d" * 64
    catalog_recipe = "e" * 64
    models.publish_status(
      params,
      installed=[
        {"recipe": built_recipe, "name": "built", "kind": models.DRIVING, "state": "built"},
        {"recipe": packaged_recipe, "name": "packaged", "kind": models.DRIVING, "state": "packaged"},
        {"recipe": monitoring_recipe, "name": "monitoring", "kind": models.MONITORING, "state": "built"},
      ],
    )
    catalog = [
      {"recipe": built_recipe, "name": "built again", "kind": models.DRIVING, "admitted": True},
      {"recipe": packaged_recipe, "name": "packaged again", "kind": models.DRIVING, "admitted": True},
      {"recipe": "f" * 64, "name": "refused", "kind": models.DRIVING, "admitted": False},
      {"recipe": monitoring_recipe, "name": "wrong kind", "kind": models.MONITORING, "admitted": True},
      {"recipe": catalog_recipe, "name": "catalog", "kind": models.DRIVING, "admitted": True},
      {"recipe": catalog_recipe, "name": "catalog duplicate", "kind": models.DRIVING, "admitted": True},
    ]
    with mock.patch.object(models, "browse", return_value={"entries": catalog}):
      entries = models.chooser_entries(params, models.DRIVING)

    self.assertEqual(
      [None if entry is None else models.selection_of(entry) for entry in entries],
      [None, built_recipe, catalog_recipe],
    )
    built = entries[1]
    catalog_entry = entries[2]
    assert built is not None
    assert catalog_entry is not None
    self.assertEqual(built["state"], "built")
    self.assertTrue(catalog_entry["admitted"])


class TestBootCommit(StoreCase):
  def custom_params(self) -> tuple[FakeParams, str]:
    digest = self.add_package("supercombo", input_shapes=SUPERCOMBO_INPUTS, slices=SUPERCOMBO_SLICES)
    self.add_build(digest, models.SUPERCOMBO.id)
    params = FakeParams({models.DRIVING_KEY: digest})
    models.commit_boot_selection(params)
    return params, digest

  def test_no_selection_commits_bundled(self):
    params = FakeParams()
    boot = models.commit_boot_selection(params)
    self.assertEqual(boot[models.DRIVING]["kind"], "bundled")
    self.assertEqual(boot[models.DRIVING]["requested"], "")
    self.assertEqual(models.boot(params), boot)
    self.assertEqual(models.model_label(params, models.DRIVING), models.BUNDLED_LABEL)

  def test_manager_continues_after_boot_selection_failure(self):
    from openpilot.system.manager import manager

    params = mock.Mock()
    params.get_bool.return_value = False
    params.all_keys.return_value = []
    metadata = SimpleNamespace(
      release_channel=False,
      channel="test",
      tested_channel=False,
      openpilot=SimpleNamespace(
        version="test-version",
        git_commit="test-commit",
        git_commit_date="test-date",
        git_origin="test-origin",
        git_normalized_origin="test-origin",
        is_dirty=False,
      ),
    )
    hardware = mock.Mock(get_serial=mock.Mock(return_value="test-serial"), get_device_type=mock.Mock(return_value="test-device"))

    with (
      mock.patch.object(manager, "save_bootlog"),
      mock.patch.object(manager, "get_build_metadata", return_value=metadata),
      mock.patch.object(manager, "Params", return_value=params),
      mock.patch.object(models, "commit_boot_selection", side_effect=RuntimeError("store failure")),
      mock.patch.object(manager, "HARDWARE", hardware),
      mock.patch.object(manager, "register", return_value="test-dongle"),
      mock.patch.object(manager.Paths, "shm_path", return_value="/tmp/moonpilot-test-shm"),
      mock.patch.object(manager.os, "mkdir", side_effect=FileExistsError),
      mock.patch.object(manager.cloudlog, "exception") as exception,
      mock.patch.object(manager.cloudlog, "bind_global"),
      mock.patch.dict(manager.os.environ, {}, clear=False),
    ):
      manager.manager_init()

    params.put.assert_any_call("Version", "test-version", block=True)
    exception.assert_called_once_with("failed to commit moonpilot model boot selection")

  def test_a_selection_with_no_build_falls_back_and_keeps_the_reason(self):
    params = FakeParams({models.DRIVING_KEY: RECIPE})
    boot = models.commit_boot_selection(params)
    self.assertEqual(boot[models.DRIVING]["kind"], "bundled")
    self.assertEqual(boot[models.DRIVING]["requested"], RECIPE)
    self.assertEqual(boot[models.DRIVING]["fallback"], models.NOT_INSTALLED)
    # The driver's selection survives, and the panel can explain it.
    self.assertEqual(models.desired(params, models.DRIVING), RECIPE)
    self.assertTrue(RECIPE[:12] in models.model_description(params, models.DRIVING))

  def test_a_legacy_selection_falls_back_to_stock(self):
    legacy = "c-" + "e" * 32
    params = FakeParams({models.DRIVING_KEY: legacy})
    entry = models.commit_boot_selection(params)[models.DRIVING]
    self.assertEqual(entry["kind"], "bundled")
    self.assertEqual(entry["requested"], legacy)
    self.assertEqual(entry["fallback"], models.UNKNOWN_SELECTION)
    self.assertEqual(models.desired(params, models.DRIVING), legacy)

  def test_a_built_selection_resolves_to_paths(self):
    digest = self.add_package("supercombo", input_shapes=SUPERCOMBO_INPUTS, slices=SUPERCOMBO_SLICES)
    self.add_build(digest, models.SUPERCOMBO.id)
    params = FakeParams({models.DRIVING_KEY: digest})
    entry = models.commit_boot_selection(params)[models.DRIVING]
    self.assertEqual(entry["kind"], "custom")
    self.assertEqual(entry["protocol"], models.SUPERCOMBO.id)
    artifact = self.artifact_sha256(digest)
    self.assertEqual(entry["members"], {"supercombo": os.path.join(models.package_dir(digest), artifact)})
    self.assertTrue(os.path.isdir(entry["build"]))
    self.assertIsNone(entry["fallback"])
    self.assertFalse(models.restart_pending(params, models.DRIVING))

  def test_a_custom_active_model_keeps_the_selection_label(self):
    params, digest = self.custom_params()
    params.put(models.ACTIVE_DRIVING_KEY, "custom (supercombo)")
    self.assertEqual(
      models.model_label(params, models.DRIVING),
      f"{digest[:12]} ({models.protocol_label(models.SUPERCOMBO.id)})",
    )

  def test_a_custom_runtime_fallback_takes_over_the_short_label(self):
    params, digest = self.custom_params()
    params.put(models.ACTIVE_DRIVING_KEY, "stock: RuntimeError: broken model")
    self.assertEqual(models.model_label(params, models.DRIVING), f"stock, {digest[:12]} did not load")
    self.assertTrue("RuntimeError: broken model" in models.model_description(params, models.DRIVING))

  def test_a_failed_build_falls_back_with_the_compilers_message(self):
    digest = self.add_package("supercombo", input_shapes=SUPERCOMBO_INPUTS, slices=SUPERCOMBO_SLICES)
    self.add_build(digest, models.SUPERCOMBO.id, state="failed", error="CUDA out of coffee")
    params = FakeParams({models.DRIVING_KEY: digest})
    entry = models.commit_boot_selection(params)[models.DRIVING]
    self.assertEqual(entry["kind"], "bundled")
    self.assertEqual(entry["fallback"], "CUDA out of coffee")

  def test_the_wrong_kind_is_refused(self):
    # A driving recipe in the monitoring param: two processes, two params, and no way for a
    # monitoring model to satisfy `modeld`'s loop.
    digest = self.add_package("supercombo", input_shapes=SUPERCOMBO_INPUTS, slices=SUPERCOMBO_SLICES)
    self.add_build(digest, models.SUPERCOMBO.id)
    params = FakeParams({models.MONITORING_KEY: digest})
    entry = models.commit_boot_selection(params)[models.MONITORING]
    self.assertEqual(entry["kind"], "bundled")
    self.assertEqual(entry["fallback"], models.WRONG_KIND)

  def test_changed_bytes_are_refused(self):
    digest = self.add_package("supercombo", input_shapes=SUPERCOMBO_INPUTS, slices=SUPERCOMBO_SLICES)
    self.add_build(digest, models.SUPERCOMBO.id)
    artifact = self.artifact_sha256(digest)
    with open(os.path.join(models.package_dir(digest), artifact), "wb") as handle:
      handle.write(b"tampered")
    params = FakeParams({models.DRIVING_KEY: digest})
    entry = models.commit_boot_selection(params)[models.DRIVING]
    self.assertEqual(entry["kind"], "bundled")
    self.assertEqual(entry["fallback"], models.ARTIFACT_CHANGED)

  def test_a_restart_is_owed_when_the_selection_moves(self):
    params = FakeParams()
    models.commit_boot_selection(params)
    self.assertFalse(models.restart_pending(params, models.DRIVING))
    models.select(params, models.DRIVING, RECIPE)
    self.assertTrue(models.restart_pending(params, models.DRIVING))
    self.assertEqual(models.model_label(params, models.DRIVING), f"{models.BUNDLED_LABEL} {models.RESTART_NOTE}")
    with self.assertRaises(ValueError):
      models.select(params, models.DRIVING, "nonsense")
    with self.assertRaises(ValueError):
      models.select(params, models.DRIVING, "c-" + "e" * 32)


class TestCapacity(unittest.TestCase):
  def test_the_three_ceilings_are_sentences(self):
    self.assertEqual(models.capacity_reason(models.MAX_ARTIFACT_BYTES + 1), models.TOO_LARGE)
    with mock.patch.object(models, "free_bytes", lambda: 0):
      self.assertEqual(models.capacity_reason(1), models.NO_SPACE)
    with mock.patch.object(models, "free_bytes", lambda: models.MAX_STORE_BYTES), mock.patch.object(models, "store_usage", lambda: models.MAX_STORE_BYTES):
      self.assertEqual(models.capacity_reason(1), models.STORE_FULL)


class TestJobReporting(unittest.TestCase):
  """What the panels render for a running job: the phase line, the bar's fraction, its label.

  Two of these are load-bearing. The fraction must be `None` for every phase without a total, because
  a bar drawn from nothing is a claim about progress no number backs. And the elapsed clock is only
  published while a job runs (`Job.started_at`), so an older worker's snapshot has no `elapsed` and
  the line must read as it did before -- a panel that required it would show `None` to the driver.
  """

  def test_job_active_matches_the_live_phases(self):
    for phase in models.PHASES:
      with self.subTest(phase=phase):
        self.assertEqual(models.job_active({"phase": phase}), phase not in ("done", "error", "canceled"))
    self.assertFalse(models.job_active(None))

  def test_the_patient_phases_carry_a_clock(self):
    for phase in models.PATIENT_PHASES:
      with self.subTest(phase=phase):
        self.assertEqual(models.job_text({"phase": phase, "elapsed": 83}), f"{models.PHASE_TEXT[phase]} 1:23")
        # No elapsed (a snapshot from a worker that did not publish one) reads as the phase alone.
        self.assertEqual(models.job_text({"phase": phase}), models.PHASE_TEXT[phase])

  def test_the_clock_is_minutes_and_hours(self):
    self.assertEqual(models.clock(0), "0:00")
    self.assertEqual(models.clock(83), "1:23")
    self.assertEqual(models.clock(3800), "1:03:20")
    self.assertEqual(models.clock(-5), "0:00")

  def test_a_download_shows_bytes_not_a_clock(self):
    job = {"phase": "downloading", "received": 24_320_000, "total": 60_881_999, "elapsed": 12.0}
    self.assertEqual(models.job_text(job), "downloading 23.2 MB of 58.1 MB")

  def test_finished_and_failed_phases_show_no_clock(self):
    for phase in ("done", "error", "canceled"):
      with self.subTest(phase=phase):
        self.assertEqual(models.job_text({"phase": phase, "elapsed": 900}), models.PHASE_TEXT[phase])

  def test_the_fraction_exists_only_where_a_total_does(self):
    self.assertIsNone(models.job_fraction(None))
    self.assertIsNone(models.job_fraction({"phase": "building", "total": 100}))
    self.assertIsNone(models.job_fraction({"phase": "waiting-network"}))
    self.assertIsNone(models.job_fraction({"phase": "downloading", "total": 0}))
    self.assertEqual(models.job_fraction({"phase": "downloading", "received": 50, "total": 100}), 0.5)
    # A download cannot exceed its declared size -- the worker refuses that -- but a bar must not
    # draw past its track if one ever did.
    self.assertEqual(models.job_fraction({"phase": "downloading", "received": 200, "total": 100}), 1.0)

  def test_the_bar_label_carries_the_percentage_and_the_bytes(self):
    job = {"phase": "downloading", "received": 24_320_000, "total": 60_881_999}
    text = models.job_progress_text(job)
    self.assertTrue(text.startswith("40%"))
    self.assertTrue(models.human_size(24_320_000) in text)
    self.assertTrue(models.human_size(60_881_999) in text)
    # With nothing to measure the label is the phase line, so the two renderers can use one call.
    self.assertEqual(models.job_progress_text({"phase": "building", "elapsed": 83}), "building 1:23")

  def test_a_live_job_publishes_its_elapsed_but_one_that_never_started_does_not(self):
    params = FakeParams()
    models.publish_status(params, job=models.Job(id="1", op="install", phase="downloading", received=5, total=10))
    # No `started_at`: a hand-made Job (and an older worker's snapshot) has no clock.
    self.assertIsNone(models.status(params)["job"]["elapsed"])
    started = models.Job(id="1", op="install", phase="building", started_at=time.monotonic() - 30)
    models.publish_status(params, job=started)
    elapsed = models.status(params)["job"]["elapsed"]
    self.assertIsNotNone(elapsed)
    self.assertGreaterEqual(float(elapsed), 30.0)
    # A nonsense value from a broken writer is dropped rather than rendered.
    params.put(models.STATUS_KEY, json.dumps({"schema": 1, "job": {"phase": "building", "elapsed": "soon"}}))
    self.assertIsNone(models.status(params)["job"]["elapsed"])


class TestWorkerPredicate(unittest.TestCase):
  """`_models_wanted` is what starts the worker and what lets the manager reap it. Both directions
  matter: a predicate that is never True never publishes a status, and one that is always True leaves
  a process running for the whole boot."""

  def wanted(self, **values) -> bool:
    from moonpilot.procs import _models_wanted

    params = FakeParams()
    for key, value in values.items():
      params.put(key, value)
    return _models_wanted(False, params, None)

  def test_a_boot_with_no_status_starts_the_worker(self):
    self.assertTrue(self.wanted())

  def test_a_published_status_with_no_request_reaps_it(self):
    self.assertFalse(self.wanted(**{models.STATUS_KEY: json.dumps(models.empty_status())}))

  def test_a_request_starts_it_again(self):
    self.assertTrue(self.wanted(**{models.STATUS_KEY: "{}", models.REQUEST_KEY: "{}"}))


class TestDeclaredSmoothing(unittest.TestCase):
  """One number for the model's timing: modeld decodes with the selection's declared smoothing, and
  controlsd's curvature reference and torque controller must time against that same value rather than
  modeld's module default (AGENTS.md, **The model marketplace**)."""

  def test_the_boot_snapshot_carries_the_declared_value(self):
    params = FakeParams()
    self.assertEqual(models.boot_configuration(params, "LAT_SMOOTH_SECONDS", 0.0), 0.0)
    snapshot = {"schema": models.SCHEMA, models.DRIVING: {"kind": "custom", "configuration": {"LAT_SMOOTH_SECONDS": 0.1}}}
    params.put(models.BOOT_KEY, json.dumps(snapshot))
    self.assertAlmostEqual(models.boot_configuration(params, "LAT_SMOOTH_SECONDS", 0.0), 0.1)
    self.assertAlmostEqual(models.boot_configuration(params, "LONG_SMOOTH_SECONDS", 0.3), 0.3)
    params.put(models.BOOT_KEY, json.dumps({"schema": models.SCHEMA, models.DRIVING: {"kind": "custom", "configuration": {"LAT_SMOOTH_SECONDS": "fast"}}}))
    self.assertEqual(models.boot_configuration(params, "LAT_SMOOTH_SECONDS", 0.0), 0.0)

  def test_controlsd_times_both_consumers_against_it(self):
    text = (ROOT / "openpilot/selfdrive/controls/controlsd.py").read_text()
    self.assertEqual(text.count('boot_configuration(self.params, "LAT_SMOOTH_SECONDS", LAT_SMOOTH_SECONDS)'), 1)
    self.assertEqual(text.count("self.sm['lateralDelay'].lateralDelay + self.moonpilot_lat_smooth"), 1)
    self.assertEqual(text.count('self.sm["lateralDelay"].lateralDelay + self.moonpilot_lat_smooth'), 1)


if __name__ == "__main__":
  unittest.main()
