"""The domain rules the model store is: protocols, structural verdicts, codecs, boot commit and
model jobs.

Every test here is a rule the rest of the feature reads rather than a restatement of the code:
`admit` (moonpilot/modelcatalog.py) is `member_reason`/`set_reason`, the worker and both panels
speak the request/status codecs, and the model processes read what `commit_boot_selection` wrote.
So a rule that moves here moves for all of them at once, which is the point of the module.
"""

import json
import os
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from moonpilot.tests.fakes import FakeParams, params
from moonpilot.tests.model_store import TempStoreMixin, write_build, write_package
from moonpilot import modelprotocols, modelstatus, models, modelview


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
SUPERCOMBO_SLICES = {name: [0, 512] for name in modelprotocols.DRIVING_SLICES}


class StoreCase(TempStoreMixin, unittest.TestCase):
  """A temp `models_root()` and shared helpers to populate it."""

  @staticmethod
  def artifact_sha256(digest: str) -> str:
    package = models.load_package(digest)
    assert package is not None
    return str(package["recipe"]["members"]["supercombo"]["artifact"]["sha256"])


class TestProtocols(unittest.TestCase):
  def test_unknown_role_set_has_no_protocol(self):
    self.assertIsNone(modelprotocols.protocol_for_roles(["supercombo", "on_policy"]))


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
      "slices": {name: [0, 8] for name in modelprotocols.DRIVING_SLICES},
      "format": "onnx",
      "targets": ["QCOM"],
    }
    base.update(overrides)
    return base

  def test_a_supercombo_member_that_declares_what_the_runtime_feeds_is_accepted(self):
    self.assertIsNone(modelprotocols.member_reason("supercombo", modelprotocols.SUPERCOMBO, self.member()))

  def test_an_input_the_runtime_does_not_supply_refuses(self):
    # The pre-`action_t` generation: `lateral_control_params`, `prev_desired_curv`, `desire`.
    member = self.member()
    member["input_shapes"]["lateral_control_params"] = [1, 2]
    self.assertEqual(modelprotocols.member_reason("supercombo", modelprotocols.SUPERCOMBO, member), modelprotocols.UNSUPPORTED_INPUT)

  def test_a_missing_required_output_refuses_the_set(self):
    member = self.member()
    member["slices"] = {name: [0, 8] for name in modelprotocols.DRIVING_SLICES - {"lane_lines"}}
    self.assertEqual(modelprotocols.set_reason(modelprotocols.SUPERCOMBO, {"supercombo": member}), f"{modelprotocols.MISSING_OUTPUTS}: lane_lines")

  def test_a_non_qcom_target_refuses(self):
    member = self.member(targets=["AMD"])
    self.assertEqual(modelprotocols.member_reason("supercombo", modelprotocols.SUPERCOMBO, member), modelprotocols.NOT_QCOM)

  def test_the_hidden_state_width_must_match_the_policy_buffer(self):
    # The one structural property the recurrence depends on: 512 features in, 512 features out. A
    # mismatch is a reshape error inside the compiled JIT, which is a crash on the road.
    member = self.member()
    member["slices"] = dict(member["slices"], hidden_state=[0, 256])
    self.assertEqual(modelprotocols.set_reason(modelprotocols.SUPERCOMBO, {"supercombo": member}), modelprotocols.HIDDEN_STATE_MISMATCH)

  def test_the_action_era_decides_which_files_a_member_may_be_in(self):
    old_head = self.member(slices={**{name: [0, 8] for name in modelprotocols.DRIVING_SLICES}, "action": [0, 2]}, published_at="2026-05-05T18:16:17-07:00")
    new_head = self.member(slices={**{name: [0, 8] for name in modelprotocols.DRIVING_SLICES}, "action": [0, 2]}, published_at="2026-06-02T18:14:34-07:00")
    # The current convention decodes `action[0,0] / max(1, v_ego)**2`; a pre-change head there would
    # be a wrong steering request, and the old convention is the split family's.
    self.assertEqual(modelprotocols.member_reason("supercombo", modelprotocols.SUPERCOMBO, old_head), modelprotocols.ACTION_ERA)
    self.assertIsNone(modelprotocols.member_reason("supercombo", modelprotocols.SUPERCOMBO, new_head))
    self.assertIsNone(
      modelprotocols.member_reason(
        "vision",
        modelprotocols.SPLIT_VISION_OFF_ON,
        {**old_head, "input_shapes": {"img": [1, 12, 128, 256], "big_img": [1, 12, 128, 256]}},
      )
    )
    old_split = {**old_head, "input_shapes": {"desire_pulse": [1, 25, 8], "features_buffer": [1, 25, 512], "traffic_convention": [1, 2]}}
    new_split = {**old_split, "published_at": "2026-06-02T18:14:34-07:00"}
    self.assertIsNone(modelprotocols.member_reason("on_policy", modelprotocols.SPLIT_VISION_OFF_ON, old_split))
    self.assertEqual(modelprotocols.member_reason("on_policy", modelprotocols.SPLIT_VISION_OFF_ON, new_split), modelprotocols.ACTION_ERA)

  def test_a_member_with_no_recorded_interface_refuses(self):
    self.assertEqual(modelprotocols.member_reason("supercombo", modelprotocols.SUPERCOMBO, {"input_shapes": {}, "slices": {}}), modelprotocols.NO_METADATA)


class TestCodecs(unittest.TestCase):
  def test_a_request_round_trips_and_a_malformed_one_is_dropped(self):
    params = FakeParams()
    request_id = modelstatus.request(params, "install", RECIPE)
    self.assertEqual(modelstatus.read_request(params), {"schema": 1, "id": request_id, "op": "install", "recipe": RECIPE})
    # A malformed request is `None`, never a retry: a loop on something unparsable is a stuck worker.
    for bad in (
      "{",
      "[]",
      json.dumps({"schema": 1, "id": "", "op": "install", "recipe": RECIPE}),
      json.dumps({"schema": 1, "id": "1", "op": "explode", "recipe": RECIPE}),
      json.dumps({"schema": 1, "id": "1", "op": "install", "recipe": "../../etc/passwd"}),
    ):
      params.put(models.REQUEST_KEY, bad)
      self.assertIsNone(modelstatus.read_request(params), bad)

  @mock.patch.object(modelstatus, "free_bytes", lambda: 0)
  def test_the_status_shape_is_fixed(self):
    params = FakeParams()
    empty = modelstatus.status(params)
    self.assertEqual(empty["job"], None)
    self.assertEqual(empty["installed"], [])
    self.assertEqual(list(empty), ["schema", "catalog", "job", "installed", "storage"])
    modelstatus.publish_status(params, job=modelstatus.Job(id="1", op="install", selection=RECIPE, phase="downloading", received=5, total=10))
    status = modelstatus.status(params)
    self.assertEqual(status["job"]["phase"], "downloading")
    self.assertEqual(status["job"]["received"], 5)
    # An unknown phase is a bug in the writer, not a rendering case: the entry is dropped.
    params.put(models.STATUS_KEY, json.dumps({"schema": 1, "job": {"phase": "teleporting"}}))
    self.assertIsNone(modelstatus.status(params)["job"])
    params.put(models.STATUS_KEY, "not json")
    self.assertEqual(modelstatus.status(params), empty)

  @mock.patch.object(modelstatus, "free_bytes", lambda: 0)
  def test_status_ignores_legacy_selection_entries(self):
    params = FakeParams()
    modelstatus.publish_status(params, installed=[{"recipe": RECIPE, "name": "a"}, {"composition": "c-" + "e" * 32, "name": "legacy"}])
    installed = modelstatus.status(params)["installed"]
    self.assertEqual(len(installed), 1)
    self.assertEqual(modelview.row_selection(installed[0]), RECIPE)


class TestChooserEntries(unittest.TestCase):
  @mock.patch.object(modelstatus, "free_bytes", lambda: 0)
  def test_stock_built_and_catalog_entries_are_ordered_and_deduplicated(self):
    params = FakeParams()
    built_recipe = "b" * 64
    packaged_recipe = "c" * 64
    monitoring_recipe = "d" * 64
    catalog_recipe = "e" * 64
    modelstatus.publish_status(
      params,
      installed=[
        {"recipe": built_recipe, "name": "built", "kind": modelprotocols.DRIVING, "state": "built"},
        {"recipe": packaged_recipe, "name": "packaged", "kind": modelprotocols.DRIVING, "state": "packaged"},
        {"recipe": monitoring_recipe, "name": "monitoring", "kind": modelprotocols.MONITORING, "state": "built"},
      ],
    )
    catalog = [
      {"recipe": built_recipe, "name": "built again", "kind": modelprotocols.DRIVING, "admitted": True},
      {"recipe": packaged_recipe, "name": "packaged again", "kind": modelprotocols.DRIVING, "admitted": True},
      {"recipe": "f" * 64, "name": "refused", "kind": modelprotocols.DRIVING, "admitted": False},
      {"recipe": monitoring_recipe, "name": "wrong kind", "kind": modelprotocols.MONITORING, "admitted": True},
      {"recipe": catalog_recipe, "name": "catalog", "kind": modelprotocols.DRIVING, "admitted": True},
      {"recipe": catalog_recipe, "name": "catalog duplicate", "kind": modelprotocols.DRIVING, "admitted": True},
    ]
    with mock.patch.object(modelview, "browse", return_value={"entries": catalog}):
      entries = modelview.chooser_entries(params, modelprotocols.DRIVING)

    self.assertEqual(
      [None if entry is None else modelview.row_selection(entry) for entry in entries],
      [None, built_recipe, catalog_recipe],
    )
    built = entries[1]
    catalog_entry = entries[2]
    assert built is not None
    assert catalog_entry is not None
    self.assertEqual(built["state"], "built")
    self.assertTrue(catalog_entry["admitted"])


class TestPicker(unittest.TestCase):
  """What the pickers group, name, star and search. The catalog owns the names: a row's folder and
  short name are its own reviewed claims, and everything else falls back to the protocol and model
  class the fork admits -- so these tests pin both halves, and that nothing is invented between."""

  def _params(self, **values) -> FakeParams:
    params = FakeParams(on=False, overrides=values)
    # A built model with no catalog row: the fallback grouping has only its protocol to go on.
    modelstatus.publish_status(params, installed=[{"recipe": "b" * 64, "name": "built", "kind": modelprotocols.DRIVING,
                                               "state": "built", "protocol": "comma.supercombo.v1"}])
    return params

  def _browse(self, entries):
    return mock.patch.object(modelview, "browse", return_value={"revision": "r", "generated_at": "", "entries": entries})

  def test_the_catalogs_folder_wins_and_the_fallback_is_the_protocol_and_class(self):
    self.assertEqual(modelview.entry_group(None), modelview.GROUP_STOCK)
    self.assertEqual(modelview.entry_group({"folder": "2026 World Models"}), "2026 World Models")
    self.assertEqual(modelview.entry_group({"protocol": "comma.supercombo.v1", "model_class": "big"}), "supercombo · big")
    # Whatever the row carries: an installed model whose catalog entry is gone still knows its protocol.
    self.assertEqual(modelview.entry_group({"protocol": "comma.split-vision-policy.v1"}), "vision + policy")
    self.assertEqual(modelview.entry_group({"kind": modelprotocols.MONITORING}), "Driver monitoring")
    # A group name is carried verbatim, never normalized away.
    self.assertEqual(modelview.entry_group({"folder": "Master Models", "protocol": "comma.supercombo.v1"}), "Master Models")

  def test_stock_leads_then_the_starred_models_then_the_groups_newest_first(self):
    params = self._params()
    entries = [
      {"recipe": "c" * 64, "name": "newer", "kind": modelprotocols.DRIVING, "admitted": True, "folder": "World Models", "updated_at": "2026-03-26"},
      {"recipe": "d" * 64, "name": "older", "kind": modelprotocols.DRIVING, "admitted": True, "folder": "Legacy Models", "updated_at": "2020-11-11"},
    ]
    with self._browse(entries):
      groups = modelview.chooser_groups(params, modelprotocols.DRIVING)
      modelview.set_favorite(params, "d" * 64, True)
      starred = modelview.chooser_groups(params, modelprotocols.DRIVING)

    self.assertEqual([label for label, _rows in groups], ["", "supercombo", "World Models", "Legacy Models"])
    self.assertEqual(groups[0][1], [None])
    self.assertEqual(
      [label for label, _rows in starred], ["", modelview.GROUP_FAVORITES, "supercombo", "World Models", "Legacy Models"]
    )
    self.assertEqual([modelview.row_selection(row) for row in starred[1][1] if row is not None], ["d" * 64])
    self.assertEqual([modelview.row_selection(row) for row in starred[-1][1] if row is not None], ["d" * 64])

  def test_an_installed_model_keeps_the_group_the_catalog_gave_it(self):
    params = self._params()
    entries = [{"recipe": "b" * 64, "name": "built", "kind": modelprotocols.DRIVING, "admitted": True,
                "folder": "World Models", "short_name": "BUILT", "model_class": "big"}]
    with self._browse(entries):
      groups = modelview.chooser_groups(params, modelprotocols.DRIVING)
    self.assertEqual([label for label, _rows in groups], ["", "World Models"])
    built = groups[1][1][0]
    assert built is not None
    self.assertEqual(modelview.entry_display(built, modelprotocols.DRIVING), "built")
    self.assertEqual(built["short_name"], "BUILT")

  def test_starred_rows_stay_in_their_group_and_copy_into_favorites(self):
    params = self._params()
    entries = [
      {"recipe": "c" * 64, "name": "a", "kind": modelprotocols.DRIVING, "admitted": True, "folder": "One"},
      {"recipe": "d" * 64, "name": "b", "kind": modelprotocols.DRIVING, "admitted": True, "folder": "Two"},
      {"recipe": "e" * 64, "name": "c", "kind": modelprotocols.MONITORING, "admitted": True, "folder": "One"},
    ]
    with self._browse(entries):
      groups = modelview.chooser_groups(params, modelprotocols.DRIVING)
      modelview.set_favorite(params, "c" * 64, True)
      starred = modelview.chooser_groups(params, modelprotocols.DRIVING)
    rows = [modelview.row_selection(row) for _label, group_rows in groups for row in group_rows if row]
    self.assertEqual(sorted(rows), sorted(set(rows)))
    # The monitoring row is not in the driving picker. The star adds one Favorites copy.
    self.assertNotIn("e" * 64, rows)
    self.assertEqual(sum(len(group_rows) for _label, group_rows in starred), len(rows) + 2)
    self.assertTrue(any(label == "One" for label, _rows in starred))
    self.assertEqual(
      [modelview.row_selection(row) for row in dict(starred)["One"] if row is not None], ["c" * 64]
    )
    self.assertEqual([modelview.row_selection(row) for row in starred[1][1] if row is not None], ["c" * 64])

  def test_a_generated_name_loses_the_kind_the_picker_already_says(self):
    generated = {"name": "Driving · split · standard · 2026-03-26 · bf430805", "name_kind": "generated"}
    published = {"name": "OP Model 10 V3 (April 19, 2026)", "name_kind": "published", "short_name": "OPM10V3"}
    self.assertEqual(modelview.entry_display(generated, modelprotocols.DRIVING), "split · standard · 2026-03-26 · bf430805")
    self.assertEqual(modelview.entry_display(published, modelprotocols.DRIVING), "OP Model 10 V3 (April 19, 2026)")
    self.assertEqual(modelview.entry_display(None, modelprotocols.DRIVING), modelview.BUNDLED_LABEL)
    self.assertEqual(modelview.entry_display({}, modelprotocols.DRIVING), "")

  def test_search_matches_the_short_name_the_group_and_the_date(self):
    published = {"recipe": "b" * 64, "name": "OP Model 10 V3 (April 19, 2026)", "short_name": "OPM10V3",
                 "folder": "2026 World Models", "updated_at": "2026-04-19", "kind": modelprotocols.DRIVING, "admitted": True}
        # subsequence, not substring
    self.assertTrue(modelview.matches_query(published, "opm10", modelprotocols.DRIVING))
    self.assertTrue(modelview.matches_query(published, "world models", modelprotocols.DRIVING))
    self.assertTrue(modelview.matches_query(published, "2026-04", modelprotocols.DRIVING))
    self.assertTrue(modelview.matches_query(None, "stock", modelprotocols.DRIVING))
    self.assertFalse(modelview.matches_query(published, "tomb raider", modelprotocols.DRIVING))
    self.assertTrue(modelview.matches_query(published, "  ", modelprotocols.DRIVING))

  def test_search_keeps_a_group_that_has_a_match_and_drops_one_that_has_not(self):
    groups = [("World Models", [{"short_name": "OPM10V3", "name": "OP Model 10 V3"}]), ("Legacy Models", [{"short_name": "ND", "name": "Notre Dame"}])]
    kept = modelview.search_groups(groups, "opm10")
    self.assertEqual([label for label, _rows in kept], ["World Models"])
    self.assertEqual(modelview.search_groups(groups, "  "), groups)

  def test_the_footer_says_what_choosing_the_row_does(self):
    params = self._params()
    built = {"recipe": "b" * 64, "name": "built", "state": "built"}
    catalog = {"recipe": "c" * 64, "name": "catalog"}
    self.assertEqual(modelview.chooser_hint(None, modelprotocols.DRIVING, params, True), modelview.CHOOSER_HINT_STOCK)
    self.assertEqual(modelview.chooser_hint(built, modelprotocols.DRIVING, params, True), modelview.CHOOSER_HINT_BUILT)
    self.assertEqual(modelview.chooser_hint(catalog, modelprotocols.DRIVING, params, True), modelview.CHOOSER_HINT_INSTALL)
    self.assertTrue(modelview.CHOOSER_HINT_OFFROAD in modelview.chooser_hint(built, modelprotocols.DRIVING, params, False))
    models.select(params, modelprotocols.DRIVING, "b" * 64)
    self.assertEqual(modelview.chooser_hint(built, modelprotocols.DRIVING, params, True), modelview.CHOOSER_HINT_SELECTED)

  def test_only_a_built_selection_can_be_chosen(self):
    params = self._params()
    self.assertTrue(modelview.is_built(params, modelprotocols.DRIVING, ""))
    self.assertTrue(modelview.is_built(params, modelprotocols.DRIVING, "b" * 64))
    self.assertFalse(modelview.is_built(params, modelprotocols.DRIVING, "c" * 64))
    self.assertFalse(modelview.is_built(params, modelprotocols.MONITORING, "b" * 64))

  def test_a_star_survives_a_stale_digest_and_the_bundled_model_cannot_be_starred(self):
    params = FakeParams()
    modelview.set_favorite(params, "b" * 64, True)
    self.assertEqual(modelview.favorites(params), {"b" * 64})
    modelview.set_favorite(params, "not-a-recipe", True)
    self.assertEqual(modelview.favorites(params), {"b" * 64})
    modelview.set_favorite(params, "", True)
    self.assertEqual(modelview.favorites(params), {"b" * 64})
    modelview.set_favorite(params, "b" * 64, False)
    self.assertEqual(modelview.favorites(params), set())
    self.assertEqual(params.get(models.FAVS_KEY), "")


class TestPagePredicates(unittest.TestCase):
  def test_models_page_visibility_and_actions_share_one_state(self):
    params = FakeParams()
    models.commit_boot_selection(params)
    self.assertEqual(
      modelview.model_page_state(params, {"job": None}, False),
      {"job_present": False, "cancel": False, "refresh": True, "reboot": False, "reboot_enabled": False},
    )
    self.assertEqual(
      modelview.model_page_state(params, {"job": {"phase": modelstatus.PHASE_DOWNLOADING}}, True),
      {"job_present": True, "cancel": True, "refresh": False, "reboot": False, "reboot_enabled": True},
    )

  def test_installed_row_predicates_share_presence_and_actions(self):
    status = {"job": None}
    self.assertEqual(modelview.installed_row_state(None, status, True), {
      "present": False, "remove": False, "rebuild": False, "rebuild_enabled": False,
    })
    self.assertEqual(modelview.installed_row_state({"state": models.BUILT}, status, True), {
      "present": True, "remove": True, "rebuild": False, "rebuild_enabled": True,
    })
    busy = {"job": {"phase": modelstatus.PHASE_BUILDING}}
    self.assertEqual(modelview.installed_row_state({"state": models.PACKAGED}, busy, True), {
      "present": True, "remove": False, "rebuild": True, "rebuild_enabled": False,
    })


class TestBrowseCache(StoreCase):
  """Both trees resolve their row callables every frame, so the index must be parsed once per file
  version -- the worker's atomic rewrite is what invalidates the slot."""

  def _write_browse(self, revision: str, names: list[str]) -> None:
    models.write_json(
      models.browse_file(),
      {
        "schema": models.SCHEMA,
        "revision": revision,
        "generated_at": "2026-09-19T00:00:00Z",
        "entries": [
          {"recipe": f"{index:064x}", "name": name, "kind": modelprotocols.DRIVING, "admitted": True} for index, name in enumerate(names)
        ],
      },
    )

  def test_an_unchanged_index_is_parsed_once(self):
    self._write_browse("r1", ["a"])
    first = modelview.browse()
    self.assertIs(modelview.browse(), first, "an unchanged index must not be re-parsed")
    self.assertEqual([e["name"] for e in modelview.admitted_entries(modelprotocols.DRIVING)], ["a"])

  def test_a_rewrite_is_picked_up_and_refilters(self):
    self._write_browse("r1", ["a"])
    self.assertEqual(modelview.admitted_entries(modelprotocols.DRIVING)[0]["name"], "a")
    self._write_browse("r2", ["a", "b"])
    self.assertEqual(modelview.browse()["revision"], "r2", "the worker's rewrite must invalidate the slot")
    self.assertEqual([e["name"] for e in modelview.admitted_entries(modelprotocols.DRIVING)], ["a", "b"])

  def test_a_missing_index_reads_empty_and_never_serves_stale(self):
    self._write_browse("r1", ["a"])
    self.assertEqual(modelview.browse()["revision"], "r1")
    os.remove(models.browse_file())
    self.assertEqual(modelview.browse(), {"revision": "", "generated_at": "", "entries": []})


class TestBootCommit(StoreCase):
  def custom_params(self) -> tuple[FakeParams, str]:
    digest = write_package("supercombo", input_shapes=SUPERCOMBO_INPUTS, slices=SUPERCOMBO_SLICES)
    write_build(digest, modelprotocols.SUPERCOMBO.id)
    params = FakeParams(on=False, overrides={models.DRIVING_KEY: digest})
    models.commit_boot_selection(params)
    return params, digest

  def test_no_selection_commits_bundled(self):
    params = FakeParams()
    boot = models.commit_boot_selection(params)
    self.assertEqual(boot[modelprotocols.DRIVING]["kind"], "bundled")
    self.assertEqual(boot[modelprotocols.DRIVING]["requested"], "")
    self.assertEqual(models.boot(params), boot)
    self.assertEqual(modelview.model_label(params, modelprotocols.DRIVING), modelview.BUNDLED_LABEL)

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
    params = FakeParams(on=False, overrides={models.DRIVING_KEY: RECIPE})
    boot = models.commit_boot_selection(params)
    self.assertEqual(boot[modelprotocols.DRIVING]["kind"], "bundled")
    self.assertEqual(boot[modelprotocols.DRIVING]["requested"], RECIPE)
    self.assertEqual(boot[modelprotocols.DRIVING]["fallback"], modelprotocols.NOT_INSTALLED)
    # The driver's selection survives, and the panel can explain it.
    self.assertEqual(models.desired(params, modelprotocols.DRIVING), RECIPE)
    self.assertTrue(RECIPE[:12] in modelview.model_description(params, modelprotocols.DRIVING))

  def test_a_legacy_selection_falls_back_to_stock(self):
    legacy = "c-" + "e" * 32
    params = FakeParams(on=False, overrides={models.DRIVING_KEY: legacy})
    entry = models.commit_boot_selection(params)[modelprotocols.DRIVING]
    self.assertEqual(entry["kind"], "bundled")
    self.assertEqual(entry["requested"], legacy)
    self.assertEqual(entry["fallback"], models.UNKNOWN_SELECTION)
    self.assertEqual(models.desired(params, modelprotocols.DRIVING), legacy)

  def test_a_built_selection_resolves_to_paths(self):
    digest = write_package("supercombo", input_shapes=SUPERCOMBO_INPUTS, slices=SUPERCOMBO_SLICES)
    write_build(digest, modelprotocols.SUPERCOMBO.id)
    params = FakeParams(on=False, overrides={models.DRIVING_KEY: digest})
    entry = models.commit_boot_selection(params)[modelprotocols.DRIVING]
    self.assertEqual(entry["kind"], "custom")
    self.assertEqual(entry["protocol"], modelprotocols.SUPERCOMBO.id)
    artifact = self.artifact_sha256(digest)
    self.assertEqual(entry["members"], {"supercombo": os.path.join(models.package_dir(digest), artifact)})
    self.assertTrue(os.path.isdir(entry["build"]))
    self.assertIsNone(entry["fallback"])
    self.assertFalse(models.restart_pending(params, modelprotocols.DRIVING))

  def test_a_custom_active_model_keeps_the_selection_label(self):
    params, digest = self.custom_params()
    params.put(models.ACTIVE_DRIVING_KEY, "custom (supercombo)")
    self.assertEqual(
      modelview.model_label(params, modelprotocols.DRIVING),
      f"{digest[:12]} ({modelview.protocol_label(modelprotocols.SUPERCOMBO.id)})",
    )

  def test_a_custom_runtime_fallback_takes_over_the_short_label(self):
    params, digest = self.custom_params()
    params.put(models.ACTIVE_DRIVING_KEY, "stock: RuntimeError: broken model")
    self.assertEqual(modelview.model_label(params, modelprotocols.DRIVING), f"stock, {digest[:12]} did not load")
    self.assertTrue("RuntimeError: broken model" in modelview.model_description(params, modelprotocols.DRIVING))

  def test_a_failed_build_falls_back_with_the_compilers_message(self):
    digest = write_package("supercombo", input_shapes=SUPERCOMBO_INPUTS, slices=SUPERCOMBO_SLICES)
    write_build(digest, modelprotocols.SUPERCOMBO.id, state="failed", error="CUDA out of coffee")
    params = FakeParams(on=False, overrides={models.DRIVING_KEY: digest})
    entry = models.commit_boot_selection(params)[modelprotocols.DRIVING]
    self.assertEqual(entry["kind"], "bundled")
    self.assertEqual(entry["fallback"], "CUDA out of coffee")

  def test_the_wrong_kind_is_refused(self):
    # A driving recipe in the monitoring param: two processes, two params, and no way for a
    # monitoring model to satisfy `modeld`'s loop.
    digest = write_package("supercombo", input_shapes=SUPERCOMBO_INPUTS, slices=SUPERCOMBO_SLICES)
    write_build(digest, modelprotocols.SUPERCOMBO.id)
    params = FakeParams(on=False, overrides={models.MONITORING_KEY: digest})
    entry = models.commit_boot_selection(params)[modelprotocols.MONITORING]
    self.assertEqual(entry["kind"], "bundled")
    self.assertEqual(entry["fallback"], models.WRONG_KIND)

  def test_changed_bytes_are_refused(self):
    digest = write_package("supercombo", input_shapes=SUPERCOMBO_INPUTS, slices=SUPERCOMBO_SLICES)
    write_build(digest, modelprotocols.SUPERCOMBO.id)
    artifact = self.artifact_sha256(digest)
    with open(os.path.join(models.package_dir(digest), artifact), "wb") as handle:
      handle.write(b"tampered")
    params = FakeParams(on=False, overrides={models.DRIVING_KEY: digest})
    entry = models.commit_boot_selection(params)[modelprotocols.DRIVING]
    self.assertEqual(entry["kind"], "bundled")
    self.assertEqual(entry["fallback"], models.ARTIFACT_CHANGED)

  def test_a_restart_is_owed_when_the_selection_moves(self):
    params = FakeParams()
    models.commit_boot_selection(params)
    self.assertFalse(models.restart_pending(params, modelprotocols.DRIVING))
    models.select(params, modelprotocols.DRIVING, RECIPE)
    self.assertTrue(models.restart_pending(params, modelprotocols.DRIVING))
    self.assertEqual(modelview.model_label(params, modelprotocols.DRIVING), f"{modelview.BUNDLED_LABEL} {modelview.RESTART_NOTE}")
    with self.assertRaises(ValueError):
      models.select(params, modelprotocols.DRIVING, "nonsense")
    with self.assertRaises(ValueError):
      models.select(params, modelprotocols.DRIVING, "c-" + "e" * 32)


class TestCapacity(unittest.TestCase):
  def test_the_three_ceilings_are_sentences(self):
    self.assertEqual(modelstatus.capacity_reason(modelstatus.MAX_ARTIFACT_BYTES + 1), modelstatus.TOO_LARGE)
    with mock.patch.object(modelstatus, "free_bytes", lambda: 0):
      self.assertEqual(modelstatus.capacity_reason(1), modelstatus.NO_SPACE)
    with (
      mock.patch.object(modelstatus, "free_bytes", lambda: modelstatus.MAX_STORE_BYTES),
      mock.patch.object(modelstatus, "store_usage", lambda: modelstatus.MAX_STORE_BYTES),
    ):
      self.assertEqual(modelstatus.capacity_reason(1), modelstatus.STORE_FULL)


class TestJobReporting(unittest.TestCase):
  """What the panels render for a running job: the phase line, the bar's fraction, its label.

  Two of these are load-bearing. The fraction must be `None` for every phase without a total, because
  a bar drawn from nothing is a claim about progress no number backs. And the elapsed clock is only
  published while a job runs (`Job.started_at`), so an older worker's snapshot has no `elapsed` and
  the line must read as it did before -- a panel that required it would show `None` to the driver.
  """

  def test_job_active_matches_the_live_phases(self):
    for phase in modelstatus.PHASES:
      with self.subTest(phase=phase):
        self.assertEqual(modelstatus.job_active({"phase": phase}), phase in modelstatus.ACTIVE_PHASES)
    self.assertFalse(modelstatus.job_active(None))

  def test_the_patient_phases_carry_a_clock(self):
    for phase in modelstatus.PATIENT_PHASES:
      with self.subTest(phase=phase):
        self.assertEqual(modelview.job_text({"phase": phase, "elapsed": 83}), f"{modelview.PHASE_TEXT[phase]} 1:23")
        # No elapsed (a snapshot from a worker that did not publish one) reads as the phase alone.
        self.assertEqual(modelview.job_text({"phase": phase}), modelview.PHASE_TEXT[phase])

  def test_the_clock_is_minutes_and_hours(self):
    self.assertEqual(modelview.clock(0), "0:00")
    self.assertEqual(modelview.clock(83), "1:23")
    self.assertEqual(modelview.clock(3800), "1:03:20")
    self.assertEqual(modelview.clock(-5), "0:00")

  def test_a_download_shows_bytes_not_a_clock(self):
    job = {"phase": "downloading", "received": 24_320_000, "total": 60_881_999, "elapsed": 12.0}
    self.assertEqual(modelview.job_text(job), "downloading 23.2 MB of 58.1 MB")

  def test_finished_and_failed_phases_show_no_clock(self):
    for phase in (modelstatus.PHASE_DONE, modelstatus.PHASE_ERROR, modelstatus.PHASE_CANCELED):
      with self.subTest(phase=phase):
        self.assertEqual(modelview.job_text({"phase": phase, "elapsed": 900}), modelview.PHASE_TEXT[phase])

  def test_the_fraction_exists_only_where_a_total_does(self):
    self.assertIsNone(modelview.job_fraction(None))
    self.assertIsNone(modelview.job_fraction({"phase": "building", "total": 100}))
    self.assertIsNone(modelview.job_fraction({"phase": "waiting-network"}))
    self.assertIsNone(modelview.job_fraction({"phase": "downloading", "total": 0}))
    self.assertEqual(modelview.job_fraction({"phase": "downloading", "received": 50, "total": 100}), 0.5)
    # A download cannot exceed its declared size -- the worker refuses that -- but a bar must not
    # draw past its track if one ever did.
    self.assertEqual(modelview.job_fraction({"phase": "downloading", "received": 200, "total": 100}), 1.0)

  def test_the_bar_label_carries_the_percentage_and_the_bytes(self):
    job = {"phase": "downloading", "received": 24_320_000, "total": 60_881_999}
    text = modelview.job_progress_text(job)
    self.assertTrue(text.startswith("40%"))
    self.assertTrue(modelview.human_size(24_320_000) in text)
    self.assertTrue(modelview.human_size(60_881_999) in text)
    # With nothing to measure the label is the phase line, so the two renderers can use one call.
    self.assertEqual(modelview.job_progress_text({"phase": "building", "elapsed": 83}), "building 1:23")

  @mock.patch.object(modelstatus, "free_bytes", lambda: 0)
  def test_a_live_job_publishes_its_elapsed_but_one_that_never_started_does_not(self):
    params = FakeParams()
    modelstatus.publish_status(params, job=modelstatus.Job(id="1", op="install", phase="downloading", received=5, total=10))
    # No `started_at`: a hand-made Job (and an older worker's snapshot) has no clock.
    self.assertIsNone(modelstatus.status(params)["job"]["elapsed"])
    started = modelstatus.Job(id="1", op="install", phase="building", started_at=time.monotonic() - 30)
    modelstatus.publish_status(params, job=started)
    elapsed = modelstatus.status(params)["job"]["elapsed"]
    self.assertIsNotNone(elapsed)
    self.assertGreaterEqual(float(elapsed), 30.0)
    # A nonsense value from a broken writer is dropped rather than rendered.
    params.put(models.STATUS_KEY, json.dumps({"schema": 1, "job": {"phase": "building", "elapsed": "soon"}}))
    self.assertIsNone(modelstatus.status(params)["job"]["elapsed"])


class TestWorkerPredicate(unittest.TestCase):
  """The manager's modelsd row starts an initial listing and later requests."""

  def wanted(self, **values) -> bool:
    from openpilot.system.manager.process_config import managed_processes

    store = params(overrides=values)
    return managed_processes["modelsd"].should_run(False, store, None)


  def test_a_boot_with_no_status_starts_the_worker(self):
    self.assertTrue(self.wanted())

  def test_a_published_status_with_no_request_reaps_it(self):
    self.assertFalse(self.wanted(**{models.STATUS_KEY: json.dumps(modelstatus.empty_status())}))

  def test_a_request_starts_it_again(self):
    self.assertTrue(self.wanted(**{models.STATUS_KEY: "{}", models.REQUEST_KEY: "{}"}))


class TestDeclaredSmoothing(unittest.TestCase):
  """One number for the model's timing: modeld decodes with the selection's declared smoothing, and
  controlsd's curvature reference and torque controller must time against that same value rather than
  modeld's module default (moonpilot/docs/models.md)."""

  def test_the_boot_snapshot_carries_the_declared_value(self):
    params = FakeParams()
    self.assertEqual(models.boot_configuration(params, "LAT_SMOOTH_SECONDS", 0.0), 0.0)
    snapshot = {"schema": models.SCHEMA, modelprotocols.DRIVING: {"kind": "custom", "configuration": {"LAT_SMOOTH_SECONDS": 0.1}}}
    params.put(models.BOOT_KEY, json.dumps(snapshot))
    self.assertAlmostEqual(models.boot_configuration(params, "LAT_SMOOTH_SECONDS", 0.0), 0.1)
    self.assertAlmostEqual(models.boot_configuration(params, "LONG_SMOOTH_SECONDS", 0.3), 0.3)
    params.put(
      models.BOOT_KEY,
      json.dumps({"schema": models.SCHEMA, modelprotocols.DRIVING: {"kind": "custom", "configuration": {"LAT_SMOOTH_SECONDS": "fast"}}}),
    )
    self.assertEqual(models.boot_configuration(params, "LAT_SMOOTH_SECONDS", 0.0), 0.0)



if __name__ == "__main__":
  unittest.main()
