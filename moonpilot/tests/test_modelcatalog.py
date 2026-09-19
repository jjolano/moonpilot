"""Admission and composition, against real recipes from the openmodels catalog.

The fixture (`fixtures/catalog.json`) is a trimmed snapshot at the catalog's own revision
`2026-09-19T06:48:29Z`: the recipe and profile *documents* are copied byte for byte -- their digests
are hashes of those bytes, so a fixture that edited them would fail to load at all -- with the
profiles' prose fields dropped (nothing reads them and they are en-GB). Every digest here was read
off the live catalog, and `test_fixture_is_a_valid_snapshot_twice_over` re-checks the copy against
the vendored SDK, which is also what makes this file the vendored copy's test.

What each case is for:

- the pinned stock recipe and two archived supercombo generations, admitted;
- a vision+policy and a vision+off+on+policy set, admitted, and a *second* vision+policy recipe with
  a different `LAT_SMOOTH_SECONDS` so the composition rule's refusal is real;
- a split with an action head from before the 2026-06-01 change, admitted under the family's pinned
  decode, and one from after it, refused -- the one pair where a wrong answer is a wrong steering
  request;
- two driver-monitoring models, one with the sleep heads and one without;
- two refusals that are not about the model at all: a split whose profile does not record the
  feature port, and a driver-monitoring model missing the heads this tree reads.
"""

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from moonpilot import modelcatalog, models
from moonpilot.vendor.openmodels import client, contracts, metadata

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "moonpilot/tests/fixtures/catalog.json"

STOCK = "84dbb0fb5e2052fbe34cacf2f216de31d0478560b33bb59a4bbb306eb834eed4"
ARCHIVED_ACTION = "01420ce0c87bf29e75bcc57ea9618f7a7bb3a64c47866acc504fe9f68019103c"
VISION_POLICY = "3fcfe14f14a2dc0e75ec37f91df42e702015cbe29eefaf755836781b0b11159f"
VISION_POLICY_OTHER = "051a266d64675ba7ceda62671b6f4ad4395619014223817db002d97178e0a333"
VISION_OFF_ON = "5f533ea72befe84bebcb8d88062625404b43215f645ca9ce09b3a4bbb114f2bc"
SPLIT_OLD_ACTION = "03a42d6ffbf4518710a75706a3ecd9a91b8ddad0a8de2b1fca33c9fe10ddbd67"
SPLIT_NEW_ACTION = "148c4b3658079ebb26053500fb55366fb63f804e0b976e1e99ff434d5ce795ec"
DM_WITH_SLEEP = "5e8b3b3b3c0e2b1a2218671a4776f961a2ee7d63548143f1ab9431ae99799ca0"
DM_WITHOUT_SLEEP = "016469e1b109c8028f29dbc5ee935bb76ca58f3ed31f30a3d2df8a2dc6b1cc2a"
DM_MISSING_HEADS = "3870150bbb21a19f29730ba327ce0595f425a37e17f51291700b54381f0d9032"
SPLIT_NO_FEATURE_PORT = "17b12acc43b4ee54e44cb9c7ea2a95a3f274b16d05499840f295610e659816fc"


class CatalogCase(unittest.TestCase):
  """The fixture, loaded once per test, with a temp store so nothing touches the real one."""

  @classmethod
  def setUpClass(cls):
    cls.raw = FIXTURE.read_bytes()
    cls.snapshot = json.loads(cls.raw)

  def setUp(self):
    self.tmp = tempfile.mkdtemp()
    patcher = mock.patch.object(models.paths, "data_dir", lambda feature: os.path.join(self.tmp, feature))
    patcher.start()
    self.addCleanup(patcher.stop)
    self.catalog = client.Catalog(self.raw, base_url="https://catalog.invalid/")

  def admit(self, digest: str) -> dict:
    return modelcatalog.admit(self.catalog, digest)


class TestFixture(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.raw = FIXTURE.read_bytes()
    cls.snapshot = json.loads(cls.raw)

  def test_fixture_is_a_valid_snapshot_twice_over(self):
    # Once through the vendored loader, which verifies every manifest digest against the document it
    # names -- so a hand-edited fixture fails here -- and once by rebuilding each document through
    # `Manifest.create`, which is the same identity check from the other side and the vendored
    # copy's own test.
    catalog = client.Catalog(self.raw, base_url="https://catalog.invalid/")
    self.assertEqual(catalog.revision, contracts.sha256(self.raw))
    self.assertEqual(len(catalog.data["entries"]), len(self.snapshot["entries"]))
    for digest, raw in self.snapshot["documents"].items():
      self.assertEqual(contracts.Manifest.create(json.loads(raw)).id, digest, digest)

  def test_every_recipe_names_a_profile_the_fixture_carries(self):
    for digest, raw in self.snapshot["documents"].items():
      document = json.loads(raw)
      if document["type"] == "recipe":
        self.assertTrue(document["profile"] in self.snapshot["documents"], digest)


def _case(name: str):
  def test(self):
    verdict = self.admit(getattr(self, name))
    self.assertTrue(verdict["admitted"], f"{name}: {verdict['reason']}")
    self.assertTrue(verdict["protocol"])

  return test


class TestAdmission(CatalogCase):
  def test_the_pinned_stock_recipe_admits_as_supercombo(self):
    verdict = self.admit(STOCK)
    self.assertTrue(verdict["admitted"], verdict["reason"])
    self.assertEqual(verdict["protocol"], models.SUPERCOMBO.id)
    # The catalog's honest caveats ride along and are never flattened into "compatible".
    self.assertTrue(any("requires_runner_check" in note for note in verdict["notes"]))
    self.assertTrue(any("consumer_owned" in note for note in verdict["notes"]))

  def test_an_archived_supercombo_with_an_action_head_still_admits(self):
    verdict = self.admit(ARCHIVED_ACTION)
    self.assertTrue(verdict["admitted"], verdict["reason"])
    self.assertEqual(verdict["protocol"], models.SUPERCOMBO.id)

  def test_the_split_families_admit(self):
    self.assertEqual(self.admit(VISION_POLICY)["protocol"], models.SPLIT_VISION_POLICY.id)
    self.assertEqual(self.admit(VISION_POLICY_OTHER)["protocol"], models.SPLIT_VISION_POLICY.id)
    self.assertEqual(self.admit(VISION_OFF_ON)["protocol"], models.SPLIT_VISION_OFF_ON.id)

  def test_the_driver_monitoring_generations_admit_with_and_without_the_sleep_heads(self):
    self.assertEqual(self.admit(DM_WITH_SLEEP)["protocol"], models.DMONITORING.id)
    self.assertEqual(self.admit(DM_WITHOUT_SLEEP)["protocol"], models.DMONITORING.id)

  def test_the_action_eras_are_split_by_the_protocols_own_convention(self):
    # Both of these carry an `action` head; the older one is the family the pinned decode was
    # written for, and the newer one belongs to the convention that came with the fused models. The
    # refusal is what keeps a wrong scale factor off the road.
    self.assertTrue(self.admit(SPLIT_OLD_ACTION)["admitted"])
    refusal = self.admit(SPLIT_NEW_ACTION)
    self.assertFalse(refusal["admitted"])
    self.assertEqual(refusal["reason"], f"on_policy: {models.ACTION_ERA}")
    self.assertEqual(refusal["protocol"], models.SPLIT_VISION_OFF_ON.id)

  def test_a_missing_feature_port_refuses_a_split(self):
    refusal = self.admit(
      "17b12acc43b4"
      if "17b12acc43b4" in self.snapshot["documents"]
      else next(
        d
        for d, raw in self.snapshot["documents"].items()
        if json.loads(raw)["type"] == "recipe" and "vision" in json.loads(raw)["members"] and not json.loads(raw)["members"]["vision"]["outputs"]
      )
    )
    self.assertFalse(refusal["admitted"])
    self.assertTrue(refusal["reason"].startswith("structure_unknown"), refusal["reason"])

  def test_a_driver_monitoring_model_without_the_read_heads_refuses(self):
    without = [
      digest
      for digest, raw in self.snapshot["documents"].items()
      if json.loads(raw)["type"] == "recipe"
      and "dmonitoring" in json.loads(raw)["members"]
      and "face_descs_lhd" not in (json.loads(raw)["members"]["dmonitoring"]["metadata"].get("output_slices") or {})
    ]
    self.assertTrue(without, "the fixture lost its minimized DM recipe")
    refusal = self.admit(without[0])
    self.assertFalse(refusal["admitted"])
    self.assertTrue(refusal["reason"].startswith(models.MISSING_OUTPUTS), refusal["reason"])

  def test_a_recipe_the_catalog_does_not_have_is_refused_by_name(self):
    verdict = self.admit("f" * 64)
    self.assertFalse(verdict["admitted"])
    self.assertTrue(verdict["reason"])


class TestBrowse(CatalogCase):
  def test_the_index_is_one_entry_per_recipe_with_its_verdict(self):
    index = modelcatalog.browse_index(self.catalog)
    self.assertEqual(index["total"], len(self.snapshot["entries"]))
    self.assertEqual(index["revision"], self.catalog.revision)
    entries = {entry["recipe"]: entry for entry in index["entries"]}
    self.assertTrue(STOCK in entries)
    self.assertTrue(entries[STOCK]["admitted"])
    self.assertEqual(entries[STOCK]["roles"], ["supercombo"])
    self.assertTrue(entries[SPLIT_NEW_ACTION]["reason"].endswith(models.ACTION_ERA))
    for entry in index["entries"]:
      self.assertEqual(
        set(entry),
        {
          "recipe",
          "composition",
          "name",
          "kind",
          "family",
          "model_class",
          "archived",
          "roles",
          "size",
          "targets",
          "available",
          "updated_at",
          "protocol",
          "admitted",
          "reason",
          "notes",
        },
      )

  def test_the_index_is_newest_first(self):
    index = modelcatalog.browse_index(self.catalog)
    dates = [entry["updated_at"] for entry in index["entries"]]
    self.assertEqual(dates, sorted(dates, reverse=True))

  def test_writing_the_index_is_what_the_panels_read(self):
    written = modelcatalog.write_browse(self.catalog)
    # `models.browse()` is the panels' reader: the same entries, without the writer's `schema`/`total`.
    self.assertEqual(models.browse()["revision"], written["revision"])
    self.assertEqual(models.browse()["entries"], written["entries"])


class TestCompose(CatalogCase):
  def test_members_that_disagree_on_smoothing_cannot_be_one_model(self):
    verdict = modelcatalog.compose(self.catalog, models.SPLIT_VISION_POLICY.id, {"vision": VISION_POLICY, "on_policy": VISION_POLICY_OTHER})
    self.assertIsNone(verdict["record"])
    self.assertEqual(verdict["reason"], models.MEMBERS_DISAGREE.format(key="LAT_SMOOTH_SECONDS"))

  def test_picks_from_two_protocols_refuse(self):
    verdict = modelcatalog.compose(self.catalog, models.SPLIT_VISION_POLICY.id, {"vision": VISION_POLICY, "on_policy": SPLIT_NEW_ACTION})
    self.assertIsNotNone(verdict["reason"])
    self.assertIsNone(verdict["record"])

  def test_a_role_set_the_protocol_does_not_have_refuses(self):
    verdict = modelcatalog.compose(self.catalog, models.SPLIT_VISION_POLICY.id, {"vision": VISION_POLICY})
    self.assertTrue(models.UNKNOWN_ROLE in str(verdict["reason"]))

  def test_one_recipe_under_every_role_is_not_a_composition(self):
    # The reference run at the catalog's own revision: a set whose roles all come from one recipe is
    # that recipe, and the digest is the selection -- a wrapper record would be a second name for one
    # model with its own build directory to keep in step.
    verdict = modelcatalog.compose(self.catalog, models.SUPERCOMBO.id, {"supercombo": STOCK})
    self.assertEqual(verdict["reason"], models.SINGLE_RECIPE)
    self.assertIsNone(verdict["record"])


class TestInterfaceVerification(unittest.TestCase):
  """A downloaded artifact is only usable if its bytes declare the interface its recipe recorded."""

  def setUp(self):
    self.tmp = tempfile.mkdtemp()
    patcher = mock.patch.object(models.paths, "data_dir", lambda feature: os.path.join(self.tmp, feature))
    patcher.start()
    self.addCleanup(patcher.stop)

  def package_from_onnx(self, *, tamper: bool = False) -> str:
    """A package whose single artifact is the bundled DM ONNX, symlinked so nothing is copied.

    The point is the *interface*: `verify_downloaded` re-reads it from those bytes and compares it
    with what the recipe recorded, which is the check that a download is what admission decided
    about.
    """
    source = ROOT / "openpilot/selfdrive/modeld/models/dmonitoring_model.onnx"
    interface = metadata.parse(str(source))
    document = {"schema": 1, "type": "recipe", "profile": "c" * 64,
                "members": {"dmonitoring": {"artifact": {"sha256": "e" * 64, "size": source.stat().st_size, "format": "onnx"},
                                            "source": {}, "configuration": {}, "missing": [], "inputs": {}, "outputs": {},
                                            "targets": ["QCOM"],
                                            "metadata": {"input_shapes": interface["input_shapes"],
                                                         "output_slices": interface["output_slices"]}}},
                "configuration": {}}
    if tamper:
      # A package that is internally consistent -- written under its own digest, like a real install
      # -- but whose recorded interface is not what its artifact declares.
      document["members"]["dmonitoring"]["metadata"]["output_slices"]["not_a_head"] = [0, 1, None]
    raw = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    recipe = hashlib.sha256(raw.encode()).hexdigest()
    os.makedirs(models.package_dir(recipe), exist_ok=True)
    with open(os.path.join(models.package_dir(recipe), "recipe.json"), "w", encoding="utf-8") as handle:
      handle.write(raw)
    os.symlink(source, os.path.join(models.package_dir(recipe), "e" * 64))
    return recipe

  def test_a_package_whose_bytes_agree_verifies(self):
    self.assertIsNone(modelcatalog.verify_downloaded(None, self.package_from_onnx()))

  def test_a_package_whose_bytes_disagree_is_a_mismatch(self):
    self.assertEqual(modelcatalog.verify_downloaded(None, self.package_from_onnx(tamper=True)),
                     modelcatalog.INTERFACE_MISMATCH)

  def test_verify_onnx_never_executes_the_file(self):
    # The only reader of a downloaded artifact: a hostile `output_slices` pickle cannot resolve
    # anything but `slice`, and an artifact with none is refused rather than trusted.
    source = ROOT / "openpilot/selfdrive/modeld/models/dmonitoring_model.onnx"
    with self.assertRaises(ValueError):
      modelcatalog.verify_onnx(str(ROOT / "moonpilot/tests/fixtures/catalog.json"))
    interface = modelcatalog.verify_onnx(str(source))
    self.assertTrue("input_img" in interface["input_shapes"])
    self.assertTrue("face_descs_lhd" in interface["slices"])
    self.assertEqual(len(interface["slices"]), 20)


if __name__ == "__main__":
  unittest.main()
