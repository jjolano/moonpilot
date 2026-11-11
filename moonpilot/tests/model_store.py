import hashlib
import json
import os

from moonpilot import modelprotocols, models


class TempStoreMixin:
  def setUp(self):
    import tempfile
    from unittest import mock


    self.tmp = tempfile.TemporaryDirectory()
    self.addCleanup(self.tmp.cleanup)
    for patcher in (
      mock.patch.object(models.paths, "data_dir", lambda feature: os.path.join(self.tmp.name, feature)),
    ):
      patcher.start()
      self.addCleanup(patcher.stop)
    os.makedirs(models.packages_dir(), exist_ok=True)


def profile_document(name: str) -> dict:
  return {
    "schema": 1,
    "type": "profile",
    "name": name,
    "slot_sets": [["supercombo"]],
    "connections": [],
    "required_configuration": list(modelprotocols.CONFIGURATION_KEYS),
    "inputs": {},
    "outputs": {},
    "state": {},
    "implementation_source": {},
  }


def write_package(role: str, *, input_shapes: dict | None = None, slices: dict | None = None,
                  configuration: dict | None = None, size: int = 4) -> str:
  artifact = bytes(size)
  artifact_digest = hashlib.sha256(artifact).hexdigest()
  agreed = configuration or {"frame_skip": 4, "LAT_SMOOTH_SECONDS": 0.0, "LONG_SMOOTH_SECONDS": 0.3}
  document = {
    "schema": 1,
    "type": "recipe",
    "profile": "c" * 64,
    "members": {role: {"artifact": {"sha256": artifact_digest, "size": size, "format": "onnx"},
                       "source": {}, "configuration": dict(agreed), "missing": [], "inputs": {}, "outputs": {},
                       "targets": ["QCOM"], "metadata": {"input_shapes": input_shapes or {}, "output_slices": slices or {}}}},
    "configuration": dict(agreed),
  }
  raw = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
  digest = hashlib.sha256(raw.encode()).hexdigest()
  os.makedirs(models.package_dir(digest), exist_ok=True)
  with open(os.path.join(models.package_dir(digest), "recipe.json"), "w", encoding="utf-8") as handle:
    handle.write(raw)
  models.write_json(os.path.join(models.package_dir(digest), "profile.json"), profile_document("comma/test/v1"))
  with open(os.path.join(models.package_dir(digest), artifact_digest), "wb") as handle:
    handle.write(artifact)
  return digest


def write_build(selection: str, protocol: str, *, state: str = "built", error: str | None = None,
                record_protocol: str | None = None, action_contract: str | None = None) -> str:
  key = models.build_key(selection, protocol, models.CAMERA_RESOLUTIONS)
  os.makedirs(os.path.dirname(models.build_model(key)), exist_ok=True)
  if state == "built":
    with open(models.build_model(key), "wb") as handle:
      handle.write(b"pickle")
  record = {"schema": 1, "state": state, "error": error,
            "protocol": record_protocol or protocol, "selection": selection}
  if action_contract is not None:
    record["action_contract"] = action_contract
  models.write_json(models.build_record(key), record)
  return key
