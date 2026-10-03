import unittest
from unittest.mock import patch

from openpilot.selfdrive.ui.soundd import ALERT_MAX_TIME, CRITICAL_MAX, AudibleAlert, Soundd


class TestSounddAlertVolume(unittest.TestCase):
  def test_warning_alert_holds_start_volume_and_keeps_critical_sound_switch(self):
    for alert in (AudibleAlert.warningSoft, AudibleAlert.warningImmediate):
      with self.subTest(alert=alert):
        soundd = Soundd.__new__(Soundd)
        soundd.current_alert = AudibleAlert.none
        soundd.current_sound = AudibleAlert.none
        soundd.current_volume = 0.37
        soundd.current_sound_frame = 0
        soundd.ramp_start_volume = 0.1
        soundd.ramp_start_time = 0.0
        soundd.pending_stop = False

        with patch("openpilot.selfdrive.ui.soundd.time.monotonic", return_value=10.0):
          soundd.update_alert(alert)

        self.assertEqual(soundd.ramp_start_volume, 0.37)
        for elapsed in (4.0, ALERT_MAX_TIME):
          with patch("openpilot.selfdrive.ui.soundd.time.monotonic", return_value=10.0 + elapsed):
            soundd.update_alert_volume()
          self.assertEqual(soundd.current_volume, 0.37)
          if elapsed < ALERT_MAX_TIME:
            self.assertEqual(soundd.current_sound, alert)
          else:
            self.assertEqual(soundd.current_sound, CRITICAL_MAX)
            self.assertEqual(soundd.current_sound_frame, 0)

  def test_non_warning_alert_does_not_change_volume_or_sound(self):
    soundd = Soundd.__new__(Soundd)
    soundd.current_alert = AudibleAlert.engage
    soundd.current_sound = AudibleAlert.engage
    soundd.current_volume = 0.37
    soundd.current_sound_frame = 7
    soundd.ramp_start_volume = 0.1
    soundd.ramp_start_time = 0.0

    soundd.update_alert_volume()

    self.assertEqual(soundd.current_volume, 0.37)
    self.assertEqual(soundd.current_sound, AudibleAlert.engage)
    self.assertEqual(soundd.current_sound_frame, 7)


if __name__ == "__main__":
  unittest.main()
