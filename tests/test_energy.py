"""Tests for bench/energy.py that need neither root nor a model.

    .venv/bin/python -m unittest tests.test_energy -v

powermetrics only runs as root, so its output is built here from the field names
it uses on Apple silicon (the format strings in /usr/bin/powermetrics:
elapsed_ns, processor.{cpu,gpu,ane}_power and _energy, processor.combined_power,
gpu.gpu_energy). A stand-in powermetrics script exercises the background
process, the stop signal and the lead-in handling end to end.
"""

import math
import os
import plistlib
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from datetime import datetime
from pathlib import Path

from bench import energy

ROOT = Path(__file__).resolve().parent.parent


def sample(cpu=1200.0, gpu=40.0, ane=0.0, ms=200.0, powers=True, energies=True, combined=True):
    """One synthetic powermetrics sample: CPU, GPU and ANE power in mW over `ms` milliseconds."""
    s = ms / 1000
    proc = {"clusters": [{"name": "E-Cluster", "freq_hz": 1.02e9, "idle_ratio": 0.8, "cpus": []},
                         {"name": "P0-Cluster", "freq_hz": 3.9e9, "idle_ratio": 0.4, "cpus": []}]}
    if powers:
        proc.update(cpu_power=cpu, gpu_power=gpu, ane_power=ane)
    if energies:  # integers, as powermetrics prints them
        proc.update(cpu_energy=round(cpu * s), gpu_energy=round(gpu * s), ane_energy=round(ane * s))
    if combined:
        proc["combined_power"] = cpu + gpu + ane
    return {"is_delta": True, "elapsed_ns": int(ms * 1e6), "hw_model": "synthetic",
            "kern_osversion": "26A434", "timestamp": datetime(2026, 10, 15, 23, 0, 0),
            "processor": proc,
            "gpu": {"freq_hz": 3.38e8, "idle_ratio": 0.95, "gpu_energy": round(gpu * s)}}


def stream(samples, sep=b"\0", cut_off_tail=True) -> bytes:
    """Samples as `powermetrics -f plist` writes them: XML plists separated by NULs."""
    parts = [plistlib.dumps(s, fmt=plistlib.FMT_XML) for s in samples]
    data = sep.join(parts) + sep
    if cut_off_tail:  # what is left when powermetrics is stopped mid-write
        data += parts[0][: len(parts[0]) // 2]
    return data


# Stand-in for /usr/bin/powermetrics: checks the arguments bench.energy passes, then
# writes one sample per interval to -o until SIGINT. FAKE_MW sets the package power.
FAKE = textwrap.dedent("""
    import os, plistlib, signal, sys, time
    a = sys.argv[1:]
    get = lambda flag: a[a.index(flag) + 1]
    if get("--samplers") != "cpu_power,gpu_power,ane_power" or get("-f") != "plist":
        sys.exit(f"unexpected arguments {a}")
    interval, mw = int(get("-i")) / 1000, float(os.environ.get("FAKE_MW", "1000"))
    stop = []
    signal.signal(signal.SIGINT, lambda *_: stop.append(1))
    with open(get("-o"), "wb") as fh:
        while not stop:
            t0 = time.monotonic()
            time.sleep(interval)
            ns = int((time.monotonic() - t0) * 1e9)
            proc = {"cpu_power": 0.5 * mw, "gpu_power": 0.1 * mw, "ane_power": 0.4 * mw,
                    "combined_power": mw}
            fh.write(plistlib.dumps({"elapsed_ns": ns, "processor": proc}) + b"\\0")
            fh.flush()
        fh.write(b"<?xml version=\\"1.0\\"?><plist><dict><key>elapsed_ns</key>")  # cut off
""")


class ParsePlists(unittest.TestCase):
    def test_nul_separated_stream_drops_cut_off_tail(self):
        got = energy.parse_plists(stream([sample(cpu=c) for c in (100, 200, 300)]))
        self.assertEqual([s["processor"]["cpu_power"] for s in got], [100, 200, 300])

    def test_separators_and_whitespace_do_not_matter(self):
        samples = [sample(cpu=c) for c in (100, 200)]
        for sep in (b"", b"\n", b"\0\n", b"\n\0\n"):
            self.assertEqual(len(energy.parse_plists(stream(samples, sep=sep))), 2, sep)

    def test_garbage_is_skipped(self):
        data = b"\0" + b"<plist version=\"1.0\"><dict><key>x</key></plist>\0" + stream([sample()])
        self.assertEqual(len(energy.parse_plists(data)), 1)
        self.assertEqual(energy.parse_plists(b""), [])

    def test_cut_off_sample_does_not_swallow_the_next(self):
        whole = plistlib.dumps(sample(cpu=700), fmt=plistlib.FMT_XML)
        data = whole[: len(whole) // 2] + b"\0" + whole + b"\0"
        self.assertEqual([s["processor"]["cpu_power"] for s in energy.parse_plists(data)], [700])


class SamplePower(unittest.TestCase):
    def test_power_fields(self):
        p = energy.sample_power(sample(cpu=1200, gpu=40, ane=760))
        self.assertEqual((p["cpu_mw"], p["gpu_mw"], p["ane_mw"], p["package_mw"]), (1200, 40, 760, 2000))
        self.assertAlmostEqual(p["seconds"], 0.2)

    def test_energy_fields_when_power_missing(self):
        # mJ over the sample / seconds = mW
        p = energy.sample_power(sample(cpu=1200, gpu=40, ane=760, powers=False, combined=False))
        self.assertAlmostEqual(p["cpu_mw"], 1200)
        self.assertAlmostEqual(p["gpu_mw"], 40)
        self.assertAlmostEqual(p["ane_mw"], 760)
        self.assertAlmostEqual(p["package_mw"], 2000)

    def test_gpu_falls_back_to_gpu_dict(self):
        s = sample(gpu=50, powers=False)
        del s["processor"]["gpu_energy"]
        self.assertAlmostEqual(energy.sample_power(s)["gpu_mw"], 50)

    def test_combined_power_is_the_package(self):
        s = sample(cpu=1000, gpu=0, ane=0)
        s["processor"]["combined_power"] = 1234.0
        self.assertEqual(energy.sample_power(s)["package_mw"], 1234.0)

    def test_missing_unit_without_combined_is_unusable(self):
        s = sample(powers=False, energies=False, combined=False)
        self.assertIsNone(energy.sample_power(s))
        s["processor"]["combined_power"] = 900.0  # package still known; units unknown
        p = energy.sample_power(s)
        self.assertEqual(p["package_mw"], 900.0)
        self.assertIsNone(p["cpu_mw"])

    def test_invalid_samples(self):
        bad = [sample(), sample(), sample(), {"elapsed_ns": 200_000_000}, {"processor": {}}]
        bad[0]["invalid"] = True
        bad[1]["processor"]["invalid"] = True
        bad[2]["elapsed_ns"] = 0
        for s in bad:
            self.assertIsNone(energy.sample_power(s))
        nan = sample(cpu=500, powers=True, energies=False)
        nan["processor"]["cpu_power"] = math.nan
        self.assertIsNone(energy.sample_power(nan)["cpu_mw"])


class MeanPower(unittest.TestCase):
    def test_time_weighted(self):
        m = energy.mean_power([sample(cpu=1000, gpu=0, ms=100), sample(cpu=4000, gpu=0, ms=300)])
        self.assertAlmostEqual(m["package_mw"], (1000 * 0.1 + 4000 * 0.3) / 0.4)
        self.assertEqual(m["samples"], 2)
        self.assertAlmostEqual(m["sampled_s"], 0.4)

    def test_lead_in_samples_dropped(self):
        # Two 200 ms samples (0-0.4 s) during the lead-in, then steady state.
        samples = [sample(cpu=9000, gpu=0)] * 2 + [sample(cpu=3000, gpu=0)] * 5
        m = energy.mean_power(samples, skip_s=0.4)
        self.assertEqual(m["samples"], 5)
        self.assertAlmostEqual(m["package_mw"], 3000)
        # A sample that starts inside the lead-in is dropped even if it ends after it.
        self.assertEqual(energy.mean_power(samples, skip_s=0.3)["samples"], 5)

    def test_unusable_samples_still_advance_the_clock(self):
        samples = [sample(cpu=9000, gpu=0), sample(cpu=9000, gpu=0), sample(cpu=3000, gpu=0)]
        samples[0]["invalid"] = True
        m = energy.mean_power(samples, skip_s=0.4)
        self.assertEqual(m["samples"], 1)
        self.assertAlmostEqual(m["package_mw"], 3000)

    def test_nothing_left(self):
        with self.assertRaises(RuntimeError):
            energy.mean_power([sample()], skip_s=1.0)

    def test_parse_then_average(self):
        data = stream([sample(cpu=2000, gpu=100, ane=900)] * 10)
        m = energy.mean_power(energy.parse_plists(data), skip_s=0.4)
        self.assertEqual(m["samples"], 8)
        self.assertAlmostEqual(m["package_mw"], 3000)
        self.assertAlmostEqual(m["ane_mw"], 900)


class Energy(unittest.TestCase):
    def test_formula(self):
        # 2 W above idle for 20 s over 10,000 inferences = 4 mJ each
        self.assertAlmostEqual(energy.energy_per_inference(3000, 1000, 20, 10_000), 4.0)

    def test_row(self):
        m = {"inferences": 500, "duration_s": 2.0, "samples": 9, "sampled_s": 1.8,
             "package_mw": 3000.0, "cpu_mw": 2000.0, "gpu_mw": 100.0, "ane_mw": 900.0}
        row = energy.row_for("Apple M4 Pro", "coreml_fp16", "coreml-cpu_and_ne", m, 1000.0)
        self.assertEqual(list(row), energy.COLUMNS)
        self.assertEqual((row["inf_per_s"], row["mj_per_inf"]), (250.0, 8.0))
        idle = energy.row_for("Apple M4 Pro", "idle", "-", {**m, "inferences": 0}, None)
        self.assertIsNone(idle["mj_per_inf"])


class WithStandIn(unittest.TestCase):
    """bench.energy.measure against a stand-in powermetrics: process handling, not physics."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        fake = Path(self.tmp.name) / "fake_powermetrics.py"
        fake.write_text(FAKE)
        self.cmd = [sys.executable, str(fake)]
        os.environ["FAKE_MW"] = "2500"

    def tearDown(self):
        os.environ.pop("FAKE_MW", None)
        self.tmp.cleanup()

    def test_load(self):
        calls = []
        m = energy.measure(lambda: (calls.append(1), time.sleep(0.002)), duration=1.0, lead_in=0.5,
                           interval_ms=100, command=self.cmd)
        self.assertGreater(m["inferences"], 50)
        self.assertGreater(len(calls), m["inferences"])  # the lead-in ran too, uncounted
        self.assertGreaterEqual(m["duration_s"], 1.0)
        self.assertGreaterEqual(m["samples"], 5)
        self.assertAlmostEqual(m["package_mw"], 2500)
        self.assertAlmostEqual(m["ane_mw"], 1000)

    def test_idle(self):
        m = energy.measure(None, duration=0.6, lead_in=0.3, interval_ms=100, command=self.cmd)
        self.assertEqual(m["inferences"], 0)
        self.assertAlmostEqual(m["package_mw"], 2500)

    def test_powermetrics_failing_is_reported(self):
        cmd = [sys.executable, "-c", "import sys; sys.exit('powermetrics must be invoked as the superuser')"]
        with self.assertRaises(SystemExit) as ctx:
            energy.measure(None, duration=0.1, lead_in=0.3, command=cmd)
        self.assertIn("superuser", str(ctx.exception))


class CommandLine(unittest.TestCase):
    @unittest.skipIf(os.geteuid() == 0, "running as root")
    def test_without_root_says_to_use_sudo(self):
        r = subprocess.run([sys.executable, "-m", "bench.energy", "--only", "coreml_fp16"],
                           cwd=ROOT, capture_output=True, text=True, timeout=60)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("sudo .venv/bin/python -B -m bench.energy --only coreml_fp16", r.stderr)


if __name__ == "__main__":
    unittest.main()
