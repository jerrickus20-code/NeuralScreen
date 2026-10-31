"""DLAA: the switch, the frame-header bits, the refusal alert and the worker side.

DLAA (native/dlaa.inl) is DLSS super resolution at 1:1 over the capture, ahead
of the neural pass. Nothing here can run it - that needs an RTX card and
NVIDIA's nvngx_dlss.dll - so this guards the wiring around it:

* the Network fold carries the switch and the click toggles the config;
* the switch rides on every frame in bits 6-7, and those bits agree with the
  worker's FRAME_FLAG_DLAA / FRAME_FLAG_DLAA_SET and with no other flag;
* NR OFF does not put the pipeline to sleep while DLAA is on (it runs on the
  bypass frames as well), and the motion field keeps flowing;
* a refusal ("[dlaa] off: ...") turns the switch back off once, with an alert,
  and the worker's [dlaa] lines reach the log at all.

Run:  runtime\\python.exe tests\\test_dlaa_controls.py
"""
import io
import json
import os
import re
import struct
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))  # the modules live in app/
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pygame
import commands
import pipeline
import protocol
import settings_io
from test_ui_buttons import build, find, paint, click

NATIVE = ROOT / "native"


def flags_of(**kwargs) -> int:
    worker = SimpleNamespace(stdin=io.BytesIO())
    protocol.send_frame(worker, 0, None, np.zeros((2, 2, 2), np.float16), False, 0,
                        no_color=True, **kwargs)
    return struct.unpack(protocol.FRAME_FMT, worker.stdin.getvalue()[:24])[3]


def main() -> int:
    failures = []

    # 1. The switch, in the Network fold; a click is a toggle, main flips it.
    pygame.init()
    menu = build()
    paint(menu)
    row = find(menu, "toggle", "dlaa")
    if row is None:
        failures.append("no DLAA switch in the Network fold")
    else:
        out = click(menu, row)
        if out != [("toggle", "dlaa")]:
            failures.append(f"the DLAA switch emits {out}, not [('toggle', 'dlaa')]")
    st = SimpleNamespace(cfg={}, dlaa_alerted=True)
    with patch.object(settings_io, "save_menu_layout") as save:
        commands.apply_menu_action(st, ("toggle", "dlaa"))
        if st.cfg.get("dlaa") is not True or st.dlaa_alerted or not save.called:
            failures.append("switching DLAA on must set the config, re-arm the "
                            "alert and save")
        commands.apply_menu_action(st, ("toggle", "dlaa"))
        if st.cfg.get("dlaa") is not False:
            failures.append("a second click must switch DLAA off")
    menu.set_state({"dlaa": True})
    paint(menu)
    fold = find(menu, "fold", "net")
    if fold is not None and "DLAA" not in str(fold.extra.get("summary", "")):
        failures.append("the Network fold's summary does not say DLAA is on")

    # 2. The frame header: bits 6-7, nothing else, and None leaves them clear.
    if flags_of(dlaa=None) & 0xC0:
        failures.append("no DLAA argument must leave bits 6-7 clear (NS_DLAA decides)")
    if flags_of(dlaa=False) & 0xC0 != protocol.FRAME_FLAG_DLAA_SET:
        failures.append("DLAA off must send the explicit bit alone")
    if flags_of(dlaa=True) & 0xC0 != protocol.FRAME_FLAG_DLAA_SET | protocol.FRAME_FLAG_DLAA:
        failures.append("DLAA on must send both bits")
    busy = flags_of(want_pixels=True, motion_small=True, bypass=True, split=0.5,
                    frame_generation=True, frame_multiplier=6, prepared=True,
                    early_reply=True, worker_scene=True)
    if busy & 0xC0:
        failures.append(f"another flag already uses bits 6-7: 0x{busy:08X}")
    host = (NATIVE / "dlss5-feed-host64.cpp").read_text(encoding="utf-8-sig")
    for name, value in (("FRAME_FLAG_DLAA", protocol.FRAME_FLAG_DLAA),
                        ("FRAME_FLAG_DLAA_SET", protocol.FRAME_FLAG_DLAA_SET)):
        m = re.search(rf"constexpr uint32_t {name} = (0x[0-9A-Fa-f]+)u;", host)
        if m is None or int(m.group(1), 16) != value:
            failures.append(f"the worker's {name} differs from protocol.py")

    # 3. The worker side is wired: configured per frame, run before the
    #    warmup and the neural pass, torn down with the textures and NGX.
    dlaa = (NATIVE / "dlaa.inl").read_text(encoding="utf-8")
    for needle, why in (('#include "dlaa.inl"', "dlaa.inl is not compiled in"),
                        ("ConfigureDlaaFrame(fh.reserved);", "the switch is never read"),
                        ("CloseDlaaResources();", "a resize keeps the old DLAA textures"),
                        ("UnloadDlaa();", "shutdown leaves the DLSS runtime loaded")):
        if needle not in host:
            failures.append(why)
    run_at = host.find("!RunDlaa(v,")
    if run_at < 0 or not (host.find("if (!warmup_done)", run_at) > run_at
                          and host.find("EvaluateVideo(v, (frame == 0", run_at) > run_at):
        failures.append("DLAA must run before the warmup and the neural pass")
    for needle in ("NVSDK_NGX_Feature_SuperSampling", "NVSDK_NGX_PerfQuality_Value_DLAA",
                   "NsGateByoDll(lib_path, \"nvngx_dlss.dll\")", "[dlaa] enabled at",
                   "[dlaa] off: %s", "[dlaa] UI: %s"):
        if needle not in dlaa:
            failures.append(f"dlaa.inl lost {needle!r}")

    # 4. NR OFF with DLAA on keeps the frames (and the motion) coming.
    idle = SimpleNamespace(paused=True, cfg={"dlaa": True}, recorder=None,
                           pending_shot=None)
    if pipeline.wants_low_cost_off(idle):
        failures.append("NR OFF must not sleep the pipeline while DLAA is on")
    idle.cfg["dlaa"] = False
    if not pipeline.wants_low_cost_off(idle):
        failures.append("NR OFF with DLAA and FG off may sleep, as before")
    main_py = (ROOT / "main.py").read_text(encoding="utf-8")
    if 'and not st.cfg.get("dlaa", False))' not in main_py:
        failures.append("the motion field is skipped in bypass while DLAA needs it")
    if '"[dlaa]"' not in (ROOT / "app" / "pipeline.py").read_text(encoding="utf-8"):
        failures.append("the worker's [dlaa] lines are dropped by the parent")

    # 5. The verdict and the alert.
    v = settings_io.dlaa_verdict
    if v(["[dlaa] off: nvngx_dlss.dll not found", "[dlaa] UI: on"]) is not False:
        failures.append("a refusal after the switch-on is a refusal")
    if v(["[dlaa] enabled at 3840x2160, motion 3840x2160, preset -", "[dlaa] UI: on"]) is not True:
        failures.append("the enabled line is the success")
    if v(["[dlaa] UI: on", "[dlaa] off: an earlier refusal"]) is not None:
        failures.append("a refusal from before the switch-on must not count")
    alerts = []
    st = SimpleNamespace(cfg={"dlaa": True}, dlaa_alerted=False, lang="en",
                         worker_logs=["[dlaa] UI: on", "[dlaa] off: nvngx_dlss.dll not found"],
                         display=SimpleNamespace(alert=lambda text, *a: alerts.append(text)))
    with patch.object(settings_io, "save_menu_layout"):
        settings_io.refresh_dlaa_ok(st)
        settings_io.refresh_dlaa_ok(st)
    if st.cfg["dlaa"] or len(alerts) != 1 or "nvngx_dlss.dll" not in alerts[0]:
        failures.append(f"a refusal must turn the switch off with one alert: {alerts}")

    # 6. The setting is a known boolean, off by default.
    if "dlaa" not in settings_io._BOOL_KEYS:
        failures.append("dlaa is not validated as a boolean")
    shipped = json.loads((ROOT / "config.default.json").read_text(encoding="utf-8"))
    if shipped.get("dlaa") is not False:
        failures.append("config.default.json does not ship DLAA off")
    if settings_io._validate_config(dict(shipped, dlaa="yes")).get("dlaa") is not True:
        failures.append("dlaa is not normalised to a bool")

    for f in failures:
        print("FAIL:", f)
    if failures:
        return 1
    print("OK: the DLAA switch, its frame bits, the bypass motion and the refusal alert hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
