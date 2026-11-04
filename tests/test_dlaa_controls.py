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
  and the worker's [dlaa] lines reach the log at all;
* the interface protection sends its rectangles for DLAA as for FG, and the
  worker hands them to DLSS as the current-colour bias mask;
* the GUI: the DLAA tile, the model picker and its DLAP message, the line
  under the switch for each state, the hotkey with no default key;
* the worker keeps what the review asked of it: four allocators for the extra
  submission, the raw capture put back when DLAA goes off, the NGX Core
  fallback, Boost's work-size pass, the still-picture hold.

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
from test_ui_buttons import build, find, paint, click, into_view

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
    if fold is None:
        failures.append("no Network fold on the main page")
    elif "DLAA" not in str(fold.extra.get("summary", "")):
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
                        ("ConfigureDlaaFrame(v, fh.reserved);", "the switch is never read"),
                        ("kFrames = 4;", "DLAA's extra submission stalls on a three-allocator ring"),
                        ("g_dlaa_raw_shown != nullptr ? g_dlaa_raw_shown : v.color.tex",
                         "the wipe's before half is not the raw capture"),
                        ("CloseDlaaResources();", "a resize keeps the old DLAA textures"),
                        ("UnloadDlaa();", "shutdown leaves the DLSS runtime loaded")):
        if needle not in host:
            failures.append(why)
    run_at = host.find("!RunDlaa(v,")
    if run_at < 0 or not (host.find("if (!warmup_done)", run_at) > run_at
                          and host.find("EvaluateVideo(v, (frame == 0", run_at) > run_at):
        failures.append("DLAA must run before the warmup and the neural pass")
    for needle, why in (
            ("NVSDK_NGX_Feature_SuperSampling", "not DLSS super resolution"),
            ("NVSDK_NGX_PerfQuality_Value_DLAA", "not the DLAA mode"),
            ('NsGateByoDll(lib_path, "nvngx_dlss.dll")', "the BYO runtime is not verified"),
            ("[dlaa] enabled at", "no success line for the client"),
            ("[dlaa] off: %s", "no refusal line for the client"),
            ("[dlaa] UI: %s", "no switch marker for the client"),
            ("NVSDK_NGX_Parameter_DLSS_Input_Bias_Current_Color_Mask",
             "the interface rectangles do not reach DLSS"),
            ("UseDlaaCore(", "no NGX Core fallback"),
            ("&NVSDK_NGX_D3D12_CreateFeature", "the fallback does not create through the core"),
            ("static void DlaaRestoreRaw(", "switching off leaves DLAA's picture on a still desktop"),
            ("kDlaaStillEvaluations", "a still picture is evaluated forever"),
            ("g_residual_pso", "Boost does not compose onto the native frame"),
            ("g_dlaa.depth.upload->Release();", "the depth staging buffer is kept"),
            ("static const unsigned from_env = [", "NS_DLAA_PRESET is read every frame")):
        if needle not in dlaa:
            failures.append(f"dlaa.inl: {why}")
    # The off switch puts the capture back before the textures go.
    configure = dlaa[dlaa.find("static void ConfigureDlaaFrame"):]
    if not (0 <= configure.find("DlaaRestoreRaw(v);") < configure.find("CloseDlaaResources();")):
        failures.append("switching off must restore the capture before closing")

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

    # 4b. The interface rectangles are sent for DLAA, FG off.
    sent = []
    st = SimpleNamespace(cfg={"fg_ui_protect": True, "dlaa": True}, gray_active=True,
                         worker=SimpleNamespace(pid=7, poll=lambda: None),
                         shm=SimpleNamespace(read_gray=lambda: np.zeros((180, 320), np.uint8)),
                         ui_detector=SimpleNamespace(update=lambda gray: [(1, 2, 3, 4)],
                                                     reset=lambda: None),
                         ui_rects_sent=None)
    with patch.object(protocol, "send_ui_rects", lambda worker, rects: sent.append(rects)):
        commands._update_ui_protection(st)
    if sent != [[(1, 2, 3, 4)]]:
        failures.append(f"the interface protection does not serve DLAA: {sent}")

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

    # 7. The GUI: the tile, the model picker, the line under the switch, the
    #    hotkey, and the model reaching the worker.
    menu = build()
    paint(menu)
    tile = find(menu, "tile", "dlaa")
    if tile is None or click(menu, tile) != [("toggle", "dlaa")]:
        failures.append("no DLAA tile on the main page, or it is not the switch")
    picker = find(menu, "choice", "dlaa_model")
    if picker is None:
        failures.append("no DLAA model picker in the Network block")
    elif list(picker.payload or []) != list(settings_io.DLAA_MODELS):
        failures.append(f"the model picker offers {picker.payload}")
    st = SimpleNamespace(cfg={"dlaa_model": "auto"})
    with patch.object(settings_io, "save_menu_layout"):
        commands.apply_menu_action(st, ("dlaa_model", "K"))
        commands.apply_menu_action(st, ("dlaa_model", "Z"))
    if st.cfg["dlaa_model"] != "K":
        failures.append("picking a model does not set it, or a bad one is taken")

    def hint_for(state):
        menu.set_state(state)
        paint(menu)
        row = find(menu, "toggle", "dlaa")
        return str(row.extra.get("hint", "")) if row else ""
    running = hint_for({"dlaa": True, "dlaa_status": {
        "state": "running", "size": "2496×1404", "boost": True, "model": "", "core": True}})
    for part in ("2496×1404", "Boost", "auto", "NVIDIA driver"):
        if part not in running:
            failures.append(f"the running line {running!r} does not say {part!r}")
    if "starting" not in hint_for({"dlaa": True, "dlaa_status": {}}):
        failures.append("switched on with no answer yet must say starting")
    if "nvngx_dlss.dll" not in hint_for({"dlaa": False, "dlaa_status": {
            "state": "failed", "reason": "missing"}}):
        failures.append("a missing DLL is not named under the switch")
    if "HDR" not in hint_for({"dlaa": False, "dlaa_status": {"state": "failed", "reason": "hdr"}}):
        failures.append("HDR is not named under the switch")
    if "SDR" not in hint_for({"dlaa": False, "dlaa_status": {"state": "off"}}):
        failures.append("off, the line does not say what DLAA is for")

    parse = settings_io.dlaa_state_from_logs
    got = parse(["[dlaa] enabled at 3840x2160 (frame 3840x2160), motion 2496x1404, "
                 "preset K, direct; flat depth and estimated motion", "[dlaa] UI: on"])
    if got != {"state": "running", "size": "3840×2160", "boost": False, "model": "K",
               "auto": False, "core": False}:
        failures.append(f"the running line parses as {got}")
    # Auto resolved by the runtime, Boost, and the cost line newer than it.
    got = parse(["[dlaa] cost 6.1 ms avg, 8.2 ms max, 118 frames (the whole DLAA submission)",
                 "[dlaa] enabled at 1664x1040 (frame 2560x1600, Boost), motion 1664x1040, "
                 "preset auto=K, direct; flat depth and estimated motion", "[dlaa] UI: on"])
    if got != {"state": "running", "size": "1664×1040", "boost": True, "model": "K",
               "auto": True, "core": False, "ms": 6.1}:
        failures.append(f"the Boost/auto/cost lines parse as {got}")
    line = hint_for({"dlaa": True, "dlaa_status": got})
    for part in ("K (auto)", "6.1 ms", "Boost"):
        if part not in line:
            failures.append(f"the status line {line!r} lacks {part!r}")
    if parse(["[dlaa] off: the HDR presentation shows ..."]) != {"state": "failed", "reason": "hdr"}:
        failures.append("an HDR refusal is not told apart")
    if parse(["[host] nothing about DLAA"], {"state": "running"}) != {"state": "running"}:
        failures.append("with no DLAA line in the window the last state must stand")

    import hotkeys
    import overlay_ui
    if any(cmd == "dlaa" for *_x, cmd, _n in hotkeys.build_bindings({}).values()):
        failures.append("DLAA must have no key until the user gives it one")
    if not any(cmd == "dlaa" for *_x, cmd, _n in hotkeys.build_bindings({"dlaa": "F9"}).values()):
        failures.append("a key given to DLAA is not bound")
    if "dlaa" not in [cmd for cmd, _label in overlay_ui.HOTKEY_ROWS]:
        failures.append("DLAA is not in the menu's key list")
    if 'elif cmd == "dlaa":' not in (ROOT / "app" / "commands.py").read_text(encoding="utf-8"):
        failures.append("the DLAA hotkey command is not handled")

    worker = SimpleNamespace(stdin=io.BytesIO())
    protocol.send_dlaa_model(worker, "M", 0.5, True)
    if struct.unpack(protocol.FRAME_FMT, worker.stdin.getvalue()) != (
            protocol.DLAA_PRESET_MAGIC, 13, 1, 500, 0):
        failures.append("DLAP does not carry the model, the mask bit and the strength")
    worker = SimpleNamespace(stdin=io.BytesIO())
    protocol.send_dlaa_model(worker, "auto", 0.0, False)
    if struct.unpack(protocol.FRAME_FMT, worker.stdin.getvalue())[3] != 1:
        failures.append("a strength of 0 must not be sent as 0 (the worker reads that as unset)")
    m = re.search(r"DLAA_PRESET_MAGIC\s*=\s*(0x[0-9A-Fa-f]+)u;", host)
    if m is None or int(m.group(1), 16) != protocol.DLAA_PRESET_MAGIC:
        failures.append("the worker's DLAP magic differs from protocol.py")
    sent_models = []
    st = SimpleNamespace(cfg={"dlaa_model": "L"}, worker_logs=["[host] up"],
                         worker=SimpleNamespace(pid=5, poll=lambda: None))
    with patch.object(protocol, "send_dlaa_model",
                      lambda w, model, strength, mask: sent_models.append((model, strength, mask))):
        commands.update_dlaa_model(st)                 # no [dlaa] line yet: an old worker?
        st.worker_logs.append("[dlaa] UI: off")
        commands.update_dlaa_model(st)
        commands.update_dlaa_model(st)                 # unchanged: not again
        st.cfg["dlaa_model"] = "K"
        commands.update_dlaa_model(st)
        st.cfg["dlaa_strength"] = 0.5                  # the slider and the mask ride along
        commands.update_dlaa_model(st)
        st.cfg["dlaa_motion_mask"] = True
        commands.update_dlaa_model(st)
    if sent_models != [("L", 1.0, False), ("K", 1.0, False), ("K", 0.5, False), ("K", 0.5, True)]:
        failures.append(f"the model is sent {sent_models}, not once per worker and change")

    # 8. The strength slider (Boost only), the two mask switches, the counter.
    menu.set_state({"dlaa": True, "nr_small": False})
    paint(menu)
    if find(menu, "slider", "dlaa_strength") is not None:
        failures.append("the strength slider is laid out without Boost, where it changes nothing")
    menu.set_state({"dlaa": True, "nr_small": True, "work_size": "1664x1040"})
    paint(menu)
    sl = find(menu, "slider", "dlaa_strength")
    if sl is None:
        failures.append("no strength slider under Boost")
    else:
        sl = into_view(menu, sl)
        track = sl.extra.get("track")
        out = menu.handle_event(pygame.event.Event(
            pygame.MOUSEBUTTONDOWN, {"pos": (track.x + track.w // 2, track.centery), "button": 1}))
        if not out or out[0][0] != "dlaa_strength":
            failures.append(f"the strength slider emits {out}")
    st = SimpleNamespace(cfg={}, display=SimpleNamespace(menu=SimpleNamespace(set_state=lambda d: None)))
    with patch.object(settings_io, "save_menu_layout"):
        commands.apply_menu_action(st, ("dlaa_strength", 0.25))
        commands.apply_menu_action(st, ("dlaa_strength", 7))
        commands.apply_menu_action(st, ("toggle", "dlaa_ui_protect"))
        commands.apply_menu_action(st, ("toggle", "dlaa_motion_mask"))
    if st.cfg.get("dlaa_strength") != 1.0 or st.cfg.get("dlaa_ui_protect") is not False \
            or st.cfg.get("dlaa_motion_mask") is not True:
        failures.append(f"the new controls do not reach the config: {st.cfg}")
    for key in ("dlaa_ui_protect", "dlaa_motion_mask"):
        if find(menu, "toggle", key) is None:
            failures.append(f"no {key} switch in the Network block")
    # DLAA's protection is on by default and serves DLAA with FG's switch off.
    sent = []
    st = SimpleNamespace(cfg={"dlaa": True, "fg_ui_protect": False}, gray_active=True,
                         worker=SimpleNamespace(pid=7, poll=lambda: None),
                         shm=SimpleNamespace(read_gray=lambda: np.zeros((180, 320), np.uint8)),
                         ui_detector=SimpleNamespace(update=lambda gray: [(1, 2, 3, 4)],
                                                     reset=lambda: None),
                         ui_rects_sent=None)
    with patch.object(protocol, "send_ui_rects", lambda worker, rects: sent.append(rects)):
        commands._update_ui_protection(st)
    if sent != [[(1, 2, 3, 4)]]:
        failures.append(f"DLAA's own protection switch is not on by default: {sent}")
    s_en = overlay_ui.STRINGS["en"]
    hud = overlay_ui.status_readings(
        {}, {"fps": 55.0, "nr": False, "gpu_ok": True, "dlaa_running": True}, s_en)
    if not any("DLAA" in line and "55.0" in line for line in hud):
        failures.append(f"with NR off and DLAA running the counter shows {hud}")
    hud = overlay_ui.status_readings(
        {}, {"fps": 55.0, "nr": False, "gpu_ok": True, "dlaa_running": False}, s_en)
    if any("DLAA" in line for line in hud):
        failures.append("with DLAA not running the counter must not claim a DLAA rate")

    for f in failures:
        print("FAIL:", f)
    if failures:
        return 1
    print("OK: the DLAA switch, its frame bits, the bypass motion and the refusal alert hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
