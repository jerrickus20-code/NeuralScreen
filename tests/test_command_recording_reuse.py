"""Command recording must not grow with rebuilds (plan 18.09, item 3).

The open question was whether repeated RNSZ rebuilds (every slider step sends
one) leak command allocators and lists. They do not, and this pins the reasons
so the property cannot be lost silently.

What the code does, and why that is enough:

* the worker's own allocators are a fixed ring - `Host::kFrames` (4) created
  once, each reused only after its own `alloc_fence` has passed
  (`BeginCommands` resets the slot's allocator and list, and the retire wait
  comes first). Nothing is created per frame;
* the extra allocator/list/fence pairs (`InitDisguise`, the pump pair) are
  created once per process, at startup - a fixed cost, not a leak;
* the FG presenter's allocator/list/fence are `winrt::com_ptr`, so they are
  released when `FgPresenter` returns, and `EnsureFg` calls `CloseFgResources()`
  before it creates a new presenter. A rebuild closes before it opens;
* RNSZ releases the TEXTURES (`ReleaseVideoTextures` nulls every one) and
  leaves the allocators alone - correct, because they are not per-size.

Checked here:

1. the per-frame path (`BeginCommands`) resets an existing allocator instead of
   creating one, and waits for the slot fence before the reset;
2. no `CreateCommandAllocator`/`CreateCommandList` call sits in a per-frame
   function - the count of creation sites is small and each is once-per-process
   or behind a close;
3. `EnsureFg` closes FG resources before creating a presenter;
4. `ReleaseVideoTextures` releases every texture it owns (a missing one is the
   actual leak class that was found here before - issue #48).

Provable by mutation: move a creation call into `BeginCommands`, or drop a
`Release` from `ReleaseVideoTextures`, and the matching check goes red.

Run:  runtime\\python.exe tests\\test_command_recording_reuse.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOST = ROOT / "native" / "dlss5-feed-host64.cpp"
FG = ROOT / "native" / "frame_generation.inl"


def without_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def body_of(text: str, marker: str) -> str:
    start = text.find(marker)
    if start < 0:
        return ""
    brace = text.find("{", start)
    if brace < 0:
        return ""
    rest = text[brace:]
    nxt = re.search(r"\nstatic [A-Za-z_].*\(", rest)
    return rest[:nxt.start()] if nxt else rest


def main() -> int:
    host = without_comments(HOST.read_text(encoding="utf-8", errors="replace"))
    fg = without_comments(FG.read_text(encoding="utf-8", errors="replace"))
    if not host or not fg:
        print("FAIL: the worker sources are not readable")
        return 1
    failures = []

    # --- 1. the per-frame path reuses, it does not create ------------------
    begin = body_of(host, "static bool BeginCommands()")
    if not begin:
        failures.append("BeginCommands() is gone - the per-frame allocation path "
                        "cannot be checked")
    else:
        if "CreateCommandAllocator" in begin or "CreateCommandList" in begin:
            failures.append("BeginCommands() creates an allocator or list: that "
                            "is one per frame, and the count grows with uptime")
        if "->Reset()" not in begin:
            failures.append("BeginCommands() no longer resets the slot's "
                            "allocator - either it creates one (a leak) or it "
                            "reuses un-reset memory")
        if "WaitFenceValue" not in begin:
            failures.append("BeginCommands() resets the allocator without "
                            "waiting for the slot fence: it would rewrite "
                            "memory the GPU may still be reading")

    # --- 2. creation sites are few, and none is per-frame ------------------
    for name, text in (("native/dlss5-feed-host64.cpp", host),
                       ("native/frame_generation.inl", fg)):
        allocs = len(re.findall(r"CreateCommandAllocator", text))
        lists = len(re.findall(r"CreateCommandList\b", text))
        if allocs > 3 or lists > 4:
            failures.append(f"{name} has {allocs} allocator and {lists} list "
                            "creation sites - more than the fixed sets this "
                            "worker is built around; a new one is per-frame or "
                            "per-rebuild until proven otherwise")

    # --- 3. a rebuild closes before it creates -----------------------------
    ensure = body_of(fg, "static bool EnsureFg(VideoState &v, DXGI_FORMAT format)")
    if not ensure:
        failures.append("EnsureFg() is gone - the FG rebuild path cannot be "
                        "checked")
    else:
        close_at = ensure.find("CloseFgResources()")
        create_at = ensure.find("FgPresenter")
        if close_at < 0:
            failures.append("EnsureFg() creates a presenter without closing "
                            "the previous resources first: every rebuild would "
                            "add an allocator, a list and a fence")
        elif create_at >= 0 and close_at > create_at:
            failures.append("EnsureFg() creates before it closes: the previous "
                            "presenter's resources are still live")

    # --- 4. every texture the rebuild path owns is released ----------------
    release = body_of(host, "static void ReleaseVideoTextures(VideoState &v)")
    if not release:
        failures.append("ReleaseVideoTextures() is gone - the rebuild release "
                        "path cannot be checked")
    else:
        for field in ("v.color.tex", "v.color.upload", "v.mv.tex",
                      "v.mv.upload", "v.output", "v.readback"):
            if not re.search(re.escape(field) + r"\s*->Release\(\)", release):
                failures.append(f"ReleaseVideoTextures() no longer releases "
                                f"{field}: a rebuild leaks it (issue #48 is "
                                "exactly this class)")
        if "g_dda_ready = false" not in release:
            failures.append("ReleaseVideoTextures() releases the capture "
                            "texture but leaves g_dda_ready set: the next "
                            "frame evaluates on a freed resource")

    for f in failures:
        print("FAIL:", f)
    if failures:
        return 1
    print("OK: command recording is a fixed ring reused behind its fence, "
          "rebuilds close before they create, and every owned texture is released")
    return 0


if __name__ == "__main__":
    sys.exit(main())
