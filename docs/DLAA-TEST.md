# DLAA test build

A test build of NeuralScreen **v2.6.4** with DLSS anti-aliasing (DLAA) of the
captured picture, before the neural pass.

Test build 4 ran on an RTX 4060 Laptop: DLAA started, ran with Boost on and
off and shut down cleanly. This build adds what the review asked for, none of
it run on a GPU yet: a cost line, a cheaper Boost path with a strength slider,
two masks against ghosting, and the counter for DLAA with NR off.

## Install

**Full install - `NeuralScreen-<tag>-full.zip` (the easy one).**

1. Download it and unpack it anywhere (its own folder; your v2.6.4 folder can
   stay as it is to compare).
2. Run `NeuralScreen.exe`. Windows may warn about an unknown publisher: *More
   info* -> *Run anyway*.

Everything is inside: v2.6.4 with this build laid over it, and NVIDIA's DLSS
runtime `native\nvngx_dlss.dll` (from the DLSS SDK, NVIDIA-signed, its license
beside it in `native\NVIDIA-DLSS-LICENSE.txt`). The first start runs the
compatibility check (a few seconds). Delete the folder to remove it; switch
autostart off first if you turned it on.

**Patch - `NeuralScreen-<tag>-patch.zip` (small, for an existing v2.6.4 folder).**

1. Close NeuralScreen (tray icon -> Quit, or Ctrl+Alt+Q).
2. Extract the zip into the v2.6.4 folder
   (<https://github.com/perseval-BLR/NeuralScreen/releases/tag/v2.6.4>) and let
   it replace the files.
3. Put NVIDIA's `nvngx_dlss.dll` into `native\libraries\` - it is not in the
   patch. The DLSS SDK has it at
   <https://github.com/NVIDIA/DLSS/tree/main/lib/Windows_x86_64/rel>, and many
   games ship it next to their executable. Without it DLAA asks the NVIDIA
   driver instead, which may or may not have a copy of its own; if not, the
   switch turns off and says the DLL is missing.
4. Start `NeuralScreen.exe`. Extract the v2.6.4 archive again to undo.

## Try it

- **DLAA tile** - on the main page next to FG, and in the collapsed bar.
- **Network** block - under the DLAA switch a line says what it is doing:
  *starting...*, *running - 2560×1600 - model K (auto) - 6.1 ms*, or why it
  did not start (DLL missing, HDR on, or the runtime refused). The
  milliseconds are DLAA's own GPU time per frame, from the worker's
  `[dlaa] cost` line every two seconds. Below it, **DLAA model** (Auto, J, K,
  L, M), with **Boost** on the **DLAA strength** slider, and two switches:
  **Protect the interface in DLAA** (on) and **Trust the frame where motion
  is unsure** (off, experimental - less ghosting on moving objects).
- **Keys** tab - *DLAA on/off* has no key until you give it one.
- With NR off and DLAA on, the on-screen counter reads *DLAA 55.0*.

Worth trying, about 20 s each on the same moving scene, then send the log:
Boost on / DLAA off, Boost on / DLAA on, Boost off / DLAA off, Boost off /
DLAA on - the `[dlaa] cost` lines and the fps lines tell DLAA's price apart
from NR's. Then each model (the cost line changes with it); the strength
slider under Boost; a game with a HUD with the interface protection on and
off; and the motion switch on a scene with fast-moving objects.

What DLAA here can and cannot do: a captured frame has no sub-pixel jitter
and no depth, so DLAA smooths edges and shimmer over time; it cannot add
detail the screen never had, as it does inside a game.

HDR is not supported - with HDR compatibility on, DLAA switches itself off.

## What to send back

`NeuralScreen.log` from the program folder, or just its lines starting with
`[dlaa]` - they say which route started DLAA (the DLL directly or the NVIDIA
driver), the size, the model, and any refusal. Plus the frame rate with DLAA
off and on, and your GPU.

## Undo

Full install: delete its folder. Patch: extract the v2.6.4 archive over the
folder again (DLAA's settings in `config.json` are ignored by v2.6.4).
