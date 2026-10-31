# DLAA test build

A test patch for NeuralScreen **v2.6.4**: DLSS anti-aliasing (DLAA) of the
captured picture, before the neural pass. It is not a full release - it goes
on top of an existing v2.6.4 folder.

**Untested on a GPU so far.** The worker in this zip was compiled from the
`dlaa` branch; this is its first run on real hardware. Keep your v2.6.4
archive to go back.

## Install

1. Close NeuralScreen (tray icon -> Quit, or Ctrl+Alt+Q).
2. Have the **v2.6.4** folder: the full archive from
   <https://github.com/perseval-BLR/NeuralScreen/releases/tag/v2.6.4>, unpacked.
3. Extract this zip **into that folder** and let it replace the files.
4. Put NVIDIA's **`nvngx_dlss.dll`** into `native\libraries\` in that folder.
   It is not in this zip. NVIDIA's DLSS SDK has it at
   <https://github.com/NVIDIA/DLSS/tree/main/lib/Windows_x86_64/rel>, and many
   games ship it next to their executable. Without it DLAA asks the NVIDIA
   driver instead, which may or may not have a copy of its own; if not, the
   switch turns off and says the DLL is missing.
5. Start `NeuralScreen.exe`. The first start re-checks compatibility (a few
   seconds), because the worker changed.

## Try it

- **DLAA tile** - on the main page next to FG, and in the collapsed bar.
- **Network** block - under the DLAA switch a line says what it is doing:
  *starting...*, *running - 3840×2160 - model K*, or why it did not start
  (DLL missing, HDR on, or the runtime refused). Below it, **DLAA model**:
  Auto, J, K, L, M.
- **Keys** tab - *DLAA on/off* has no key until you give it one.
- **Protect the interface in FG/DLAA** (Network) keeps a still HUD from
  smearing over a moving picture.

Worth trying: DLAA alone with NR off; with NR on; with **Boost** on (DLAA then
runs at Boost's size, cheaper and coarser); each model; a game with a HUD,
with and without interface protection; and the frame rate with DLAA off/on.

HDR is not supported - with HDR compatibility on, DLAA switches itself off.

## What to send back

`NeuralScreen.log` from the program folder, or just its lines starting with
`[dlaa]` - they say which route started DLAA (the DLL directly or the NVIDIA
driver), the size, the model, and any refusal. Plus the frame rate with DLAA
off and on, and your GPU.

## Undo

Extract the v2.6.4 archive over the folder again (DLAA's settings in
`config.json` are ignored by v2.6.4).
