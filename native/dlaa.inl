// DLAA: DLSS super resolution at 1:1 (NVSDK_NGX_Feature_SuperSampling,
// PerfQuality_Value_DLAA) over the captured frame, before the neural pass.
//
// Without Boost the result is copied back into v.color.tex, so everything
// downstream of the capture - NR, the bypass present, FG, the recording - sees
// the anti-aliased frame and needs no change of its own. The raw frame is kept
// beside it (g_dlaa.raw): the wipe's "before" half shows it, and switching
// DLAA off puts it back at once, even on a desktop that does not change.
//
// With Boost DLAA runs at Boost's work size, on the frame scaled down the way
// NR's input is. When the neural pass runs on the frame, it is handed that
// work-size result directly (g_dlaa_feeds_nr, EvaluateVideo) and NR's own
// residual composite adds DLAA's correction (DLAA(small) - small) at DLAA's
// strength next to the network's: one downscale and one composite per frame
// instead of two of each. When the neural pass is off, DLAA composes the full
// frame itself with the same residual pass. Either way the native frame keeps
// its sharpness and the anti-aliasing is coarser - the trade Boost makes.
//
// What a capture can and cannot give DLSS. Motion: the same current ->
// previous field NR is handed (in work-size pixels). Depth: none - a flat
// plane, as FG's. Jitter: none - the game already resolved its frame, so DLAA
// here is temporal anti-aliasing and stabilisation of what the screen shows,
// not the sub-pixel reconstruction it does inside an engine. Two things tell
// DLSS to trust the current frame instead of its history (the
// DLSS.Input.Bias.Current.Color.Mask): the still-interface rectangles the
// client finds (UIRC, as FG's protection), and, opt-in, the blocks where
// NVOFA's own cost says its motion is unreliable.
//
// The runtime is NVIDIA's public nvngx_dlss.dll: native\libraries\ first
// (verified like every BYO file), then native\, initialised directly as
// nvngx_dlssg.dll is. When that route is missing or refused, the feature is
// asked of NGX Core - the driver's own loader, initialised for NR already.
// Opt-in, like FG. Its GPU time is measured on every frame and logged every
// two seconds ("[dlaa] cost", the line the menu's status reads).
static PFN_NR_Evaluate g_dlaa_evaluate = nullptr;
// The exception code of the last DLSS call that faulted inside NVIDIA's
// runtime, 0 when it did not. Guarded for the same reason as FG's (#145): an
// old driver that faults instead of refusing must not end the worker.
static DWORD g_dlaa_fault = 0;

static NVSDK_NGX_Result DlaaCreateGuarded(PFN_NR_Create create, ID3D12GraphicsCommandList *list,
                                          const NVSDK_NGX_Parameter *params, NVSDK_NGX_Handle **out)
{
    g_dlaa_fault = 0;
    __try
    {
        if (TestFailureOnce("dlaa-create-fault"))
            RaiseException(EXCEPTION_ACCESS_VIOLATION, 0, 0, nullptr);
        return create(list, NVSDK_NGX_Feature_SuperSampling, params, out);
    }
    __except (EXCEPTION_EXECUTE_HANDLER) { g_dlaa_fault = GetExceptionCode(); }
    return NVSDK_NGX_Result_Fail;
}

static NVSDK_NGX_Result DlaaEvaluateGuarded(ID3D12GraphicsCommandList *list,
    NVSDK_NGX_Handle *handle, NVSDK_NGX_Parameter *params)
{
    g_dlaa_fault = 0;
    __try
    {
        if (TestFailureOnce("dlaa-evaluate-fault"))
            RaiseException(EXCEPTION_ACCESS_VIOLATION, 0, 0, nullptr);
        return g_dlaa_evaluate(list, handle, params, nullptr);
    }
    __except (EXCEPTION_EXECUTE_HANDLER) { g_dlaa_fault = GetExceptionCode(); }
    return NVSDK_NGX_Result_Fail;
}

// A still picture is evaluated this many times (DLSS's history settles on it)
// and then shown from the last result, as NR does (kStillEvaluations): apps
// that present the same picture again and again (#183) would otherwise keep
// the GPU busy for nothing.
static constexpr unsigned kDlaaStillEvaluations = 4;

// The shader-visible descriptor slots, all written once per (re)create.
enum DlaaSlot : UINT {
    kDlaaMotionSrc = 0, kDlaaMotionDst = 1,           // the motion stretch (full size)
    kDlaaScaleSrc = 2, kDlaaScaleDst = 3,             // colour -> work size (Boost)
    kDlaaResNative = 4, kDlaaResIn = 5, kDlaaResOut = 6, kDlaaResDst = 7, // the residual (Boost)
    kDlaaMaskUav = 8,                                 // the interface mask's clear
    kDlaaCostSrc = 9, kDlaaCostMaskUav = 10,          // NVOFA's cost -> the mask
    kDlaaSlots = 11
};

// The mask's second source: where NVOFA's cost says the motion is a guess.
// cost (R8_UINT, per flow block, higher = worse match) is read at the block
// the pixel falls in; the mask pixel is set, never cleared, so the rectangles
// drawn before stay. The threshold is baked in (NS_DLAA_COST_T, default
// kDlaaCostThreshold): the scale root signature has four constants, all used.
static const char kDlaaCostMaskHlsl[] =
    "Texture2D<uint>    gCost : register(t0);\n"
    "RWTexture2D<float> gMask : register(u0);\n"
    "cbuffer S : register(b0) { uint gW; uint gH; uint gCW; uint gCH; };\n"
    "[numthreads(8, 8, 1)]\n"
    "void CSMain(uint3 id : SV_DispatchThreadID)\n"
    "{\n"
    "    if (id.x >= gW || id.y >= gH) return;\n"
    "    uint2 c = uint2(id.x * gCW / gW, id.y * gCH / gH);\n"
    "    if (gCost[c] > DLAA_COST_T) gMask[id.xy] = 1.0;\n"
    "}\n";
static constexpr unsigned kDlaaCostThreshold = 32;

static struct DlaaState {
    HMODULE module = nullptr;
    bool via_core = false;
    unsigned runtime_major = 0;   // nvngx_dlss.dll's file version (0: unknown or NGX Core)
    PFN_NR_Create create = nullptr;
    PFN_NR_Release release = nullptr;
    NVSDK_NGX_Parameter *params = nullptr;
    NVSDK_NGX_Handle *feature = nullptr;
    // Render-size textures. output: UNORDERED_ACCESS at rest; depth (flat),
    // mask and work_in (Boost's input - not "small", which rpcndr.h defines as
    // char): NON_PIXEL_SHADER_RESOURCE at rest.
    VideoTex depth;
    ID3D12Resource *output = nullptr, *mask = nullptr, *work_in = nullptr;
    // Frame-size textures. raw: the capture as it came, COPY_DEST at rest.
    // full: Boost's composite, UNORDERED_ACCESS at rest. motion: the field
    // stretched to the frame's size when the work size is smaller (without
    // Boost), UNORDERED_ACCESS at rest.
    ID3D12Resource *raw = nullptr, *full = nullptr, *motion = nullptr;
    // Its own descriptors: the shared scale and residual sets are rebound by
    // MOTS and by NR within a frame, and a shader-visible heap is read when the
    // list runs, not when it is recorded (see BindScale4Descriptors).
    ID3D12DescriptorHeap *heap = nullptr, *cpu_heap = nullptr;
    ID3D12Resource *bound_color = nullptr, *bound_mv = nullptr, *bound_cost = nullptr;
    ID3D12PipelineState *cost_pso = nullptr;   // kDlaaCostMaskHlsl, on g_scale_rs
    int cost_pso_state = 0;                    // 0 not tried, 1 ready, -1 failed
    UINT w = 0, height = 0;   // the render size (DLSS's input and output)
    UINT fw = 0, fh = 0;      // the frame's size
    UINT mw = 0, mh = 0;      // the motion field's size
    unsigned preset = 0;      // the model the feature was created with
    bool boost = false;
    // The interface rectangles the mask was last drawn from (count 0: none).
    uint32_t mask_count = UINT32_MAX;
    uint16_t mask_rects[kUiRectsMax][4] = {};
    // The feature has accumulated a frame: false after every (re)create, so
    // the first evaluation of a new feature starts its history.
    bool history = false;
    unsigned still_evals = 0;
    // v.color.tex holds DLAA's result rather than the capture (raw has that).
    bool color_is_result = false;
    // output holds DLAA's result for the capture on screen (the feed mode).
    bool fed_valid = false;
    // The runtime refused or faulted: off until the switch is turned on again.
    bool failed = false;
} g_dlaa;
static int g_dlaa_ui_enabled = -1;

// The model the menu picked (DLAP): 0 = the runtime's own choice, else the
// NGX preset value (J..M); -1 until a client sends one. The strength of
// DLAA's correction under Boost (DLAP too), -1 until sent.
static int g_dlaa_preset_ui = -1;
static float g_dlaa_strength_ui = -1.0f;

static bool DlaaRequested()
{
    static const bool requested = [] {
        char value[8] = {};
        return GetEnvironmentVariableA("NS_DLAA", value, sizeof(value)) == 1 && value[0] == '1';
    }();
    return (g_dlaa_ui_enabled < 0 ? requested : g_dlaa_ui_enabled != 0) && !g_dlaa.failed;
}

// The DLSS model for DLAA: the menu's choice, else NS_DLAA_PRESET=J|K|L|M
// (read once), else the runtime's own (K, the transformer model, on 310.x).
static unsigned DlaaPreset()
{
    static const unsigned from_env = [] {
        char value[8] = {};
        if (GetEnvironmentVariableA("NS_DLAA_PRESET", value, sizeof(value)) != 1) return 0u;
        const char letter = static_cast<char>(value[0] & ~0x20);
        if (letter < 'J' || letter > 'M') return 0u;
        return static_cast<unsigned>(NVSDK_NGX_DLSS_Hint_Render_Preset_J) + (letter - 'J');
    }();
    if (g_dlaa_preset_ui < 0) return from_env;
    const unsigned ui = static_cast<unsigned>(g_dlaa_preset_ui);
    return (ui >= NVSDK_NGX_DLSS_Hint_Render_Preset_J && ui <= NVSDK_NGX_DLSS_Hint_Render_Preset_M)
        ? ui : 0u;
}

// How much of DLAA's correction the Boost composite adds (0..1): the menu's
// slider, else NS_DLAA_STRENGTH (read once), else all of it.
static float DlaaStrength()
{
    static const float from_env = [] {
        char value[16] = {};
        if (GetEnvironmentVariableA("NS_DLAA_STRENGTH", value, sizeof(value)) == 0) return 1.0f;
        const float f = static_cast<float>(atof(value));
        return f >= 0.0f && f <= 1.0f ? f : 1.0f;
    }();
    return g_dlaa_strength_ui >= 0.0f ? g_dlaa_strength_ui : from_env;
}

// ---- GPU time: two timestamps around every DLAA submission ----------------
// Always on, unlike the NS_PHASE stages: the number is what the status line
// shows and what the models are compared by.
static struct DlaaTiming {
    ID3D12QueryHeap *heap = nullptr;
    ID3D12Resource *readback = nullptr;
    UINT64 freq = 0;
    UINT64 fence[Host::kFrames] = {};
    int state = 0;   // 0 not tried, 1 ready, -1 failed
    double sum = 0.0, max = 0.0;
    unsigned n = 0;
    ULONGLONG report_tick = 0;
} g_dlaa_time;

static bool DlaaTimingReady()
{
    auto &t = g_dlaa_time;
    if (t.state != 0) return t.state == 1;
    t.state = -1;
    D3D12_QUERY_HEAP_DESC qd = {};
    qd.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    qd.Count = 2 * Host::kFrames;
    if (FAILED(h.dev->CreateQueryHeap(&qd, IID_PPV_ARGS(&t.heap)))) return false;
    D3D12_HEAP_PROPERTIES rb = {};
    rb.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC bd = {};
    bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bd.Width = sizeof(UINT64) * 2 * Host::kFrames; bd.Height = 1; bd.DepthOrArraySize = 1;
    bd.MipLevels = 1; bd.SampleDesc.Count = 1;
    bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    if (FAILED(h.dev->CreateCommittedResource(&rb, D3D12_HEAP_FLAG_NONE, &bd,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&t.readback)))) return false;
    if (FAILED(h.queue->GetTimestampFrequency(&t.freq)) || t.freq == 0) return false;
    t.state = 1;
    return true;
}

// Read the slots whose submissions completed, and say the average every 2 s.
static void DlaaCollectTiming()
{
    auto &t = g_dlaa_time;
    if (t.state != 1) return;
    const UINT64 completed = h.fence->GetCompletedValue();
    if (completed == UINT64_MAX) return;
    for (int slot = 0; slot < Host::kFrames; ++slot)
    {
        if (t.fence[slot] == 0 || t.fence[slot] > completed) continue;
        t.fence[slot] = 0;
        D3D12_RANGE r = { sizeof(UINT64) * slot * 2, sizeof(UINT64) * (slot + 1) * 2 };
        void *mapped = nullptr;
        if (FAILED(t.readback->Map(0, &r, &mapped)) || mapped == nullptr) continue;
        const UINT64 *ts = static_cast<const UINT64 *>(mapped) + slot * 2;
        if (ts[1] > ts[0])
        {
            const double ms = static_cast<double>(ts[1] - ts[0]) * 1000.0 / static_cast<double>(t.freq);
            t.sum += ms;
            if (ms > t.max) t.max = ms;
            ++t.n;
        }
        D3D12_RANGE none = { 0, 0 };
        t.readback->Unmap(0, &none);
    }
    const ULONGLONG now = GetTickCount64();
    if (t.report_tick == 0) t.report_tick = now;
    if (now - t.report_tick >= 2000 && t.n > 0)
    {
        // "[dlaa] cost" is a contract: settings_io parses it for the status line.
        Log("[dlaa] cost %.1f ms avg, %.1f ms max, %u frames (the whole DLAA submission)",
            t.sum / t.n, t.max, t.n);
        t.sum = t.max = 0.0; t.n = 0;
        t.report_tick = now;
    }
}

static void CloseDlaaTiming()
{
    auto &t = g_dlaa_time;
    if (t.readback) { t.readback->Release(); t.readback = nullptr; }
    if (t.heap) { t.heap->Release(); t.heap = nullptr; }
    t.state = 0;
    for (auto &f : t.fence) f = 0;
}

// Put the capture back where DLAA's result was: the switch went off, or DLAA
// stopped. Without it a desktop that does not change would keep showing the
// anti-aliased picture until it did.
static void DlaaRestoreRaw(VideoState &v)
{
    if (!g_dlaa.color_is_result || g_dlaa.raw == nullptr || v.color.tex == nullptr ||
        v.color.tex != g_dlaa.bound_color || g_submission_failed)
        return;
    g_dlaa.color_is_result = false;
    if (!BeginCommands()) return;
    D3D12_RESOURCE_BARRIER pre[] = {
        Transition(g_dlaa.raw, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE),
        Transition(v.color.tex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                   D3D12_RESOURCE_STATE_COPY_DEST),
    };
    h.list->ResourceBarrier(_countof(pre), pre);
    h.list->CopyResource(v.color.tex, g_dlaa.raw);
    D3D12_RESOURCE_BARRIER post[] = {
        Transition(g_dlaa.raw, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COPY_DEST),
        Transition(v.color.tex, D3D12_RESOURCE_STATE_COPY_DEST,
                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
    };
    h.list->ResourceBarrier(_countof(post), post);
    (void)EndCommands();
}

static void CloseDlaaResources()
{
    g_dlaa_raw_shown = nullptr;
    g_dlaa_feeds_nr = false;
    g_dlaa_feed_out = g_dlaa_feed_raw = nullptr;
    g_dlaa.color_is_result = false;
    g_dlaa.fed_valid = false;
    if (g_submission_failed) return;
    if (g_dlaa.feature == nullptr && g_dlaa.output == nullptr && g_dlaa.depth.tex == nullptr &&
        g_dlaa.raw == nullptr && g_dlaa.mask == nullptr && g_dlaa.work_in == nullptr &&
        g_dlaa.full == nullptr && g_dlaa.motion == nullptr)
        return;
    // A frame recorded with these may still be on the queue: a toggle or a
    // resize is rare, a wait here is not on the frame path.
    if (h.fence != nullptr && h.fence_value != 0)
        (void)WaitFenceValue(h.fence, h.fence_value, 2000, "dlaa-close");
    if (g_dlaa.feature) { g_dlaa.release(g_dlaa.feature); g_dlaa.feature = nullptr; }
    if (g_dlaa.depth.tex) { g_dlaa.depth.tex->Release(); g_dlaa.depth.tex = nullptr; }
    if (g_dlaa.depth.upload) { g_dlaa.depth.upload->Release(); g_dlaa.depth.upload = nullptr; }
    for (ID3D12Resource **r : { &g_dlaa.output, &g_dlaa.mask, &g_dlaa.work_in, &g_dlaa.raw,
                                &g_dlaa.full, &g_dlaa.motion })
        if (*r) { (*r)->Release(); *r = nullptr; }
    g_dlaa.bound_color = g_dlaa.bound_mv = g_dlaa.bound_cost = nullptr;
    g_dlaa.w = g_dlaa.height = g_dlaa.fw = g_dlaa.fh = g_dlaa.mw = g_dlaa.mh = 0;
    g_dlaa.mask_count = UINT32_MAX;
    g_dlaa.history = false;
    g_dlaa.still_evals = 0;
}

// A refusal: logged with the prefix the client reads ("[dlaa] off:"), the
// capture put back, and the switch latched off until the user turns it on.
static void DlaaRefuse(VideoState &v, const char *fmt, ...)
{
    char text[256];
    va_list args;
    va_start(args, fmt);
    vsnprintf(text, sizeof(text), fmt, args);
    va_end(args);
    Log("[dlaa] off: %s", text);
    g_dlaa.failed = true;
    DlaaRestoreRaw(v);
    CloseDlaaResources();
}

// On shutdown: the runtime's parameter block and the module too.
static void UnloadDlaa()
{
    CloseDlaaResources();
    if (g_submission_failed) return;
    CloseDlaaTiming();
    if (g_dlaa.cost_pso) { g_dlaa.cost_pso->Release(); g_dlaa.cost_pso = nullptr; }
    g_dlaa.cost_pso_state = 0;
    if (g_dlaa.heap) { g_dlaa.heap->Release(); g_dlaa.heap = nullptr; }
    if (g_dlaa.cpu_heap) { g_dlaa.cpu_heap->Release(); g_dlaa.cpu_heap = nullptr; }
    if (g_dlaa.params) { NVSDK_NGX_D3D12_DestroyParameters(g_dlaa.params); g_dlaa.params = nullptr; }
    g_dlaa.create = nullptr; g_dlaa.release = nullptr; g_dlaa_evaluate = nullptr;
    if (g_dlaa.module) { FreeLibrary(g_dlaa.module); g_dlaa.module = nullptr; }
    g_dlaa.via_core = false;
    g_dlaa.runtime_major = 0;
}

// The direct runtime goes; NGX Core answers instead. False when the core was
// never initialised (it is for NR, so this is the exception).
static bool UseDlaaCore(const char *why)
{
    if (!h.ngx_inited) return false;
    if (g_dlaa.module) { FreeLibrary(g_dlaa.module); g_dlaa.module = nullptr; }
    // The SDK declares the parameter block non-const where our pointers take
    // it const; the call itself is identical (as NS_NGX_VIA_CORE does for NR).
    g_dlaa.create = reinterpret_cast<PFN_NR_Create>(&NVSDK_NGX_D3D12_CreateFeature);
    g_dlaa.release = reinterpret_cast<PFN_NR_Release>(&NVSDK_NGX_D3D12_ReleaseFeature);
    g_dlaa_evaluate = reinterpret_cast<PFN_NR_Evaluate>(&NVSDK_NGX_D3D12_EvaluateFeature_C);
    g_dlaa.via_core = true;
    g_dlaa.runtime_major = 0;
    Log("[dlaa] %s - asking NGX Core (the driver's own loader) instead", why);
    return true;
}

// The DLL's file version, for the log and for naming the model "auto" means.
static unsigned DlaaRuntimeMajor(const wchar_t *path)
{
    DWORD unused = 0;
    const DWORD n = GetFileVersionInfoSizeW(path, &unused);
    if (n == 0) return 0;
    std::vector<BYTE> info(n);
    VS_FIXEDFILEINFO *fi = nullptr;
    UINT len = 0;
    if (!GetFileVersionInfoW(path, 0, n, info.data()) ||
        !VerQueryValueW(info.data(), L"\\", reinterpret_cast<void **>(&fi), &len) || !fi)
        return 0;
    Log("[dlaa] DLSS runtime %u.%u.%u.%u", HIWORD(fi->dwFileVersionMS), LOWORD(fi->dwFileVersionMS),
        HIWORD(fi->dwFileVersionLS), LOWORD(fi->dwFileVersionLS));
    return HIWORD(fi->dwFileVersionMS);
}

static bool LoadDlaaRuntime(VideoState &v)
{
    if (g_dlaa.module || g_dlaa.via_core) return true;
    if (g_dlaa.params == nullptr &&
        NVSDK_NGX_FAILED(NVSDK_NGX_D3D12_AllocateParameters(&g_dlaa.params)))
    {
        g_dlaa.params = nullptr;
        DlaaRefuse(v, "no NGX parameter block");
        return false;
    }
    // The BYO library folder next to the worker, then the worker's own
    // directory - the same order and the same verification as FG's runtime.
    wchar_t directory[MAX_PATH] = {};
    GetModuleFileNameW(nullptr, directory, MAX_PATH);
    if (auto slash = wcsrchr(directory, L'\\')) *(slash + 1) = 0;
    wchar_t lib_path[MAX_PATH]; wcscpy_s(lib_path, directory);
    wcscat_s(lib_path, L"libraries\\nvngx_dlss.dll");
    const wchar_t *loaded_from = nullptr;
    if (GetFileAttributesW(lib_path) != INVALID_FILE_ATTRIBUTES)
    {
        if (NsGateByoDll(lib_path, "nvngx_dlss.dll"))
        {
            g_dlaa.module = LoadLibraryW(lib_path);
            if (g_dlaa.module)
            {
                loaded_from = lib_path;
                Log("[dlaa] DLSS runtime from native\\libraries\\ (BYO, verified NVIDIA signature)");
            }
        }
        else
            Log("[dlaa] BYO refused: nvngx_dlss.dll is not a NVIDIA-signed runtime (%ls)", lib_path);
    }
    wchar_t path[MAX_PATH]; wcscpy_s(path, directory);
    wcscat_s(path, L"nvngx_dlss.dll");
    if (!g_dlaa.module && GetFileAttributesW(path) != INVALID_FILE_ATTRIBUTES)
    {
        if (NsGateByoDll(path, "nvngx_dlss.dll"))
        {
            g_dlaa.module = LoadLibraryW(path);
            if (g_dlaa.module) loaded_from = path;
        }
        else Log("[dlaa] nvngx_dlss.dll next to the worker is not a NVIDIA-signed runtime");
    }
    const char *direct_failure = nullptr;
    if (!g_dlaa.module)
        direct_failure = "nvngx_dlss.dll not found in native\\libraries\\ or native\\";
    else
    {
        g_dlaa.runtime_major = DlaaRuntimeMajor(loaded_from);
        auto init = reinterpret_cast<PFN_NR_InitExt>(GetProcAddress(g_dlaa.module, "NVSDK_NGX_D3D12_Init_Ext"));
        g_dlaa.create = reinterpret_cast<PFN_NR_Create>(GetProcAddress(g_dlaa.module, "NVSDK_NGX_D3D12_CreateFeature"));
        g_dlaa.release = reinterpret_cast<PFN_NR_Release>(GetProcAddress(g_dlaa.module, "NVSDK_NGX_D3D12_ReleaseFeature"));
        g_dlaa_evaluate = reinterpret_cast<PFN_NR_Evaluate>(GetProcAddress(g_dlaa.module, "NVSDK_NGX_D3D12_EvaluateFeature"));
        if (!init || !g_dlaa.create || !g_dlaa.release || !g_dlaa_evaluate)
            direct_failure = "the DLSS runtime is missing an entry point";
        else
        {
            auto result = init(0x1000000ULL, directory, h.dev, NVSDK_NGX_Version_API, g_dlaa.params);
            if (TestFailureOnce("dlaa-init")) result = NVSDK_NGX_Result_Fail;
            Log("[dlaa] Init_Ext -> 0x%08X (%s)", result, NgxResultName(result));
            if (NVSDK_NGX_FAILED(result)) direct_failure = "the DLSS runtime refused to initialise";
        }
    }
    if (direct_failure == nullptr)
    {
        Log("[dlaa] runtime: nvngx_dlss.dll, called directly");
        return true;
    }
    g_dlaa.create = nullptr; g_dlaa.release = nullptr; g_dlaa_evaluate = nullptr;
    g_dlaa.runtime_major = 0;
    if (UseDlaaCore(direct_failure)) return true;
    if (g_dlaa.module) { FreeLibrary(g_dlaa.module); g_dlaa.module = nullptr; }
    DlaaRefuse(v, "%s - put NVIDIA's nvngx_dlss.dll into native\\libraries\\", direct_failure);
    return false;
}

// What "auto" resolves to, for the log and the status line: the 310 runtimes
// use K for DLAA (the SDK header says so); anything else is not assumed.
static const char *DlaaPresetName(unsigned preset)
{
    static char text[16];
    if (preset != 0)
    {
        text[0] = static_cast<char>('A' + preset - 1); text[1] = 0;
        return text;
    }
    return g_dlaa.runtime_major >= 310 ? "auto=K" : "auto";
}

// The cost-mask shader, compiled once; on the scale root signature.
static bool EnsureDlaaCostPipeline()
{
    if (g_dlaa.cost_pso_state != 0) return g_dlaa.cost_pso_state == 1;
    g_dlaa.cost_pso_state = -1;
    if (g_scale_rs == nullptr) return false;
    HMODULE compiler = LoadLibraryExW(L"d3dcompiler_47.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    auto compile = compiler ? reinterpret_cast<PFN_D3DCompile_>(GetProcAddress(compiler, "D3DCompile")) : nullptr;
    if (compile == nullptr) return false;
    unsigned threshold = kDlaaCostThreshold;
    char env[16] = {};
    if (GetEnvironmentVariableA("NS_DLAA_COST_T", env, sizeof(env)) > 0)
    {
        const unsigned t = static_cast<unsigned>(strtoul(env, nullptr, 10));
        if (t >= 1 && t <= 255) threshold = t;
    }
    char define[16];
    snprintf(define, sizeof(define), "%u", threshold);
    const D3D_SHADER_MACRO macros[] = { { "DLAA_COST_T", define }, { nullptr, nullptr } };
    ID3DBlob *code = nullptr, *errors = nullptr;
    HRESULT hr = compile(kDlaaCostMaskHlsl, sizeof(kDlaaCostMaskHlsl) - 1, "dlaa_cost.hlsl", macros, nullptr,
                         "CSMain", "cs_5_0", 0, 0, &code, &errors);
    if (FAILED(hr) || code == nullptr)
    {
        Log("[dlaa] cost-mask shader compile failed 0x%08X: %s", hr,
            errors ? static_cast<const char *>(errors->GetBufferPointer()) : "(no log)");
        if (errors) errors->Release();
        return false;
    }
    if (errors) errors->Release();
    D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
    pd.pRootSignature = g_scale_rs;
    pd.CS.pShaderBytecode = code->GetBufferPointer();
    pd.CS.BytecodeLength = code->GetBufferSize();
    hr = h.dev->CreateComputePipelineState(&pd, IID_PPV_ARGS(&g_dlaa.cost_pso));
    code->Release();
    if (FAILED(hr)) { Log("[dlaa] cost-mask pipeline failed 0x%08X", hr); return false; }
    g_dlaa.cost_pso_state = 1;
    Log("[dlaa] motion-confidence mask ready (cost above %u marks the motion unreliable)", threshold);
    return true;
}

// Every descriptor the frame uses, written once per (re)create: the textures
// they name only change with a resize, and a resize closes everything first.
static bool BindDlaaDescriptors(VideoState &v)
{
    const UINT stride = h.dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    if (g_dlaa.heap == nullptr)
    {
        D3D12_DESCRIPTOR_HEAP_DESC hd = {};
        hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        hd.NumDescriptors = kDlaaSlots;
        hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        if (FAILED(h.dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&g_dlaa.heap)))) return false;
        hd.NumDescriptors = 1;
        hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;   // ClearUnorderedAccessView's CPU handle
        if (FAILED(h.dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&g_dlaa.cpu_heap)))) return false;
    }
    const D3D12_CPU_DESCRIPTOR_HANDLE base = g_dlaa.heap->GetCPUDescriptorHandleForHeapStart();
    auto at = [&](UINT slot) { D3D12_CPU_DESCRIPTOR_HANDLE c = base; c.ptr += static_cast<SIZE_T>(slot) * stride; return c; };
    auto srv = [&](ID3D12Resource *r, DXGI_FORMAT f, UINT slot) {
        D3D12_SHADER_RESOURCE_VIEW_DESC sd = {};
        sd.Format = f;
        sd.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        sd.Texture2D.MipLevels = 1;
        h.dev->CreateShaderResourceView(r, &sd, at(slot));
    };
    auto uav = [&](ID3D12Resource *r, DXGI_FORMAT f, D3D12_CPU_DESCRIPTOR_HANDLE c) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
        ud.Format = f;
        ud.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        h.dev->CreateUnorderedAccessView(r, nullptr, &ud, c);
    };
    if (g_dlaa.motion)
    {
        srv(v.mv.tex, DXGI_FORMAT_R16G16_FLOAT, kDlaaMotionSrc);
        uav(g_dlaa.motion, DXGI_FORMAT_R16G16_FLOAT, at(kDlaaMotionDst));
    }
    if (g_dlaa.boost)
    {
        srv(v.color.tex, DXGI_FORMAT_R8G8B8A8_UNORM, kDlaaScaleSrc);
        uav(g_dlaa.work_in, DXGI_FORMAT_R8G8B8A8_UNORM, at(kDlaaScaleDst));
        srv(v.color.tex, DXGI_FORMAT_R8G8B8A8_UNORM, kDlaaResNative);
        srv(g_dlaa.work_in, DXGI_FORMAT_R8G8B8A8_UNORM, kDlaaResIn);
        srv(g_dlaa.output, DXGI_FORMAT_R8G8B8A8_UNORM, kDlaaResOut);
        uav(g_dlaa.full, DXGI_FORMAT_R8G8B8A8_UNORM, at(kDlaaResDst));
    }
    uav(g_dlaa.mask, DXGI_FORMAT_R8_UNORM, at(kDlaaMaskUav));
    uav(g_dlaa.mask, DXGI_FORMAT_R8_UNORM, at(kDlaaCostMaskUav));
    uav(g_dlaa.mask, DXGI_FORMAT_R8_UNORM, g_dlaa.cpu_heap->GetCPUDescriptorHandleForHeapStart());
    g_dlaa.bound_color = v.color.tex;
    g_dlaa.bound_mv = v.mv.tex;
    g_dlaa.bound_cost = nullptr;   // written by the cost pass when NVOFA has a buffer
    return true;
}

static D3D12_GPU_DESCRIPTOR_HANDLE DlaaGpuSlot(UINT slot)
{
    D3D12_GPU_DESCRIPTOR_HANDLE g = g_dlaa.heap->GetGPUDescriptorHandleForHeapStart();
    g.ptr += static_cast<UINT64>(slot) *
             h.dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    return g;
}

static bool CreateDlaaFeature()
{
    auto p = g_dlaa.params;
    p->Reset();
    p->Set(NVSDK_NGX_Parameter_CreationNodeMask, 1u);
    p->Set(NVSDK_NGX_Parameter_VisibilityNodeMask, 1u);
    p->Set(NVSDK_NGX_Parameter_Width, g_dlaa.w); p->Set(NVSDK_NGX_Parameter_Height, g_dlaa.height);
    p->Set(NVSDK_NGX_Parameter_OutWidth, g_dlaa.w); p->Set(NVSDK_NGX_Parameter_OutHeight, g_dlaa.height);
    p->Set(NVSDK_NGX_Parameter_PerfQualityValue, static_cast<int>(NVSDK_NGX_PerfQuality_Value_DLAA));
    // Motion at the render size, which for DLAA is the output size. The frame
    // is the 8-bit SDR desktop: no HDR flag, no exposure.
    p->Set(NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags,
           static_cast<int>(NVSDK_NGX_DLSS_Feature_Flags_MVLowRes));
    p->Set(NVSDK_NGX_Parameter_DLSS_Enable_Output_Subrects, 0);
    if (g_dlaa.preset != 0) p->Set(NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_DLAA, g_dlaa.preset);
    if (!BeginCommands()) return false;
    const auto result = DlaaCreateGuarded(g_dlaa.create, h.list, p, &g_dlaa.feature);
    if (g_dlaa_fault != 0)
    {
        // Never execute a list NVIDIA's runtime faulted in, never trust a
        // handle it may have half-written.
        AbortCommands();
        g_dlaa.feature = nullptr;
        ReportFailure("dlaa-create", "seh", static_cast<HRESULT>(g_dlaa_fault));
        Log("[dlaa] CreateFeature raised 0x%08X (caught)", g_dlaa_fault);
        return false;
    }
    if (!WaitFenceValue(h.fence, EndCommands(), 30000, "dlaa-create") || NVSDK_NGX_FAILED(result) ||
        !g_dlaa.feature)
    {
        g_dlaa.feature = nullptr;
        Log("[dlaa] CreateFeature failed 0x%08X (%s)%s", result, NgxResultName(result),
            g_dlaa.via_core ? " through NGX Core" : "");
        return false;
    }
    return true;
}

static bool EnsureDlaa(VideoState &v)
{
    const UINT fw = v.upscale ? v.full_w : v.w, fh = v.upscale ? v.full_h : v.hgt;
    const bool boost = v.nr_small && (v.w != fw || v.hgt != fh);
    const UINT w = boost ? v.w : fw, height = boost ? v.hgt : fh;
    if (g_dlaa.feature && g_dlaa.fw == fw && g_dlaa.fh == fh && g_dlaa.mw == v.w &&
        g_dlaa.mh == v.hgt && g_dlaa.boost == boost && g_dlaa.bound_color == v.color.tex &&
        g_dlaa.bound_mv == v.mv.tex && g_dlaa.preset == DlaaPreset()) return true;
    DlaaRestoreRaw(v);
    CloseDlaaResources();
    if (!LoadDlaaRuntime(v)) return false;
    // The stretch and the Boost passes are the worker's own scale and
    // residual shaders.
    const bool stretch = !boost && (v.w != fw || v.hgt != fh);
    if (!EnsureScalePipeline())
    {
        DlaaRefuse(v, "the scale pipeline is unavailable");
        return false;
    }
    g_dlaa.w = w; g_dlaa.height = height; g_dlaa.fw = fw; g_dlaa.fh = fh;
    g_dlaa.mw = v.w; g_dlaa.mh = v.hgt; g_dlaa.boost = boost;
    g_dlaa.preset = DlaaPreset();
    if (!CreateDlaaFeature())
    {
        // The direct runtime said no (not a fault): NGX Core gets one try.
        const bool retry = g_dlaa_fault == 0 && !g_dlaa.via_core &&
                           UseDlaaCore("the direct runtime refused CreateFeature");
        if (!retry || !CreateDlaaFeature())
        {
            if (!g_submission_failed)
                DlaaRefuse(v, "the DLSS runtime refused DLAA at %ux%u%s", w, height,
                           g_dlaa.via_core ? " (NGX Core too)" : "");
            return false;
        }
    }
    if (!CreateVideoTex(g_dlaa.depth, w, height, DXGI_FORMAT_R32_FLOAT, w * 4))
    { DlaaRefuse(v, "the depth texture could not be created"); return false; }
    {
        std::vector<float> depth(static_cast<size_t>(w) * height, 0.5f);
        if (!FillUpload(g_dlaa.depth, reinterpret_cast<BYTE *>(depth.data()), w * 4, height))
        { DlaaRefuse(v, "the depth texture could not be filled"); return false; }
    }
    g_dlaa.output = MakeTex(w, height, DXGI_FORMAT_R8G8B8A8_UNORM, true);
    g_dlaa.mask = MakeTex(w, height, DXGI_FORMAT_R8_UNORM, true);
    g_dlaa.raw = MakeTex(fw, fh, DXGI_FORMAT_R8G8B8A8_UNORM, false);
    if (boost)
    {
        g_dlaa.work_in = MakeTex(w, height, DXGI_FORMAT_R8G8B8A8_UNORM, true);
        g_dlaa.full = MakeTex(fw, fh, DXGI_FORMAT_R8G8B8A8_UNORM, true);
    }
    if (stretch) g_dlaa.motion = MakeTex(fw, fh, DXGI_FORMAT_R16G16_FLOAT, true);
    if (!g_dlaa.output || !g_dlaa.mask || !g_dlaa.raw || (boost && (!g_dlaa.work_in || !g_dlaa.full)) ||
        (stretch && !g_dlaa.motion))
    { DlaaRefuse(v, "the working textures could not be created"); return false; }
    if (!BindDlaaDescriptors(v)) { DlaaRefuse(v, "descriptor heap"); return false; }
    if (!BeginCommands()) return false;
    CopyUpload(h.list, g_dlaa.depth);
    D3D12_RESOURCE_BARRIER b[7] = {};
    UINT n = 0;
    b[n++] = Transition(g_dlaa.depth.tex, D3D12_RESOURCE_STATE_COPY_DEST,
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    b[n++] = Transition(g_dlaa.output, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    b[n++] = Transition(g_dlaa.mask, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    b[n++] = Transition(g_dlaa.raw, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    if (g_dlaa.work_in)
        b[n++] = Transition(g_dlaa.work_in, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    if (g_dlaa.full)
        b[n++] = Transition(g_dlaa.full, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (g_dlaa.motion)
        b[n++] = Transition(g_dlaa.motion, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    h.list->ResourceBarrier(n, b);
    if (!WaitFenceValue(h.fence, EndCommands(), 30000, "dlaa-create"))
    { DlaaRefuse(v, "the GPU did not finish setting DLAA up"); return false; }
    // The flat plane is on the GPU now; its staging copy (33 MB at 4K) goes.
    g_dlaa.depth.upload->Release(); g_dlaa.depth.upload = nullptr;
    // "[dlaa] enabled at" is a contract: settings_io parses it for the status line.
    Log("[dlaa] enabled at %ux%u (frame %ux%u%s), motion %ux%u, preset %s, %s; flat depth and "
        "estimated motion", w, height, fw, fh, boost ? ", Boost" : "",
        v.w, v.hgt, DlaaPresetName(g_dlaa.preset), g_dlaa.via_core ? "NGX Core" : "direct");
    return true;
}

// The bias mask: 1 where DLSS should trust the current frame - the still
// interface the client found, and (opt-in) where NVOFA's cost says the motion
// is a guess. Returns whether a mask is handed to DLSS this frame. The
// rectangles alone are redrawn only when they change; with the cost pass the
// mask is rebuilt on every evaluation, since the cost is.
static bool DrawDlaaMask()
{
    const uint32_t count = (std::min)(g_ui_rects.count, kUiRectsMax);
    auto &f = g_nvofa;
    const bool cost = g_dlaa_cost_wanted && f.cost != nullptr && f.valid && EnsureDlaaCostPipeline();
    const bool rects_changed = count != g_dlaa.mask_count ||
        memcmp(g_dlaa.mask_rects, g_ui_rects.rects, sizeof(g_dlaa.mask_rects[0]) * count) != 0;
    if (!cost && !rects_changed) return count > 0;
    g_dlaa.mask_count = count;
    memcpy(g_dlaa.mask_rects, g_ui_rects.rects, sizeof(g_dlaa.mask_rects[0]) * count);
    if (!cost && count == 0) return false;   // no mask is handed to DLSS then
    auto to_uav = Transition(g_dlaa.mask, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                             D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    h.list->ResourceBarrier(1, &to_uav);
    ID3D12DescriptorHeap *heaps[] = { g_dlaa.heap };
    h.list->SetDescriptorHeaps(1, heaps);
    const D3D12_GPU_DESCRIPTOR_HANDLE gpu = DlaaGpuSlot(kDlaaMaskUav);
    const D3D12_CPU_DESCRIPTOR_HANDLE cpu = g_dlaa.cpu_heap->GetCPUDescriptorHandleForHeapStart();
    const FLOAT zero[4] = { 0, 0, 0, 0 }, one[4] = { 1, 1, 1, 1 };
    h.list->ClearUnorderedAccessViewFloat(gpu, cpu, g_dlaa.mask, zero, 0, nullptr);
    D3D12_RESOURCE_BARRIER between = {};
    between.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    between.UAV.pResource = g_dlaa.mask;
    h.list->ResourceBarrier(1, &between);
    D3D12_RECT rects[kUiRectsMax];
    UINT n = 0;
    for (uint32_t r = 0; r < count; ++r)
    {
        // The same 1/65535ths of the frame FG reads (frame_generation.inl).
        const auto &q = g_ui_rects.rects[r];
        const LONG x0 = static_cast<LONG>(static_cast<uint64_t>(q[0]) * g_dlaa.w / 65535u);
        const LONG y0 = static_cast<LONG>(static_cast<uint64_t>(q[1]) * g_dlaa.height / 65535u);
        const LONG x1 = (std::min)(static_cast<LONG>(g_dlaa.w),
                                   x0 + static_cast<LONG>(static_cast<uint64_t>(q[2]) * g_dlaa.w / 65535u));
        const LONG y1 = (std::min)(static_cast<LONG>(g_dlaa.height),
                                   y0 + static_cast<LONG>(static_cast<uint64_t>(q[3]) * g_dlaa.height / 65535u));
        if (x1 > x0 && y1 > y0) rects[n++] = { x0, y0, x1, y1 };
    }
    if (n > 0)
    {
        h.list->ClearUnorderedAccessViewFloat(gpu, cpu, g_dlaa.mask, one, n, rects);
        h.list->ResourceBarrier(1, &between);
    }
    if (cost)
    {
        if (f.cost.get() != g_dlaa.bound_cost)
        {
            // NVOFA's buffer is new (its session was rebuilt): its descriptor
            // is reissued. The buffer changes only with a rebuild, which waits
            // for every submission, so no list in flight reads the old one.
            D3D12_SHADER_RESOURCE_VIEW_DESC sd = {};
            sd.Format = DXGI_FORMAT_R8_UINT;
            sd.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            sd.Texture2D.MipLevels = 1;
            D3D12_CPU_DESCRIPTOR_HANDLE c = g_dlaa.heap->GetCPUDescriptorHandleForHeapStart();
            c.ptr += static_cast<SIZE_T>(kDlaaCostSrc) *
                     h.dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            h.dev->CreateShaderResourceView(f.cost.get(), &sd, c);
            g_dlaa.bound_cost = f.cost.get();
        }
        const auto desc = f.cost->GetDesc();
        auto cost_in = Transition(f.cost.get(), D3D12_RESOURCE_STATE_COMMON,
                                  D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        h.list->ResourceBarrier(1, &cost_in);
        h.list->SetComputeRootSignature(g_scale_rs);
        h.list->SetPipelineState(g_dlaa.cost_pso);
        const UINT sizes[4] = { g_dlaa.w, g_dlaa.height, static_cast<UINT>(desc.Width), desc.Height };
        h.list->SetComputeRoot32BitConstants(0, 4, sizes, 0);
        h.list->SetComputeRootDescriptorTable(1, DlaaGpuSlot(kDlaaCostSrc));
        h.list->Dispatch((g_dlaa.w + 7) / 8, (g_dlaa.height + 7) / 8, 1);
        auto cost_out = Transition(f.cost.get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                   D3D12_RESOURCE_STATE_COMMON);
        h.list->ResourceBarrier(1, &cost_out);
    }
    auto to_srv = Transition(g_dlaa.mask, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                             D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    h.list->ResourceBarrier(1, &to_srv);
    return true;
}

// Copy `result` (UNORDERED_ACCESS at rest) over v.color.tex.
static void DlaaShow(VideoState &v, ID3D12Resource *result)
{
    D3D12_RESOURCE_BARRIER pre[] = {
        Transition(v.color.tex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                   D3D12_RESOURCE_STATE_COPY_DEST),
        Transition(result, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
    };
    h.list->ResourceBarrier(_countof(pre), pre);
    h.list->CopyResource(v.color.tex, result);
    D3D12_RESOURCE_BARRIER post[] = {
        Transition(v.color.tex, D3D12_RESOURCE_STATE_COPY_DEST,
                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
        Transition(result, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
    };
    h.list->ResourceBarrier(_countof(post), post);
}

// The network takes DLAA's work-size result this frame (EvaluateVideo).
static void DlaaFeedNr()
{
    g_dlaa_feeds_nr = true;
    g_dlaa_feed_out = g_dlaa.output;
    g_dlaa_feed_raw = g_dlaa.work_in;
    g_dlaa_feed_strength = DlaaStrength();
}

// Called on every frame, after the capture and the motion upload and ahead of
// everything that reads v.color.tex, in a submission of its own (queue order
// puts it before the neural pass and the present). Nothing waits on it.
//
// color_written: the capture (or the pipe) wrote v.color.tex this frame.
// still: it wrote the same picture as before. nr_consumes: the neural pass
// runs on this frame under Boost and takes the work-size result (the full
// frame is left as captured). False only for a failure the frame loop has to
// end on; a refusal turns DLAA off and the frame goes on.
static bool RunDlaa(VideoState &v, bool color_written, bool still, bool reset, bool nr_consumes)
{
    if (color_written) { g_dlaa.color_is_result = false; g_dlaa.fed_valid = false; }
    g_dlaa_raw_shown = nullptr;
    g_dlaa_feeds_nr = false;
    if (!DlaaRequested() || !v.color_in_srv || !v.inputs_ready) return true;
    DlaaCollectTiming();
    const bool model_changed = g_dlaa.feature != nullptr && g_dlaa.preset != DlaaPreset();
    // Nothing new, and the result for this picture exists already: on screen
    // (v.color.tex) or, in the feed mode, in output for the network - unless
    // the model changed, which has to show on a still desktop too (EnsureDlaa
    // puts the capture back and the new model runs on it).
    const bool feed = nr_consumes && g_dlaa.boost;
    if (!color_written && !model_changed)
    {
        if (feed && g_dlaa.fed_valid) { DlaaFeedNr(); return true; }
        if (!feed && g_dlaa.color_is_result) { g_dlaa_raw_shown = g_dlaa.raw; return true; }
    }
    // The HDR presentation composes the native HDR capture, not v.color: DLAA
    // would change only the SDR proxy under it.
    if (g_hdr_capture)
    {
        DlaaRefuse(v, "the HDR presentation shows the native HDR capture; DLAA works on SDR only");
        return true;
    }
    if (!EnsureDlaa(v)) return !g_submission_failed;
    const bool feeding = nr_consumes && g_dlaa.boost;
    const bool fresh_history = reset || !g_dlaa.history;
    if (!(color_written && still)) g_dlaa.still_evals = 0;
    ID3D12Resource *result = g_dlaa.boost ? g_dlaa.full : g_dlaa.output;
    // A picture DLSS has settled on: shown from the last result, no evaluation.
    const bool hold = color_written && still && !fresh_history &&
                      g_dlaa.still_evals >= kDlaaStillEvaluations;
    if (hold && feeding)
    {
        // output still holds the result; the network takes it. No commands.
        g_dlaa.fed_valid = true;
        DlaaFeedNr();
        return true;
    }
    if (!BeginCommands()) return false;
    if (hold)
    {
        DlaaShow(v, result);
        if (EndCommands() == 0) return false;
        g_dlaa.color_is_result = true;
        g_dlaa_raw_shown = g_dlaa.raw;
        return true;
    }
    const int slot = h.frame_slot;
    const bool timed = DlaaTimingReady();
    if (timed) h.list->EndQuery(g_dlaa_time.heap, D3D12_QUERY_TYPE_TIMESTAMP, slot * 2);
    if (!feeding)
    {
        // The capture as it came, kept for the wipe and for switching off.
        D3D12_RESOURCE_BARRIER pre = Transition(v.color.tex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                                D3D12_RESOURCE_STATE_COPY_SOURCE);
        h.list->ResourceBarrier(1, &pre);
        h.list->CopyResource(g_dlaa.raw, v.color.tex);
        D3D12_RESOURCE_BARRIER post = Transition(v.color.tex, D3D12_RESOURCE_STATE_COPY_SOURCE,
                                                 D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        h.list->ResourceBarrier(1, &post);
    }
    ID3D12DescriptorHeap *heaps[] = { g_dlaa.heap };
    ID3D12Resource *input = v.color.tex, *motion = v.mv.tex;
    float scale_x = 1.0f, scale_y = 1.0f;
    if (g_dlaa.boost)
    {
        // The frame at the work size, as NR's input is made (area reduction).
        auto to_uav = Transition(g_dlaa.work_in, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                 D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        h.list->ResourceBarrier(1, &to_uav);
        h.list->SetDescriptorHeaps(1, heaps);
        h.list->SetComputeRootSignature(g_scale_rs);
        h.list->SetPipelineState(g_scale4_pso);
        const UINT sizes[4] = { g_dlaa.w, g_dlaa.height, g_dlaa.fw, g_dlaa.fh };
        h.list->SetComputeRoot32BitConstants(0, 4, sizes, 0);
        h.list->SetComputeRootDescriptorTable(1, DlaaGpuSlot(kDlaaScaleSrc));
        h.list->Dispatch((g_dlaa.w + 7) / 8, (g_dlaa.height + 7) / 8, 1);
        auto to_srv = Transition(g_dlaa.work_in, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                 D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        h.list->ResourceBarrier(1, &to_srv);
        input = g_dlaa.work_in;
    }
    else if (g_dlaa.motion != nullptr)
    {
        // The field is in work-size pixels; stretched to the frame's size, its
        // values are scaled by DLSS's own MV scale rather than by the shader.
        h.list->SetDescriptorHeaps(1, heaps);
        h.list->SetComputeRootSignature(g_scale_rs);
        h.list->SetPipelineState(g_scale_pso);
        const UINT sizes[4] = { g_dlaa.w, g_dlaa.height, v.w, v.hgt };
        h.list->SetComputeRoot32BitConstants(0, 4, sizes, 0);
        h.list->SetComputeRootDescriptorTable(1, DlaaGpuSlot(kDlaaMotionSrc));
        h.list->Dispatch((g_dlaa.w + 7) / 8, (g_dlaa.height + 7) / 8, 1);
        auto to_srv = Transition(g_dlaa.motion, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                 D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        h.list->ResourceBarrier(1, &to_srv);
        motion = g_dlaa.motion;
        scale_x = static_cast<float>(g_dlaa.w) / static_cast<float>(v.w);
        scale_y = static_cast<float>(g_dlaa.height) / static_cast<float>(v.hgt);
    }
    const bool masked = DrawDlaaMask();
    auto p = g_dlaa.params;
    p->Reset();
    p->Set(NVSDK_NGX_Parameter_Color, input);
    p->Set(NVSDK_NGX_Parameter_Output, g_dlaa.output);
    p->Set(NVSDK_NGX_Parameter_Depth, g_dlaa.depth.tex);
    p->Set(NVSDK_NGX_Parameter_MotionVectors, motion);
    if (masked) p->Set(NVSDK_NGX_Parameter_DLSS_Input_Bias_Current_Color_Mask, g_dlaa.mask);
    p->Set(NVSDK_NGX_Parameter_Jitter_Offset_X, 0.0f);
    p->Set(NVSDK_NGX_Parameter_Jitter_Offset_Y, 0.0f);
    p->Set(NVSDK_NGX_Parameter_Reset, fresh_history ? 1 : 0);
    p->Set(NVSDK_NGX_Parameter_MV_Scale_X, scale_x);
    p->Set(NVSDK_NGX_Parameter_MV_Scale_Y, scale_y);
    p->Set(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width, g_dlaa.w);
    p->Set(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height, g_dlaa.height);
    p->Set(NVSDK_NGX_Parameter_DLSS_Pre_Exposure, 1.0f);
    p->Set(NVSDK_NGX_Parameter_DLSS_Exposure_Scale, 1.0f);
    const NVSDK_NGX_Result evaluated = DlaaEvaluateGuarded(h.list, g_dlaa.feature, p);
    if (g_dlaa_fault != 0)
    {
        // Never execute a list NVIDIA's runtime faulted in.
        AbortCommands();
        ReportFailure("dlaa-evaluate", "seh", static_cast<HRESULT>(g_dlaa_fault));
        DlaaRefuse(v, "EvaluateFeature raised 0x%08X (caught)", g_dlaa_fault);
        return !g_submission_failed;
    }
    if (NVSDK_NGX_FAILED(evaluated))
    {
        // Nothing in the list ran yet; dropping it leaves the frame as captured.
        AbortCommands();
        DlaaRefuse(v, "EvaluateFeature failed 0x%08X (%s)", evaluated, NgxResultName(evaluated));
        return !g_submission_failed;
    }
    if (g_dlaa.boost && !feeding)
    {
        // native + (DLAA(small) - small) * strength, with NR's residual pass
        // on DLAA's own descriptors: the neural pass is off, so DLAA composes
        // the full frame itself.
        auto out_srv = Transition(g_dlaa.output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                  D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        h.list->ResourceBarrier(1, &out_srv);
        h.list->SetDescriptorHeaps(1, heaps);
        h.list->SetComputeRootSignature(g_residual_rs);
        h.list->SetPipelineState(g_residual_pso);
        UINT32 rc[4] = { g_dlaa.fw, g_dlaa.fh, 0, 0 };
        const float strength = DlaaStrength(), none = 0.0f;
        memcpy(&rc[2], &strength, sizeof(float));
        memcpy(&rc[3], &none, sizeof(float));
        h.list->SetComputeRoot32BitConstants(0, 4, rc, 0);
        h.list->SetComputeRootDescriptorTable(1, DlaaGpuSlot(kDlaaResNative));
        h.list->Dispatch((g_dlaa.fw + 7) / 8, (g_dlaa.fh + 7) / 8, 1);
        auto out_uav = Transition(g_dlaa.output, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                  D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        h.list->ResourceBarrier(1, &out_uav);
    }
    else if (g_dlaa.motion != nullptr)
    {
        auto back = Transition(g_dlaa.motion, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        h.list->ResourceBarrier(1, &back);
    }
    // The result replaces the capture: every consumer reads v.color.tex. In
    // the feed mode the network reads output instead and the frame stays raw.
    if (!feeding) DlaaShow(v, result);
    if (timed)
    {
        h.list->EndQuery(g_dlaa_time.heap, D3D12_QUERY_TYPE_TIMESTAMP, slot * 2 + 1);
        h.list->ResolveQueryData(g_dlaa_time.heap, D3D12_QUERY_TYPE_TIMESTAMP, slot * 2, 2,
                                 g_dlaa_time.readback, sizeof(UINT64) * slot * 2);
    }
    const UINT64 fence = EndCommands();
    if (fence == 0) return false;
    if (timed) g_dlaa_time.fence[slot] = fence;
    g_dlaa.history = true;
    if (feeding) { g_dlaa.fed_valid = true; DlaaFeedNr(); }
    else { g_dlaa.color_is_result = true; g_dlaa_raw_shown = g_dlaa.raw; }
    if (color_written && still) ++g_dlaa.still_evals;
    return true;
}

// The switch rides on every frame, like FG's (FRAME_FLAG_DLAA_SET says the
// client sends it, FRAME_FLAG_DLAA is its state).
static void ConfigureDlaaFrame(VideoState &v, uint32_t flags)
{
    if (!(flags & FRAME_FLAG_DLAA_SET)) return;
    const int enabled = (flags & FRAME_FLAG_DLAA) != 0;
    if (g_dlaa_ui_enabled == enabled) return;
    // Off: the capture goes back on screen now, not when the desktop next
    // changes.
    DlaaRestoreRaw(v);
    CloseDlaaResources();
    g_dlaa_ui_enabled = enabled;
    // A fresh switch-on is a fresh attempt: a runtime dropped into
    // native\libraries\ meanwhile gets its chance without a restart.
    // A runtime dropped in after NGX Core answered for the missing one gets
    // the direct route back.
    if (enabled && g_dlaa.failed && g_dlaa.via_core)
    {
        g_dlaa.via_core = false;
        g_dlaa.create = nullptr; g_dlaa.release = nullptr; g_dlaa_evaluate = nullptr;
    }
    if (enabled) g_dlaa.failed = false;
    Log("[dlaa] UI: %s", enabled ? "on" : "off");
}
