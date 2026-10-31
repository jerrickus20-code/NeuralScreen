// DLAA: DLSS super resolution at 1:1 (NVSDK_NGX_Feature_SuperSampling,
// PerfQuality_Value_DLAA) over the captured frame, before the neural pass.
//
// The result is copied back into v.color.tex, so everything downstream of the
// capture - NR, Boost's residual anchor, the bypass present, FG, the
// recording - sees the anti-aliased frame and needs no change of its own. The
// raw frame is kept beside it (g_dlaa.raw): the wipe's "before" half shows it,
// and switching DLAA off puts it back at once, even on a desktop that does not
// change.
//
// What a capture can and cannot give DLSS. Motion: the same current ->
// previous field NR is handed (in work-size pixels). Depth: none - a flat
// plane, as FG's. Jitter: none - the game already resolved its frame, so DLAA
// here is temporal anti-aliasing and stabilisation of what the screen shows,
// not the sub-pixel reconstruction it does inside an engine. Where the client
// found a still interface over a moving picture (the UIRC rectangles FG's
// interface protection uses), DLSS is told to trust the current frame
// (DLSS.Input.Bias.Current.Color.Mask) instead of dragging the HUD along with
// the motion behind it.
//
// Two sizes. Without Boost DLAA runs at the frame's size, as DLAA does. With
// Boost it runs at Boost's work size, on the frame scaled down the way NR's
// input is, and its correction is composed onto the native frame with NR's own
// residual pass (native + DLAA(small) - small): a fraction of the cost, and the
// native frame keeps its sharpness - coarser anti-aliasing, the same trade
// Boost makes for the network.
//
// The runtime is NVIDIA's public nvngx_dlss.dll: native\libraries\ first
// (verified like every BYO file), then native\, initialised directly as
// nvngx_dlssg.dll is. When that route is missing or refused, the feature is
// asked of NGX Core - the driver's own loader, initialised for NR already.
// Opt-in, like FG.
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
    kDlaaSlots = 9
};

static struct DlaaState {
    HMODULE module = nullptr;
    bool via_core = false;
    PFN_NR_Create create = nullptr;
    PFN_NR_Release release = nullptr;
    NVSDK_NGX_Parameter *params = nullptr;
    NVSDK_NGX_Handle *feature = nullptr;
    // Render-size textures. output: UNORDERED_ACCESS at rest; depth (flat),
    // mask and small (Boost's input): NON_PIXEL_SHADER_RESOURCE at rest.
    VideoTex depth;
    ID3D12Resource *output = nullptr, *mask = nullptr, *small = nullptr;
    // Frame-size textures. raw: the capture as it came, COPY_DEST at rest.
    // full: Boost's composite, UNORDERED_ACCESS at rest. motion: the field
    // stretched to the frame's size when the work size is smaller (without
    // Boost), UNORDERED_ACCESS at rest.
    ID3D12Resource *raw = nullptr, *full = nullptr, *motion = nullptr;
    // Its own descriptors: the shared scale and residual sets are rebound by
    // MOTS and by NR within a frame, and a shader-visible heap is read when the
    // list runs, not when it is recorded (see BindScale4Descriptors).
    ID3D12DescriptorHeap *heap = nullptr, *cpu_heap = nullptr;
    ID3D12Resource *bound_color = nullptr, *bound_mv = nullptr;
    UINT w = 0, height = 0;   // the render size (DLSS's input and output)
    UINT fw = 0, fh = 0;      // the frame's size
    UINT mw = 0, mh = 0;      // the motion field's size
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
    // The runtime refused or faulted: off until the switch is turned on again.
    bool failed = false;
} g_dlaa;
static int g_dlaa_ui_enabled = -1;

static bool DlaaRequested()
{
    static const bool requested = [] {
        char value[8] = {};
        return GetEnvironmentVariableA("NS_DLAA", value, sizeof(value)) == 1 && value[0] == '1';
    }();
    return (g_dlaa_ui_enabled < 0 ? requested : g_dlaa_ui_enabled != 0) && !g_dlaa.failed;
}

// NS_DLAA_PRESET=J|K|L|M: the DLSS model for DLAA. Unset: the runtime's own
// choice (K, the transformer model, on the 310 runtimes). Read once.
static unsigned DlaaPreset()
{
    static const unsigned preset = [] {
        char value[8] = {};
        if (GetEnvironmentVariableA("NS_DLAA_PRESET", value, sizeof(value)) != 1) return 0u;
        const char letter = static_cast<char>(value[0] & ~0x20);
        if (letter < 'J' || letter > 'M') return 0u;
        return static_cast<unsigned>(NVSDK_NGX_DLSS_Hint_Render_Preset_J) + (letter - 'J');
    }();
    return preset;
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
    g_dlaa.color_is_result = false;
    if (g_submission_failed) return;
    if (g_dlaa.feature == nullptr && g_dlaa.output == nullptr && g_dlaa.depth.tex == nullptr &&
        g_dlaa.raw == nullptr && g_dlaa.mask == nullptr && g_dlaa.small == nullptr &&
        g_dlaa.full == nullptr && g_dlaa.motion == nullptr)
        return;
    // A frame recorded with these may still be on the queue: a toggle or a
    // resize is rare, a wait here is not on the frame path.
    if (h.fence != nullptr && h.fence_value != 0)
        (void)WaitFenceValue(h.fence, h.fence_value, 2000, "dlaa-close");
    if (g_dlaa.feature) { g_dlaa.release(g_dlaa.feature); g_dlaa.feature = nullptr; }
    if (g_dlaa.depth.tex) { g_dlaa.depth.tex->Release(); g_dlaa.depth.tex = nullptr; }
    if (g_dlaa.depth.upload) { g_dlaa.depth.upload->Release(); g_dlaa.depth.upload = nullptr; }
    for (ID3D12Resource **r : { &g_dlaa.output, &g_dlaa.mask, &g_dlaa.small, &g_dlaa.raw,
                                &g_dlaa.full, &g_dlaa.motion })
        if (*r) { (*r)->Release(); *r = nullptr; }
    g_dlaa.bound_color = g_dlaa.bound_mv = nullptr;
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
    if (g_dlaa.heap) { g_dlaa.heap->Release(); g_dlaa.heap = nullptr; }
    if (g_dlaa.cpu_heap) { g_dlaa.cpu_heap->Release(); g_dlaa.cpu_heap = nullptr; }
    if (g_dlaa.params) { NVSDK_NGX_D3D12_DestroyParameters(g_dlaa.params); g_dlaa.params = nullptr; }
    g_dlaa.create = nullptr; g_dlaa.release = nullptr; g_dlaa_evaluate = nullptr;
    if (g_dlaa.module) { FreeLibrary(g_dlaa.module); g_dlaa.module = nullptr; }
    g_dlaa.via_core = false;
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
    Log("[dlaa] %s - asking NGX Core (the driver's own loader) instead", why);
    return true;
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
    if (GetFileAttributesW(lib_path) != INVALID_FILE_ATTRIBUTES)
    {
        if (NsGateByoDll(lib_path, "nvngx_dlss.dll"))
        {
            g_dlaa.module = LoadLibraryW(lib_path);
            if (g_dlaa.module)
                Log("[dlaa] DLSS runtime from native\\libraries\\ (BYO, verified NVIDIA signature)");
        }
        else
            Log("[dlaa] BYO refused: nvngx_dlss.dll is not a NVIDIA-signed runtime (%ls)", lib_path);
    }
    if (!g_dlaa.module)
    {
        wchar_t path[MAX_PATH]; wcscpy_s(path, directory);
        wcscat_s(path, L"nvngx_dlss.dll");
        if (GetFileAttributesW(path) != INVALID_FILE_ATTRIBUTES)
        {
            if (NsGateByoDll(path, "nvngx_dlss.dll")) g_dlaa.module = LoadLibraryW(path);
            else Log("[dlaa] nvngx_dlss.dll next to the worker is not a NVIDIA-signed runtime");
        }
    }
    const char *direct_failure = nullptr;
    if (!g_dlaa.module)
        direct_failure = "nvngx_dlss.dll not found in native\\libraries\\ or native\\";
    else
    {
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
    if (UseDlaaCore(direct_failure)) return true;
    if (g_dlaa.module) { FreeLibrary(g_dlaa.module); g_dlaa.module = nullptr; }
    DlaaRefuse(v, "%s - put NVIDIA's nvngx_dlss.dll into native\\libraries\\", direct_failure);
    return false;
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
        uav(g_dlaa.small, DXGI_FORMAT_R8G8B8A8_UNORM, at(kDlaaScaleDst));
        srv(v.color.tex, DXGI_FORMAT_R8G8B8A8_UNORM, kDlaaResNative);
        srv(g_dlaa.small, DXGI_FORMAT_R8G8B8A8_UNORM, kDlaaResIn);
        srv(g_dlaa.output, DXGI_FORMAT_R8G8B8A8_UNORM, kDlaaResOut);
        uav(g_dlaa.full, DXGI_FORMAT_R8G8B8A8_UNORM, at(kDlaaResDst));
    }
    uav(g_dlaa.mask, DXGI_FORMAT_R8_UNORM, at(kDlaaMaskUav));
    uav(g_dlaa.mask, DXGI_FORMAT_R8_UNORM, g_dlaa.cpu_heap->GetCPUDescriptorHandleForHeapStart());
    g_dlaa.bound_color = v.color.tex;
    g_dlaa.bound_mv = v.mv.tex;
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
    if (DlaaPreset() != 0) p->Set(NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_DLAA, DlaaPreset());
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
        g_dlaa.bound_mv == v.mv.tex) return true;
    DlaaRestoreRaw(v);
    CloseDlaaResources();
    if (!LoadDlaaRuntime(v)) return false;
    // The stretch and the Boost passes are the worker's own scale and
    // residual shaders.
    const bool stretch = !boost && (v.w != fw || v.hgt != fh);
    if ((stretch || boost) && !EnsureScalePipeline())
    {
        DlaaRefuse(v, "the scale pipeline is unavailable");
        return false;
    }
    g_dlaa.w = w; g_dlaa.height = height; g_dlaa.fw = fw; g_dlaa.fh = fh;
    g_dlaa.mw = v.w; g_dlaa.mh = v.hgt; g_dlaa.boost = boost;
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
        g_dlaa.small = MakeTex(w, height, DXGI_FORMAT_R8G8B8A8_UNORM, true);
        g_dlaa.full = MakeTex(fw, fh, DXGI_FORMAT_R8G8B8A8_UNORM, true);
    }
    if (stretch) g_dlaa.motion = MakeTex(fw, fh, DXGI_FORMAT_R16G16_FLOAT, true);
    if (!g_dlaa.output || !g_dlaa.mask || !g_dlaa.raw || (boost && (!g_dlaa.small || !g_dlaa.full)) ||
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
    if (g_dlaa.small)
        b[n++] = Transition(g_dlaa.small, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    if (g_dlaa.full)
        b[n++] = Transition(g_dlaa.full, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (g_dlaa.motion)
        b[n++] = Transition(g_dlaa.motion, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    h.list->ResourceBarrier(n, b);
    if (!WaitFenceValue(h.fence, EndCommands(), 30000, "dlaa-create"))
    { DlaaRefuse(v, "the GPU did not finish setting DLAA up"); return false; }
    // The flat plane is on the GPU now; its staging copy (33 MB at 4K) goes.
    g_dlaa.depth.upload->Release(); g_dlaa.depth.upload = nullptr;
    const unsigned preset = DlaaPreset();
    Log("[dlaa] enabled at %ux%u (frame %ux%u%s), motion %ux%u, preset %c, %s; flat depth and "
        "estimated motion", w, height, fw, fh, boost ? ", Boost: composed onto the native frame" : "",
        v.w, v.hgt, preset ? static_cast<char>('A' + preset - 1) : '-',
        g_dlaa.via_core ? "NGX Core" : "direct");
    return true;
}

// The interface mask: 1 where the client found a still interface, 0 elsewhere.
// Redrawn only when the rectangles change.
static void DrawDlaaMask()
{
    const uint32_t count = (std::min)(g_ui_rects.count, kUiRectsMax);
    if (count == g_dlaa.mask_count &&
        memcmp(g_dlaa.mask_rects, g_ui_rects.rects, sizeof(g_dlaa.mask_rects[0]) * count) == 0)
        return;
    g_dlaa.mask_count = count;
    memcpy(g_dlaa.mask_rects, g_ui_rects.rects, sizeof(g_dlaa.mask_rects[0]) * count);
    if (count == 0) return;   // no mask is handed to DLSS then
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
    if (n > 0) h.list->ClearUnorderedAccessViewFloat(gpu, cpu, g_dlaa.mask, one, n, rects);
    auto to_srv = Transition(g_dlaa.mask, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                             D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    h.list->ResourceBarrier(1, &to_srv);
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

// Called on every frame, after the capture and the motion upload and ahead of
// everything that reads v.color.tex, in a submission of its own (queue order
// puts it before the neural pass and the present). Nothing waits on it.
//
// color_written: the capture (or the pipe) wrote v.color.tex this frame.
// still: it wrote the same picture as before. False only for a failure the
// frame loop has to end on; a refusal turns DLAA off and the frame goes on.
static bool RunDlaa(VideoState &v, bool color_written, bool still, bool reset)
{
    if (color_written) g_dlaa.color_is_result = false;
    g_dlaa_raw_shown = nullptr;
    if (!DlaaRequested() || !v.color_in_srv || !v.inputs_ready) return true;
    // Nothing new, and the frame on screen is DLAA's already.
    if (!color_written && g_dlaa.color_is_result)
    {
        g_dlaa_raw_shown = g_dlaa.raw;
        return true;
    }
    // The HDR presentation composes the native HDR capture, not v.color: DLAA
    // would change only the SDR proxy under it.
    if (g_hdr_capture)
    {
        DlaaRefuse(v, "the HDR presentation shows the native HDR capture; DLAA works on SDR only");
        return true;
    }
    if (!EnsureDlaa(v)) return !g_submission_failed;
    const bool fresh_history = reset || !g_dlaa.history;
    if (!(color_written && still)) g_dlaa.still_evals = 0;
    ID3D12Resource *result = g_dlaa.boost ? g_dlaa.full : g_dlaa.output;
    if (!BeginCommands()) return false;
    // A picture DLSS has settled on: shown from the last result, no evaluation.
    if (color_written && still && !fresh_history && g_dlaa.still_evals >= kDlaaStillEvaluations)
    {
        DlaaShow(v, result);
        if (EndCommands() == 0) return false;
        g_dlaa.color_is_result = true;
        g_dlaa_raw_shown = g_dlaa.raw;
        return true;
    }
    // The capture as it came, kept for the wipe and for switching off.
    {
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
        auto to_uav = Transition(g_dlaa.small, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                 D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        h.list->ResourceBarrier(1, &to_uav);
        h.list->SetDescriptorHeaps(1, heaps);
        h.list->SetComputeRootSignature(g_scale_rs);
        h.list->SetPipelineState(g_scale4_pso);
        const UINT sizes[4] = { g_dlaa.w, g_dlaa.height, g_dlaa.fw, g_dlaa.fh };
        h.list->SetComputeRoot32BitConstants(0, 4, sizes, 0);
        h.list->SetComputeRootDescriptorTable(1, DlaaGpuSlot(kDlaaScaleSrc));
        h.list->Dispatch((g_dlaa.w + 7) / 8, (g_dlaa.height + 7) / 8, 1);
        auto to_srv = Transition(g_dlaa.small, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                 D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        h.list->ResourceBarrier(1, &to_srv);
        input = g_dlaa.small;
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
    DrawDlaaMask();
    auto p = g_dlaa.params;
    p->Reset();
    p->Set(NVSDK_NGX_Parameter_Color, input);
    p->Set(NVSDK_NGX_Parameter_Output, g_dlaa.output);
    p->Set(NVSDK_NGX_Parameter_Depth, g_dlaa.depth.tex);
    p->Set(NVSDK_NGX_Parameter_MotionVectors, motion);
    if (g_dlaa.mask_count > 0 && g_dlaa.mask_count != UINT32_MAX)
        p->Set(NVSDK_NGX_Parameter_DLSS_Input_Bias_Current_Color_Mask, g_dlaa.mask);
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
    if (g_dlaa.boost)
    {
        // native + (DLAA(small) - small), with NR's residual pass on DLAA's
        // own descriptors.
        auto out_srv = Transition(g_dlaa.output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                  D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        h.list->ResourceBarrier(1, &out_srv);
        h.list->SetDescriptorHeaps(1, heaps);
        h.list->SetComputeRootSignature(g_residual_rs);
        h.list->SetPipelineState(g_residual_pso);
        UINT32 rc[3] = { g_dlaa.fw, g_dlaa.fh, 0 };
        const float strength = 1.0f;
        memcpy(&rc[2], &strength, sizeof(float));
        h.list->SetComputeRoot32BitConstants(0, 3, rc, 0);
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
    // The result replaces the capture: every consumer reads v.color.tex.
    DlaaShow(v, result);
    if (EndCommands() == 0) return false;
    g_dlaa.history = true;
    g_dlaa.color_is_result = true;
    g_dlaa_raw_shown = g_dlaa.raw;
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
