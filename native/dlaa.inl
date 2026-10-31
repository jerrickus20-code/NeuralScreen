// DLAA: DLSS super resolution at 1:1 (NVSDK_NGX_Feature_SuperSampling,
// PerfQuality_Value_DLAA) over the captured frame, before the neural pass.
//
// The result is copied back into v.color.tex, so everything downstream of the
// capture - NR, Boost's residual anchor, the bypass present, FG, the wipe, the
// recording - sees the anti-aliased frame and needs no change of its own.
//
// What a capture can and cannot give DLSS. Motion: the same current ->
// previous field NR is handed (in work-size pixels; stretched to the colour's
// size when the work size is smaller). Depth: none - a flat plane, as FG's.
// Jitter: none - the game already resolved its frame, so DLAA here is temporal
// anti-aliasing and stabilisation of what the screen shows, not the
// sub-pixel reconstruction it does inside an engine. Opt-in, like FG.
//
// The runtime is NVIDIA's public nvngx_dlss.dll: native\libraries\ first
// (verified like every BYO file), then native\. It is not bundled.
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

static struct DlaaState {
    HMODULE module = nullptr;
    PFN_NR_Create create = nullptr;
    PFN_NR_Release release = nullptr;
    NVSDK_NGX_Parameter *params = nullptr;
    NVSDK_NGX_Handle *feature = nullptr;
    VideoTex depth;                     // flat; NON_PIXEL_SHADER_RESOURCE at rest
    ID3D12Resource *output = nullptr;   // UNORDERED_ACCESS at rest
    // The motion at the colour's size, when the work size is smaller (the
    // processing resolution below the screen's). UNORDERED_ACCESS at rest.
    ID3D12Resource *motion = nullptr;
    // Its own descriptor pair for the stretch: the shared motion pair is
    // rebound by MOTS, and a shader-visible heap is read when the list runs,
    // not when it is recorded (see BindScale4Descriptors).
    ID3D12DescriptorHeap *heap = nullptr;
    ID3D12Resource *heap_src = nullptr, *heap_dst = nullptr;
    UINT w = 0, height = 0, mw = 0, mh = 0;
    unsigned preset = 0;
    // The feature has accumulated a frame: false after every (re)create, so
    // the first evaluation of a new feature starts its history.
    bool history = false;
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
// choice (K, the transformer model, on the 310 runtimes).
static unsigned DlaaPreset()
{
    char value[8] = {};
    if (GetEnvironmentVariableA("NS_DLAA_PRESET", value, sizeof(value)) != 1) return 0;
    const char letter = static_cast<char>(value[0] & ~0x20);
    if (letter < 'J' || letter > 'M') return 0;
    return static_cast<unsigned>(NVSDK_NGX_DLSS_Hint_Render_Preset_J) + (letter - 'J');
}

// A refusal: logged with the prefix the client reads ("[dlaa] off:") and the
// switch latched off until the user turns it on again.
static void DlaaRefuse(const char *fmt, ...)
{
    char text[256];
    va_list args;
    va_start(args, fmt);
    vsnprintf(text, sizeof(text), fmt, args);
    va_end(args);
    Log("[dlaa] off: %s", text);
    g_dlaa.failed = true;
}

static void CloseDlaaResources()
{
    if (g_submission_failed) return;
    if (g_dlaa.feature == nullptr && g_dlaa.output == nullptr && g_dlaa.depth.tex == nullptr &&
        g_dlaa.motion == nullptr)
        return;
    // A frame recorded with these may still be on the queue: a toggle or a
    // resize is rare, a wait here is not on the frame path.
    if (h.fence != nullptr && h.fence_value != 0)
        (void)WaitFenceValue(h.fence, h.fence_value, 2000, "dlaa-close");
    if (g_dlaa.feature) { g_dlaa.release(g_dlaa.feature); g_dlaa.feature = nullptr; }
    if (g_dlaa.depth.tex) { g_dlaa.depth.tex->Release(); g_dlaa.depth.tex = nullptr; }
    if (g_dlaa.depth.upload) { g_dlaa.depth.upload->Release(); g_dlaa.depth.upload = nullptr; }
    if (g_dlaa.output) { g_dlaa.output->Release(); g_dlaa.output = nullptr; }
    if (g_dlaa.motion) { g_dlaa.motion->Release(); g_dlaa.motion = nullptr; }
    g_dlaa.heap_src = g_dlaa.heap_dst = nullptr;
    g_dlaa.w = g_dlaa.height = g_dlaa.mw = g_dlaa.mh = 0;
    g_dlaa.history = false;
}

// On shutdown: the runtime's parameter block and the module too.
static void UnloadDlaa()
{
    CloseDlaaResources();
    if (g_submission_failed) return;
    if (g_dlaa.heap) { g_dlaa.heap->Release(); g_dlaa.heap = nullptr; }
    if (g_dlaa.params) { NVSDK_NGX_D3D12_DestroyParameters(g_dlaa.params); g_dlaa.params = nullptr; }
    g_dlaa.create = nullptr; g_dlaa.release = nullptr; g_dlaa_evaluate = nullptr;
    if (g_dlaa.module) { FreeLibrary(g_dlaa.module); g_dlaa.module = nullptr; }
}

static bool LoadDlaaRuntime()
{
    if (g_dlaa.module) return true;
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
    if (!g_dlaa.module)
    {
        DlaaRefuse("nvngx_dlss.dll not found - put NVIDIA's DLSS runtime into native\\libraries\\");
        return false;
    }
    // Any failure from here on unloads the runtime again, so the next switch-on
    // starts from nothing instead of creating on a runtime never initialised.
    auto unload = [] {
        if (g_dlaa.params) { NVSDK_NGX_D3D12_DestroyParameters(g_dlaa.params); g_dlaa.params = nullptr; }
        g_dlaa.create = nullptr; g_dlaa.release = nullptr; g_dlaa_evaluate = nullptr;
        FreeLibrary(g_dlaa.module); g_dlaa.module = nullptr;
        return false;
    };
    auto init = reinterpret_cast<PFN_NR_InitExt>(GetProcAddress(g_dlaa.module, "NVSDK_NGX_D3D12_Init_Ext"));
    g_dlaa.create = reinterpret_cast<PFN_NR_Create>(GetProcAddress(g_dlaa.module, "NVSDK_NGX_D3D12_CreateFeature"));
    g_dlaa.release = reinterpret_cast<PFN_NR_Release>(GetProcAddress(g_dlaa.module, "NVSDK_NGX_D3D12_ReleaseFeature"));
    g_dlaa_evaluate = reinterpret_cast<PFN_NR_Evaluate>(GetProcAddress(g_dlaa.module, "NVSDK_NGX_D3D12_EvaluateFeature"));
    if (!init || !g_dlaa.create || !g_dlaa.release || !g_dlaa_evaluate ||
        NVSDK_NGX_FAILED(NVSDK_NGX_D3D12_AllocateParameters(&g_dlaa.params)))
    {
        DlaaRefuse("the DLSS runtime is missing an entry point or parameters");
        return unload();
    }
    auto result = init(0x1000000ULL, directory, h.dev, NVSDK_NGX_Version_API, g_dlaa.params);
    if (TestFailureOnce("dlaa-init")) result = NVSDK_NGX_Result_Fail;
    Log("[dlaa] Init_Ext -> 0x%08X", result);
    if (NVSDK_NGX_FAILED(result))
    {
        DlaaRefuse("the DLSS runtime refused to initialise (0x%08X %s)", result, NgxResultName(result));
        return unload();
    }
    return true;
}

// The motion stretch's descriptor pair, reissued only when a texture changes.
static bool BindDlaaMotion(ID3D12Resource *src)
{
    if (g_dlaa.heap == nullptr)
    {
        D3D12_DESCRIPTOR_HEAP_DESC hd = {};
        hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        hd.NumDescriptors = 2;
        hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        if (FAILED(h.dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&g_dlaa.heap)))) return false;
    }
    if (src == g_dlaa.heap_src && g_dlaa.motion == g_dlaa.heap_dst) return true;
    const UINT stride = h.dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_CPU_DESCRIPTOR_HANDLE cpu = g_dlaa.heap->GetCPUDescriptorHandleForHeapStart();
    D3D12_SHADER_RESOURCE_VIEW_DESC sd = {};
    sd.Format = DXGI_FORMAT_R16G16_FLOAT;
    sd.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    sd.Texture2D.MipLevels = 1;
    h.dev->CreateShaderResourceView(src, &sd, cpu);
    D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
    ud.Format = DXGI_FORMAT_R16G16_FLOAT;
    ud.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    cpu.ptr += stride;
    h.dev->CreateUnorderedAccessView(g_dlaa.motion, nullptr, &ud, cpu);
    g_dlaa.heap_src = src;
    g_dlaa.heap_dst = g_dlaa.motion;
    return true;
}

static bool EnsureDlaa(VideoState &v)
{
    const UINT w = v.upscale ? v.full_w : v.w, height = v.upscale ? v.full_h : v.hgt;
    const unsigned preset = DlaaPreset();
    if (g_dlaa.feature && g_dlaa.w == w && g_dlaa.height == height && g_dlaa.mw == v.w &&
        g_dlaa.mh == v.hgt && g_dlaa.preset == preset) return true;
    CloseDlaaResources();
    if (!LoadDlaaRuntime()) return false;
    const bool stretch = v.w != w || v.hgt != height;
    if (stretch && !EnsureScalePipeline())
    {
        DlaaRefuse("the motion stretch pipeline is unavailable");
        return false;
    }
    auto p = g_dlaa.params;
    p->Reset();
    p->Set(NVSDK_NGX_Parameter_CreationNodeMask, 1u);
    p->Set(NVSDK_NGX_Parameter_VisibilityNodeMask, 1u);
    p->Set(NVSDK_NGX_Parameter_Width, w); p->Set(NVSDK_NGX_Parameter_Height, height);
    p->Set(NVSDK_NGX_Parameter_OutWidth, w); p->Set(NVSDK_NGX_Parameter_OutHeight, height);
    p->Set(NVSDK_NGX_Parameter_PerfQualityValue, static_cast<int>(NVSDK_NGX_PerfQuality_Value_DLAA));
    // Motion at the render size, which for DLAA is the output size. The frame
    // is the 8-bit SDR desktop: no HDR flag, no exposure.
    p->Set(NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags,
           static_cast<int>(NVSDK_NGX_DLSS_Feature_Flags_MVLowRes));
    p->Set(NVSDK_NGX_Parameter_DLSS_Enable_Output_Subrects, 0);
    if (preset != 0) p->Set(NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_DLAA, preset);
    if (!BeginCommands()) return false;
    const auto result = DlaaCreateGuarded(g_dlaa.create, h.list, p, &g_dlaa.feature);
    if (g_dlaa_fault != 0)
    {
        // Never execute a list NVIDIA's runtime faulted in, never trust a
        // handle it may have half-written.
        AbortCommands();
        g_dlaa.feature = nullptr;
        ReportFailure("dlaa-create", "seh", static_cast<HRESULT>(g_dlaa_fault));
        DlaaRefuse("CreateFeature raised 0x%08X (caught)", g_dlaa_fault);
        return false;
    }
    if (!WaitFenceValue(h.fence, EndCommands(), 30000, "dlaa-create") || NVSDK_NGX_FAILED(result) ||
        !g_dlaa.feature)
    {
        g_dlaa.feature = nullptr;
        DlaaRefuse("CreateFeature failed 0x%08X (%s)", result, NgxResultName(result));
        return false;
    }
    if (!CreateVideoTex(g_dlaa.depth, w, height, DXGI_FORMAT_R32_FLOAT, w * 4))
    { DlaaRefuse("the depth texture could not be created"); return false; }
    std::vector<float> depth(static_cast<size_t>(w) * height, 0.5f);
    if (!FillUpload(g_dlaa.depth, reinterpret_cast<BYTE *>(depth.data()), w * 4, height))
    { DlaaRefuse("the depth texture could not be filled"); return false; }
    if (!BeginCommands()) return false;
    CopyUpload(h.list, g_dlaa.depth);
    D3D12_RESOURCE_BARRIER b[2] = {
        Transition(g_dlaa.depth.tex, D3D12_RESOURCE_STATE_COPY_DEST,
                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
    };
    UINT barriers = 1;
    g_dlaa.output = MakeTex(w, height, DXGI_FORMAT_R8G8B8A8_UNORM, true);
    if (g_dlaa.output == nullptr) { AbortCommands(); DlaaRefuse("the output texture could not be created"); return false; }
    b[barriers++] = Transition(g_dlaa.output, D3D12_RESOURCE_STATE_COMMON,
                               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    h.list->ResourceBarrier(barriers, b);
    if (stretch)
    {
        g_dlaa.motion = MakeTex(w, height, DXGI_FORMAT_R16G16_FLOAT, true);
        if (g_dlaa.motion == nullptr) { AbortCommands(); DlaaRefuse("the motion texture could not be created"); return false; }
        auto to_uav = Transition(g_dlaa.motion, D3D12_RESOURCE_STATE_COMMON,
                                 D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        h.list->ResourceBarrier(1, &to_uav);
    }
    if (!WaitFenceValue(h.fence, EndCommands(), 30000, "dlaa-create"))
    { DlaaRefuse("the GPU did not finish setting DLAA up"); return false; }
    g_dlaa.w = w; g_dlaa.height = height; g_dlaa.mw = v.w; g_dlaa.mh = v.hgt;
    g_dlaa.preset = preset;
    Log("[dlaa] enabled at %ux%u, motion %ux%u, preset %c; flat depth and estimated motion",
        w, height, v.w, v.hgt,
        preset ? static_cast<char>('A' + preset - 1) : '-');
    return true;
}

// One DLAA pass over v.color.tex, in its own submission ahead of the neural
// pass: queue order makes the network read the anti-aliased frame. Nothing
// waits on it. False only for a failure the frame loop has to end on; a
// refusal turns DLAA off and lets the frame go on without it.
static bool RunDlaa(VideoState &v, bool reset)
{
    if (!DlaaRequested() || !v.color_in_srv || !v.inputs_ready) return true;
    // The HDR presentation composes the native HDR capture, not v.color: DLAA
    // would change only the SDR proxy under it. Said once; the switch goes off.
    if (g_hdr_capture)
    {
        DlaaRefuse("the HDR presentation shows the native HDR capture; DLAA works on SDR only");
        return true;
    }
    if (!EnsureDlaa(v))
    {
        // Whatever was built before the refusal goes with it.
        CloseDlaaResources();
        return !g_submission_failed;
    }
    const UINT w = g_dlaa.w, height = g_dlaa.height;
    if (!BeginCommands()) return false;
    ID3D12Resource *motion = v.mv.tex;
    float scale_x = 1.0f, scale_y = 1.0f;
    if (g_dlaa.motion != nullptr)
    {
        // The field is in work-size pixels; stretched to the colour's size, its
        // values are scaled by DLSS's own MV scale rather than by the shader.
        if (!BindDlaaMotion(v.mv.tex)) { AbortCommands(); DlaaRefuse("descriptor heap"); return true; }
        ID3D12DescriptorHeap *heaps[] = { g_dlaa.heap };
        h.list->SetDescriptorHeaps(1, heaps);
        h.list->SetComputeRootSignature(g_scale_rs);
        h.list->SetPipelineState(g_scale_pso);
        const UINT sizes[4] = { w, height, v.w, v.hgt };
        h.list->SetComputeRoot32BitConstants(0, 4, sizes, 0);
        h.list->SetComputeRootDescriptorTable(1, g_dlaa.heap->GetGPUDescriptorHandleForHeapStart());
        h.list->Dispatch((w + 7) / 8, (height + 7) / 8, 1);
        auto to_srv = Transition(g_dlaa.motion, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                 D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        h.list->ResourceBarrier(1, &to_srv);
        motion = g_dlaa.motion;
        scale_x = static_cast<float>(w) / static_cast<float>(v.w);
        scale_y = static_cast<float>(height) / static_cast<float>(v.hgt);
    }
    auto p = g_dlaa.params;
    p->Reset();
    p->Set(NVSDK_NGX_Parameter_Color, v.color.tex);
    p->Set(NVSDK_NGX_Parameter_Output, g_dlaa.output);
    p->Set(NVSDK_NGX_Parameter_Depth, g_dlaa.depth.tex);
    p->Set(NVSDK_NGX_Parameter_MotionVectors, motion);
    p->Set(NVSDK_NGX_Parameter_Jitter_Offset_X, 0.0f);
    p->Set(NVSDK_NGX_Parameter_Jitter_Offset_Y, 0.0f);
    p->Set(NVSDK_NGX_Parameter_Reset, (reset || !g_dlaa.history) ? 1 : 0);
    p->Set(NVSDK_NGX_Parameter_MV_Scale_X, scale_x);
    p->Set(NVSDK_NGX_Parameter_MV_Scale_Y, scale_y);
    p->Set(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width, w);
    p->Set(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height, height);
    p->Set(NVSDK_NGX_Parameter_DLSS_Pre_Exposure, 1.0f);
    p->Set(NVSDK_NGX_Parameter_DLSS_Exposure_Scale, 1.0f);
    const NVSDK_NGX_Result result = DlaaEvaluateGuarded(h.list, g_dlaa.feature, p);
    if (g_dlaa_fault != 0)
    {
        AbortCommands();
        ReportFailure("dlaa-evaluate", "seh", static_cast<HRESULT>(g_dlaa_fault));
        DlaaRefuse("EvaluateFeature raised 0x%08X (caught)", g_dlaa_fault);
        CloseDlaaResources();
        return true;
    }
    if (NVSDK_NGX_FAILED(result))
    {
        // The list holds nothing that ran yet; dropping it leaves the frame
        // as captured.
        AbortCommands();
        DlaaRefuse("EvaluateFeature failed 0x%08X (%s)", result, NgxResultName(result));
        CloseDlaaResources();
        return true;
    }
    // The result replaces the capture: every consumer reads v.color.tex.
    D3D12_RESOURCE_BARRIER pre[] = {
        Transition(v.color.tex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                   D3D12_RESOURCE_STATE_COPY_DEST),
        Transition(g_dlaa.output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                   D3D12_RESOURCE_STATE_COPY_SOURCE),
    };
    h.list->ResourceBarrier(_countof(pre), pre);
    h.list->CopyResource(v.color.tex, g_dlaa.output);
    D3D12_RESOURCE_BARRIER post[3] = {
        Transition(v.color.tex, D3D12_RESOURCE_STATE_COPY_DEST,
                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
        Transition(g_dlaa.output, D3D12_RESOURCE_STATE_COPY_SOURCE,
                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
    };
    UINT count = 2;
    if (g_dlaa.motion != nullptr)
        post[count++] = Transition(g_dlaa.motion, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    h.list->ResourceBarrier(count, post);
    if (EndCommands() == 0) return false;
    g_dlaa.history = true;
    return true;
}

// The switch rides on every frame, like FG's (FRAME_FLAG_DLAA_SET says the
// client sends it, FRAME_FLAG_DLAA is its state).
static void ConfigureDlaaFrame(uint32_t flags)
{
    if (!(flags & FRAME_FLAG_DLAA_SET)) return;
    const int enabled = (flags & FRAME_FLAG_DLAA) != 0;
    if (g_dlaa_ui_enabled == enabled) return;
    CloseDlaaResources();
    g_dlaa_ui_enabled = enabled;
    // A fresh switch-on is a fresh attempt: a runtime dropped into
    // native\libraries\ meanwhile gets its chance without a restart.
    if (enabled) g_dlaa.failed = false;
    Log("[dlaa] UI: %s", enabled ? "on" : "off");
}
