# File: 5_cpu_vs_gpu/cpu_vs_gpu.jl

using LinearAlgebra, MKL, BenchmarkTools, Plots, Measures
using CUDA
include("../1_annexes/system_info.jl")

# --------------------------------------------------------
# Configuration
# --------------------------------------------------------

# Set to true to save the plot as a file, false to display it in Julia
save_plot = false

# --------------------------------------------------------
# Interactive prompts for user input
# --------------------------------------------------------

println("Choose floating-point type:")
println("1. Float32")
println("2. Float64")
choice = parse(Int, readline())
if choice == 1
    T = Float32
elseif choice == 2
    T = Float64
else
    error("Invalid choice. Please select 1 (Float32) or 2 (Float64).")
end

Ns = 256:256:8192

println("Number of N sizes: $(length(Ns))")
println(" ")
println("Comparative GFLOPS benchmarks (CPU single/multi vs GPU) for single operation (MxM, MxV)")
println("Data type: $(T)")
println(" ")

# CPU part
GFLOPS_peak_raw = cpu_info(true)

@inline function peak_gflops_for(gflops_fp32::Real, ::Type{Float32})
    return float(gflops_fp32)
end
@inline function peak_gflops_for(gflops_fp32::Real, ::Type{Float64})
    return float(gflops_fp32) / 2
end

const GFLOPS_peak = peak_gflops_for(GFLOPS_peak_raw, T)

N_cores     = num_physical_cores()
N_threads   = 12                            # Adjust to your CPU

# GPU part
const GPU_GFLOPS_peak_raw = gpu_info(true)

@inline function gpu_peak_gflops_for(gflops_fp32::Real, ::Type{Float32})
    return float(gflops_fp32)
end

# Attempts to read the official attribute; if it fails or seems inconsistent, uses heuristic.
function _fp32_to_fp64_ratio()
    dev = CUDA.device()
    # 1) Direct attempt via attribute
    r = try
        Int(CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO))
    catch
        0
    end

    name = try CUDA.name(dev) catch; "" end
    cc   = try CUDA.capability(dev) catch; v"0.0" end

    # 2) Correction for Ampere/Ada consumption (RTX 30/40): force 64
    #    unless it’s a known HPC GPU (A100/A800/H100/H800) where FP64 ≈ FP32/2.
    if r > 0
        if cc >= v"8.0" && occursin(r"(?i)\b(geforce|rtx)\b", name) &&
           !occursin(r"(?i)\b(A100|A800|H100|H800|L40|L40S)\b", name)
            return 64
        else
            return r
        end
    end

    # 3) Fallback by architecture/name
    if occursin(r"(?i)\b(A100|A800|H100|H800)\b", name)
        return 2             # FP64 ≈ FP32/2 in HPC GPUs
    elseif cc >= v"8.0"      # Ampere/Ada consumption (RTX 30/40)
        return 64
    elseif cc >= v"7.0"      # Volta/Turing
        return 32
    else
        return 32
    end
end

@inline function gpu_peak_gflops_for(gflops_fp32::Real, ::Type{Float64})
    ratio = _fp32_to_fp64_ratio()
    return float(gflops_fp32) / ratio
end

# --- Calculation and log ---------------------------------------------------------

const max_gpu_gflops = round(gpu_peak_gflops_for(GPU_GFLOPS_peak_raw, T), digits = 2)
println("Theoretical max GPU GFLOPS ($(T)): ", max_gpu_gflops, " GFLOPS",
        "  |  FP32:FP64 ratio used = ", _fp32_to_fp64_ratio(),
        "  |  device = ", try CUDA.name(CUDA.device()) catch; "unknown" end)

# --------------------------------------------------------
# Storage
# --------------------------------------------------------

# Measured GFLOPS CPU
gflops_MxM_single = Float64[]
gflops_MxM_multi  = Float64[]
gflops_MxV_single = Float64[]
gflops_MxV_multi  = Float64[]

# Theoretical GFLOPS CPU
gflops_theoretical_single = Float64[]
gflops_theoretical_multi  = Float64[]

# Measured GFLOPS GPU
gflops_MxM_gpu = Float64[]
gflops_MxV_gpu = Float64[]

# Theoretical GFLOPS GPU
gflops_theoretical_gpu = Float64[]

# Speedups
speedup_MxM = Float64[]
speedup_MxV = Float64[]

# --------------------------------------------------------
# GPU preparation
# --------------------------------------------------------
CUDA.allowscalar(false)  # Avoid accidental scalar access
CUDA.synchronize()

# --------------------------------------------------------
# Main loop
# --------------------------------------------------------

for N in Ns
    GC.gc()
    sleep(0.01)

    A     = rand(T, N, N)
    B_mat = rand(T, N, N)
    B_vec = rand(T, N)      # 1D vector for GEMV

    # ---------------------------
    # GEMM / GEMV benchmarks CPU
    # ---------------------------

    # Single-thread
    BLAS.set_num_threads(1)
    bench_mxm_single = @benchmark $A * $B_mat samples=500 evals=1
    dt_mxm_single = minimum(bench_mxm_single).time * 1e-9
    push!(gflops_MxM_single, 2 * N^3 / dt_mxm_single / 1e9)

    bench_mxv_single = @benchmark $A * $B_vec samples=500 evals=1
    dt_mxv_single = minimum(bench_mxv_single).time * 1e-9
    push!(gflops_MxV_single, 2 * N^2 / dt_mxv_single / 1e9)

    # Multi-thread (single operation parallelized by BLAS)
    BLAS.set_num_threads(N_threads)
    bench_mxm_multi = @benchmark $A * $B_mat samples=500 evals=1
    dt_mxm_multi = minimum(bench_mxm_multi).time * 1e-9
    push!(gflops_MxM_multi, 2 * N^3 / dt_mxm_multi / 1e9)

    bench_mxv_multi = @benchmark $A * $B_vec samples=500 evals=1
    dt_mxv_multi = minimum(bench_mxv_multi).time * 1e-9
    push!(gflops_MxV_multi, 2 * N^2 / dt_mxv_multi / 1e9)

    # Theoretical CPU (constant per N)
    push!(gflops_theoretical_multi,  GFLOPS_peak)
    push!(gflops_theoretical_single, GFLOPS_peak / N_cores)

    # GPU part
    try
        CUDA.reclaim()   # Free memory at pool level if available
    catch
        # Older versions of CUDA.jl may not have reclaim()
    end
    sleep(0.01)

    # Data on GPU (without transferring in each benchmark)
    A_gpu     = CUDA.rand(T, N, N)
    B_mat_gpu = CUDA.rand(T, N, N)
    B_vec_gpu = CUDA.rand(T, N)

    # Warm-up to compile and preheat cuBLAS/kernels
    CUDA.@sync A_gpu * B_mat_gpu
    CUDA.@sync A_gpu * B_vec_gpu

    # ---------------------------
    # GEMM / GEMV benchmarks (single operation; evals=1)
    # ---------------------------

    bench_mxm = @benchmark begin
        CUDA.@sync $A_gpu * $B_mat_gpu
    end samples=500 evals=1
    dt_mxm = minimum(bench_mxm).time * 1e-9
    push!(gflops_MxM_gpu, 2 * N^3 / dt_mxm / 1e9)

    bench_mxv = @benchmark begin
        CUDA.@sync $A_gpu * $B_vec_gpu
    end samples=500 evals=1
    dt_mxv = minimum(bench_mxv).time * 1e-9
    push!(gflops_MxV_gpu, 2 * N^2 / dt_mxv / 1e9)

    # Theoretical GPU
    push!(gflops_theoretical_gpu, max_gpu_gflops)

    # Speedup calculation
    push!(speedup_MxM, gflops_MxM_gpu[end] / gflops_MxM_multi[end])
    push!(speedup_MxV, gflops_MxV_gpu[end] / gflops_MxV_multi[end])

    # Short console log
    println("N = $N | MxM_s:$(round(gflops_MxM_single[end],digits=1))  MxM_m:$(round(gflops_MxM_multi[end],digits=1))  ",
            "MxV_s:$(round(gflops_MxV_single[end],digits=1))  MxV_m:$(round(gflops_MxV_multi[end],digits=1)) | ",
            "MxM_gpu:$(round(gflops_MxM_gpu[end],digits=1))  MxV_gpu:$(round(gflops_MxV_gpu[end],digits=1)) | ",
            "Speedup_MxM:$(round(speedup_MxM[end],digits=1))  Speedup_MxV:$(round(speedup_MxV[end],digits=1))")
end

Ns_vec = collect(Ns)

# --------------------------------------------------------
# Plot defaults (force horizontal text)
# --------------------------------------------------------
default(
    markerstrokewidth=0,
    legendfontsize=12,
    margins=5mm,
    xtickfont=font(12, "sans-serif", :black, rotation=0),
    ytickfont=font(12, "sans-serif", :black, rotation=0),
    xguidefont=font(12, "sans-serif", :black, rotation=0),
    yguidefont=font(12, "sans-serif", :black, rotation=0)
)

mkpath("figures/5_cpu_vs_gpu")

# Speedup plot
yg_max_speedup = maximum(vcat(speedup_MxM, speedup_MxV)) * 1.02

p_speedup = plot(
    Ns_vec, speedup_MxM, label="MxM Speedup (GPU / CPU Multi)", lw=1, color=:blue, marker=:circle, markersize=4,
    xlabel="Matrix size N", ylabel="Speedup", legend=:topleft,
    ylims=(0, yg_max_speedup), grid=true, size=(1200, 600),
    legendfontsize=12,
    xtickfont=font(12, "sans-serif", :black, rotation=0),
    ytickfont=font(12, "sans-serif", :black, rotation=0),
    xguidefont=font(12, "sans-serif", :black, rotation=0),
    yguidefont=font(12, "sans-serif", :black, rotation=0)
)
plot!(p_speedup, Ns_vec, speedup_MxV, label="MxV Speedup (GPU / CPU Multi)", lw=1, marker=:circle, markersize=4, color=:green)

plot!(p_speedup, Ns_vec, fill(1, length(Ns_vec)), lw=2, ls=:dash, color=:black, label="1x Baseline")

# Display or save the plot based on save_plot
if save_plot
    savefig(p_speedup, "figures/5_cpu_vs_gpu/$(T)_GFLOPS_speedup.png")
    println("Saved figure:")
    println("  - figures/5_cpu_vs_gpu/$(T)_GFLOPS_speedup.png")
else
    display(p_speedup)
    println("Speedup plot displayed (not saved).")
end