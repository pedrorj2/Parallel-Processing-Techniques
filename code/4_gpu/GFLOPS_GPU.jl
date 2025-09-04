# File: 4_gpu/GFLOPS_GPU.jl

using CUDA
using BenchmarkTools
using Plots, Measures
using LinearAlgebra
include("../1_annexes/system_info.jl")   # For gpu_info(true)

# --------------------------------------------------------
# Configuration
# --------------------------------------------------------

# Set to true to save the plots as files, false to display in Julia
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

println("Choose matrix size range:")
println("1. 256:256:8192")
println("2. 32:32:512 (for low N)")
choice_range = parse(Int, readline())
if choice_range == 1
    Ns = 256:256:8192
    file_suffix = ""
elseif choice_range == 2
    Ns = 32:32:512
    file_suffix = "_lowN"
else
    error("Invalid choice. Please select 1 (256:256:8192) or 2 (32:32:512).")
end

println("Number of N sizes: $(length(Ns))")
println()
println("GFLOPS benchmarks (MxM, MxV) on GPU (CUDA) with $T")
println()

# --------------------------------------------------------
# Peak GFLOPS helpers (theoretical lines)
# --------------------------------------------------------

const GPU_GFLOPS_peak_fp32_raw = gpu_info(true)  # assumes FP32 peak from your helper

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

    # 2) Correction for consumer Ampere/Ada (RTX 30/40): force 64
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
    elseif cc >= v"8.0"      # Ampere/Ada consumer (RTX 30/40)
        return 64
    elseif cc >= v"7.0"      # Volta/Turing typical
        return 32
    else
        return 32
    end
end

@inline function gpu_peak_gflops_for(gflops_fp32::Real, ::Type{Float64})
    ratio = _fp32_to_fp64_ratio()
    return float(gflops_fp32) / ratio
end

const FP32_to_FP64_RATIO = _fp32_to_fp64_ratio()
const MAX_GFLOPS_FP32 = round(gpu_peak_gflops_for(GPU_GFLOPS_peak_fp32_raw, Float32), digits = 2)
const MAX_GFLOPS_FP64 = round(gpu_peak_gflops_for(GPU_GFLOPS_peak_fp32_raw, Float64), digits = 2)

println("Theoretical peak GFLOPS:")
println("  FP32: ", MAX_GFLOPS_FP32, " | FP64: ", MAX_GFLOPS_FP64,
        "  (FP32:FP64 ratio = ", FP32_to_FP64_RATIO, ")")
println("  device = ", try CUDA.name(CUDA.device()) catch; "unknown" end)

# GPU name for filenames
gpu_name = try
    CUDA.name(CUDA.device())
catch
    "GPU"
end

_sanitize_filename(s::AbstractString) = replace(replace(replace(s, " " => "_"), "/" => "-"), "\\" => "-")
gpu_tag = _sanitize_filename(gpu_name)

# --------------------------------------------------------
# Storage
# --------------------------------------------------------

# For chosen T: MxM and MxV
gflops_MxM_T = Float64[]
gflops_MxV_T = Float64[]
gflops_theoretical_T = Float64[]   # constant line for chosen T

# Cross-type MxM comparison (for plot B)
gflops_MxM_FP32 = Float64[]
gflops_MxM_FP64 = Float64[]

# --------------------------------------------------------
# GPU preparation
# --------------------------------------------------------
CUDA.allowscalar(false)   # avoid accidental scalar slow paths
CUDA.synchronize()

# --------------------------------------------------------
# Helpers to benchmark one GEMM/GEMV pass
# --------------------------------------------------------
function bench_gemm_gflops!(dst::Vector{Float64}, N::Int, ::Type{TT}; samples=500) where {TT<:AbstractFloat}
    A = CUDA.rand(TT, N, N)
    B = CUDA.rand(TT, N, N)
    # warm-up
    CUDA.@sync A * B
    b = @benchmark begin
        CUDA.@sync $A * $B
    end samples=samples evals=1
    dt = minimum(b).time * 1e-9
    push!(dst, 2 * N^3 / dt / 1e9)
end

function bench_gemv_gflops!(dst::Vector{Float64}, N::Int, ::Type{TT}; samples=500) where {TT<:AbstractFloat}
    A = CUDA.rand(TT, N, N)
    x = CUDA.rand(TT, N)
    # warm-up
    CUDA.@sync A * x
    b = @benchmark begin
        CUDA.@sync $A * $x
    end samples=samples evals=1
    dt = minimum(b).time * 1e-9
    push!(dst, 2 * N^2 / dt / 1e9)
end

# --------------------------------------------------------
# Main loop for chosen T (MxM & MxV)
# --------------------------------------------------------
Ns_vec = collect(Ns)

println()
println("— Measuring $(T) GEMM/GEMV —")
for N in Ns_vec
    GC.gc()
    try CUDA.reclaim() catch end
    sleep(0.01)

    # chosen T
    bench_gemm_gflops!(gflops_MxM_T, N, T; samples=500)
    bench_gemv_gflops!(gflops_MxV_T, N, T; samples=500)
    # theoretical line for chosen T (constant)
    push!(gflops_theoretical_T, T === Float32 ? MAX_GFLOPS_FP32 : MAX_GFLOPS_FP64)

    println("N = $N | $(string(T)) MxM:$(round(gflops_MxM_T[end],digits=1))  MxV:$(round(gflops_MxV_T[end],digits=1))")
end

# # --------------------------------------------------------
# # Cross-type MxM comparison (Plot B)
# #   Always build FP32 and FP64 GEMM curves for the same Ns
# # --------------------------------------------------------
# println()
# println("— Measuring cross-type GEMM (FP32 vs FP64) for comparison —")
# for N in Ns_vec
#     GC.gc()
#     try CUDA.reclaim() catch end
#     sleep(0.01)

#     bench_gemm_gflops!(gflops_MxM_FP32, N, Float32; samples=300)
#     bench_gemm_gflops!(gflops_MxM_FP64, N, Float64; samples=300)

#     println("N = $N | FP32 MxM:$(round(gflops_MxM_FP32[end],digits=1))  FP64 MxM:$(round(gflops_MxM_FP64[end],digits=1))")
# end

# --------------------------------------------------------
# Plot defaults
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

mkpath("figures/4_gpu/gflops")

# --------------------------------------------------------
# Plot (A) or (C): T MxM vs MxV
# --------------------------------------------------------
yg_max_T = maximum(vcat(gflops_MxM_T, gflops_MxV_T, gflops_theoretical_T)) * 1.02
p_T = plot(
    Ns_vec, gflops_MxM_T, label="MxM ($(string(T)))", lw=1, color=:blue, linestyle=:solid, marker=:circle, markersize=4,
    xlabel="Matrix size N", ylabel="GFLOPS", legend=:topleft,
    ylims=(0, yg_max_T), grid=true, size=(1200, 600)
)
plot!(p_T, Ns_vec, gflops_MxV_T, label="MxV ($(string(T)))", lw=1, color=:green, linestyle=:solid, marker=:cross, markersize=4)
plot!(p_T, Ns_vec, gflops_theoretical_T,
      label="Theoretical peak ($(string(T))): $(round(gflops_theoretical_T[end],digits=1))",
      lw=1, linestyle=:dash, color=:black)

# Filename and title label by T
if T === Float32
    outfile_T = "figures/4_gpu/gflops/FP32_GFLOPS_$(gpu_tag)_MxM_vs_MxV$(file_suffix).png"
else
    outfile_T = "figures/4_gpu/gflops/FP64_GFLOPS_$(gpu_tag)_MxM_vs_MxV$(file_suffix).png"
end

# --------------------------------------------------------
# Plot (B): FP32 MxM vs FP64 MxM
# --------------------------------------------------------
yg_max_B = maximum(vcat(gflops_MxM_FP32, gflops_MxM_FP64, [MAX_GFLOPS_FP32, MAX_GFLOPS_FP64])) * 1.02
p_B = plot(
    Ns_vec, gflops_MxM_FP32, label="MxM (FP32)", lw=1, color=:blue, linestyle=:solid, marker=:circle, markersize=4,
    xlabel="Matrix size N", ylabel="GFLOPS", legend=:topleft,
    ylims=(0, yg_max_B), grid=true, size=(1200, 600)
)
plot!(p_B, Ns_vec, gflops_MxM_FP64, label="MxM (FP64)", lw=1, color=:red, linestyle=:solid, marker=:utriangle, markersize=4)
plot!(p_B, Ns_vec, fill(MAX_GFLOPS_FP32, length(Ns_vec)), label="Theoretical FP32: $(MAX_GFLOPS_FP32)", lw=1, linestyle=:dash, color=:black)
plot!(p_B, Ns_vec, fill(MAX_GFLOPS_FP64, length(Ns_vec)), label="Theoretical FP64: $(MAX_GFLOPS_FP64)", lw=1, linestyle=:dashdot, color=:gray)

outfile_B = "figures/4_gpu/gflops/FP32_vs_FP64_GFLOPS_$(gpu_tag)_MxM$(file_suffix).png"

# --------------------------------------------------------
# Display or save
# --------------------------------------------------------
if save_plot
    savefig(p_T, outfile_T)
    # savefig(p_B, outfile_B)
    println("Saved figures:")
    println("  - ", outfile_T)
    # println("  - ", outfile_B)
else
    display(p_T)
    display(p_B)
    println("Plots displayed (not saved).")
end
