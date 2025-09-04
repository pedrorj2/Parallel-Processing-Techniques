# File: 4_gpu/tgpu_tmemory_GPU.jl

using CUDA
using BenchmarkTools
using Plots, Measures
using LinearAlgebra
include("../1_annexes/system_info.jl")  # For gpu_info(true)

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

println("Choose matrix size range:")
println("1. 256:256:8192")
println("2. 32:16:256 (for low N)")
choice_range = parse(Int, readline())
if choice_range == 1
    Ns = 256:256:8192
    file_suffix = ""
elseif choice_range == 2
    Ns = 32:16:256
    file_suffix = "_lowN"
else
    error("Invalid choice. Please select 1 (256:256:8192) or 2 (32:16:256).")
end

println("Number of N sizes: $(length(Ns))")
println(" ")
println("Theory vs Measured benchmarks for MxM on GPU (times in μs) with $T")
println(" ")

# --------------------------------------------------------
# Device info / theoretical bandwidth and FLOPS
# --------------------------------------------------------

gpu_name = try
    CUDA.name(CUDA.device())
catch
    "GPU"
end

function _sanitize_filename(s::AbstractString)
    replace(replace(replace(s, " " => "_"), "/" => "-"), "\\" => "-")
end
gpu_tag = _sanitize_filename(gpu_name)

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

    # 2) Correction for Ampere/Ada consumer (RTX 30/40): force 64
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
    elseif cc >= v"8.0"      # Ampere/Ada consumer
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

const max_gpu_gflops = round(gpu_peak_gflops_for(GPU_GFLOPS_peak_raw, T), digits = 2)
println("Theoretical max GPU GFLOPS ($(T)): ", max_gpu_gflops, " GFLOPS",
        "  |  FP32:FP64 ratio used = ", _fp32_to_fp64_ratio(),
        "  |  device = ", gpu_name)

function gpu_theoretical_bandwidth_Bps()
    dev = CUDA.device()
    mem_clock_khz = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE) # kHz
    bus_bits      = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH) # bits
    # GDDR (DDR): 2 transfers per cycle -> bytes/s
    return 2.0 * (mem_clock_khz * 1_000.0) * (bus_bits / 8.0)
end

BW_bytes_per_s = gpu_theoretical_bandwidth_Bps()

# --------------------------------------------------------
# Storage (MxM only)
# --------------------------------------------------------

# Theoretical times (in s)
t_gpu_theo_mxm = Float64[]
t_mem_theo_mxm = Float64[]

# Measured times (in s)
t_gpu_measured_mxm = Float64[]
t_mem_measured_mxm = Float64[]

# --------------------------------------------------------
# Kernels for memory benchmarks (MxM touch patterns)
# --------------------------------------------------------

function fill_columns_kernel!(A)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if i <= size(A,1) && j <= size(A,2)
        @inbounds A[i, j] = one(eltype(A))
    end
    return
end

function fill_rows_kernel!(A)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if i <= size(A,1) && j <= size(A,2)
        @inbounds A[i, j] = one(eltype(A))
    end
    return
end

# Wrappers
@inline function launch_fill_columns!(A, threads_mat, blocks_mat)
    @cuda threads=threads_mat blocks=blocks_mat fill_columns_kernel!(A)
    return
end

@inline function launch_fill_rows!(A, threads_mat, blocks_mat)
    @cuda threads=threads_mat blocks=blocks_mat fill_rows_kernel!(A)
    return
end

# Launch config
function launch_config(N; block_size=256)
    threads_mat = (block_size, 1)
    blocks_mat  = (cld(N, block_size), N)
    return threads_mat, blocks_mat
end

# Warmup kernels
CUDA.@sync begin
    A0 = CUDA.zeros(T, 32, 32)
    threads_mat, blocks_mat = launch_config(32)
    @cuda threads=threads_mat blocks=blocks_mat fill_columns_kernel!(A0)
    @cuda threads=threads_mat blocks=blocks_mat fill_rows_kernel!(A0)
end

# Warmup CUBLAS (GEMM)
CUDA.@sync begin
    A_warm = CUDA.rand(T, 32, 32)
    B_warm = CUDA.rand(T, 32, 32)
    C_warm = CUDA.zeros(T, 32, 32)
    CUBLAS.gemm!('N', 'N', T(1), A_warm, B_warm, T(0), C_warm)
end

# --------------------------------------------------------
# Main loop (MxM only)
# --------------------------------------------------------

for N in Ns
    GC.gc()
    CUDA.reclaim()
    sleep(0.01)

    local threads_mat, blocks_mat = launch_config(N)

    # Memory benchmarks (rows + cols + allocation)
    bench_cols = @benchmark begin
        CUDA.@sync launch_fill_columns!(mat, $threads_mat, $blocks_mat)
    end setup=(mat = CUDA.zeros(T, $N, $N)) samples=300 evals=1
    dt_cols = minimum(bench_cols).time * 1e-9

    bench_rows = @benchmark begin
        CUDA.@sync launch_fill_rows!(mat, $threads_mat, $blocks_mat)
    end setup=(mat = CUDA.zeros(T, $N, $N)) samples=300 evals=1
    dt_rows = minimum(bench_rows).time * 1e-9

    bench_alloc_mat = @benchmark CUDA.zeros($T, $N, $N) samples=300 evals=1
    dt_alloc_mat = minimum(bench_alloc_mat).time * 1e-9

    push!(t_mem_measured_mxm, dt_rows + dt_cols + dt_alloc_mat)

    # Theoretical memory (s)
    bytes_mxm = 3.0 * N^2 * sizeof(T)
    push!(t_mem_theo_mxm, bytes_mxm / BW_bytes_per_s)

    # Compute benchmarks (GEMM)
    bench_gpu_mxm = @benchmark begin
        CUDA.@sync CUBLAS.gemm!('N', 'N', $T(1), A, B, $T(0), C)
    end setup=(
        A = CUDA.rand($T, $N, $N);
        B = CUDA.rand($T, $N, $N);
        C = CUDA.zeros($T, $N, $N)
    ) samples=200 evals=1
    dt_gpu_mxm = minimum(bench_gpu_mxm).time * 1e-9
    push!(t_gpu_measured_mxm, dt_gpu_mxm)

    # Theoretical compute (s)
    flops_mxm = 2 * N^3
    push!(t_gpu_theo_mxm, flops_mxm / (max_gpu_gflops * 1e9))

    println("N = $N: t_gpu_theo_mxm:$(round(t_gpu_theo_mxm[end]*1e6, digits=2))  t_mem_theo_mxm:$(round(t_mem_theo_mxm[end]*1e6, digits=2))  ",
            "t_gpu_measured_mxm:$(round(t_gpu_measured_mxm[end]*1e6, digits=2))  t_mem_measured_mxm:$(round(t_mem_measured_mxm[end]*1e6, digits=2))")
end

Ns_vec = collect(Ns)

# --------------------------------------------------------
# Plot defaults
# --------------------------------------------------------

default(
    markerstrokewidth=0,
    legendfontsize=12,
    margins=5mm,
    xtickfont=font(12,"sans-serif",:black,rotation=0),
    ytickfont=font(12,"sans-serif",:black,rotation=0),
    xguidefont=font(12,"sans-serif",:black,rotation=0),
    yguidefont=font(12,"sans-serif",:black,rotation=0)
)

mkpath("figures/4_gpu/tgpu_tmemory")

# ---------- MxM: Theory vs Measured ----------
y_max_mxm = maximum(vcat(t_gpu_theo_mxm, t_mem_theo_mxm, t_gpu_measured_mxm, t_mem_measured_mxm)) * 1.15 * 1e6  # to μs

p_mxm = plot(
    Ns_vec, t_gpu_measured_mxm .* 1e6, label="t_gpu_measured", lw=1, marker=:circle, color=:blue,
    xlabel="Matrix size N", ylabel="Time (μs)", legend=:topleft,
    ylims=(0, y_max_mxm), grid=true, size=(1200, 600)
)
plot!(p_mxm, Ns_vec, t_mem_measured_mxm .* 1e6, label="t_mem_measured", lw=1, marker=:cross, color=:blue)
plot!(p_mxm, Ns_vec, t_gpu_theo_mxm .* 1e6, label="t_gpu_theo", lw=1, linestyle=:dash, color=:black)
plot!(p_mxm, Ns_vec, t_mem_theo_mxm .* 1e6, label="t_mem_theo", lw=1, linestyle=:dash, color=:black)

# Display or save the plots based on save_plot
if save_plot
    savefig(p_mxm, "figures/4_gpu/tgpu_tmemory/$(T)_MXM_$(gpu_tag)$file_suffix.png")
    println("Saved figure:")
    println("  - figures/4_gpu/tgpu_tmemory/$(T)_MXM_$(gpu_tag)$file_suffix.png")
else
    display(p_mxm)
    println("Plot displayed (not saved).")
end
