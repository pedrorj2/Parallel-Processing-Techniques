# File: 3_cpu/tcpu_tmemory.jl

using LinearAlgebra, BenchmarkTools, Plots, Measures
using MKL
include("../1_annexes/system_info.jl")

# --------------------------------------------------------
# Configuration
# --------------------------------------------------------

# Set to true to save the plot as a file, false to display it in Julia
save_plot = true

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

println("\nChoose matrix size range:")
println("1. 50:50:2000")
println("2. 5:5:60 (for low N)")
choice_range = parse(Int, readline())
if choice_range == 1
    Ns = 50:50:2000
    file_suffix = ""
elseif choice_range == 2
    Ns = 20:20:300
    file_suffix = "_lowN"
else
    error("Invalid choice. Please select 1 (50:50:2000) or 2 (20:20:300).")
end

N_cores     = num_physical_cores()
N_threads   = 12                            # Adjust to your CPU
RAM_frequency_MHz = 3200                    # Simplified model for theoretical memory time

println("Number of N sizes: $(length(Ns))")
println(" ")
println("Theory vs Real benchmarks for MxM and MxV on CPU with $T (times in ns)")
println(" ")

GFLOPS_peak_raw = cpu_info(true)           # Reported multi-core peak performance
@inline function peak_gflops_for(gflops_fp32::Real, ::Type{Float32})
    return float(gflops_fp32)
end
@inline function peak_gflops_for(gflops_fp32::Real, ::Type{Float64})
    return float(gflops_fp32) / 2
end

const GFLOPS_peak = peak_gflops_for(GFLOPS_peak_raw, T)

# --------------------------------------------------------
# Storage
# --------------------------------------------------------

# Theoretical times (in ns)
t_cpu_theo_mxm = Float64[]
t_cpu_theo_mxv = Float64[]
t_mem_theo_mxm = Float64[]
t_mem_theo_mxv = Float64[]

# Real times (in ns)
t_cpu_measured_mxm = Float64[]
t_cpu_measured_mxv = Float64[]
t_mem_measured_mxm = Float64[]
t_mem_measured_mxv = Float64[]

# --------------------------------------------------------
# Kernels
# --------------------------------------------------------

# Row-wise access (inner loop j) — inefficient (column-major)
function mem_access_rows!(A)
    M = size(A, 1)
    @inbounds for i in 1:M
        for j in 1:M
            A[i, j] = 1f0
        end
    end
    return A
end

# Column-wise access (inner loop i) — efficient (column-major)
function mem_access_columns!(A)
    M = size(A, 1)
    @inbounds for j in 1:M
        for i in 1:M
            A[i, j] = 1f0
        end
    end
    return A
end

function mem_vector!(b)
    M = length(b)
    @inbounds for i in 1:M
        b[i] = 1f0
    end
    return b
end

function mem_allocate_matrix(M)
    A = zeros(T, M, M)
    @inbounds A[M, M] = 1
    return A
end

function mem_allocate_vector(M)
    b = zeros(T, M)
    @inbounds b[M] = 1
    return b
end

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
    # Theoretical times (in ns)
    # ---------------------------

    # Theoretical CPU: based on GFLOPS
    RAM_Hz = RAM_frequency_MHz * 1e6
    t_cpu_theo_mxm_val = (2 * N^3) / (GFLOPS_peak * 1e9) * 1e9  # Convert to ns
    t_cpu_theo_mxv_val = (2 * N^2) / (GFLOPS_peak * 1e9) * 1e9  # Convert to ns
    push!(t_cpu_theo_mxm, t_cpu_theo_mxm_val)
    push!(t_cpu_theo_mxv, t_cpu_theo_mxv_val)

    # Theoretical memory (simplified model; convert MHz to Hz)
    t_mem_theo_mxm_val = (3 * N^2) / RAM_Hz * 1e9  # Convert to ns
    t_mem_theo_mxv_val = (N^2 + 2 * N) / RAM_Hz * 1e9  # Convert to ns
    push!(t_mem_theo_mxm, t_mem_theo_mxm_val)
    push!(t_mem_theo_mxv, t_mem_theo_mxv_val)

    # ---------------------------
    # Real times (in ns)
    # ---------------------------

    # Real CPU (multiplication benchmark)
    BLAS.set_num_threads(N_threads)
    bench_mxm = @benchmark $A * $B_mat samples=300 evals=1
    t_cpu_measured_mxm_val = minimum(bench_mxm).time  # In ns (BenchmarkTools uses ns by default)
    push!(t_cpu_measured_mxm, t_cpu_measured_mxm_val)

    bench_mxv = @benchmark $A * $B_vec samples=300 evals=1
    t_cpu_measured_mxv_val = minimum(bench_mxv).time  # In ns
    push!(t_cpu_measured_mxv, t_cpu_measured_mxv_val)

    # Real memory (access benchmark)
    bench_cols = @benchmark mem_access_columns!(mat) setup=(mat = zeros(T, $N, $N)) samples=300 evals=1
    dt_cols = minimum(bench_cols).time  # In ns
    bench_rows = @benchmark mem_access_rows!(mat) setup=(mat = zeros(T, $N, $N)) samples=300 evals=1
    dt_rows = minimum(bench_rows).time  # In ns
    bench_alloc_mat = @benchmark mem_allocate_matrix($N) samples=300 evals=1
    dt_alloc_mat = minimum(bench_alloc_mat).time  # In ns
    bench_vec = @benchmark mem_vector!(vec) setup=(vec = zeros(T, $N)) samples=300 evals=1
    dt_vec = minimum(bench_vec).time  # In ns
    bench_alloc_vec = @benchmark mem_allocate_vector($N) samples=300 evals=1
    dt_alloc_vec = minimum(bench_alloc_vec).time  # In ns

    t_mem_measured_mxm_val = (dt_rows + dt_cols + dt_alloc_mat)  # In ns
    t_mem_measured_mxv_val = (dt_rows + dt_vec + dt_alloc_vec)   # In ns
    push!(t_mem_measured_mxm, t_mem_measured_mxm_val)
    push!(t_mem_measured_mxv, t_mem_measured_mxv_val)

    # Short console log
    println("N = $N | t_cpu_theo_mxm:$(round(t_cpu_theo_mxm[end],digits=2))  t_mem_theo_mxm:$(round(t_mem_theo_mxm[end],digits=2))  ",
            "t_cpu_measured_mxm:$(round(t_cpu_measured_mxm[end],digits=2))  t_mem_measured_mxm:$(round(t_mem_measured_mxm[end],digits=2))  ",
            "t_cpu_theo_mxv:$(round(t_cpu_theo_mxv[end],digits=2))  t_mem_theo_mxv:$(round(t_mem_theo_mxv[end],digits=2))  ",
            "t_cpu_measured_mxv:$(round(t_cpu_measured_mxv[end],digits=2))  t_mem_measured_mxv:$(round(t_mem_measured_mxv[end],digits=2))")
end

Ns_vec = collect(Ns)

# --------------------------------------------------------
# Plot defaults (force horizontal text)
# --------------------------------------------------------
default(
    markerstrokewidth=0,
    legendfontsize=14,
    margins=10mm,
    xtickfont=font(14, "sans-serif", :black, rotation=0),
    ytickfont=font(14, "sans-serif", :black, rotation=0),
    xguidefont=font(14, "sans-serif", :black, rotation=0),
    yguidefont=font(14, "sans-serif", :black, rotation=0)
)

mkpath("figures/3_cpu/tcpu_tmemory")

# ---------- MxM: Theory vs Real ----------
y_max_mxm = maximum(vcat(t_cpu_theo_mxm, t_mem_theo_mxm, t_cpu_measured_mxm, t_mem_measured_mxm)) * 1.15

p_mxm = plot(
    Ns_vec, t_cpu_theo_mxm, label="t_cpu_theo", lw=2, linestyle=:solid, color=:blue,
    xlabel="Matrix size N", ylabel="Time (ns)", legend=:topleft,
    ylims=(0, y_max_mxm), grid=true, size=(1600, 800),
    xtickfont=font(14, "sans-serif", :black, rotation=0),
    ytickfont=font(14, "sans-serif", :black, rotation=0),
    xguidefont=font(14, "sans-serif", :black, rotation=0),
    yguidefont=font(14, "sans-serif", :black, rotation=0)
)
plot!(p_mxm, Ns_vec, t_mem_theo_mxm, label="t_mem_theo", lw=2, linestyle=:solid, color=:red)
plot!(p_mxm, Ns_vec, t_cpu_measured_mxm, label="t_cpu_measured", lw=2, linestyle=:dash, marker=:circle, color=:blue)
plot!(p_mxm, Ns_vec, t_mem_measured_mxm, label="t_mem_measured", lw=2, linestyle=:dash, marker=:circle, color=:red)

# ---------- MxV: Theory vs Real ----------
y_max_mxv = maximum(vcat(t_cpu_theo_mxv, t_mem_theo_mxv, t_cpu_measured_mxv, t_mem_measured_mxv)) * 1.25

p_mxv = plot(
    Ns_vec, t_cpu_theo_mxv, label="t_cpu_theo", lw=2, linestyle=:solid, color=:green,
    xlabel="Matrix size N", ylabel="Time (ns)", legend=:topleft,
    ylims=(0, y_max_mxv), grid=true, size=(1600, 800),
    xtickfont=font(12, "sans-serif", :black, rotation=0),
    ytickfont=font(12, "sans-serif", :black, rotation=0),
    xguidefont=font(12, "sans-serif", :black, rotation=0),
    yguidefont=font(12, "sans-serif", :black, rotation=0)
)
plot!(p_mxv, Ns_vec, t_mem_theo_mxv, label="t_mem_theo", lw=2, linestyle=:solid, color=:red)
plot!(p_mxv, Ns_vec, t_cpu_measured_mxv, label="t_cpu_measured", lw=2, linestyle=:dash, marker=:circle, color=:green)
plot!(p_mxv, Ns_vec, t_mem_measured_mxv, label="t_mem_measured", lw=2, linestyle=:dash, marker=:circle, color=:red)

# Display or save the plots based on save_plot
if save_plot
    savefig(p_mxm, "figures/3_cpu/tcpu_tmemory/presentacion$(T)_MXM_Ryzen5_5600X$file_suffix.png")
    savefig(p_mxv, "figures/3_cpu/tcpu_tmemory/presentacion$(T)_MXV_Ryzen5_5600X$file_suffix.png")
    println("Saved figures:")
    println("  - figures/3_cpu/tcpu_tmemory/presentacion$(T)_MXM_Ryzen5_5600X$file_suffix.png")
    println("  - figures/3_cpu/tcpu_tmemory/presentacion$(T)_MXV_Ryzen5_5600X$file_suffix.png")
else
    display(p_mxm)
    display(p_mxv)
    println("Plots displayed (not saved).")
end
