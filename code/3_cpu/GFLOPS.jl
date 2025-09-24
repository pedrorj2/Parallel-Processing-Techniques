# File: 3_cpu/GFLOPS.jl

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
println("1. 50:50:2000 (normal range)")
println("2. 20:10:200 (low N range)")
choice_range = parse(Int, readline())
if choice_range == 1
    Ns = 50:50:2000
    file_suffix = ""
elseif choice_range == 2
    Ns = 20:10:200
    file_suffix = "_lowN"
else
    error("Invalid choice. Please select 1 (50:50:2000) or 2 (20:10:200).")
end

println("Number of N sizes: $(length(Ns))")
println(" ")
println("GFLOPS benchmarks for a single operation (MxM, MxV) on CPU with $T")
println(" ")

GFLOPS_peak_raw = cpu_info(true)

@inline function peak_gflops_for(gflops_fp32::Real, ::Type{Float32})
    return float(gflops_fp32)
end
@inline function peak_gflops_for(gflops_fp32::Real, ::Type{Float64})
    return float(gflops_fp32) / 2
end

const GFLOPS_peak = peak_gflops_for(GFLOPS_peak_raw, T)

N_cores = num_physical_cores()
N_threads = 12  # Adjust to your CPU

# --------------------------------------------------------
# Storage
# --------------------------------------------------------

# GFLOPS measured
gflops_MxM_single = Float64[]
gflops_MxM_multi = Float64[]
gflops_MxV_single = Float64[]
gflops_MxV_multi = Float64[]

# GFLOPS theoretical
gflops_theoretical_single = Float64[]
gflops_theoretical_multi = Float64[]

# --------------------------------------------------------
# Main loop
# --------------------------------------------------------

for N in Ns
    GC.gc()
    sleep(0.5)

    A = rand(T, N, N)
    B_mat = rand(T, N, N)
    B_vec = rand(T, N)  # 1D vector for GEMV

    # ---------------------------
    # GEMM / GEMV benchmarks
    # ---------------------------

    # Single-thread
    BLAS.set_num_threads(1)
    bench_mxm_single = @benchmark $A * $B_mat samples=400 evals=1
    dt_mxm_single = minimum(bench_mxm_single).time * 1e-9
    push!(gflops_MxM_single, 2 * N^3 / dt_mxm_single / 1e9)

    bench_mxv_single = @benchmark $A * $B_vec samples=400 evals=1
    dt_mxv_single = minimum(bench_mxv_single).time * 1e-9
    push!(gflops_MxV_single, 2 * N^2 / dt_mxv_single / 1e9)

    # Multi-thread
    BLAS.set_num_threads(N_threads)
    bench_mxm_multi = @benchmark $A * $B_mat samples=400 evals=1
    dt_mxm_multi = minimum(bench_mxm_multi).time * 1e-9
    push!(gflops_MxM_multi, 2 * N^3 / dt_mxm_multi / 1e9)

    bench_mxv_multi = @benchmark $A * $B_vec samples=400 evals=1
    dt_mxv_multi = minimum(bench_mxv_multi).time * 1e-9
    push!(gflops_MxV_multi, 2 * N^2 / dt_mxv_multi / 1e9)

    # Theoretical (constant per N)
    push!(gflops_theoretical_multi, GFLOPS_peak)
    push!(gflops_theoretical_single, GFLOPS_peak / N_cores)

    # Short console log
    println("N = $N | MxM_s:$(round(gflops_MxM_single[end],digits=1))  MxM_m:$(round(gflops_MxM_multi[end],digits=1))  ",
            "MxV_s:$(round(gflops_MxV_single[end],digits=1))  MxV_m:$(round(gflops_MxV_multi[end],digits=1))")
end

Ns_vec = collect(Ns)

# --------------------------------------------------------
# Plot defaults
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

mkpath("figures/3_cpu/gflops")

yg_max = maximum(vcat(gflops_MxM_single, gflops_MxM_multi, gflops_MxV_single, gflops_MxV_multi,
                      gflops_theoretical_single, gflops_theoretical_multi)) * 1.02

# --------------------------------------------------------
# Plot 1: Single Core - MxM vs MxV vs Theoretical
# --------------------------------------------------------
p_single = plot(
    Ns_vec, gflops_MxM_single, label="MxM Single", lw=2, color=:blue, linestyle=:dash, marker=:circle, markersize=4,
    xlabel="Matrix size N", ylabel="GFLOPS", legend=:topleft,
    ylims=(0, yg_max), grid=true, size=(1600, 800),
    xtickfont=font(14, "sans-serif", :black, rotation=0),
    ytickfont=font(14, "sans-serif", :black, rotation=0),
    xguidefont=font(14, "sans-serif", :black, rotation=0),
    yguidefont=font(14, "sans-serif", :black, rotation=0)
)
plot!(p_single, Ns_vec, gflops_MxV_single, label="MxV Single", lw=2, color=:green, linestyle=:dash, marker=:circle, markersize=4)
plot!(p_single, Ns_vec, gflops_theoretical_single, label="Theoretical Single: $(round(gflops_theoretical_single[end],digits=1)) GFLOPS", lw=2, linestyle=:solid, color=:black)

# --------------------------------------------------------
# Plot 2: Multi Core - MxM vs MxV vs Theoretical
# --------------------------------------------------------
p_multi = plot(
    Ns_vec, gflops_MxM_multi, label="MxM Multi", lw=2, color=:blue, linestyle=:dash, marker=:circle, markersize=4,
    xlabel="Matrix size N", ylabel="GFLOPS", legend=:topleft,
    ylims=(0, yg_max), grid=true, size=(1600, 800),
    xtickfont=font(14, "sans-serif", :black, rotation=0),
    ytickfont=font(14, "sans-serif", :black, rotation=0),
    xguidefont=font(14, "sans-serif", :black, rotation=0),
    yguidefont=font(14, "sans-serif", :black, rotation=0)
)
plot!(p_multi, Ns_vec, gflops_MxV_multi, label="MxV Multi", lw=2, color=:green, linestyle=:dash, marker=:circle, markersize=4)
plot!(p_multi, Ns_vec, gflops_theoretical_multi, label="Theoretical Multi: $(round(gflops_theoretical_multi[end],digits=1)) GFLOPS", lw=2, linestyle=:solid, color=:black)

# --------------------------------------------------------
# Display or save the plots based on save_plot
# --------------------------------------------------------
if save_plot
    savefig(p_single, "figures/3_cpu/gflops/presentacion$(T)_GFLOPS_Ryzen5_5600X_Single_Core$file_suffix.png")
    savefig(p_multi, "figures/3_cpu/gflops/presentacion$(T)_GFLOPS_Ryzen5_5600X_Multi_Core$file_suffix.png")
    println("Saved figures:")
    println("  - figures/3_cpu/gflops/presentacion$(T)_GFLOPS_Ryzen5_5600X_Single_Core$file_suffix.png")
    println("  - figures/3_cpu/gflops/presentacion$(T)_GFLOPS_Ryzen5_5600X_Multi_Core$file_suffix.png")
else
    display(p_single)
    display(p_multi)
    println("Plots displayed (not saved).")
end
