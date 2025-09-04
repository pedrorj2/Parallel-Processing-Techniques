# File: 3_cpu/GFLOPS.jl

using LinearAlgebra, BenchmarkTools, Plots, Measures
using MKL
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

# Matrix size ranges
Ns = 50:50:2000  # For plots A, B, C, E
Ns_low = 20:10:200  # For plot D
file_suffix = ""
file_suffix_low = "_lowN"

println("Number of N sizes (main): $(length(Ns))")
println("Number of N sizes (low N): $(length(Ns_low))")
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

# GFLOPS measured (main range)
gflops_MxM_single = Float64[]
gflops_MxM_multi = Float64[]
gflops_MxV_single = Float64[]
gflops_MxV_multi = Float64[]

# GFLOPS measured (low N range for Plot D)
gflops_MxM_single_low = Float64[]
gflops_MxM_multi_low = Float64[]

# GFLOPS theoretical
gflops_theoretical_single = Float64[]
gflops_theoretical_multi = Float64[]
gflops_theoretical_single_low = Float64[]
gflops_theoretical_multi_low = Float64[]

# --------------------------------------------------------
# Main loop for Ns (Plots A, B, C, E)
# --------------------------------------------------------

for N in Ns
    GC.gc()
    sleep(0.01)

    A = rand(T, N, N)
    B_mat = rand(T, N, N)
    B_vec = rand(T, N)  # 1D vector for GEMV

    # ---------------------------
    # GEMM / GEMV benchmarks
    # ---------------------------

    # Single-thread
    BLAS.set_num_threads(1)
    bench_mxm_single = @benchmark $A * $B_mat samples=500 evals=1
    dt_mxm_single = minimum(bench_mxm_single).time * 1e-9
    push!(gflops_MxM_single, 2 * N^3 / dt_mxm_single / 1e9)

    bench_mxv_single = @benchmark $A * $B_vec samples=500 evals=1
    dt_mxv_single = minimum(bench_mxv_single).time * 1e-9
    push!(gflops_MxV_single, 2 * N^2 / dt_mxv_single / 1e9)

    # Multi-thread
    BLAS.set_num_threads(N_threads)
    bench_mxm_multi = @benchmark $A * $B_mat samples=500 evals=1
    dt_mxm_multi = minimum(bench_mxm_multi).time * 1e-9
    push!(gflops_MxM_multi, 2 * N^3 / dt_mxm_multi / 1e9)

    bench_mxv_multi = @benchmark $A * $B_vec samples=500 evals=1
    dt_mxv_multi = minimum(bench_mxv_multi).time * 1e-9
    push!(gflops_MxV_multi, 2 * N^2 / dt_mxv_multi / 1e9)

    # Theoretical (constant per N)
    push!(gflops_theoretical_multi, GFLOPS_peak)
    push!(gflops_theoretical_single, GFLOPS_peak / N_cores)

    # Short console log
    println("N = $N | MxM_s:$(round(gflops_MxM_single[end],digits=1))  MxM_m:$(round(gflops_MxM_multi[end],digits=1))  ",
            "MxV_s:$(round(gflops_MxV_single[end],digits=1))  MxV_m:$(round(gflops_MxV_multi[end],digits=1))")
end

# --------------------------------------------------------
# Main loop for Ns_low (Plot D)
# --------------------------------------------------------

for N in Ns_low
    GC.gc()
    sleep(0.01)

    A = rand(T, N, N)
    B_mat = rand(T, N, N)

    # Single-thread
    BLAS.set_num_threads(1)
    bench_mxm_single = @benchmark $A * $B_mat samples=500 evals=1
    dt_mxm_single = minimum(bench_mxm_single).time * 1e-9
    push!(gflops_MxM_single_low, 2 * N^3 / dt_mxm_single / 1e9)

    # Multi-thread
    BLAS.set_num_threads(N_threads)
    bench_mxm_multi = @benchmark $A * $B_mat samples=500 evals=1
    dt_mxm_multi = minimum(bench_mxm_multi).time * 1e-9
    push!(gflops_MxM_multi_low, 2 * N^3 / dt_mxm_multi / 1e9)

    # Theoretical (constant per N)
    push!(gflops_theoretical_multi_low, GFLOPS_peak)
    push!(gflops_theoretical_single_low, GFLOPS_peak / N_cores)

    # Short console log
    println("Low N = $N | MxM_s:$(round(gflops_MxM_single_low[end],digits=1))  MxM_m:$(round(gflops_MxM_multi_low[end],digits=1))")
end

Ns_vec = collect(Ns)
Ns_vec_low = collect(Ns_low)

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

mkpath("figures/3_cpu/gflops")

yg_max = maximum(vcat(gflops_MxM_single, gflops_MxM_multi, gflops_MxV_single, gflops_MxV_multi,
                      gflops_theoretical_single, gflops_theoretical_multi)) * 1.02
yg_max_low = maximum(vcat(gflops_MxM_single_low, gflops_MxM_multi_low,
                          gflops_theoretical_single_low, gflops_theoretical_multi_low)) * 1.02

# --------------------------------------------------------
# Plot A: FP32 MxM Single vs MxV Single
# --------------------------------------------------------
p_gflops_A = plot(
    Ns_vec, gflops_MxM_single, label="MxM Single", lw=1, color=:blue, linestyle=:solid, marker=:cross, markersize=4,
    xlabel="Matrix size N", ylabel="GFLOPS", legend=:topleft,
    ylims=(0, yg_max), grid=true, size=(1200, 600),
    xtickfont=font(12, "sans-serif", :black, rotation=0),
    ytickfont=font(12, "sans-serif", :black, rotation=0),
    xguidefont=font(12, "sans-serif", :black, rotation=0),
    yguidefont=font(12, "sans-serif", :black, rotation=0)
)
plot!(p_gflops_A, Ns_vec, gflops_MxV_single, label="MxV Single", lw=1, color=:green, linestyle=:solid, marker=:cross, markersize=4)

# --------------------------------------------------------
# Plot B: FP32 MxM Multi vs MxV Multi
# --------------------------------------------------------
p_gflops_B = plot(
    Ns_vec, gflops_MxM_multi, label="MxM Multi", lw=1, color=:blue, linestyle=:solid, marker=:circle, markersize=4,
    xlabel="Matrix size N", ylabel="GFLOPS", legend=:topleft,
    ylims=(0, yg_max), grid=true, size=(1200, 600),
    xtickfont=font(12, "sans-serif", :black, rotation=0),
    ytickfont=font(12, "sans-serif", :black, rotation=0),
    xguidefont=font(12, "sans-serif", :black, rotation=0),
    yguidefont=font(12, "sans-serif", :black, rotation=0)
)
plot!(p_gflops_B, Ns_vec, gflops_MxV_multi, label="MxV Multi", lw=1, color=:green, linestyle=:solid, marker=:circle, markersize=4)

# --------------------------------------------------------
# Plot C: FP32 MxM Single vs MxM Multi vs Theoretical
# --------------------------------------------------------
p_gflops_C = plot(
    Ns_vec, gflops_MxM_single, label="MxM Single", lw=1, color=:blue, linestyle=:solid, marker=:cross, markersize=4,
    xlabel="Matrix size N", ylabel="GFLOPS", legend=:topleft,
    ylims=(0, yg_max), grid=true, size=(1200, 600),
    xtickfont=font(12, "sans-serif", :black, rotation=0),
    ytickfont=font(12, "sans-serif", :black, rotation=0),
    xguidefont=font(12, "sans-serif", :black, rotation=0),
    yguidefont=font(12, "sans-serif", :black, rotation=0)
)
plot!(p_gflops_C, Ns_vec, gflops_MxM_multi, label="MxM Multi", lw=1, color=:blue, linestyle=:solid, marker=:circle, markersize=4)
plot!(p_gflops_C, Ns_vec, gflops_theoretical_single, label="Theoretical Single: $(round(gflops_theoretical_single[end],digits=1)) GFLOPS", lw=1, linestyle=:solid, color=:black)
plot!(p_gflops_C, Ns_vec, gflops_theoretical_multi, label="Theoretical Multi: $(round(gflops_theoretical_multi[end],digits=1)) GFLOPS", lw=1, linestyle=:solid, color=:black)

# --------------------------------------------------------
# Plot D: FP32 Zoom low N MxM Single vs MxM Multi vs Theoretical
# --------------------------------------------------------
p_gflops_D = plot(
    Ns_vec_low, gflops_MxM_single_low, label="MxM Single", lw=1, color=:blue, linestyle=:solid, marker=:cross, markersize=4,
    xlabel="Matrix size N", ylabel="GFLOPS", legend=:topleft,
    ylims=(0, yg_max_low), grid=true, size=(1200, 600),
    xtickfont=font(12, "sans-serif", :black, rotation=0),
    ytickfont=font(12, "sans-serif", :black, rotation=0),
    xguidefont=font(12, "sans-serif", :black, rotation=0),
    yguidefont=font(12, "sans-serif", :black, rotation=0)
)
plot!(p_gflops_D, Ns_vec_low, gflops_MxM_multi_low, label="MxM Multi", lw=1, color=:blue, linestyle=:solid, marker=:circle, markersize=4)
plot!(p_gflops_D, Ns_vec_low, gflops_theoretical_single_low, label="Theoretical Single: $(round(gflops_theoretical_single_low[end],digits=1)) GFLOPS", lw=1, linestyle=:solid, color=:black)
plot!(p_gflops_D, Ns_vec_low, gflops_theoretical_multi_low, label="Theoretical Multi: $(round(gflops_theoretical_multi_low[end],digits=1)) GFLOPS", lw=1, linestyle=:solid, color=:black)

# --------------------------------------------------------
# Display or save the plots based on save_plot
# --------------------------------------------------------
if save_plot
    savefig(p_gflops_A, "figures/3_cpu/gflops/$(T)_GFLOPS_Ryzen5_5600X_MxM_vs_MxV_Single$file_suffix.png")
    savefig(p_gflops_B, "figures/3_cpu/gflops/$(T)_GFLOPS_Ryzen5_5600X_MxM_vs_MxV_Multi$file_suffix.png")
    savefig(p_gflops_C, "figures/3_cpu/gflops/$(T)_GFLOPS_Ryzen5_5600X_MxM_Theo$file_suffix.png")
    savefig(p_gflops_D, "figures/3_cpu/gflops/$(T)_GFLOPS_Ryzen5_5600X_MxM_Theo$file_suffix_low.png")
    println("Saved figures:")
    println("  - figures/3_cpu/gflops/$(T)_GFLOPS_Ryzen5_5600X_MxM_vs_MxV_Single$file_suffix.png")
    println("  - figures/3_cpu/gflops/$(T)_GFLOPS_Ryzen5_5600X_MxM_vs_MxV_Multi$file_suffix.png")
    println("  - figures/3_cpu/gflops/$(T)_GFLOPS_Ryzen5_5600X_MxM_Theo$file_suffix.png")
    println("  - figures/3_cpu/gflops/$(T)_GFLOPS_Ryzen5_5600X_MxM_Theo$file_suffix_low.png")
else
    display(p_gflops_A)
    display(p_gflops_B)
    display(p_gflops_C)
    display(p_gflops_D)
    println("Plots displayed (not saved).")
end