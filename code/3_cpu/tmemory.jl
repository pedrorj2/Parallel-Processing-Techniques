# File: code/3_cpu/tmemory.jl

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

println("\nChoose matrix size range:")
println("1. 32:8:1000")
println("2. 8:2:200 (for low N)")
choice_range = parse(Int, readline())
if choice_range == 1
    Ns = 32:8:1000
    file_suffix = ""
elseif choice_range == 2
    Ns = 8:2:200
    file_suffix = "_lowN"
else
    error("Invalid choice. Please select 1 (32:8:1000) or 2 (8:2:200).")
end

RAM_frequency_MHz = 3200  # Simplified model for theoretical memory time

println("Number of N sizes: $(length(Ns))")
println(" ")
println("MEMORY benchmarks for a single operation (MxM, MxV) on CPU with $T")
println(" ")

# --------------------------------------------------------
# Storage
# --------------------------------------------------------

# Measured memory times (s)
dt_rows_arr      = Float64[]  # Write entire matrix row-wise
dt_cols_arr      = Float64[]  # Write entire matrix column-wise
dt_alloc_mat_arr = Float64[]  # Matrix allocation
dt_vec_arr       = Float64[]  # Write entire vector
dt_alloc_vec_arr = Float64[]  # Vector allocation

# "Real total" times (simplified model, s)
dt_mem_measured_total_mxm = Float64[]  # access_rows + access_columns + allocation_mat
dt_mem_measured_total_mxv = Float64[]  # access_rows + access_vec + allocation_vec

# Theoretical memory times (s)
t_mem_theo_mxm = Float64[]   # (3*N^2)/RAM_Hz (simplified model)
t_mem_theo_mxv = Float64[]   # (N^2 + 2*N)/RAM_Hz

# --------------------------------------------------------
# Kernels (single operation, no Nt)
# --------------------------------------------------------

# Row-wise access (inner loop j) — inefficient in column-major
function mem_access_rows!(A)
    M = size(A, 1)
    @inbounds for i in 1:M
        for j in 1:M
            A[i, j] = 1f0
        end
    end
    return A
end

# Column-wise access (inner loop i) — efficient in column-major
function mem_access_columns!(A)
    M = size(A, 1)
    @inbounds for j in 1:M
        for i in 1:M
            A[i, j] = 1f0
        end
    end
    return A
end

# Vector write — sequential access
function mem_vector!(b)
    M = length(b)
    @inbounds for i in 1:M
        b[i] = 1f0
    end
    return b
end

# Allocate matrix and set last element to 1
function mem_allocate_matrix(M)
    A = zeros(T, M, M)
    @inbounds A[M, M] = 1
    return A
end

# Allocate vector and set last element to 1
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

    # ---------------------------
    # Memory (cold arrays per sample)
    # ---------------------------

    bench_rows = @benchmark mem_access_rows!(mat) setup=(mat = zeros(T, $N, $N)) samples=300 evals=1
    dt_rows = minimum(bench_rows).time * 1e-9
    push!(dt_rows_arr, dt_rows)

    bench_cols = @benchmark mem_access_columns!(mat) setup=(mat = zeros(T, $N, $N)) samples=300 evals=1
    dt_cols = minimum(bench_cols).time * 1e-9
    push!(dt_cols_arr, dt_cols)

    bench_alloc_mat = @benchmark mem_allocate_matrix($N) samples=300 evals=1
    dt_alloc_mat = minimum(bench_alloc_mat).time * 1e-9
    push!(dt_alloc_mat_arr, dt_alloc_mat)

    bench_vec = @benchmark mem_vector!(vec) setup=(vec = zeros(T, $N)) samples=300 evals=1
    dt_vec = minimum(bench_vec).time * 1e-9
    push!(dt_vec_arr, dt_vec)

    bench_alloc_vec = @benchmark mem_allocate_vector($N) samples=300 evals=1
    dt_alloc_vec = minimum(bench_alloc_vec).time * 1e-9
    push!(dt_alloc_vec_arr, dt_alloc_vec)

    # "Real total"
    push!(dt_mem_measured_total_mxm, dt_rows + dt_cols + dt_alloc_mat)
    push!(dt_mem_measured_total_mxv, dt_rows + dt_vec + dt_alloc_vec)
    println("N = $N: t_memory MxM measured = $(dt_mem_measured_total_mxm[end] * 1e6) μs, t_memory MxV measured = $(dt_mem_measured_total_mxv[end] * 1e6) μs")

    # Theoretical memory (simplified model)
    RAM_Hz = RAM_frequency_MHz * 1e6
    push!(t_mem_theo_mxm, (3 * N^2) / RAM_Hz)
    push!(t_mem_theo_mxv, (N^2 + 2 * N) / RAM_Hz)
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

mkpath("figures/3_cpu/tmemory")

# Convert to microseconds for readability
measured_mxm_us  = dt_mem_measured_total_mxm .* 1e6
measured_mxv_us  = dt_mem_measured_total_mxv .* 1e6
theo_mxm_us  = t_mem_theo_mxm .* 1e6
theo_mxv_us  = t_mem_theo_mxv .* 1e6

ym_max_mxm = maximum(vcat(measured_mxm_us, theo_mxm_us)) * 1.25
ym_max_mxv = maximum(vcat(measured_mxv_us, theo_mxv_us)) * 1.25

# ---------- MEMORY: separate figure for MxM ----------
p_mem_mxm = plot(
    Ns_vec, theo_mxm_us, label="t_mem_theo_mxm", lw=1, linestyle=:dash, color=:black,
    xlabel="Matrix size N", ylabel="Time (μs)", legend=:topleft,
    ylims=(0, ym_max_mxm), grid=true, size=(1200, 600)
)
plot!(p_mem_mxm, Ns_vec, measured_mxm_us, label="t_mem_measured_mxm", lw=1, marker=:circle, color=:blue)

# ---------- MEMORY: separate figure for MxV ----------
p_mem_mxv = plot(
    Ns_vec, theo_mxv_us, label="t_mem_theo_mxv", lw=1, linestyle=:dash, color=:black,
    xlabel="Matrix size N", ylabel="Time (μs)", legend=:topleft,
    ylims=(0, ym_max_mxv), grid=true, size=(1200, 600)
)
plot!(p_mem_mxv, Ns_vec, measured_mxv_us, label="t_mem_measured_mxv", lw=1, marker=:circle, color=:green)

# Display or save the plots based on save_plot
if save_plot
    savefig(p_mem_mxm, "figures/3_cpu/tmemory/$(T)_MEMORY_MxM_Ryzen5_5600X$file_suffix.png")
    savefig(p_mem_mxv, "figures/3_cpu/tmemory/$(T)_MEMORY_MxV_Ryzen5_5600X$file_suffix.png")
    println("Saved figures:")
    println("  - figures/3_cpu/tmemory/$(T)_MEMORY_MxM_Ryzen5_5600X$file_suffix.png")
    println("  - figures/3_cpu/tmemory/$(T)_MEMORY_MxV_Ryzen5_5600X$file_suffix.png")
else
    display(p_mem_mxm)
    display(p_mem_mxv)
    println("Plots displayed (not saved).")
end