# File: 8_wave_equation_solvers/benchmark_laplacian.jl

using Printf
using Plots
using OffsetArrays
using LinearAlgebra
using Base.Threads
using BenchmarkTools
using Measures
using MKL

BLAS.set_num_threads(12)

include("../7_global_interpolation/D2_matrix.jl")
include("../7_global_interpolation/D1_matrix.jl")
include("../utilities/offset.jl")
include("../utilities/nodes.jl")

# Include system_info.jl for CPU information (assumed to provide cpu_info and num_physical_cores)
include("../1_annexes/system_info.jl")

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

# Laplacian functions adapted from the provided codes
function laplacian_1D(u_vec, D2x)
    return D2x * u_vec
end

function laplacian_2D_MxM(u_mat, D2x, D2yT)
    return D2x * u_mat .+ u_mat * D2yT
end

function laplacian_2D_MxV(U::OffsetArray{T}, D2x::OffsetArray{T}, D2y::OffsetArray{T}, Nx, Ny) where {T}
    L = OffsetArray(zeros(T, Nx + 1, Ny + 1), 0:Nx, 0:Ny)
    @inbounds @threads for j in 0:Ny
        for i in 0:Nx
            d2x = zero(T)
            for ii in 0:Nx
                d2x += D2x[i, ii] * U[ii, j]
            end
            d2y = zero(T)
            for jj in 0:Ny
                d2y += D2y[j, jj] * U[i, jj]
            end
            L[i, j] = d2x + d2y
        end
    end
    return L
end

# Function to get average time per call using BenchmarkTools
function get_avg_time(f, args...)
    GC.gc()
    bench = @benchmark $f($args...) samples = 300 evals = 1
    return minimum(bench).time * 1e-9  # Convert to seconds
end

# Peak GFLOPS calculation
@inline function peak_gflops_for(gflops_fp32::Real, ::Type{Float32})
    return float(gflops_fp32)
end
@inline function peak_gflops_for(gflops_fp32::Real, ::Type{Float64})
    return float(gflops_fp32) / 2
end

# Main benchmark function
function benchmark_laplacians()
    nodetype = :chebyshev_lobatto
    Ns = 50:50:2000  # Match the range from gflops_1operation.jl

    gflops_1d_single = Float64[]
    gflops_1d_multi = Float64[]
    gflops_MxM_single = Float64[]
    gflops_MxM_multi = Float64[]
    gflops_MxV_single = Float64[]
    gflops_MxV_multi = Float64[]
    gflops_theoretical_single = Float64[]
    gflops_theoretical_multi = Float64[]

    # Get CPU information
    GFLOPS_peak_raw = cpu_info(true)
    GFLOPS_peak = peak_gflops_for(GFLOPS_peak_raw, T)
    N_cores = num_physical_cores()
    N_threads = 12

    for N in Ns
        @printf("Benchmarking for N=%d\n", N)
        stencil = N + 1  # Global interpolation: stencil = N + 1

        # --- Nodes ---
        xnodes = nodetype == :chebyshev ? convert.(T, chebyshev_nodes(N)) :
                 nodetype == :chebyshev_lobatto ? convert.(T, chebyshev_lobatto_nodes(N)) :
                 nodetype == :equispaced ? convert.(T, equispaced_nodes(N)) :
                 error("Unsupported node type")
        ynodes = nodetype == :chebyshev ? convert.(T, chebyshev_nodes(N)) :
                 nodetype == :chebyshev_lobatto ? convert.(T, chebyshev_lobatto_nodes(N)) :
                 nodetype == :equispaced ? convert.(T, equispaced_nodes(N)) :
                 error("Unsupported node type")

        # --- Differentiation matrices ---
        D2x = OffsetArray(build_D2_matrix(xnodes, stencil, T), 0:N, 0:N)
        D2y = OffsetArray(build_D2_matrix(ynodes, stencil, T), 0:N, 0:N)
        D2yT = OffsetArray(permutedims(D2y), 0:N, 0:N)

        M = N + 1

        # 1D setup
        u1d_vec_parent = rand(T, M)
        u1d_vec = OffsetArray(u1d_vec_parent, 0:N)

        # 2D setup
        u2d_mat_parent = rand(T, M, M)
        u2d_mat = OffsetArray(u2d_mat_parent, 0:N, 0:N)

        # 1D benchmarks
        BLAS.set_num_threads(1)
        t_1d_single = get_avg_time(laplacian_1D, u1d_vec, D2x)
        flops_1d = 2 * M * M  # FLOPs for matrix-vector multiply
        push!(gflops_1d_single, flops_1d / t_1d_single / 1e9)

        BLAS.set_num_threads(N_threads)
        t_1d_multi = get_avg_time(laplacian_1D, u1d_vec, D2x)
        push!(gflops_1d_multi, flops_1d / t_1d_multi / 1e9)

        # 2D MxM (matrix-matrix) benchmarks
        BLAS.set_num_threads(1)
        t_MxM_single = get_avg_time(laplacian_2D_MxM, u2d_mat, D2x, D2yT)
        flops_2d = 4 * M * M * M  # Two matrix-matrix multiplies, each ~2 M^3 FLOPs
        push!(gflops_MxM_single, flops_2d / t_MxM_single / 1e9)

        BLAS.set_num_threads(N_threads)
        t_MxM_multi = get_avg_time(laplacian_2D_MxM, u2d_mat, D2x, D2yT)
        push!(gflops_MxM_multi, flops_2d / t_MxM_multi / 1e9)

        # 2D MxV (looped matrix-vector) benchmarks
        BLAS.set_num_threads(1)
        t_MxV_single = get_avg_time(laplacian_2D_MxV, u2d_mat, D2x, D2y, N, N)
        push!(gflops_MxV_single, flops_2d / t_MxV_single / 1e9)

        BLAS.set_num_threads(N_threads)
        t_MxV_multi = get_avg_time(laplacian_2D_MxV, u2d_mat, D2x, D2y, N, N)
        push!(gflops_MxV_multi, flops_2d / t_MxV_multi / 1e9)

        # Theoretical GFLOPS
        push!(gflops_theoretical_single, GFLOPS_peak / N_cores)
        push!(gflops_theoretical_multi, GFLOPS_peak)

        # Console log
        println("1D_MxV_s:$(round(gflops_1d_single[end], digits=1)) 1D_MxV_m:$(round(gflops_1d_multi[end], digits=1))\n",
            "2D_MxM_s:$(round(gflops_MxM_single[end], digits=1)) 2D_MxM_m:$(round(gflops_MxM_multi[end], digits=1))\n",
            "2D_MxV_s:$(round(gflops_MxV_single[end], digits=1)) 2D_MxV_m:$(round(gflops_MxV_multi[end], digits=1))\n")
    end

    default(
        markerstrokewidth=0,
        legendfontsize=12,
        margins=5mm,
        xtickfont=font(12, "sans-serif", :black, rotation=0),
        ytickfont=font(12, "sans-serif", :black, rotation=0),
        xguidefont=font(12, "sans-serif", :black, rotation=0),
        yguidefont=font(12, "sans-serif", :black, rotation=0)
    )

    mkpath("figures/8_wave_equation_solvers")

    Ns_vec = collect(Ns)

    yg_max = maximum(vcat(
        gflops_1d_single, gflops_1d_multi,
        gflops_MxM_single, gflops_MxM_multi,
        gflops_MxV_single, gflops_MxV_multi,
        gflops_theoretical_single, gflops_theoretical_multi
    )) * 1.02

    # -------------------------
    # Figure 1: 1D MxV vs Theoretical
    # -------------------------
    p_1d = plot(
        Ns_vec, gflops_1d_single,
        label="1D MxV Single", lw=1, marker=:cross, markersize=4, color=:green,
        xlabel="Matrix size N", ylabel="GFLOPS",
        legend=:topleft, ylims=(0, yg_max), grid=true, size=(1200, 600)
    )
    plot!(p_1d, Ns_vec, gflops_1d_multi, label="1D MxV Multi", lw=1, marker=:circle, markersize=4, color=:green)
    plot!(p_1d, Ns_vec, gflops_theoretical_single,
        label="Theoretical Single: $(round(gflops_theoretical_single[end], digits=1)) GFLOPS",
        lw=1, color=:black, linestyle=:solid)
    plot!(p_1d, Ns_vec, gflops_theoretical_multi,
        label="Theoretical Multi: $(round(gflops_theoretical_multi[end], digits=1)) GFLOPS",
        lw=1, color=:black, linestyle=:solid)

    savefig(p_1d, "figures/8_wave_equation_solvers/$(T)_GFLOPS_laplacian_1D_MxV.png")
    println("Saved figure: figures/8_wave_equation_solvers/$(T)_GFLOPS_laplacian_1D_MxV.png")

    # -------------------------
    # Figure 2: 2D MxM vs Theoretical
    # -------------------------
    p_mxm = plot(
        Ns_vec, gflops_MxM_single,
        label="2D MxM Single", lw=1, marker=:cross, markersize=4, color=:blue,
        xlabel="Matrix size N", ylabel="GFLOPS",
        legend=:topleft, ylims=(0, yg_max), grid=true, size=(1200, 600)
    )
    plot!(p_mxm, Ns_vec, gflops_MxM_multi, label="2D MxM Multi", lw=1, marker=:circle, markersize=4, color=:blue)
    plot!(p_mxm, Ns_vec, gflops_theoretical_single,
        label="Theoretical Single: $(round(gflops_theoretical_single[end], digits=1)) GFLOPS",
        lw=1, color=:black, linestyle=:solid)
    plot!(p_mxm, Ns_vec, gflops_theoretical_multi,
        label="Theoretical Multi: $(round(gflops_theoretical_multi[end], digits=1)) GFLOPS",
        lw=1, color=:black, linestyle=:solid)

    savefig(p_mxm, "figures/8_wave_equation_solvers/$(T)_GFLOPS_laplacian_2D_MxM.png")
    println("Saved figure: figures/8_wave_equation_solvers/$(T)_GFLOPS_laplacian_2D_MxM.png")

    # -------------------------
    # Figure 3: 2D MxV vs Theoretical
    # -------------------------
    p_mxv = plot(
        Ns_vec, gflops_MxV_single,
        label="2D MxV Single", lw=1, marker=:cross, markersize=4, color=:red,
        xlabel="Matrix size N", ylabel="GFLOPS",
        legend=:topleft, ylims=(0, yg_max), grid=true, size=(1200, 600)
    )
    plot!(p_mxv, Ns_vec, gflops_MxV_multi, label="2D MxV Multi", lw=1, marker=:circle, color=:red, markersize=4)
    plot!(p_mxv, Ns_vec, gflops_theoretical_single,
        label="Theoretical Single: $(round(gflops_theoretical_single[end], digits=1)) GFLOPS",
        lw=1, color=:black, linestyle=:solid)
    plot!(p_mxv, Ns_vec, gflops_theoretical_multi,
        label="Theoretical Multi: $(round(gflops_theoretical_multi[end], digits=1)) GFLOPS",
        lw=1, color=:black, linestyle=:solid)

    savefig(p_mxv, "figures/8_wave_equation_solvers/$(T)_GFLOPS_laplacian_2D_MxV.png")
    println("Saved figure: figures/8_wave_equation_solvers/$(T)_GFLOPS_laplacian_2D_MxV.png")
end

benchmark_laplacians()

