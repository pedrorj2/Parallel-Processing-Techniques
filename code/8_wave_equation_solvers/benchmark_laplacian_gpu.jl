# File: 8_wave_equation_solvers/benchmark_laplacian_gpu.jl

using Printf
using Plots
using LinearAlgebra
using Base.Threads
using BenchmarkTools
using Measures
using CUDA

CUDA.allowscalar(false)

include("../7_global_interpolation/D2_matrix.jl")
include("../7_global_interpolation/D1_matrix.jl")
include("../utilities/nodes.jl")
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

# Function to get GPU peak GFLOPS
function gpu_peak_gflops(::Type{T}) where {T}
    fp32_peak = gpu_info(true)
    if T == Float32
        return fp32_peak
    else
        ratio = 1/64 # For Ampere and later architectures
        return fp32_peak * ratio
    end
end

function laplacian_2D_MxM(u_mat, D2x, D2yT)
    return D2x * u_mat .+ u_mat * D2yT
end

# FUNCIÓN CORREGIDA para obtener tiempo promedio en GPU
function get_avg_time_gpu(f, args...)
    GC.gc()
    CUDA.synchronize()
    
    # Warmup
    f(args...)
    CUDA.synchronize()
    
    # Usar bloque begin-end para evitar problemas de scope
    bench = @benchmark begin
        result = $f($args...)
        CUDA.synchronize()
        result
    end samples=300 evals=1
    
    return minimum(bench).time * 1e-9  # Convert to seconds
end

# Main benchmark function
function benchmark_laplacians()
    nodetype = :chebyshev
    Ns = 50:50:4000

    gflops_MxM_gpu = Float64[]
    gflops_theoretical_gpu = Float64[]

    GFLOPS_peak = gpu_peak_gflops(T)

    for N in Ns
        @printf("Benchmarking for N=%d\n", N)
        stencil = N + 1

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
        D2x_offset = build_D2_matrix(xnodes, stencil, T)
        D2y_offset = build_D2_matrix(ynodes, stencil, T)
        D2x = parent(D2x_offset)
        D2y = parent(D2y_offset)
        D2yT = transpose(D2y)

        M = N + 1

        # 2D setup
        u2d_mat = rand(T, M, M)

        # Move to GPU
        D2x_gpu = CuArray(D2x)
        D2yT_gpu = CuArray(D2yT)
        u2d_mat_gpu = CuArray(u2d_mat)

        # 2D MxM benchmarks on GPU
        t_MxM_gpu = get_avg_time_gpu(laplacian_2D_MxM, u2d_mat_gpu, D2x_gpu, D2yT_gpu)
        flops_2d = 4 * M * M * M
        push!(gflops_MxM_gpu, flops_2d / t_MxM_gpu / 1e9)

        # Theoretical GFLOPS
        push!(gflops_theoretical_gpu, GFLOPS_peak)

        # Console log
        println("2D_MxM_gpu:$(round(gflops_MxM_gpu[end], digits=1))")
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
        gflops_MxM_gpu,
        gflops_theoretical_gpu
    )) * 1.02

    # -------------------------
    # Figure 1: 2D MxM vs Theoretical
    # -------------------------
    p_mxm = plot(
        Ns_vec, gflops_MxM_gpu,
        label="2D MxM GPU", lw=1, marker=:circle, markersize=4, color=:blue,
        xlabel="Matrix size N", ylabel="GFLOPS",
        legend=:topleft, ylims=(0, yg_max), grid=true, size=(1200, 600)
    )
    plot!(p_mxm, Ns_vec, gflops_theoretical_gpu,
        label="Theoretical GPU: $(round(gflops_theoretical_gpu[end], digits=1)) GFLOPS",
        lw=1, color=:black, linestyle=:solid)

    savefig(p_mxm, "figures/8_wave_equation_solvers/$(T)_GFLOPS_laplacian_2D_MxM_gpu.png")
    println("Saved figure: figures/8_wave_equation_solvers/$(T)_GFLOPS_laplacian_2D_MxM_gpu.png")
end

benchmark_laplacians()