# File: 8_wave_equation_solvers/benchmark_versions/wave1D_benchmark.jl

using Printf
using Plots
using OffsetArrays
using LinearAlgebra
using BenchmarkTools
using Measures

# # Uncomment if using MKL, Float64 have issues with Julia MKL in Windows
# # with OffsetArrays Matrix-Matrix multiplications, Float32 works fine.
# # If you want to use MKL and Float64, try running it in a Linux environment
# using MKL 

BLAS.set_num_threads(12)

include("../../7_global_interpolation/D2_matrix.jl")
include("../../7_global_interpolation/D1_matrix.jl")

include("../../utilities/offset.jl")
include("../../utilities/nodes.jl")

# -------------------------------------------------
# Interactive prompts for user input
# -------------------------------------------------

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

if T == Float32
    println("Warning: For large Nx values (>100), Float32 may cause numerical instability.")
end
println("\nSelect spatial node count Nx from the following options:")
allowed_Nx = [25, 50, 75, 100, 125, 150, 175, 200]
for (i, nx) in enumerate(allowed_Nx)
    println("$i. Nx = $nx")
end

print("\nEnter choice (1-8): ")
nx_choice = parse(Int, readline())
if nx_choice < 1 || nx_choice > length(allowed_Nx)
    error("Invalid choice. Please select a number between 1 and $(length(allowed_Nx)).")
end

Nx = allowed_Nx[nx_choice]

println("\nSelected: $T with Nx = $Nx")
println("Interpolation stencil size: $(Nx+1) nodes")
println("Starting simulation...")

# -------------------------------------------------
# Helper functions
# -------------------------------------------------

"""
    save_plots_at_time!(U, xnodes, time_val, A, Nx, nodetype, output_dir, curr)

Generate both original and interpolated plots for a given time point.
"""
function save_plots_at_time!(U::OffsetArray{T}, xnodes, time_val::T, A::T, Nx, nodetype, output_dir, curr) where {T}
    # Plot state at nodes
    p = plot(parent(xnodes), parent(U[:, 0, curr]),
        title=@sprintf("State at t=%.2f s", time_val),
        xlabel="x", ylabel="u(x)",
        ylim=(-2A, 2A), size=(600, 400),
        label=string(nodetype),
        fontsize=12, titlefontsize=12, guidefontsize=12, legendfontsize=12, tickfontsize=12)
    scatter!(parent(xnodes), parent(U[:, 0, curr]),
        marker=:circle, markersize=3,
        label="$(Nx+1) nodes",
        color=:royalblue1)
    savefig(joinpath(output_dir, @sprintf("state_t_%.2f.png", time_val)))
    println("Plot saved: state_t_$(time_val).png")

    # Interpolated state
    Nfine = 100
    xfine = range(first(xnodes), last(xnodes), length=Nfine + 1)
    ℓx = OffsetArray([evalL(xnodes, i, xfine[k]) for i in 0:Nx, k in 1:Nfine+1], 0:Nx, 1:Nfine+1)
    u_fine = zeros(T, Nfine + 1)
    @inbounds for k in 1:Nfine+1
        s = zero(T)
        for i in 0:Nx
            s += U[i, 0, curr] * ℓx[i, k]
        end
        u_fine[k] = s
    end

    p = plot(xfine, u_fine,
        title=@sprintf("Interpolated state at t=%.2f s", time_val),
        xlabel="x", ylabel="u(x)",
        lw=2, ylim=(-2A, 2A), size=(600, 400),
        label=string(nodetype),
        fontsize=12, titlefontsize=12, guidefontsize=12, legendfontsize=12, tickfontsize=12)
    scatter!(parent(xnodes), parent(U[:, 0, curr]),
        marker=:circle, markersize=3,
        label="$(Nx+1) nodes",
        color=:royalblue1)
    savefig(joinpath(output_dir, @sprintf("interpolated_state_t_%.2f.png", time_val)))
    println("Plot saved: interpolated_state_t_$(time_val).png")
end

"""
    apply_initial_conditions!(U, xnodes, A, σ, x_c, Nx)

Set U[i,0,0] = Gaussian pulse of amplitude A, width σ, centered at x_c,
and U[i,1,0] = 0 (velocity).
"""
function apply_initial_conditions!(U::OffsetArray{T}, xnodes, A::T, σ::T, x_c::T, Nx) where {T}
    @inbounds for i in 0:Nx
        U[i, 0, 0] = A * exp(-((xnodes[i] - x_c)^2) / (2 * σ^2)) # Gaussian initial condition
        U[i, 1, 0] = zero(T)
    end
end

"""
    apply_dirichlet!(U, n, Nx)

Enforce zero Dirichlet boundary conditions on both fields at time index n.
Fields are at index 0 (u) and 1 (v).
"""
function apply_dirichlet!(U::OffsetArray{T}, n, Nx) where {T}
    U[0, 0, n] = zero(T)
    U[Nx, 0, n] = zero(T)
    U[0, 1, n] = zero(T)
    U[Nx, 1, n] = zero(T)
end

"""
    laplacian_1D(U, D2x)

Compute the Laplacian of U[:,0,n] (position field).
"""
function laplacian_1D(U::OffsetArray{T}, D2x::OffsetArray{T}, n) where {T}
    return D2x * U[:, 0, n]
end

"""
    derivative_1D(U, D1x)

Compute the spatial derivative of U[:,0,n] (position field).
"""
function derivative_1D(U::OffsetArray{T}, D1x::OffsetArray{T}, n) where {T}
    return D1x * U[:, 0, n]
end

# -------------------------------------------------
# Main simulation function
# -------------------------------------------------
function main(T::Type, Nx::Int)
    # --- Parameters ---
    L = T(1.0)
    c = T(1.0)
    dt = T(1e-4) * (L / Nx)
    T_time = T(1.0)
    Nt = Int(floor(T_time / dt))
    A = T(1.0)
    σ = T(0.075)
    x_c = L / T(2)

    stencil = Nx + 1  # Global interpolation stencil
    nodetype = :chebyshev # :equispaced, :chebyshev, :chebyshev_lobatto

    # --- Nodes ---
    xnodes = nodetype == :equispaced ? convert.(T, equispaced_nodes(Nx)) :
             nodetype == :chebyshev ? convert.(T, chebyshev_nodes(Nx)) :
             nodetype == :chebyshev_lobatto ? convert.(T, chebyshev_lobatto_nodes(Nx)) :
             error("Unsupported node type: $nodetype")

    # --- Differentiation matrices ---
    D2x = OffsetArray(build_D2_matrix(xnodes, stencil, T), 0:Nx, 0:Nx)

    # --- State tensor U[x, field=0→u,1→v, t=0:1] ---
    U = OffsetArray(zeros(T, Nx + 1, 2, 2), 0:Nx, 0:1, 0:1)

    # --- Physical tensor F[x, comp=0→f_u,1→f_v] ---
    F = OffsetArray(zeros(T, Nx + 1, 2), 0:Nx, 0:1)

    # --- Initial conditions at n=0 ---
    apply_initial_conditions!(U, xnodes, A, σ, x_c, Nx)
    apply_dirichlet!(U, 0, Nx)

    # --- Output directory ---
    output_dir = "code/8_wave_equation_solvers/benchmark_versions/output/output1D_$(Nx)_benchmark"
    isdir(output_dir) || (mkpath(output_dir); println("Created directory $output_dir"))

    # --- Time stepping with intermediate saves ---
    println("Starting 1D wave benchmark with $(stencil)-node stencil and $nodetype nodes…")
    println("Using $T with Nx=$Nx")
    println("Total time steps: $Nt")

    # Define times to save plots
    times_to_save = T[0.0, 0.25, 0.5, 0.75, 1.0]
    indices_to_save = floor.(Int, times_to_save ./ dt)

    # Save initial state (t=0)
    curr = 0
    save_plots_at_time!(U, xnodes, T(0.0), A, Nx, nodetype, output_dir, curr)

    t_start = time()
    for n in 0:Nt-1
        curr = n % 2
        next = (n + 1) % 2
        apply_dirichlet!(U, curr, Nx)
        L = laplacian_1D(U, D2x, curr)

        @views begin
            F[:, 0] .= U[:, 1, curr] # f_u = v
            F[:, 1] .= c^2 .* L      # f_v = c^2 * Δu
            U[:, :, next] .= U[:, :, curr] .+ dt .* F
        end

        # Check if we need to save plots at this time step
        if (n + 1) in indices_to_save
            time_val = (n + 1) * dt
            save_plots_at_time!(U, xnodes, time_val, A, Nx, nodetype, output_dir, next)
        end

    end

    apply_dirichlet!(U, Nt % 2, Nx)
    println("Time stepping completed in $(round(time() - t_start, digits=2)) s")
end

# -------------------------------------------------
# Execute simulation
# -------------------------------------------------
main(T, Nx)
