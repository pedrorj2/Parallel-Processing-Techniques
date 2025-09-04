# File: 8_wave_equation_solvers/benchmark_versions/wave2D_MxV_benchmark.jl

using Printf
using Plots
using OffsetArrays
using LinearAlgebra
using Base.Threads
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
println("\nSelect spatial node count Nx=Ny from the following options:")
allowed_Nx = [25, 50, 75, 100, 125, 150, 175, 200]
for (i, nx) in enumerate(allowed_Nx)
    println("$i. Nx = Ny = $nx")
end

print("\nEnter choice (1-8): ")
nx_choice = parse(Int, readline())
if nx_choice < 1 || nx_choice > length(allowed_Nx)
    error("Invalid choice. Please select a number between 1 and $(length(allowed_Nx)).")
end

Nx = allowed_Nx[nx_choice]
Ny = Nx  # Square grid

println("\nSelected: $T with Nx = Ny = $Nx")
println("Interpolation stencil size: $(Nx+1) nodes")
println("Starting simulation...")

# -------------------------------------------------
# Helper functions
# -------------------------------------------------

"""
    save_plots_at_time!(U, xnodes, ynodes, time_val, A, Nx, Ny, nodetype, output_dir, curr)

Generate both original and interpolated heatmap plots for a given time point.
"""
function save_plots_at_time!(U::OffsetArray{T}, xnodes, ynodes, time_val::T, A::T, Nx, Ny, nodetype, output_dir, curr) where {T}
    # Plot state at nodes
    x_vals = collect(xnodes)
    y_vals = collect(ynodes)
    p = heatmap(x_vals, y_vals, U[:, :, 0, curr]',
                xlabel="x", ylabel="y",
                title=@sprintf("State at t=%.2f s", time_val),
                clim=(-A, A), color=:inferno, size=(600, 600),
                fontsize=12)
    savefig(joinpath(output_dir, @sprintf("state_t_%.2f.png", time_val)))
    println("Plot saved: state_t_$(time_val).png")

    # Interpolated state
    Nfine = 100
    xfine = range(first(xnodes), last(xnodes), length=Nfine+1)
    yfine = range(first(ynodes), last(ynodes), length=Nfine+1)
    ℓx = OffsetArray([evalL(xnodes, i, xfine[k]) for i in 0:Nx, k in 1:Nfine+1], 0:Nx, 1:Nfine+1)
    ℓy = OffsetArray([evalL(ynodes, j, yfine[l]) for j in 0:Ny, l in 1:Nfine+1], 0:Ny, 1:Nfine+1)

    u_fine = zeros(T, Nfine+1, Nfine+1)
    @inbounds for l in 1:Nfine+1
        for k in 1:Nfine+1
            s = zero(T)
            for i in 0:Nx, j in 0:Ny
                s += U[i, j, 0, curr] * ℓx[i, k] * ℓy[j, l]
            end
            u_fine[k, l] = s
        end
    end
    
    p = heatmap(xfine, yfine, u_fine',
                clim=(-A, A), color=:inferno,
                xlabel="x", ylabel="y",
                title=@sprintf("Interpolated state at t=%.2f s", time_val),
                size=(600, 600),
                fontsize=12)
    savefig(joinpath(output_dir, @sprintf("interpolated_state_t_%.2f.png", time_val)))
    println("Plot saved: interpolated_state_t_$(time_val).png")
end

"""
    apply_initial_conditions!(U, xnodes, ynodes, A, σ, x_c, y_c, Nx, Ny)

Set U[i,j,0,0] = Gaussian pulse of amplitude A, width σ, centered at (x_c,y_c),
and U[i,j,1,0] = 0 (velocity).
"""
function apply_initial_conditions!(U::OffsetArray{T}, xnodes, ynodes, A::T, σ::T, x_c::T, y_c::T, Nx, Ny) where {T}
    @inbounds for j in 0:Ny
        for i in 0:Nx
            U[i, j, 0, 0] = A * exp(-((xnodes[i] - x_c)^2 + (ynodes[j] - y_c)^2) / (2 * σ^2))
            U[i, j, 1, 0] = zero(T)
        end
    end
end

"""
    apply_dirichlet!(U, n, Nx, Ny)

Enforce zero Dirichlet boundary conditions on both fields at time index n.
Fields are at index 0 (u) and 1 (v).
"""
function apply_dirichlet!(U::OffsetArray{T}, n, Nx, Ny) where {T}
    @inbounds for j in 0:Ny
        U[0, j, 0, n] = zero(T)
        U[Nx, j, 0, n] = zero(T)
        U[0, j, 1, n] = zero(T)
        U[Nx, j, 1, n] = zero(T)
    end
    @inbounds for i in 0:Nx
        U[i, 0, 0, n] = zero(T)
        U[i, Ny, 0, n] = zero(T)
        U[i, 0, 1, n] = zero(T)
        U[i, Ny, 1, n] = zero(T)
    end
end

"""
    laplacian_2D(U, n, D2x, D2y, Nx, Ny)

Compute the Laplacian of U[:,:,0,n] (position field) using explicit matrix-vector loops.
"""
function laplacian_2D(U::OffsetArray{T}, n, D2x::OffsetArray{T}, D2y::OffsetArray{T}, Nx, Ny) where {T}
    L = OffsetArray(zeros(T, Nx+1, Ny+1), 0:Nx, 0:Ny)
    @inbounds @threads for j in 0:Ny
        for i in 0:Nx
            d2x = zero(T)
            for ii in 0:Nx
                d2x += D2x[i, ii] * U[ii, j, 0, n]
            end
            d2y = zero(T)
            for jj in 0:Ny
                d2y += D2y[j, jj] * U[i, jj, 0, n]
            end
            L[i, j] = d2x + d2y
        end
    end
    return L
end

# -------------------------------------------------
# Main simulation function
# -------------------------------------------------
function main(T::Type, Nx::Int, Ny::Int)
    # --- Parameters ---
    Lx, Ly = T(1.0), T(1.0)
    c = T(1.0)
    dt = T(1e-4) * min(Lx/Nx, Ly/Ny)
    T_time = T(1.0)
    Nt = Int(floor(T_time/dt))
    A = T(1.0)
    σ = T(0.075)
    x_c, y_c = Lx/T(2), Ly/T(2)

    stencil = Nx + 1  # Global interpolation stencil
    nodetype = :chebyshev_lobatto # :equispaced, :chebyshev, :chebyshev_lobatto

    # --- Nodes ---
    xnodes = nodetype == :chebyshev ? convert.(T, chebyshev_nodes(Nx)) :
             nodetype == :chebyshev_lobatto ? convert.(T, chebyshev_lobatto_nodes(Nx)) :
             nodetype == :equispaced ? convert.(T, equispaced_nodes(Nx)) :
             error("Unsupported node type")
    ynodes = nodetype == :chebyshev ? convert.(T, chebyshev_nodes(Ny)) :
             nodetype == :chebyshev_lobatto ? convert.(T, chebyshev_lobatto_nodes(Ny)) :
             nodetype == :equispaced ? convert.(T, equispaced_nodes(Ny)) :
             error("Unsupported node type")

    # --- Differentiation matrices ---
    D2x = OffsetArray(build_D2_matrix(xnodes, stencil, T), 0:Nx, 0:Nx)
    D2y = OffsetArray(build_D2_matrix(ynodes, stencil, T), 0:Ny, 0:Ny)

    # --- State tensor U[x, y, field=0→u,1→v, t=0:1] ---
    U = OffsetArray(zeros(T, Nx+1, Ny+1, 2, 2), 0:Nx, 0:Ny, 0:1, 0:1)

    # --- Physical tensor F[x, y, comp=0→f_u,1→f_v] ---
    F = OffsetArray(zeros(T, Nx+1, Ny+1, 2), 0:Nx, 0:Ny, 0:1)

    # --- Initial & boundary conditions at n=0 ---
    apply_initial_conditions!(U, xnodes, ynodes, A, σ, x_c, y_c, Nx, Ny)
    apply_dirichlet!(U, 0, Nx, Ny)

    # --- Output directory ---
    output_dir = "code/8_wave_equation_solvers/benchmark_versions/output/output2D_$(Nx)_MxV_benchmark"
    isdir(output_dir) || (mkpath(output_dir); println("Created directory $output_dir"))

    # --- Time stepping with intermediate saves ---
    println("Starting 2D MxV benchmark with $(stencil)-node stencil and $nodetype nodes…")
    println("Using $T with Nx = Ny = $Nx")
    println("Total time steps: $Nt")

    # Define times to save plots
    times_to_save = T[0.0, 0.33, 0.66, 1.0]
    indices_to_save = floor.(Int, times_to_save ./ dt)

    # Save initial state (t=0)
    curr = 0
    save_plots_at_time!(U, xnodes, ynodes, T(0.0), A, Nx, Ny, nodetype, output_dir, curr)

    t_start = time()
    for n in 0:Nt-1
        curr = n % 2
        next = (n + 1) % 2
        apply_dirichlet!(U, curr, Nx, Ny)
        L = laplacian_2D(U, curr, D2x, D2y, Nx, Ny)

        @views begin
            F[:, :, 0] .= U[:, :, 1, curr] # f_u = v
            F[:, :, 1] .= c^2 .* L         # f_v = c^2 * Δu
            U[:, :, :, next] .= U[:, :, :, curr] .+ dt .* F
        end

        # Check if we need to save plots at this time step
        if (n + 1) in indices_to_save
            time_val = (n + 1) * dt
            save_plots_at_time!(U, xnodes, ynodes, time_val, A, Nx, Ny, nodetype, output_dir, next)
        end
    end
    
    apply_dirichlet!(U, Nt % 2, Nx, Ny)
    println("Time stepping completed in $(round(time() - t_start, digits=2)) s")

    BLAS.set_num_threads(1)
end

# -------------------------------------------------
# Execute simulation
# -------------------------------------------------
main(T, Nx, Ny)
