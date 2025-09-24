# File: 8_wave_equation_solvers/animation_versions/wave2D_MxV.jl

# These versions are the original version with Explicit Euler scheme and with 0 to N indexing for the spatial dimension.

using Printf
using Plots
using OffsetArrays
using Base.Threads
using LinearAlgebra
using MKL

BLAS.set_num_threads(12)

include("../../7_global_interpolation/D2_matrix.jl")
include("../../7_global_interpolation/D1_matrix.jl")
include("../../utilities/offset.jl")
include("../../utilities/nodes.jl")

# -------------------------------------------------
# Helper functions
# -------------------------------------------------

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
                d2y +=  D2y[j, jj] * U[i, jj, 0, n]
            end
            L[i, j] = d2x + d2y
        end
    end
    return L
end

"""
    gradient_2D(U, n, D1x, D1y, Nx, Ny)

Compute the spatial gradients of U[:,:,0,n] (position field) using explicit matrix-vector loops.
Returns dudx, dudy.
"""
function gradient_2D(U::OffsetArray{T}, n, D1x::OffsetArray{T}, D1y::OffsetArray{T}, Nx, Ny) where {T}
    dudx = OffsetArray(zeros(T, Nx+1, Ny+1), 0:Nx, 0:Ny)
    dudy = OffsetArray(zeros(T, Nx+1, Ny+1), 0:Nx, 0:Ny)
    @inbounds @threads for j in 0:Ny
        for i in 0:Nx
            dx = zero(T)
            for ii in 0:Nx
                dx += D1x[i, ii] * U[ii, j, 0, n]
            end
            dy = zero(T)
            for jj in 0:Ny
                dy += U[i, jj, 0, n] * D1y[j, jj]
            end
            dudx[i, j] = dx
            dudy[i, j] = dy
        end
    end
    return dudx, dudy
end

# -------------------------------------------------
# Main simulation
# -------------------------------------------------
function main(T::Type=Float32)
    # --- Parameters ---
    Nx, Ny = 25, 25
    Lx, Ly = T(1.0), T(1.0)
    c = T(1.0)
    dt = T(1e-4) * min(Lx/Nx, Ly/Ny)
    T_time = T(2.0)
    Nt = Int(floor(T_time/dt))
    A = T(1.0)
    σ = T(0.075)
    x_c, y_c = Lx/T(2), Ly/T(2)

    stencil = 26
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
    D2yT = OffsetArray(permutedims(D2y), 0:Ny, 0:Ny)


    D1x = OffsetArray(build_D1_matrix(xnodes, stencil, T), 0:Nx, 0:Nx)
    D1y = OffsetArray(build_D1_matrix(ynodes, stencil, T), 0:Ny, 0:Ny)
    D1yT = OffsetArray(permutedims(D1y), 0:Ny, 0:Ny)

    # --- State tensor U[x, y, field=0→u,1→v, t] ---
    U = OffsetArray(zeros(T, Nx+1, Ny+1, 2, Nt+1), 0:Nx, 0:Ny, 0:1, 0:Nt)

    # --- Physical tensor F[x, y, comp=0→f_u,1→f_v] ---
    F = OffsetArray(zeros(T, Nx+1, Ny+1, 2), 0:Nx, 0:Ny, 0:1)

    # --- Initial & boundary conditions at n=0 ---
    apply_initial_conditions!(U, xnodes, ynodes, A, σ, x_c, y_c, Nx, Ny)
    apply_dirichlet!(U, 0, Nx, Ny)

    # --- Time stepping ---
    println("Starting 2D MxV simulation with $stencil-node stencil and $nodetype nodes…")
    println("Using $(BLAS.get_num_threads()) BLAS threads with $T")
    t_start = time()
    for n in 0:Nt-1
        apply_dirichlet!(U, n, Nx, Ny)
        L = laplacian_2D(U, n, D2x, D2y, Nx, Ny)

        @views begin
            F[:, :, 0] .= U[:, :, 1, n] # f_u = v
            F[:, :, 1] .= c^2 .* L      # f_v = c^2 * Δu
            U[:, :, :, n+1] .= U[:, :, :, n] .+ dt .* F
        end

        if n % 10000 == 0
            @printf("Step %d of %d completed\n", n, Nt)
        end
    end
    apply_dirichlet!(U, Nt, Nx, Ny)
    println("Time stepping completed in $(round(time() - t_start, digits=2)) s")

    BLAS.set_num_threads(1)

    # --- Output directory ---
    output_dir = "code/8_wave_equation_solvers/animation_versions/output/output2D_$(Nx)_MxV"
    isdir(output_dir) || (mkpath(output_dir); println("Created directory $output_dir"))

    # --- Energy computation ---
    energy_kin = zeros(T, Nt+1)
    energy_pot = zeros(T, Nt+1)
    energy_total = zeros(T, Nt+1)

    for n in 0:Nt
        dudx, dudy = gradient_2D(U, n, D1x, D1y, Nx, Ny)
        Ek, Ep = zero(T), zero(T)
        @inbounds for j in 1:Ny-1
            for i in 1:Nx-1
                Δx = xnodes[i] - xnodes[i-1]
                Δy = ynodes[j] - ynodes[j-1]
                Ek += T(0.5) * U[i, j, 1, n]^2 * Δx * Δy
                Ep += T(0.5) * c^2 * (dudx[i, j]^2 + dudy[i, j]^2) * Δx * Δy
            end
        end
        energy_kin[n+1] = Ek
        energy_pot[n+1] = Ep
        energy_total[n+1] = Ek + Ep
    end
    println("Energy evaluation completed")

    # --- Plot energy vs time ---
    t_vals = collect(0:Nt) .* dt
    plot(t_vals, energy_kin, label="Kinetic", xlabel="Time", ylabel="Energy",
         title=@sprintf("2D, %s, dt=%.2e, stencil=%d", string(nodetype), dt, stencil))
    plot!(t_vals, energy_pot, label="Potential", lw=2)
    plot!(t_vals, energy_total, label="Total", lw=2, linestyle=:dash)
    savefig(joinpath(output_dir, "energy_MxV.png"))
    println("Energy plot saved to $(joinpath(output_dir, "energy_MxV.png"))")

    # --- Animation ---
    x_vals = collect(xnodes)
    y_vals = collect(ynodes)
    anim = @animate for n in 0:3000:Nt
        heatmap(x_vals, y_vals, U[:, :, 0, n]',
                xlabel="x", ylabel="y",
                title=@sprintf("t=%.2f", n * dt),
                clim=(-A, A), color=:inferno, size=(600, 600))
    end
    gif(anim, joinpath(output_dir, "original_MxV.gif"), fps=30)
    println("Animation saved to $(joinpath(output_dir, "original_MxV.gif"))")

    # --- Interpolated animation ---
    Nfine = 100
    xfine = range(first(xnodes), last(xnodes), length=Nfine+1)
    yfine = range(first(ynodes), last(ynodes), length=Nfine+1)
    ℓx = OffsetArray([evalL(xnodes, i, xfine[k]) for i in 0:Nx, k in 1:Nfine+1], 0:Nx, 1:Nfine+1)
    ℓy = OffsetArray([evalL(ynodes, j, yfine[l]) for j in 0:Ny, l in 1:Nfine+1], 0:Ny, 1:Nfine+1)

    anim_interp = @animate for n in 0:3000:Nt
        u_fine = zeros(T, Nfine+1, Nfine+1)
        @inbounds for l in 1:Nfine+1
            for k in 1:Nfine+1
                s = zero(T)
                for i in 0:Nx, j in 0:Ny
                    s += U[i, j, 0, n] * ℓx[i, k] * ℓy[j, l]
                end
                u_fine[k, l] = s
            end
        end
        heatmap(xfine, yfine, u_fine',
                clim=(-A, A), color=:inferno,
                xlabel="x", ylabel="y",
                title=@sprintf("t=%.2f", n * dt),
                size=(600, 600))
    end
    gif(anim_interp, joinpath(output_dir, "evaluated$(Nfine)_MxV.gif"), fps=30)
    println("Interpolated animation saved to $(joinpath(output_dir, "evaluated$(Nfine)_MxV.gif"))")
end

main(Float32)