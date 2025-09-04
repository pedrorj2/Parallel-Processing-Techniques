# File: 8_wave_equation_solvers/animation_versions/wave1D.jl

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
    apply_initial_conditions!(U, xnodes, A, σ, x_c, Nx)

Set U[i,0,0] = Gaussian pulse of amplitude A, width σ, centered at x_c,
and U[i,1,0] = 0 (velocity).
"""
function apply_initial_conditions!(U::OffsetArray{T}, xnodes, A::T, σ::T, x_c::T, Nx) where {T}
    @inbounds @threads for i in 0:Nx
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
function laplacian_1D(U::OffsetArray{T}, D2x::OffsetArray{T}) where {T}
    return D2x * U[:, 0]
end

"""
    derivative_1D(U, D1x)

Compute the spatial derivative of U[:,0,n] (position field).
"""
function derivative_1D(U::OffsetArray{T}, D1x::OffsetArray{T}) where {T}
    return D1x * U[:, 0]
end

# -------------------------------------------------
# Main simulation function
# -------------------------------------------------
function main(T::Type=Float32)
    # --- Parameters ---
    Nx = 25
    L = T(1.0)
    c = T(1.0)
    dt = T(1e-4) * (L / Nx)
    T_time = T(1.0)
    Nt = Int(floor(T_time / dt))
    A = T(1.0)
    σ = T(0.075)
    x_c = L / T(2)

    stencil = 26
    nodetype = :chebyshev # :equispaced, :chebyshev, :chebyshev_lobatto

    # --- Nodes ---
    xnodes = nodetype == :equispaced ? convert.(T, equispaced_nodes(Nx)) :
             nodetype == :chebyshev ? convert.(T, chebyshev_nodes(Nx)) :
             nodetype == :chebyshev_lobatto ? convert.(T, chebyshev_lobatto_nodes(Nx)) :
             error("Unsupported node type: $nodetype")

    # --- Differentiation matrices ---
    D2x = OffsetArray(build_D2_matrix(xnodes, stencil, T), 0:Nx, 0:Nx)
    D1x = OffsetArray(build_D1_matrix(xnodes, stencil, T), 0:Nx, 0:Nx)

    # --- State tensor U[x, field=0→u,1→v, t] ---
    U = OffsetArray(zeros(T, Nx+1, 2, Nt+1), 0:Nx, 0:1, 0:Nt)

    # --- Physical tensor F[x, comp=0→f_u,1→f_v] ---
    F = OffsetArray(zeros(T, Nx+1, 2), 0:Nx, 0:1)

    # --- Initial & boundary conditions at n=0 ---
    apply_initial_conditions!(U, xnodes, A, σ, x_c, Nx)
    apply_dirichlet!(U, 0, Nx)

    # --- Time stepping ---
    println("Starting 1D wave simulation with $stencil-node stencil and $nodetype nodes…")
    println("Using $(BLAS.get_num_threads()) BLAS threads and $T")
    t_start = time()
    for n in 0:Nt-1
        apply_dirichlet!(U, n, Nx)
        L = laplacian_1D(U[:, :, n], D2x)

        @views begin
            F[:, 0] .= U[:, 1, n] # f_u = v
            F[:, 1] .= c^2 .* L   # f_v = c^2 * Δu
            U[:, :, n+1] .= U[:, :, n] .+ dt .* F
        end

        if mod(n, 100000) == 0
            @printf("Step %d of %d completed\n", n, Nt)
        end
    end
    apply_dirichlet!(U, Nt, Nx)
    println("Time stepping completed in $(round(time() - t_start, digits=2)) s")

    # --- Output directory ---
    output_dir = "code/8_wave_equation_solvers/animation_versions/output/output1D_$(Nx)"
    isdir(output_dir) || (mkpath(output_dir); println("Created directory $output_dir"))

    # --- Energy computation ---
    energy_kin = zeros(T, Nt+1)
    energy_pot = zeros(T, Nt+1)
    energy_total = zeros(T, Nt+1)

    for n in 0:Nt
        dudx = derivative_1D(U[:, :, n], D1x)
        Ek, Ep = zero(T), zero(T)
        @inbounds for i in 1:Nx
            Δx = xnodes[i] - xnodes[i-1]
            Ek += T(0.5) * U[i, 1, n]^2 * Δx
            Ep += T(0.5) * c^2 * dudx[i]^2 * Δx
        end
        energy_kin[n+1] = Ek
        energy_pot[n+1] = Ep
        energy_total[n+1] = Ek + Ep
    end
    println("Energy evaluation completed")

    # --- Plot energy vs time ---
    t_vals = collect(0:Nt) .* dt
    plot(t_vals, energy_kin, label="Kinetic", xlabel="Time", ylabel="Energy",
         title=@sprintf("1D, %s, dt=%.2e, stencil=%d", string(nodetype), dt, stencil))
    plot!(t_vals, energy_pot, label="Potential", lw=2)
    plot!(t_vals, energy_total, label="Total", lw=2, linestyle=:dash)
    savefig(joinpath(output_dir, "energy.png"))
    println("Energy plot saved to $(joinpath(output_dir, "energy.png"))")

    # --- Field animation with nodal markers ---
    anim = @animate for n in 0:3000:Nt
        p = plot(parent(xnodes), parent(U[:, 0, n]),
                 title=@sprintf("t=%.2f", n * dt),
                 ylim=(-2A, 2A), size=(600, 600),
                 label=string(nodetype))
        scatter!(parent(xnodes), parent(U[:, 0, n]),
                 marker=:circle, markersize=3,
                 label="$(Nx+1) exact nodes",
                 color=:royalblue1)
    end
    gif(anim, joinpath(output_dir, "original.gif"), fps=30)
    println("Animation saved to $(joinpath(output_dir, "original.gif"))")

    # --- Interpolated animation ---
    Nfine = 100
    xfine = range(first(xnodes), last(xnodes), length=Nfine+1)
    ℓx = OffsetArray([evalL(xnodes, i, xfine[k]) for i in 0:Nx, k in 1:Nfine+1], 0:Nx, 1:Nfine+1)

    anim_interp = @animate for n in 0:3000:Nt
        u_fine = zeros(T, Nfine+1)
        @inbounds @threads for k in 1:Nfine+1
            s = zero(T)
            for i in 0:Nx
                s += U[i, 0, n] * ℓx[i, k]
            end
            u_fine[k] = s
        end
        p = plot(xfine, u_fine,
                 title=@sprintf("t=%.2f", n * dt),
                 lw=2, ylim=(-2A, 2A), size=(600, 600),
                 label=string(nodetype))
        scatter!(parent(xnodes), parent(U[:, 0, n]),
                 marker=:circle, markersize=3,
                 label="$(Nx+1) exact nodes",
                 color=:royalblue1)
    end
    gif(anim_interp, joinpath(output_dir, "evaluated$(Nfine).gif"), fps=30)
    println("Interpolated animation saved to $(joinpath(output_dir, "evaluated$(Nfine).gif"))")
end

main(Float32)