# File: code/6_state_formulation/version3/version3_1D.jl

using Printf
using Plots
using OffsetArrays

function apply_initial_conditions(u, v, Nx, dx, A, sigma, x_c)
    for i in 0:Nx
        x = i * dx
        u[i, 0] = A * exp(-(x - x_c)^2 / (2 * sigma^2))
        v[i + (Nx+1), 0] = 0.0
    end
    return u, v
end

function laplacian_1D(u, Nx, dx)
    L = OffsetArray(zeros(Nx + 1), 0:Nx)
    for i in 1:Nx-1
        L[i] = (u[i+1] - 2u[i] + u[i-1]) / dx^2
    end
    return L
end

function derivative_1D(u, Nx, dx)
    dudx = OffsetArray(zeros(Nx + 1), 0:Nx)
    for i in 1:Nx-1
        dudx[i] = (u[i+1] - u[i-1]) / (2 * dx)
    end
    return dudx
end

function apply_dirichlet(u, v, n, Nx)
    u[0, n] = 0.0
    u[Nx, n] = 0.0
    v[Nx+1, n] = 0.0
    v[2Nx+1, n] = 0.0
    return u, v
end

function main()
    # -------------------------
    # Simulation parameters
    # -------------------------
    Nx = 100                 # spatial points (indices 0..Nx)
    L = 1.0                  # string length
    dx = L / Nx              # spatial step
    c = 1.0                  # wave speed

    dt = 0.0005 * dx         # time step
    T = 10.0                 # total simulated time
    Nt = Int(floor(T / dt))  # time steps (indices 0..Nt)

    # Gaussian initial pulse
    A = 1.0                  # amplitude
    sigma = 0.075            # width
    x_c = L / 2              # center

    # -------------------------
    # State tensor U: concatenated u and v
    #   - u: rows 0..Nx
    #   - v: rows Nx+1..2Nx+1
    # -------------------------
    N_total = 2 * (Nx + 1)
    U = OffsetArray(zeros(N_total, Nt+1), 0:(N_total-1), 0:Nt)
    u = OffsetArray(@view(U[0:Nx, :]), (0:Nx, 0:Nt))
    v = OffsetArray(@view(U[Nx+1:2Nx+1, :]), (Nx+1:2Nx+1, 0:Nt))

    # -------------------------------------------------
    # Initialization
    # -------------------------------------------------
    u, v = apply_initial_conditions(u, v, Nx, dx, A, sigma, x_c)
    u, v = apply_dirichlet(u, v, 0, Nx)

    # -------------------------
    # Explicit Euler time stepping
    # -------------------------
    t_ini = time()

    for n in 0:Nt-1
        u, v = apply_dirichlet(u, v, n, Nx)
        L_vec = laplacian_1D(u[:, n], Nx, dx)

        for i in 1:Nx-1
            u[i, n+1] = u[i, n] + dt * v[i + (Nx+1), n]
            v[i + (Nx+1), n+1] = v[i + (Nx+1), n] + dt * c^2 * L_vec[i]
        end

        if mod(n, 10000) == 0
            @printf("Step %d of %d completed\n", n, Nt)
        end
    end

    t_fin1 = time()
    println("Time stepping completed")

    # Energy
    energy_kin = zeros(Nt + 1)
    energy_pot = zeros(Nt + 1)
    energy_total = zeros(Nt + 1)

    for n in 0:Nt
        Ek = 0.0
        Ep = 0.0
        dudx = derivative_1D(u[:, n], Nx, dx)
        for i in 1:Nx-1
            Ek += 0.5 * v[i + (Nx+1), n]^2 * dx
            Ep += 0.5 * c^2 * dudx[i]^2 * dx
        end
        energy_kin[n+1] = Ek
        energy_pot[n+1] = Ep
        energy_total[n+1] = Ek + Ep
    end

    t_fin2 = time()
    println("Energy evaluation completed")

    # Energy plot
    t_vals = collect(0:Nt) .* dt
    plot(t_vals, energy_kin, label="Kinetic", xlabel="Time", ylabel="Energy",
        title="1‑D wave v3, dt = $dt", lw=2)
    plot!(t_vals, energy_pot, label="Potential", lw=2)
    plot!(t_vals, energy_total, label="Total", lw=2, linestyle=:dash)

    savefig("code/6_state_formulation/version3/version3_energy.png")
    println("Energy plot saved")

    t_fin3 = time()

    # Animation
    x_vals = collect(0:Nx) .* dx
    anim = @animate for n in 0:1000:Nt
        plot(x_vals, collect(u[:, n]), xlabel="x", ylabel="u",
            title="t = $(round(n*dt, digits=2))", lw=2, legend=false,
            ylim=(-A, A))
    end
    gif(anim, "code/6_state_formulation/version3/version3_animation.gif", fps=30)

    t_fin4 = time()

    # Timing info
    open("code/6_state_formulation/version3/version3_timings.txt", "w") do file
        write(file, "1‑D wave – version 3 (state tensor U)\n")
        write(file, @sprintf("Time stepping: %.3f s\n", t_fin1 - t_ini))
        write(file, @sprintf("Energy calc:   %.3f s\n", t_fin2 - t_fin1))
        write(file, @sprintf("Energy plot:   %.3f s\n", t_fin3 - t_fin2))
        write(file, @sprintf("GIF render:    %.3f s\n", t_fin4 - t_fin3))
        write(file, @sprintf("Total:         %.3f s\n", t_fin4 - t_ini))
    end
    println("Timings saved")
end

main()
