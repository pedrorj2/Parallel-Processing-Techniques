# File: code/6_state_formulation/version1/version1_1D.jl

using Printf
using Plots
using OffsetArrays

# -------------------------
# Simulation parameters
# -------------------------
Nx = 100                 # spatial points (indices 0..Nx)
L  = 1.0                 # string length
dx = L / Nx              # spatial step
c  = 1.0                 # wave speed

dt = 0.0005 * dx          # time step
T  = 10.0                 # total simulated time
Nt = Int(floor(T / dt))  # time steps (indices 0..Nt)

# Gaussian initial pulse
A     = 1.0              # amplitude
sigma = 0.075            # width
x_c   = L / 2            # center

# -------------------------
# State arrays: u(x,t) and v(x,t)
# OffsetArray lets us index from 0
# -------------------------
u = OffsetArray(zeros(Nx+1, Nt+1), 0:Nx, 0:Nt)   # displacement
v = OffsetArray(zeros(Nx+1, Nt+1), 0:Nx, 0:Nt)   # velocity

# -------------------------------------------------
# Initialization
# -------------------------------------------------
for i in 0:Nx
    x = i * dx
    u[i, 0] = A * exp(-((x - x_c)^2) / (2*sigma^2))
    v[i, 0] = 0.0
end

# -------------------------
# Explicit Euler time stepping
# -------------------------
t_ini = time()

for n in 0:Nt-1

    # Dirichlet boundaries (u = v = 0)
    u[0,  n] = 0;  u[Nx, n] = 0
    v[0,  n] = 0;  v[Nx, n] = 0

    # Interior points (1:Nx-1)
    for i in 1:Nx-1
        u[i, n+1] = u[i, n] + dt * v[i, n]
        v[i, n+1] = v[i, n] + dt * c^2 *
                    (u[i+1, n] - 2u[i, n] + u[i-1, n]) / dx^2 # centered, 2nd order
    end

    if mod(n, 10000) == 0
        @printf("Step %d of %d completed\n", n, Nt)
    end
end

t_fin1 = time()
println("Time stepping completed")

# -------------------------
# Energy in time
# -------------------------
energy_kin   = zeros(Nt+1)
energy_pot   = zeros(Nt+1)
energy_total = zeros(Nt+1)

for n in 0:Nt
    Ek = 0.0
    Ep = 0.0
    for i in 1:Nx-1
        Ek += 0.5 * v[i, n]^2 * dx
        dudx = (u[i+1, n] - u[i-1, n]) / (2*dx)   # centered, 2nd‑order
        # dudx = (u[i, n] - u[i-1, n]) / dx     # forward, 1st‑order
        # dudx = (u[i+1, n] - u[i, n]) / dx     # backward, 1st‑order
        Ep += 0.5 * c^2 * dudx^2 * dx
    end
    energy_kin[n+1]   = Ek
    energy_pot[n+1]   = Ep
    energy_total[n+1] = Ek + Ep
end

t_fin2 = time()
println("Energy evaluation completed")

# -------------------------
# Plot energy vs. time
# -------------------------
t_vals = collect(0:Nt) .* dt

plot(t_vals, energy_kin,   label="Kinetic",
     xlabel="Time", ylabel="Energy",
     title="1‑D wave, dt = $dt (explicit Euler)", lw=2)
plot!(t_vals, energy_pot,   label="Potential", lw=2)
plot!(t_vals, energy_total, label="Total",     lw=2, linestyle=:dash)

savefig("code/6_state_formulation/version1/version1_energy.png")
println("Energy plot saved")

t_fin3 = time()

# -------------------------
# Animation of the field
# -------------------------
x_vals = collect(0:Nx) .* dx
anim = @animate for n in 0:1000:Nt
    plot(x_vals, collect(u[:, n]),
         xlabel="x", ylabel="u",
         title="t = $(round(n*dt, digits=2))",
         lw=2, legend=false, ylim=(-A, A))
end
gif(anim, "code/6_state_formulation/version1/version1_animation.gif", fps=60)

t_fin4 = time()

# -------------------------
# Save timing info
# -------------------------
open("code/6_state_formulation/version1/version1_timings.txt", "w") do file
    write(file, "1‑D wave (explicit Euler) timings\n")
    write(file, @sprintf("Time stepping: %.3f s\n", t_fin1 - t_ini))
    write(file, @sprintf("Energy calc:   %.3f s\n", t_fin2 - t_fin1))
    write(file, @sprintf("Energy plot:   %.3f s\n", t_fin3 - t_fin2))
    write(file, @sprintf("GIF render:    %.3f s\n", t_fin4 - t_fin3))
    write(file, @sprintf("Total:         %.3f s\n", t_fin4 - t_ini))
end
println("Timings saved")
