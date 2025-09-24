# File: 8_wave_equation_solvers/benchmark_versions/wave2D_MxM_benchmark.jl

# These versions are like the original ones with the modular scheme implemetation, still with 0 to N indexing for the spatial dimension.
# The recomended versions are in the optimized_versions folder.

using Printf
using Plots
using OffsetArrays
using LinearAlgebra
using BenchmarkTools
using Measures

# using Profile
# using ProfileView

using MKL

BLAS.set_num_threads(12)

include("../../7_global_interpolation/D2_matrix.jl")
include("../../7_global_interpolation/D1_matrix.jl")
include("../../utilities/offset.jl")
include("../../utilities/nodes.jl")
include("../../utilities/temporal_schemes.jl")  # Include our temporal schemes

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
    println("Warning: For large Nx and Ny values (>100), Float32 may cause numerical instability.")
end

println("\nSelect spatial node count Nx=Ny from the following options:")
allowed_Nx = [25, 50, 75, 100, 125, 150, 175, 200, 300, 400, 500]
for (i, nx) in enumerate(allowed_Nx)
    println("$i. Nx = Ny = $nx")
end

print("\nEnter choice (1-11): ")
nx_choice = parse(Int, readline())
if nx_choice < 1 || nx_choice > length(allowed_Nx)
    error("Invalid choice. Please select a number between 1 and $(length(allowed_Nx)).")
end

Nx = allowed_Nx[nx_choice]
Ny = Nx  # Square grid

# Choose temporal scheme
println("\nChoose temporal integration scheme:")
println("1. Euler")
println("2. RK4")
println("3. RK2")
println("4. Euler Semi-Implicit (velocity first)")
print("Enter choice (1-4): ")
scheme_choice = parse(Int, readline())

scheme_name, scheme_function = if scheme_choice == 1
    ("Euler", Euler)
elseif scheme_choice == 2
    ("RK4", RK4)
elseif scheme_choice == 3
    ("RK2", RK2)
elseif scheme_choice == 4
    ("EulerImplicit", EulerImplicit)
else
    error("Invalid choice. Please select 1 (Euler), 2 (RK4), 3 (RK2), or 4 (EulerImplicit).")
end


println("\nSelected: $T with Nx = Ny = $Nx using $scheme_name scheme")
println("Interpolation stencil size: $(Nx+1) nodes")
if Nx >= 100
    println("Note: Interpolated plots will be skipped for performance (Nx ≥ 100)")
end
println("Starting simulation...")

# -------------------------------------------------
# Helper functions
# -------------------------------------------------

"""
    create_output_directory(Nx, scheme_name, T)

Create organized output directory structure based on parameters.
"""
function create_output_directory(Nx, scheme_name, T)
    base_dir = "code/8_wave_equation_solvers/scheme_versions/output/wave2D_MxM"

    # Create hierarchical directory structure
    precision_str = T == Float32 ? "Float32" : "Float64"
    output_dir = joinpath(base_dir,
        scheme_name,
        "$(Nx)x$(Nx)_nodes",
        precision_str)

    if !isdir(output_dir)
        mkpath(output_dir)
        println("Created directory: $output_dir")
    end

    return output_dir
end

# """
#     save_original_plot_at_time!(U, xnodes, ynodes, time_val, A, Nx, Ny, nodetype, output_dir, scheme_name)

# Generate only the original heatmap plot (no interpolation) for performance on large grids.
# """
# function save_original_plot_at_time!(U::OffsetArray{T,3}, xnodes, ynodes, time_val::T, A::T, Nx, Ny, nodetype, output_dir, scheme_name) where {T}
#     # Plot state at nodes only
#     x_vals = collect(xnodes)
#     y_vals = collect(ynodes)
#     p = heatmap(x_vals, y_vals, U[:, :, 0]',
#                 xlabel="x", ylabel="y",
#                 title=@sprintf("%s: State at t=%.2f s", scheme_name, time_val),
#                 clim=(-A, A), color=:inferno, size=(600, 600),
#                 fontsize=12)

#     filename = @sprintf("%s_state_t_%.2f.png", scheme_name, time_val)
#     savefig(joinpath(output_dir, filename))
#     println("Plot saved: $filename")
# end

# """
#     save_plots_at_time!(U, xnodes, ynodes, time_val, A, Nx, Ny, nodetype, output_dir, scheme_name)

# Generate both original and interpolated heatmap plots for a given time point.
# Now includes scheme name in filenames.
# """
# function save_plots_at_time!(U::OffsetArray{T,3}, xnodes, ynodes, time_val::T, A::T, Nx, Ny, nodetype, output_dir, scheme_name) where {T}
#     # Plot state at nodes
#     x_vals = collect(xnodes)
#     y_vals = collect(ynodes)
#     p = heatmap(x_vals, y_vals, U[:, :, 0]',
#                 xlabel="x", ylabel="y",
#                 title=@sprintf("%s: State at t=%.2f s", scheme_name, time_val),
#                 clim=(-A, A), color=:inferno, size=(600, 600),
#                 fontsize=12)

#     filename = @sprintf("%s_state_t_%.2f.png", scheme_name, time_val)
#     savefig(joinpath(output_dir, filename))
#     println("Plot saved: $filename")

#     # Interpolated state
#     Nfine = 100
#     xfine = range(first(xnodes), last(xnodes), length=Nfine+1)
#     yfine = range(first(ynodes), last(ynodes), length=Nfine+1)
#     ℓx = OffsetArray([evalL(xnodes, i, xfine[k]) for i in 0:Nx, k in 1:Nfine+1], 0:Nx, 1:Nfine+1)
#     ℓy = OffsetArray([evalL(ynodes, j, yfine[l]) for j in 0:Ny, l in 1:Nfine+1], 0:Ny, 1:Nfine+1)

#     u_fine = zeros(T, Nfine+1, Nfine+1)
#     @inbounds for l in 1:Nfine+1
#         for k in 1:Nfine+1
#             s = zero(T)
#             for i in 0:Nx, j in 0:Ny
#                 s += U[i, j, 0] * ℓx[i, k] * ℓy[j, l]
#             end
#             u_fine[k, l] = s
#         end
#     end

#     p = heatmap(xfine, yfine, u_fine',
#                 clim=(-A, A), color=:inferno,
#                 xlabel="x", ylabel="y",
#                 title=@sprintf("%s: Interpolated state at t=%.2f s", scheme_name, time_val),
#                 size=(600, 600),
#                 fontsize=12)

#     interp_filename = @sprintf("%s_interpolated_state_t_%.2f.png", scheme_name, time_val)
#     savefig(joinpath(output_dir, interp_filename))
#     println("Plot saved: $interp_filename")
# end

"""
    save_simulation_info(output_dir, scheme_name, T, Nx, Ny, dt, Nt, execution_time, nodetype)

Save simulation parameters and performance info to a text file.
"""
function save_simulation_info(output_dir, scheme_name, T, Nx, Ny, dt, Nt, execution_time, nodetype)
    info_file = joinpath(output_dir, "simulation_info.txt")

    # Calculate GFLOPs based on scheme
    scheme_steps = if scheme_name == "Euler"
        1
    elseif scheme_name == "RK2"
        2
    elseif scheme_name == "RK4"
        4
    else
        1  # default fallback
    end

    total_ops = Nt * scheme_steps * 2 * 2 * (Nx + 1)^3
    gflops = total_ops / execution_time / 1e9  # Convert to GFLOPs

    open(info_file, "w") do f
        println(f, "=== SIMULATION PARAMETERS ===")
        println(f, "Method: MxM (Matrix-Matrix)")
        println(f, "Temporal Scheme: $scheme_name")
        println(f, "Precision: $T")
        println(f, "Grid Size: $(Nx) x $(Ny)")
        println(f, "Time Step: $dt")
        println(f, "Total Steps: $Nt")
        println(f, "Total Time: $(Nt * dt)")
        println(f, "Execution Time: $(round(execution_time, digits=2)) seconds")
        println(f, "GFLOPs: $(round(gflops, digits=2))")
        println(f, "Total Operations: $(total_ops)")
        println(f, "Stencil Size: $(Nx+1) nodes")
        println(f, "Node Type: $nodetype")
    end

    println("Simulation info saved to: $info_file")
end

"""
    apply_initial_conditions!(U, xnodes, ynodes, A, σ, x_c, y_c, Nx, Ny)

Set U[i,j,0] = Gaussian pulse of amplitude A, width σ, centered at (x_c,y_c),
and U[i,j,1] = 0 (velocity).
"""
function apply_initial_conditions!(U::OffsetArray{T,3}, xnodes, ynodes, A::T, σ::T, x_c::T, y_c::T, Nx, Ny) where {T}
    @inbounds for j in 0:Ny
        for i in 0:Nx
            U[i, j, 0] = A * exp(-((xnodes[i] - x_c)^2 + (ynodes[j] - y_c)^2) / (2 * σ^2))
            U[i, j, 1] = zero(T)
        end
    end
end

"""
    apply_dirichlet!(U, Nx, Ny)

Enforce zero Dirichlet boundary conditions on both fields (u and v) for a 3D state slice.
"""
function apply_dirichlet!(U::OffsetArray{T,3}, Nx, Ny) where {T}
    @inbounds for j in 0:Ny
        U[0, j, 0] = zero(T)
        U[Nx, j, 0] = zero(T)
        U[0, j, 1] = zero(T)
        U[Nx, j, 1] = zero(T)
    end
    @inbounds for i in 0:Nx
        U[i, 0, 0] = zero(T)
        U[i, Ny, 0] = zero(T)
        U[i, 0, 1] = zero(T)
        U[i, Ny, 1] = zero(T)
    end
end

"""
    laplacian_2D(U, D2x, D2yT)

Compute Laplacian of the position field (U[:,:,0]) for 3D arrays.
"""
function laplacian_2D(U::OffsetArray{T,3}, D2x::OffsetArray{T,2}, D2yT::OffsetArray{T,2}) where {T}
    return D2x * U[:, :, 0] .+ U[:, :, 0] * D2yT
end

"""
    Wave2D(U, t, params)

Right-hand side function for 2D wave equation following F(U,t) pattern.
Returns dU/dt where U contains the state [u, v] fields.
"""
function Wave2D(U::OffsetArray{T,3}, t::T, params, F) where {T}
    # Unpack parameters
    D2x, D2yT, c, Nx, Ny = params

    apply_dirichlet!(U, Nx, Ny)

    # Compute Laplacian
    L = laplacian_2D(U, D2x, D2yT)

    # Wave equation: d²u/dt² = c²∇²u
    # As first-order system: du/dt = v, dv/dt = c²∇²u
    @views begin
        F[:, :, 0] .= U[:, :, 1]      # f_u = v
        F[:, :, 1] .= c^2 .* L        # f_v = c²∇²u
    end

    return F
end

# -------------------------------------------------
# Main simulation function
# -------------------------------------------------
function main(T::Type, Nx::Int, Ny::Int, scheme_name::String, scheme_function)
    # Parameters
    Lx, Ly = T(1.0), T(1.0)
    c = T(1.0)
    dt = T(1e-2) * min(Lx / Nx, Ly / Ny)
    T_time = T(1.0)
    Nt = Int(floor(T_time / dt))
    A = T(1.0)
    σ = T(0.075)
    x_c, y_c = Lx / T(2), Ly / T(2)

    stencil = Nx + 1  # Global interpolation stencil
    nodetype = :chebyshev # :equispaced, :chebyshev, :chebyshev_lobatto

    # Nodes
    xnodes = nodetype == :chebyshev ? convert.(T, chebyshev_nodes(Nx)) :
             nodetype == :chebyshev_lobatto ? convert.(T, chebyshev_lobatto_nodes(Nx)) :
             nodetype == :equispaced ? convert.(T, equispaced_nodes(Nx)) :
             error("Unsupported node type")
    ynodes = nodetype == :chebyshev ? convert.(T, chebyshev_nodes(Ny)) :
             nodetype == :chebyshev_lobatto ? convert.(T, chebyshev_lobatto_nodes(Ny)) :
             nodetype == :equispaced ? convert.(T, equispaced_nodes(Ny)) :
             error("Unsupported node type")

    # Differentiation matrices
    D2x = OffsetArray(build_D2_matrix(xnodes, stencil, T), 0:Nx, 0:Nx)
    D2y = OffsetArray(build_D2_matrix(ynodes, stencil, T), 0:Ny, 0:Ny)
    D2yT = OffsetArray(permutedims(parent(D2y)), 0:Ny, 0:Ny)

    # Pack parameters for the RHS function
    wave_params = (D2x, D2yT, c, Nx, Ny)

    # State tensors
    U = OffsetArray(zeros(T, Nx + 1, Ny + 1, 2), 0:Nx, 0:Ny, 0:1)
    # Create output array with same structure
    F = similar(U)

    # Create RHS function closure following F(U, t) pattern
    function rhs(U, t)
        return Wave2D(U, t, wave_params, F)
    end


    # Initial & boundary conditions at n=0
    apply_initial_conditions!(U, xnodes, ynodes, A, σ, x_c, y_c, Nx, Ny)
    apply_dirichlet!(U, Nx, Ny)

    # Create organized output directory
    output_dir = create_output_directory(Nx, scheme_name, T)

    # Time stepping with intermediate saves
    println("Starting 2D MxM benchmark with $(stencil)-node stencil and $nodetype nodes…")
    println("Using $T with Nx = Ny = $Nx and $scheme_name temporal scheme")
    println("Total time steps: $Nt")
    println("Output directory: $output_dir")

    # # Define times to save plots
    # times_to_save = T[0.0, 0.33, 0.66, 1.0]
    # indices_to_save = floor.(Int, times_to_save ./ dt)

    # # Save initial state (t=0) - conditional interpolation
    # if Nx < 99
    #     save_plots_at_time!(U, xnodes, ynodes, T(0.0), A, Nx, Ny, nodetype, output_dir, scheme_name)
    # else
    #     save_original_plot_at_time!(U, xnodes, ynodes, T(0.0), A, Nx, Ny, nodetype, output_dir, scheme_name)
    # end

    # Main time integration loop with temporal scheme abstraction
    t_start = time()
    # Profile.clear()  # Clear previous profile data
    # ProfileView.@profview begin  # Qualified with ProfileView to resolve ambiguity
        for n in 0:Nt-1
            t_current = n * dt

            # Use the selected temporal scheme
            U .= scheme_function(U, dt, t_current, rhs)

            # # Check if we need to save plots at this time step
            # if (n + 1) in indices_to_save
            #     time_val = (n + 1) * dt

            #     # Conditional plotting based on grid size for performance
            #     if Nx < 99
            #         save_plots_at_time!(U, xnodes, ynodes, time_val, A, Nx, Ny, nodetype, output_dir, scheme_name)
            #     else
            #         save_original_plot_at_time!(U, xnodes, ynodes, time_val, A, Nx, Ny, nodetype, output_dir, scheme_name)
            #     end
            # end
        end
    # end

    execution_time = time() - t_start
    apply_dirichlet!(U, Nx, Ny)

    # Save simulation information
    save_simulation_info(output_dir, scheme_name, T, Nx, Ny, dt, Nt, execution_time, nodetype)

    println("Time stepping completed in $(round(execution_time, digits=2)) s")
    println("All outputs saved to: $output_dir")
end

# -------------------------------------------------
# Execute simulation
# -------------------------------------------------
main(T, Nx, Ny, scheme_name, scheme_function)