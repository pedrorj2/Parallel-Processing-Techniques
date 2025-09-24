# File: 8_wave_equation_solvers/benchmark_versions/wave2D_MxV_benchmark_unified.jl

# These versions are like the original ones with the modular scheme implemetation, but with 1 to N+1 indexing for the spatial dimension.
# The change was made as I encountered issues with CUDA.jl and OffsetArrays.jl compatibility. These could be explored further in the future.
# The recomended versions are in the optimized_versions folder.

using Printf
using Plots
using OffsetArrays
using LinearAlgebra
using BenchmarkTools
using Measures
using Profile
using ProfileView

# Conditional GPU support
using MKL
HAS_CUDA = false
try
    using CUDA
    global HAS_CUDA = CUDA.functional()
catch
    # HAS_CUDA already set to false
end

include("../../7_global_interpolation/D2_matrix.jl")
include("../../7_global_interpolation/D1_matrix.jl")
include("../../utilities/offset.jl")  # Still needed for build functions
include("../../utilities/nodes.jl")
include("../../utilities/temporal_schemes.jl")

# -------------------------------------------------
# Interactive prompts for user input
# -------------------------------------------------

# Choose CPU vs GPU
println("Choose computation backend:")
println("1. CPU")
if HAS_CUDA
    println("2. GPU (CUDA)")
    print("Enter choice (1-2): ")
    backend_choice = parse(Int, readline())
    use_gpu = backend_choice == 2
else
    println("2. GPU (CUDA) - Not available")
    print("Enter choice (1): ")
    backend_choice = parse(Int, readline())
    use_gpu = false
end

if backend_choice < 1 || backend_choice > (HAS_CUDA ? 2 : 1)
    error("Invalid choice.")
end

# Set array type and threading
if use_gpu
    ArrayType = CuArray
    BLAS.set_num_threads(1)  # No need for CPU threading when using GPU
    println("Using GPU computation")
else
    ArrayType = Array
    BLAS.set_num_threads(12)
    println("Using CPU computation with 12 threads")
end

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
allowed_Nx = use_gpu ? [25, 50, 75, 100, 125, 150, 175, 200, 300, 400, 500, 1000, 1500, 2000, 3000] :
             [25, 50, 75, 100, 125, 150, 175, 200, 300, 400, 500]

for (i, nx) in enumerate(allowed_Nx)
    println("$i. Nx = Ny = $nx")
end

print("\nEnter choice (1-$(length(allowed_Nx))): ")
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

backend_str = use_gpu ? "GPU" : "CPU"
println("\nSelected: $T with Nx = Ny = $Nx using $scheme_name scheme on $backend_str")
println("Interpolation stencil size: $(Nx+1) nodes")
if Nx >= 100
    println("Note: Interpolated plots will be skipped for performance (Nx ≥ 100)")
end
println("Starting simulation...")

# -------------------------------------------------
# Helper functions
# -------------------------------------------------

"""
    create_output_directory(Nx, scheme_name, T, use_gpu)
Create organized output directory structure based on parameters.
"""
function create_output_directory(Nx, scheme_name, T, use_gpu)
    base_dir = use_gpu ? "code/8_wave_equation_solvers/scheme_versions/output/wave2D_MxV_GPU" :
               "code/8_wave_equation_solvers/scheme_versions/output/wave2D_MxV_CPU"

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

"""
    save_original_plot_at_time!(U, xnodes, ynodes, time_val, A, Nx, Ny, nodetype, output_dir, scheme_name, use_gpu)
Generate only the original heatmap plot (no interpolation) for performance on large grids.
"""
function save_original_plot_at_time!(U::AbstractArray{T,3}, xnodes, ynodes, time_val::T, A::T, Nx, Ny, nodetype, output_dir, scheme_name, use_gpu) where {T}
    U_cpu = use_gpu ? Array(U) : U
    # Plot state at nodes only
    x_vals = collect(xnodes)
    y_vals = collect(ynodes)
    p = heatmap(x_vals, y_vals, U_cpu[:, :, 1]',
        xlabel="x", ylabel="y",
        title=@sprintf("%s: State at t=%.2f s", scheme_name, time_val),
        clim=(-A, A), color=:inferno, size=(600, 600),
        fontsize=12)

    filename = @sprintf("%s_state_t_%.2f.png", scheme_name, time_val)
    savefig(joinpath(output_dir, filename))
    println("Plot saved: $filename")
end

"""
    save_plots_at_time!(U, xnodes, ynodes, time_val, A, Nx, Ny, nodetype, output_dir, scheme_name, use_gpu)
Generate both original and interpolated heatmap plots for a given time point.
"""
function save_plots_at_time!(U::AbstractArray{T,3}, xnodes, ynodes, time_val::T, A::T, Nx, Ny, nodetype, output_dir, scheme_name, use_gpu) where {T}
    U_cpu = use_gpu ? Array(U) : U
    xnodes_cpu = use_gpu ? Array(xnodes) : xnodes
    ynodes_cpu = use_gpu ? Array(ynodes) : ynodes

    # Plot state at nodes
    x_vals = collect(xnodes_cpu)
    y_vals = collect(ynodes_cpu)
    p = heatmap(x_vals, y_vals, U_cpu[:, :, 1]',
        xlabel="x", ylabel="y",
        title=@sprintf("%s: State at t=%.2f s", scheme_name, time_val),
        clim=(-A, A), color=:inferno, size=(600, 600),
        fontsize=12)

    filename = @sprintf("%s_state_t_%.2f.png", scheme_name, time_val)
    savefig(joinpath(output_dir, filename))
    println("Plot saved: $filename")

    # Interpolated state - Convert back to OffsetArray for evalL
    Nfine = 100
    xfine = range(first(xnodes_cpu), last(xnodes_cpu), length=Nfine + 1)
    yfine = range(first(ynodes_cpu), last(ynodes_cpu), length=Nfine + 1)

    # Convert to OffsetArray temporarily for evalL function
    xnodes_offset = OffsetArray(xnodes_cpu, 0:Nx)
    ynodes_offset = OffsetArray(ynodes_cpu, 0:Ny)

    ℓx = [evalL(xnodes_offset, i, xfine[k]) for i in 0:Nx, k in 1:Nfine+1]  # Use 0-based for OffsetArray
    ℓy = [evalL(ynodes_offset, j, yfine[l]) for j in 0:Ny, l in 1:Nfine+1]  # Use 0-based for OffsetArray

    u_fine = zeros(T, Nfine + 1, Nfine + 1)
    @inbounds for l in 1:Nfine+1
        for k in 1:Nfine+1
            s = zero(T)
            for i in 1:Nx+1, j in 1:Ny+1
                s += U_cpu[i, j, 1] * ℓx[i, k] * ℓy[j, l]  # Use 1-based indexing for regular arrays
            end
            u_fine[k, l] = s
        end
    end

    p = heatmap(xfine, yfine, u_fine',
        clim=(-A, A), color=:inferno,
        xlabel="x", ylabel="y",
        title=@sprintf("%s: Interpolated state at t=%.2f s", scheme_name, time_val),
        size=(600, 600),
        fontsize=12)

    interp_filename = @sprintf("%s_interpolated_state_t_%.2f.png", scheme_name, time_val)
    savefig(joinpath(output_dir, interp_filename))
    println("Plot saved: $interp_filename")
end

"""
    save_simulation_info(output_dir, scheme_name, T, Nx, Ny, dt, Nt, execution_time, nodetype, use_gpu)
Save simulation parameters and performance info to a text file.
"""
function save_simulation_info(output_dir, scheme_name, T, Nx, Ny, dt, Nt, execution_time, nodetype, use_gpu)
    info_file = joinpath(output_dir, "simulation_info.txt")

    scheme_steps = if scheme_name == "Euler"
        1
    elseif scheme_name == "RK2"
        2
    elseif scheme_name == "RK4"
        4
    else
        1
    end

    total_ops = Nt * scheme_steps * 2 * 2 * (Nx + 1)^3
    gflops = total_ops / execution_time / 1e9

    backend_str = use_gpu ? "GPU" : "CPU"

    open(info_file, "w") do f
        println(f, "=== SIMULATION PARAMETERS ===")
        println(f, "Method: MxV (Matrix-Vector) on $backend_str")
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
Set U[:,:,1] = Gaussian pulse of amplitude A, width σ, centered at (x_c,y_c),
and U[:,:,2] = 0 (velocity). Uses vectorized operations for efficiency.
"""
function apply_initial_conditions!(U::AbstractArray{T,3}, xnodes::AbstractArray{T,1}, ynodes::AbstractArray{T,1}, A::T, σ::T, x_c::T, y_c::T, Nx, Ny) where {T}
    xs = reshape(xnodes, :, 1)
    ys = reshape(ynodes, 1, :)
    dist_sq = (xs .- x_c) .^ 2 .+ (ys .- y_c) .^ 2
    U[:, :, 1] .= A .* exp.(-dist_sq ./ (2 * σ^2))
    U[:, :, 2] .= zero(T)
end

"""
    apply_dirichlet!(U, Nx, Ny)
Enforce zero Dirichlet boundary conditions on both fields (u and v) for a 3D state slice.
"""
function apply_dirichlet!(U::AbstractArray{T,3}, Nx, Ny) where {T}
    U[1, :, :] .= zero(T)
    U[end, :, :] .= zero(T)
    U[:, 1, :] .= zero(T)
    U[:, end, :] .= zero(T)
end

"""
    laplacian_2D_MxV(U, D2x, D2y, Nx, Ny)
Compute Laplacian using genuine Matrix-Vector operations per column/row.
"""
function laplacian_2D_MxV(U::AbstractArray{T,3}, D2x::AbstractArray{T,2}, D2y::AbstractArray{T,2}, Nx, Ny) where {T}
    L = similar(U, Nx+1, Ny+1)
    
    @inbounds for k in 1:Ny+1
        L[:, k] .= D2x * U[:,k,1]
    end
    @inbounds for i in 1:Nx+1
        L[i, :] .+= (U[i, :, 1]' * D2y)[:]  # Nota el [:]
    end

    return L
end




"""
    Wave2D(U, t, params)
Right-hand side function for 2D wave equation using MxV operations.
"""
function Wave2D(U::AbstractArray{T,3}, t::T, params, F) where {T}
    # Unpack parameters
    D2x, D2y, c, Nx, Ny = params

    apply_dirichlet!(U, Nx, Ny)

    # Compute Laplacian usando MxV con TODOS los argumentos
    L = laplacian_2D_MxV(U, D2x, D2y, Nx, Ny)  

    # Wave equation: d²u/dt² = c²∇²u
    # As first-order system: du/dt = v, dv/dt = c²∇²u
    F[:, :, 1] .= U[:, :, 2]      # f_u = v
    F[:, :, 2] .= c^2 .* L        # f_v = c²∇²u

    return F
end


# -------------------------------------------------
# Main simulation function
# -------------------------------------------------
function main(T::Type, Nx::Int, Ny::Int, scheme_name::String, scheme_function, use_gpu::Bool)
    # Parameters
    Lx, Ly = T(1.0), T(1.0)
    c = T(1.0)
    dt = T(1e-4) * min(Lx / Nx, Ly / Ny)
    T_time = T(1.0)
    Nt = Int(floor(T_time / dt))
    A = T(1.0)
    σ = T(0.075)
    x_c, y_c = Lx / T(2), Ly / T(2)

    stencil = Nx + 1
    nodetype = :chebyshev

    # Nodes (always build on CPU first, then transfer if needed)
    xnodes_cpu = if nodetype == :chebyshev
        convert.(T, chebyshev_nodes(Nx))
    elseif nodetype == :chebyshev_lobatto
        convert.(T, chebyshev_lobatto_nodes(Nx))
    elseif nodetype == :equispaced
        convert.(T, equispaced_nodes(Nx))
    else
        error("Unsupported node type")
    end
    ynodes_cpu = xnodes_cpu  # Since Ny = Nx, same

    # Convert OffsetArray to regular Array and then to target ArrayType
    xnodes = ArrayType(parent(xnodes_cpu))
    ynodes = ArrayType(parent(ynodes_cpu))

    # Differentiation matrices (build on CPU using OffsetArrays, then convert)
    D2x_cpu = build_D2_matrix(xnodes_cpu, stencil, T)  # This returns OffsetArray
    D2y_cpu = build_D2_matrix(ynodes_cpu, stencil, T)  # This returns OffsetArray
    D2x = ArrayType(parent(D2x_cpu))  # Convert to regular array then to ArrayType
    D2y = ArrayType(parent(D2y_cpu))  # Convert to regular array then to ArrayType
    D2yT = permutedims(D2y)  # permutedims works on both Array and CuArray

    # Pack parameters for the RHS function
    wave_params = (D2x, D2y, c, Nx, Ny)

    # State tensors
    U = ArrayType(zeros(T, Nx + 1, Ny + 1, 2))
    F = similar(U)

    # Create RHS function closure following F(U, t) pattern
    function rhs(U, t)
        return Wave2D(U, t, wave_params, F)
    end

    # Initial & boundary conditions at n=0
    apply_initial_conditions!(U, xnodes, ynodes, A, σ, x_c, y_c, Nx, Ny)
    apply_dirichlet!(U, Nx, Ny)

    # Create organized output directory
    output_dir = create_output_directory(Nx, scheme_name, T, use_gpu)

    # Time stepping with intermediate saves
    backend_str = use_gpu ? "GPU" : "CPU"
    println("Starting 2D MxV benchmark with $(stencil)-node stencil and $nodetype nodes on $backend_str…")
    println("Using $T with Nx = Ny = $Nx and $scheme_name temporal scheme")
    println("Total time steps: $Nt")
    println("Output directory: $output_dir")

    # Define times to save plots
    times_to_save = T[0.0, T_time/3, 2*T_time/3, T_time]
    indices_to_save = floor.(Int, times_to_save ./ dt)

    # Save initial state (t=0) - conditional interpolation
    if Nx < 99
        save_plots_at_time!(U, parent(xnodes_cpu), parent(ynodes_cpu), T(0.0), A, Nx, Ny, nodetype, output_dir, scheme_name, use_gpu)
    else
        save_original_plot_at_time!(U, parent(xnodes_cpu), parent(ynodes_cpu), T(0.0), A, Nx, Ny, nodetype, output_dir, scheme_name, use_gpu)
    end

    t_start = time()

    # Main time integration loop with temporal scheme abstraction
    # Profile.clear()  # Clear previous profile data
    # ProfileView.@profview begin
        for n in 0:Nt-1
            t_current = n * dt

            # Use the selected temporal scheme
            U .= scheme_function(U, dt, t_current, rhs)

            # Check if we need to save plots at this time step
            if (n + 1) in indices_to_save
                time_val = (n + 1) * dt

                # Conditional plotting based on grid size for performance
                if Nx < 99
                    save_plots_at_time!(U, parent(xnodes_cpu), parent(ynodes_cpu), time_val, A, Nx, Ny, nodetype, output_dir, scheme_name, use_gpu)
                else
                    save_original_plot_at_time!(U, parent(xnodes_cpu), parent(ynodes_cpu), time_val, A, Nx, Ny, nodetype, output_dir, scheme_name, use_gpu)
                end
            end
        end
    # end

    execution_time = time() - t_start
    apply_dirichlet!(U, Nx, Ny)

    # Save simulation information
    save_simulation_info(output_dir, scheme_name, T, Nx, Ny, dt, Nt, execution_time, nodetype, use_gpu)

    println("Time stepping completed in $(round(execution_time, digits=2)) s")
    println("All outputs saved to: $output_dir")
end

# -------------------------------------------------
# Execute simulation
# -------------------------------------------------
main(T, Nx, Ny, scheme_name, scheme_function, use_gpu)
