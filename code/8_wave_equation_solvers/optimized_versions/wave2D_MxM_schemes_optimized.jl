using Printf
using Plots
using OffsetArrays
using LinearAlgebra
using BenchmarkTools
using Measures
using DelimitedFiles

# This is the recommended version.

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
include("../../utilities/offset.jl")
include("../../utilities/nodes.jl")
include("../../utilities/temporal_schemes.jl")  # Usar el archivo optimizado

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
    BLAS.set_num_threads(1)
    println("Using GPU computation")
else
    ArrayType = Array
    BLAS.set_num_threads(12)
    println("Using CPU computation with 12 threads")
end

# println("Choose floating-point type:")
# println("1. Float32")
# println("2. Float64")
# choice = parse(Int, readline())
# if choice == 1
#     T = Float32
# elseif choice == 2
#     T = Float64
# else
#     error("Invalid choice. Please select 1 (Float32) or 2 (Float64).")
# end

# if T == Float32
#     println("Warning: For large Nx and Ny values (>100), Float32 may cause numerical instability.")
# end

T = Float32

println("\nSelect spatial node count Nx=Ny from the following options:")
allowed_Nx = [25, 50, 75, 100, 200, 300, 400, 500, 600, 700, 1000]


for (i, nx) in enumerate(allowed_Nx)
    println("$i. Nx = Ny = $nx")
end

print("\nEnter choice (1-$(length(allowed_Nx))): ")
nx_choice = parse(Int, readline())
if nx_choice < 1 || nx_choice > length(allowed_Nx)
    error("Invalid choice. Please select a number between 1 and $(length(allowed_Nx)).")
end

Nx = allowed_Nx[nx_choice]
Ny = Nx

# Choose temporal scheme
println("\nChoose temporal integration scheme:")
println("1. Euler")
println("2. RK4")
print("Enter choice (1-2): ")
scheme_choice = parse(Int, readline())

scheme_name, scheme_function = if scheme_choice == 1
    ("Euler", Euler_optimized)
elseif scheme_choice == 2
    ("RK4", RK4_optimized)
else
    error("Invalid choice. Please select 1 (Euler) or 2 (RK4).")
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

function save_matrix_csv(matrix, filename)
    writedlm(filename, matrix, ',')
    println("Saved CSV: $(filename)")
end

function save_step_matrices(D2x_cpu, D2x, U, u_fine, output_dir, suffix)
    save_matrix_csv(parent(D2x_cpu), joinpath(output_dir, "D2x_cpu_$suffix.csv"))
    save_matrix_csv(D2x, joinpath(output_dir, "D2x_$suffix.csv"))
    save_matrix_csv(U[:, :, 1], joinpath(output_dir, "U_$suffix.csv"))
    if u_fine !== nothing
        save_matrix_csv(u_fine, joinpath(output_dir, "u_fine_$suffix.csv"))
    end
end



"""
    create_output_directory(Nx, scheme_name, T, use_gpu)
Create organized output directory structure based on parameters.
"""
function create_output_directory(Nx, scheme_name, T, use_gpu)
    base_dir = use_gpu ? "code/8_wave_equation_solvers/optimized_versions/output/wave2D_MxM_GPU" :
               "code/8_wave_equation_solvers/optimized_versions/output/wave2D_MxM_CPU"

    precision_str = T == Float32 ? "Float32" : "Float64"
    output_dir = joinpath(base_dir, scheme_name, "$(Nx)x$(Nx)_nodes", precision_str)

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
    x_vals = collect(use_gpu ? Array(xnodes) : xnodes)
    y_vals = collect(use_gpu ? Array(ynodes) : ynodes)

    p = heatmap(x_vals, y_vals, U_cpu[:, :, 1]',
        xlabel="x", ylabel="y",
        title=@sprintf("%s: State at t=%.2f s", scheme_name, time_val),
        clim=(-A, A), color=:inferno, size=(600, 600),
        fontsize=12)
    filename = @sprintf("%s_state_t_%.2f.png", scheme_name, time_val)
    savefig(joinpath(output_dir, filename))
    println("Plot saved: $filename")
end

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

    # Interpolated state
    Nfine = 100
    xfine = range(first(xnodes_cpu), last(xnodes_cpu), length=Nfine + 1)
    yfine = range(first(ynodes_cpu), last(ynodes_cpu), length=Nfine + 1)
    xnodes_offset = OffsetArray(xnodes_cpu, 0:Nx)
    ynodes_offset = OffsetArray(ynodes_cpu, 0:Ny)
    ℓx = [evalL(xnodes_offset, i, xfine[k]) for i in 0:Nx, k in 1:Nfine+1]
    ℓy = [evalL(ynodes_offset, j, yfine[l]) for j in 0:Ny, l in 1:Nfine+1]

    u_fine = zeros(T, Nfine + 1, Nfine + 1)
    @inbounds for l in 1:Nfine+1
        for k in 1:Nfine+1
            s = zero(T)
            for i in 1:Nx+1, j in 1:Ny+1
                s += U_cpu[i, j, 1] * ℓx[i, k] * ℓy[j, l]
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

    return u_fine
end


"""
    save_simulation_info(output_dir, scheme_name, T, Nx, Ny, dt, Nt, execution_time, nodetype, use_gpu)
Save simulation parameters and performance info to a text file.
"""
function save_simulation_info(output_dir, scheme_name, T, Nx, Ny, dt, Nt, execution_time, nodetype, use_gpu)
    info_file = joinpath(output_dir, "simulation_info.txt")

    scheme_steps = if scheme_name == "Euler"
        1
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
        println(f, "Method: MxM (Matrix-Matrix) OPTIMIZED on $backend_str")
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

# -------------------------------------------------
# CPU-Optimized Core functions 
# -------------------------------------------------

"""
    apply_initial_conditions_cpu!(U, xnodes, ynodes, A, σ, x_c, y_c, Nx, Ny)
CPU-optimized initial conditions with manual loops to avoid broadcasting overhead.
"""
function apply_initial_conditions_cpu!(U::AbstractArray{T,3}, xnodes::AbstractArray{T,1}, ynodes::AbstractArray{T,1}, A::T, σ::T, x_c::T, y_c::T, Nx, Ny) where {T}
    inv_2sigma2 = T(1) / (T(2) * σ^2)

    @inbounds for j in 1:Ny+1, i in 1:Nx+1
        dx = xnodes[i] - x_c
        dy = ynodes[j] - y_c
        dist_sq = dx * dx + dy * dy
        U[i, j, 1] = A * exp(-dist_sq * inv_2sigma2)
        U[i, j, 2] = zero(T)
    end
end

"""
    apply_dirichlet_cpu!(U, Nx, Ny)
CPU-optimized Dirichlet boundary conditions with @inbounds.
"""
function apply_dirichlet_cpu!(U::AbstractArray{T,3}, Nx, Ny) where {T}
    @inbounds begin
        # Borders in x direction
        for j in 1:Ny+1, k in 1:2
            U[1, j, k] = zero(T)
            U[Nx+1, j, k] = zero(T)
        end
        # Borders in y direction  
        for i in 1:Nx+1, k in 1:2
            U[i, 1, k] = zero(T)
            U[i, Ny+1, k] = zero(T)
        end
    end
end

# -------------------------------------------------
# GPU-Optimized Core functions 
# -------------------------------------------------

"""
    apply_initial_conditions_gpu!(U, xnodes, ynodes, A, σ, x_c, y_c, Nx, Ny)
GPU-optimized initial conditions using broadcasting instead of scalar indexing.
"""
function apply_initial_conditions_gpu!(U::AbstractArray{T,3}, xnodes::AbstractArray{T,1}, ynodes::AbstractArray{T,1}, A::T, σ::T, x_c::T, y_c::T, Nx, Ny) where {T}
    # Use broadcasting instead of scalar loops
    xs = reshape(xnodes, :, 1)  # GPU-friendly reshape
    ys = reshape(ynodes, 1, :)  # GPU-friendly reshape

    # Vectorized operations (GPU-optimized)
    dist_sq = (xs .- x_c) .^ 2 .+ (ys .- y_c) .^ 2
    U[:, :, 1] .= A .* exp.(-dist_sq ./ (2 * σ^2))
    U[:, :, 2] .= zero(T)
end

"""
    apply_dirichlet_gpu!(U, Nx, Ny)
GPU-optimized Dirichlet boundary conditions using broadcasting.
"""
function apply_dirichlet_gpu!(U::AbstractArray{T,3}, Nx, Ny) where {T}
    # Use broadcasting instead of scalar loops
    U[1, :, :] .= zero(T)      # First row
    U[end, :, :] .= zero(T)    # Last row  
    U[:, 1, :] .= zero(T)      # First column
    U[:, end, :] .= zero(T)    # Last column
end

# -------------------------------------------------
# Shared optimized functions
# -------------------------------------------------

"""
    laplacian_2D!(L, U, D2x, D2yT, temp)
Optimized in-place Laplacian computation using BLAS operations.
Works for both CPU and GPU.
"""
function laplacian_2D!(L::AbstractArray{T,2}, U::AbstractArray{T,3}, D2x::AbstractArray{T,2}, D2yT::AbstractArray{T,2}, temp::AbstractArray{T,2}) where {T}
    # Use BLAS optimized matrix multiplication in-place
    mul!(temp, D2x, view(U, :, :, 1))  # temp = D2x * U[:,:,1]
    mul!(L, view(U, :, :, 1), D2yT)    # L = U[:,:,1] * D2yT

    # Add in-place: L += temp
    L .+= temp

    return L
end

"""
    Wave2D_cpu_optimized!(F, U, t, params, temp_arrays)
CPU-optimized RHS function with pre-allocated temporaries and in-place operations.
"""
function Wave2D_cpu_optimized!(F::AbstractArray{T,3}, U::AbstractArray{T,3}, t::T, params, temp_arrays) where {T}
    # Unpack parameters
    D2x, D2yT, c, Nx, Ny = params
    L, temp = temp_arrays

    apply_dirichlet_cpu!(U, Nx, Ny)

    # Compute Laplacian in-place
    laplacian_2D!(L, U, D2x, D2yT, temp)

    # Wave equation: du/dt = v, dv/dt = c²∇²u
    # Use optimized loops instead of broadcasting
    c_squared = c * c
    @inbounds for j in 1:Ny+1, i in 1:Nx+1
        F[i, j, 1] = U[i, j, 2]           # f_u = v
        F[i, j, 2] = c_squared * L[i, j]   # f_v = c²∇²u
    end

    return F
end

"""
    Wave2D_gpu_optimized!(F, U, t, params, temp_arrays)
GPU-optimized RHS function using broadcasting for scalar operations.
"""
function Wave2D_gpu_optimized!(F::AbstractArray{T,3}, U::AbstractArray{T,3}, t::T, params, temp_arrays) where {T}
    # Unpack parameters
    D2x, D2yT, c, Nx, Ny = params
    L, temp = temp_arrays

    apply_dirichlet_gpu!(U, Nx, Ny)  # Use GPU version

    # Compute Laplacian in-place (BLAS operations work fine on GPU)
    laplacian_2D!(L, U, D2x, D2yT, temp)

    # Wave equation using broadcasting (GPU-friendly)
    c_squared = c * c
    F[:, :, 1] .= U[:, :, 2]           # Broadcasting assignment
    F[:, :, 2] .= c_squared .* L       # Broadcasting multiplication

    return F
end

# -------------------------------------------------
# Main simulation function - COMPLETAMENTE OPTIMIZADA
# -------------------------------------------------

function main(T::Type, Nx::Int, Ny::Int, scheme_name::String, scheme_function, use_gpu::Bool)
    # Parameters
    Lx, Ly = T(1.0), T(1.0)
    c = T(1.0)
    A = T(1.0)
    σ = T(0.075)
    x_c, y_c = Lx / T(2), Ly / T(2)
    T_time = T(1.0)

    # ✅ CÁLCULO DEFINITIVAMENTE ROBUSTO DE dt Y Nt
    # Usar Float64 SIEMPRE para cálculos de tiempo para evitar overflow
    dx = Float64(min(Lx / Nx, Ly / Ny))
    dt_factor = 1e-2
    dt_working = dt_factor * dx
    Nt_working = Int(round(Float64(T_time) / dt_working))

    # Variables para el solver
    dt = T(dt_working)
    Nt = Nt_working

    println("Calculated dt: $dt")

    println("Steps: $Nt")

    # Verificar que no hay NaN
    if !all(isfinite.([dt_working, Float64(T_time), dx]))
        error("NaN or Inf detected in time parameters!")
    end

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
    ynodes_cpu = xnodes_cpu

    # Convert OffsetArray to regular Array and then to target ArrayType
    xnodes = ArrayType(parent(xnodes_cpu))
    ynodes = ArrayType(parent(ynodes_cpu))

    tstart = time_ns()

    D2x_cpu64 = build_D2_matrix(xnodes_cpu, stencil, Float64)
    D2y_cpu64 = build_D2_matrix(ynodes_cpu, stencil, Float64)

    tend = time_ns()
    println("D2 matrices (Float64) built in $((tend - tstart) / 1.0e9) seconds")

    D2x_cpu = convert(Array{T,2}, parent(D2x_cpu64))
    D2y_cpu = convert(Array{T,2}, parent(D2y_cpu64))


    D2x = ArrayType(parent(D2x_cpu))
    D2y = ArrayType(parent(D2y_cpu))
    D2yT = permutedims(D2y)

    # Pack parameters for the RHS function
    wave_params = (D2x, D2yT, c, Nx, Ny)

    # State tensors
    U = ArrayType(zeros(T, Nx + 1, Ny + 1, 2))
    F = similar(U)

    # PRE-ALLOCATE temporary arrays for optimization (CLAVE PARA PERFORMANCE)
    L = ArrayType(zeros(T, Nx + 1, Ny + 1))      # Laplacian result
    temp = ArrayType(zeros(T, Nx + 1, Ny + 1))   # Temporary for matrix mult
    temp_arrays = (L, temp)

    # ✅ CREAR BUFFERS PARA RK4 OPTIMIZADO
    local rk4_buffers = nothing
    local scheme_func_optimized

    if scheme_name == "RK4"
        # Crear buffers pre-allocados para RK4
        println("Creating optimized RK4 buffers...")
        rk4_buffers = RK4Buffers(U)

        # Wrapper que incluye los buffers
        scheme_func_optimized = (U, dt, t, rhs) -> RK4_optimized(U, dt, t, rhs, rk4_buffers)
        println("RK4 buffers created successfully")
    elseif scheme_name == "Euler"
        scheme_func_optimized = (U, dt, t, rhs) -> Euler_optimized(U, dt, t, rhs)
    else
        error("Unsupported scheme: $scheme_name")
    end

    # SOLUCIÓN AL PROBLEMA DE SCOPE: Declarar función fuera del if/else
    local rhs_optimized  # Declare in function scope

    # Choose device-specific optimized version and initialize conditions
    if use_gpu
        rhs_optimized = (U, t) -> Wave2D_gpu_optimized!(F, U, t, wave_params, temp_arrays)

        # GPU-optimized initial conditions
        apply_initial_conditions_gpu!(U, xnodes, ynodes, A, σ, x_c, y_c, Nx, Ny)
        apply_dirichlet_gpu!(U, Nx, Ny)
    else
        rhs_optimized = (U, t) -> Wave2D_cpu_optimized!(F, U, t, wave_params, temp_arrays)

        # CPU-optimized initial conditions
        apply_initial_conditions_cpu!(U, xnodes, ynodes, A, σ, x_c, y_c, Nx, Ny)
        apply_dirichlet_cpu!(U, Nx, Ny)
    end

    # Create organized output directory
    output_dir = create_output_directory(Nx, scheme_name, T, use_gpu)

    # Time stepping with intermediate saves
    backend_str = use_gpu ? "GPU" : "CPU"
    println("Starting 2D MxM OPTIMIZED benchmark with $(stencil)-node stencil and $nodetype nodes on $backend_str…")
    println("Using $T with Nx = Ny = $Nx and $scheme_name temporal scheme")
    println("Total time steps: $Nt")
    println("Output directory: $output_dir")

    # # Define times to save plots - CÁLCULO ROBUSTO
    # save_fractions = [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]
    # indices_to_save = [Int(round(frac * Float64(Nt))) for frac in save_fractions]

    # println("Save indices: $indices_to_save")

    # # Guardado inicial (t=0)
    # time_val_T = T(0.0)
    # if Nx < 76
    #     u_fine = save_plots_at_time!(U, parent(xnodes_cpu), parent(ynodes_cpu), time_val_T, A, Nx, Ny, nodetype, output_dir, scheme_name, use_gpu)
    #     #save_step_matrices(D2x_cpu, D2x, U, u_fine, output_dir, "t$(round(time_val_T, digits=2))")
    # else
    #     save_original_plot_at_time!(U, parent(xnodes_cpu), parent(ynodes_cpu), time_val_T, A, Nx, Ny, nodetype, output_dir, scheme_name, use_gpu)
    #     #save_step_matrices(D2x_cpu, D2x, U, nothing, output_dir, "t$(round(time_val_T, digits=2))")
    # end

    # Warm-up run to compile everything
    println("Warming up...")
    for n in 1:5
        rhs_optimized(U, T(0.0))
    end

    # ✅ MEDICIÓN FINAL ROBUSTA DE TIEMPO
    println("Starting timed run...")
    t_start_ns = time_ns()

    for n in 0:Nt-1
        t_current_robust = Float64(n) * dt_working
        t_solver = T(t_current_robust)
        U .= scheme_func_optimized(U, dt, t_solver, rhs_optimized)

        # if n in indices_to_save[2:end] || (n + 1) in indices_to_save
        #     time_val_robust = Float64(n + 1) * dt_working
        #     time_val_T = T(time_val_robust)
        #     # if !isfinite(time_val_robust)
        #     #     @warn "time_val is not finite at step $(n+1): $time_val_robust"
        #     #     continue
        #     # end
        #     if Nx < 76
        #         u_fine = save_plots_at_time!(U, parent(xnodes_cpu), parent(ynodes_cpu), time_val_T, A, Nx, Ny, nodetype, output_dir, scheme_name, use_gpu)
        #         # save_step_matrices(D2x_cpu, D2x, U, u_fine, output_dir, "t$(round(time_val_T, digits=2))")
        #     else
        #         save_original_plot_at_time!(U, parent(xnodes_cpu), parent(ynodes_cpu), time_val_T, A, Nx, Ny, nodetype, output_dir, scheme_name, use_gpu)
        #         # save_step_matrices(D2x_cpu, D2x, U, nothing, output_dir, "t$(round(time_val_T, digits=2))")
        #     end
        # end
    end

    # ✅ CÁLCULO FINAL ROBUSTO DE TIEMPO TRANSCURRIDO
    t_end_ns = time_ns()
    execution_time = (t_end_ns - t_start_ns) / 1.0e9  # Convertir a segundos

    println("=== TIMING VERIFICATION ===")
    println("t_start_ns: $t_start_ns")
    println("t_end_ns: $t_end_ns")
    println("Difference (ns): $(t_end_ns - t_start_ns)")
    println("execution_time (s): $execution_time")

    # Verificación de sanidad para timing
    if execution_time > 3600
        @warn "Execution time suspiciously large: $execution_time seconds"
    end
    if execution_time <= 0
        @warn "Non-positive execution time detected: $execution_time seconds"
        execution_time = 1.0  # Set to 1 second as fallback
    end

    # Final boundary conditions
    if use_gpu
        apply_dirichlet_gpu!(U, Nx, Ny)
    else
        apply_dirichlet_cpu!(U, Nx, Ny)
    end

    # Save simulation information
    save_simulation_info(output_dir, scheme_name, T, Nx, Ny, dt, Nt, execution_time, nodetype, use_gpu)

    println("Time stepping completed in $(round(execution_time, digits=2)) s")
    println("All outputs saved to: $output_dir")

    # ✅ CÁLCULO FINAL ROBUSTO DE PERFORMANCE
    scheme_steps = scheme_name == "Euler" ? 1 : (scheme_name == "RK2" ? 2 : 4)
    total_ops = Nt * scheme_steps * 2 * 2 * (Nx + 1)^3
    gflops = execution_time > 0 ? total_ops / execution_time / 1e9 : 0.0
    time_per_step = execution_time > 0 ? execution_time / Nt * 1000 : 0.0  # en milisegundos

    println("\n=== PERFORMANCE SUMMARY (FINAL) ===")
    println("Execution time: $(round(execution_time, digits=2)) s")
    println("Performance: $(round(gflops, digits=2)) GFLOPs")
    println("Time per step: $(round(time_per_step, digits=3)) ms")
    println("Total operations: $(total_ops)")

    # Verificaciones finales
    if time_per_step <= 0
        @warn "Non-positive time per step detected: $time_per_step ms"
    end
end

# -------------------------------------------------
# Execute simulation
# -------------------------------------------------

main(T, Nx, Ny, scheme_name, scheme_function, use_gpu)
