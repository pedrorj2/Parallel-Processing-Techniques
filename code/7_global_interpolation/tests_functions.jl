# File: 7_global_interpolation/tests_functions.jl
#
# Tests for interpolation functions and differentiation matrices.

using CSV
using Tables
using Plots
using OffsetArrays
using Printf

include("lagrange.jl")
include("D2_matrix.jl")
include("D1_matrix.jl")
include("D0_matrix.jl")
include("../utilities/nodes.jl")

# -------------------------------------------------
# Helper functions
# -------------------------------------------------

"""
    save_matrix_csv(mat::AbstractMatrix, fname::String)

Save a matrix to a CSV file, handling OffsetArray by converting to parent array.
"""
function save_matrix_csv(mat::AbstractMatrix, fname::String)
    local A = isa(mat, OffsetArray) ? parent(mat) : mat
    mkpath(dirname(fname))
    CSV.write(fname, Tables.table(A))
    println("\n💾 Matrix saved to: $fname")
end

"""
    get_valid_integer(prompt::String, min_val::Int, max_val::Int)

Prompt user for an integer input within a specified range.
"""
function get_valid_integer(prompt::String, min_val::Int, max_val::Int)
    while true
        print(prompt)
        input = strip(readline())
        try
            val = parse(Int, input)
            if min_val <= val <= max_val
                return val
            else
                println("Please enter a number between $min_val and $max_val.")
            end
        catch
            println("Invalid input. Please enter a valid integer.")
        end
    end
end

# -------------------------------------------------
# Test: iterative_lagrange.jl
# -------------------------------------------------

"""
    test_recursive(N::Int, order::Int, eval_node::Int)

Evaluate and print L_j^(order)(x_i) for all j at a specific node x_i.
"""
function test_recursive(N::Int, order::Int, eval_node::Int)
    raw_nodes = [Float64(i) / N for i in 0:N]
    xnodes = OffsetArray(raw_nodes, 0:N)
    i = clamp(eval_node, 0, N)  # Ensure eval_node is within valid range
    x = xnodes[i]
    prime = order == 0 ? "" : order == 1 ? "'" : "''"

    println("Evaluating at x_$i = $x (order = $order)")
    for j in collect(axes(xnodes, 1))
        v = evalL(xnodes, j, x; order=order)
        println("  L_$j$prime(x_$i) = $v")
    end
end

# -------------------------------------------------
# Test: D2_matrix.jl with extracted factor
# -------------------------------------------------

"""
    test_D2_matrix(T::Type, M::Int, stencil_size::Int)

Test D2 matrix with factor 1/dx² extracted, for given type T and stencil size.
"""
function test_D2_matrix(T::Type, M::Int, stencil_size::Int)
    raw_nodes = [T(i) / (M - 1) for i in 0:M-1]
    xnodes = OffsetArray(raw_nodes, 0:M-1)
    dx = xnodes[1] - xnodes[0]

    D2 = build_D2_matrix(xnodes, stencil_size)
    D2_factorless = D2 * dx^2

    println("D2 matrix $(size(D2)) with stencil size $(stencil_size) (extracted 1/dx² factor):\n")
    colwidth = 12
    header = "     " * join([rpad("j=$j", colwidth) for j in 0:M-1])
    println(header)
    println(repeat("-", length(header)))

    for i in 0:M-1
        print(rpad("i=$i", 6) * "| ")
        for j in 0:M-1
            val = D2_factorless[i, j]
            str = T === Float64 ? @sprintf("%.3f", val) : val == 0 // 1 ? "0" : string(val)
            print(rpad(str, colwidth))
        end
        println()
    end

    if T != Float64
        println("\nCommon factor: 1/dx² = ", 1 // (dx^2))
    else
        println("\nCommon factor (approximate): 1/dx² ≈ ", @sprintf("%.3f", 1 / dx^2))
        fname = "code/7_global_interpolation/outputs/D2_factorless_$(T)_M$(M)_stencil$(stencil_size).csv"
        save_matrix_csv(D2_factorless, fname)
    end
end

# -------------------------------------------------
# Test: D2_matrix.jl without extracted factor
# -------------------------------------------------

"""
    test_D2_matrix_nofactor(T::Type, M::Int, stencil_size::Int)

Test D2 matrix with factor 1/dx² included, for given type T and stencil size.
"""
function test_D2_matrix_nofactor(T::Type, M::Int, stencil_size::Int)
    raw_nodes = [T(i) / (M - 1) for i in 0:M-1]
    xnodes = OffsetArray(raw_nodes, 0:M-1)

    D2 = build_D2_matrix(xnodes, stencil_size)

    println("D2 matrix $(size(D2)) with stencil size $(stencil_size):\n")
    colwidth = 12
    header = "     " * join([rpad("j=$j", colwidth) for j in 0:M-1])
    println(header)
    println(repeat("-", length(header)))

    for i in 0:M-1
        print(rpad("i=$i", 6) * "| ")
        for j in 0:M-1
            val = D2[i, j]
            str = T === Float64 ? @sprintf("%.3f", val) : val == 0 // 1 ? "0" : string(val)
            print(rpad(str, colwidth))
        end
        println()
    end

    if T === Float64
        fname = "code/7_global_interpolation/outputs/D2_$(T)_M$(M)_stencil$(stencil_size).csv"
        save_matrix_csv(D2, fname)
    end
end

# -------------------------------------------------
# Test: D2_matrix.jl with Chebyshev nodes
# -------------------------------------------------

"""
    test_D2_matrix_chebyshev(T::Type, M::Int, stencil_size::Int)

Test D2 matrix with Chebyshev nodes, without factor 1/dx².
"""
function test_D2_matrix_chebyshev(T::Type, M::Int, stencil_size::Int)
    raw_nodes = [(1 - cos(π * i / (M - 1))) / 2 for i in 0:M-1]
    xnodes = OffsetArray(T.(raw_nodes), 0:M-1)

    D2 = build_D2_matrix(xnodes, stencil_size)

    println("D2 matrix $(size(D2)) with Chebyshev nodes:\n")
    colwidth = 12
    header = "     " * join([rpad("j=$j", colwidth) for j in 0:M-1])
    println(header)
    println(repeat("-", length(header)))

    for i in 0:M-1
        print(rpad("i=$i", 6) * "| ")
        for j in 0:M-1
            val = D2[i, j]
            str = T === Float64 ? @sprintf("%.3f", val) : val == 0 // 1 ? "0" : string(val)
            print(rpad(str, colwidth))
        end
        println()
    end

    println("\n(Note: Chebyshev nodes are non-uniformly distributed in [0,1])\n")
    println("Nodes used (x_i):\n")
    for i in 0:M-1
        @printf("  x_%d = %.8f\n", i, xnodes[i])
    end

    println("\nDifferences between consecutive nodes (Δx_i = x_{i+1} - x_i):\n")
    for i in 0:M-2
        dx = xnodes[i+1] - xnodes[i]
        @printf("  Δx_%d_%d = %.8f\n", i, i + 1, dx)
    end

    if T === Float64
        fname = "code/7_global_interpolation/outputs/D2_chebyshev_$(T)_M$(M)_stencil$(stencil_size).csv"
        save_matrix_csv(D2, fname)
    end
end

# -------------------------------------------------
# Test: D1_matrix.jl (first derivative)
# -------------------------------------------------

"""
    test_D1_matrix(T::Type, M::Int, stencil_size::Int)

Test D1 matrix with factor 1/dx extracted, for given type T and stencil size.
"""
function test_D1_matrix(T::Type, M::Int, stencil_size::Int)
    raw_nodes = [T(i) / (M - 1) for i in 0:M-1]
    xnodes = OffsetArray(raw_nodes, 0:M-1)
    dx = xnodes[1] - xnodes[0]

    D1 = build_D1_matrix(xnodes, stencil_size)
    D1_factorless = D1 * dx

    println("D1 matrix $(size(D1)) with stencil size $(stencil_size) (extracted 1/dx factor):\n")
    colwidth = 12
    header = "     " * join([rpad("j=$j", colwidth) for j in 0:M-1])
    println(header)
    println(repeat("-", length(header)))

    for i in 0:M-1
        print(rpad("i=$i", 6) * "| ")
        for j in 0:M-1
            val = D1_factorless[i, j]
            str = T === Float64 ? @sprintf("%.3f", val) : val == 0 // 1 ? "0" : string(val)
            print(rpad(str, colwidth))
        end
        println()
    end

    if T != Float64
        println("\nCommon factor: 1/dx = ", 1 // dx)
    else
        println("\nCommon factor (approximate): 1/dx ≈ ", @sprintf("%.3f", 1 / dx))
        fname = "code/7_global_interpolation/outputs/D1_factorless_$(T)_M$(M)_stencil$(stencil_size).csv"
        save_matrix_csv(D1_factorless, fname)
    end
end

# -------------------------------------------------
# Test: D0_matrix.jl (zeroth derivative)
# -------------------------------------------------

"""
    test_D0_matrix(T::Type, M::Int, stencil_size::Int)

Test D0 matrix (identity-like matrix) for given type T and stencil size.
"""
function test_D0_matrix(T::Type, M::Int, stencil_size::Int)
    raw_nodes = [T(i) / (M - 1) for i in 0:M-1]
    xnodes = OffsetArray(raw_nodes, 0:M-1)

    D0 = build_D0_matrix(xnodes, stencil_size)

    println("D0 matrix $(size(D0)) with stencil size $(stencil_size):\n")
    colwidth = 12
    header = "     " * join([rpad("j=$j", colwidth) for j in 0:M-1])
    println(header)
    println(repeat("-", length(header)))

    for i in 0:M-1
        print(rpad("i=$i", 6) * "| ")
        for j in 0:M-1
            val = D0[i, j]
            str = T === Float64 ? @sprintf("%.6f", val) : val == 0 // 1 ? "0" : string(val)
            print(rpad(str, colwidth))
        end
        println()
    end

    if T === Float64 || T === Float32
        fname = "code/7_global_interpolation/outputs/D0_$(T)_M$(M)_stencil$(stencil_size).csv"
        save_matrix_csv(D0, fname)
    else
        println("\n(Note: Rational could lead to big denominators)")
    end
end

# -------------------------------------------------
# Test: Node distributions
# -------------------------------------------------

"""
    test_nodes(N::Int)
Visualize equispaced, Chebyshev, and Chebyshev-Lobatto node distributions
and show minimum spacing for each.
"""
function test_nodes(N::Int)
    nodes = N + 1  # Number of nodes

    x_equispaced = equispaced_nodes(N)
    x_chebyshev = chebyshev_nodes(N)
    x_lobatto = chebyshev_lobatto_nodes(N)

    # Calculate minimum spacing for each distribution
    dx_eq = minimum(diff(parent(x_equispaced)))
    dx_ch = minimum(diff(parent(x_chebyshev)))
    dx_lo = minimum(diff(parent(x_lobatto)))

    # Calculate maximum spacing too for comparison
    dx_eq_max = maximum(diff(parent(x_equispaced)))
    dx_ch_max = maximum(diff(parent(x_chebyshev)))
    dx_lo_max = maximum(diff(parent(x_lobatto)))

    y_e = fill(1.5, N + 1)
    y_c = fill(1.0, N + 1)
    y_l = fill(0.5, N + 1)

    p = plot(x_equispaced, y_e;
        seriestype=:scatter,
        markersize=4,
        label=@sprintf("Equispaced                        (dx_min=%.6f)", dx_eq),
        xlabel="x",
        title="Node distributions with N=$(N) ($(nodes) nodes)",
        yticks=false)

    scatter!(p, x_chebyshev, y_c;
        markersize=4,
        label=@sprintf("Chebyshev-Gauss              (dx_min=%.6f)", dx_ch))

    scatter!(p, x_lobatto, y_l;
        markersize=4,
        label=@sprintf("Chebyshev-Gauss-Lobatto (dx_min=%.6f)", dx_lo))

    ylims!(p, 0, 1.75)

    display(p)

    # Print detailed spacing information
    println("\n" * "="^60)
    println("SPACING ANALYSIS FOR N = $N ($(nodes) nodes)")
    println("="^60)

    println("\n📏 EQUISPACED NODES:")
    @printf("   dx_min = dx_max = %.8f (constant spacing)\n", dx_eq)
    @printf("   Total range covered: [%.7f, %.7f]\n", first(x_equispaced), last(x_equispaced))

    println("\n📏 CHEBYSHEV-GAUSS NODES:")
    @printf("   dx_min = %.8f (near boundaries)\n", dx_ch)
    @printf("   dx_max = %.8f (near center)\n", dx_ch_max)
    @printf("   Ratio dx_max/dx_min = %.2f\n", dx_ch_max / dx_ch)
    @printf("   Total range covered: [%.7f, %.7f]\n", first(x_chebyshev), last(x_chebyshev))

    println("\n📏 CHEBYSHEV-GAUSS-LOBATTO NODES:")
    @printf("   dx_min = %.8f (near boundaries)\n", dx_lo)
    @printf("   dx_max = %.8f (near center)\n", dx_lo_max)
    @printf("   Ratio dx_max/dx_min = %.2f\n", dx_lo_max / dx_lo)
    @printf("   Total range covered: [%.7f, %.7f]\n", first(x_lobatto), last(x_lobatto))

    println("\n🔍 COMPARISON:")
    @printf("   Chebyshev dx_min is %.1fx smaller than equispaced\n", dx_eq / dx_ch)
    @printf("   This explains why Chebyshev needs smaller timesteps!\n")

    println("\n💡 IMPLICATION FOR WAVE EQUATION:")
    T_simulation = 1.0  # 1 second simulation

    # Time steps based on dx_min for stability
    dt_eq_optimal = 0.1 * dx_eq
    dt_ch_optimal = 0.1 * dx_ch
    dt_lo_optimal = 0.1 * dx_lo

    steps_eq = Int(floor(T_simulation / dt_eq_optimal))
    steps_ch = Int(floor(T_simulation / dt_ch_optimal))
    steps_lo = Int(floor(T_simulation / dt_lo_optimal))

    @printf("   TIME STEPS NEEDED FOR 1 SECOND SIMULATION:\n")
    @printf("   \n")
    @printf("   Equispaced:     dt = %.2e → %d steps\n", dt_eq_optimal, steps_eq)
    @printf("   Chebyshev:      dt = %.2e → %d steps\n", dt_ch_optimal, steps_ch)
    @printf("   Lobatto:        dt = %.2e → %d steps\n", dt_lo_optimal, steps_lo)

    @printf("\n   📊 COMPUTATIONAL IMPACT:\n")
    @printf("   Chebyshev needs %.1fx MORE steps than equispaced\n", steps_ch / steps_eq)


    println("="^60)
end


# -------------------------------------------------
# Interactive test menu
# -------------------------------------------------

"""
    run_tests()

Run an interactive menu to select and execute tests, prompting for N and other parameters.
"""
function run_tests()
    println("\nChoose a test:\n")
    println("1. iterative_lagrange (order 0, 1, or 2) at a specific node")
    println(" ")
    println("2. build_D2_matrix (Float64, Equispaced)")
    println("3. build_D2_matrix (Float64, Equispaced) factor 1/dx² not extracted")
    println("4. build_D2_matrix (Float32, Chebyshev )")
    println(" ")
    println("5. build_D1_matrix (Float64, Equispaced)")
    println(" ")
    println("6. build_D0_matrix (Float32, Equispaced)")
    println(" ")
    println("7. test_nodes (Equispaced vs Chebyshev vs Chebyshev-Lobatto)")
    println("\n0. Exit")

    choice = strip(readline())

    if choice == "0"
        println("Exiting...")
        return
    elseif !in(choice, ["1", "2", "3", "4", "5", "6", "7", "8"])
        println("Invalid option. Try again.")
        println("\nPress Enter to continue...")
        readline()
        run_tests()
        return
    end

    # Prompt for N (number of nodes, 0 to N)
    N = get_valid_integer("Enter N (for N+1 nodes): ", 2, 1000)
    M = N + 1 # Matrix size

    # Default parameters
    stencil_size = 3
    order = 2
    eval_node = 1

    # Prompt for additional parameters for tests requiring Lagrange or stencil
    if choice in ["1", "2", "3", "4", "5", "6"]
        stencil_size = get_valid_integer("Enter stencil size (3 to $M): ", 3, M)
        if choice == "1"
            order = get_valid_integer("Enter Lagrange order (0 for L, 1 for L', 2 for L''): ", 0, 2)
            eval_node = get_valid_integer("Enter node index to evaluate (0 to $N): ", 0, N)
        end
    end

    # Execute the selected test
    if choice == "1"
        test_recursive(N, order, eval_node)
    elseif choice == "2"
        test_D2_matrix(Float64, M, stencil_size)
    elseif choice == "3"
        test_D2_matrix_nofactor(Float64, M, stencil_size)
    elseif choice == "4"
        test_D2_matrix_chebyshev(Float32, M, stencil_size)
    elseif choice == "5"
        test_D1_matrix(Float64, M, stencil_size)
    elseif choice == "6"
        test_D0_matrix(Float32, M, stencil_size)
    elseif choice == "7"
        test_nodes(N)
    end

    println("\nPress Enter to continue...")
    readline()

    run_tests()
end

# Execute menu
run_tests()