include("code/1_annexes/system_info.jl")

# Function to print a horizontal line for borders
function print_line(width=50)
    println("─" ^ width)
end

# Generic function to display a menu with borders, updated to handle empty line separators
function show_menu(title, options, width=50, prompt="Select an option: ")
    # Ensure width is at least as long as the title plus some padding
    effective_width = max(width, length(title) + 4)  # Add 4 for borders and padding
    
    # Top border
    println("┌" * "─" ^ (effective_width - 2) * "┐")
    
    # Title, centered
    title_padding = " " ^ max(0, (effective_width - length(title) - 2) ÷ 2)
    println("│" * title_padding * title * " " ^ (effective_width - length(title) - length(title_padding) - 2) * "│")
    
    # Separator line
    println("├" * "─" ^ (effective_width - 2) * "┤")
    
    # Options with empty line handling
    for (idx, desc) in options
        if idx == "" && desc == ""
            # Empty line separator
            println("│" * " " ^ (effective_width - 2) * "│")
        else
            padding = " " ^ max(0, effective_width - length("  $idx. $desc") - 3)
            println("│  $idx. $desc$padding │")
        end
    end
    
    # Bottom border
    println("└" * "─" ^ (effective_width - 2) * "┘")
    print("\n$prompt")
end

# Main menu options
function get_main_menu_options()
    [
        (1, "Annexes"),
        (2, "Foundational Concepts"),
        (3, "CPU Architecture and Performance"),
        (4, "GPU Architecture and Performance"),
        (5, "CPU vs GPU Comparison"),
        (6, "State Formulation"),
        (7, "Global Interpolation"),
        (8, "Wave Equation Solvers"),
        ("", ""),
        (0, "Exit")
    ]
end

# Annexes submenu options
function get_annexes_menu_options()
    [
        (1, "Julia Initial Setup"),
        (2, "System Information"),
        ("", ""),  # Empty line separator
        (0, "Back to Main Menu (0 or Enter)")
    ]
end

# Foundational concepts submenu options
function get_foundational_concepts_options()
    [
        (1, "Operation to Data (operation-to-data.jl)"),
        ("", ""),  # Empty line separator
        (0, "Back to Main Menu (0 or Enter)")
    ]
end

# CPU submenu options
function get_cpu_menu_options()
    [
        (1, "GFLOPS vs N (CPU)"),
        (2, "t_CPU vs t_memory (CPU)"),
        (3, "Cache memory conflict spikes (CPU)"),
        ("", ""),  # Empty line separator
        (0, "Back to Main Menu (0 or Enter)")
    ]
end

# GPU submenu options
function get_gpu_menu_options()
    [
        (1, "GFLOPS vs N (GPU)"),
        (2, "t_GPU vs t_memory (GPU)"),
        ("", ""),  # Empty line separator
        (0, "Back to Main Menu (0 or Enter)")
    ]
end

# CPU vs GPU submenu options
function get_cpu_vs_gpu_menu_options()
    [
        (1, "CPU vs GPU speedup comparison"),
        ("", ""),  # Empty line separator
        (0, "Back to Main Menu (0 or Enter)")
    ]
end

# State Formulation submenu options
function get_state_formulation_options()
    [
        (1, "Explicit Implementation"),
        (2, "Modular Implementation"),
        (3, "State Vector Formulation"),
        (4, "Physical Vector Formulation"),
        ("", ""),  # Empty line separator
        (0, "Back to Main Menu (0 or Enter)")
    ]
end

# Global Interpolation submenu options
function get_global_interpolation_options()
    [
        (1, "Test Functions"),
        ("", ""),  # Empty line separator
        (0, "Back to Main Menu (0 or Enter)")
    ]
end

# Wave Equation Solvers submenu options - updated structure
function get_wave_equation_solvers_options()
    [
        (1, "Laplacian Benchmark"),
        ("", ""),  # Empty line separator
        (2, "Wave 2D MxM Benchmark"),
        (3, "Wave 2D MxV Benchmark"),
        ("", ""),  # Empty line separator
        (4, "Wave 1D MxV Animation (Nx=25) (26 nodes)"),
        (5, "Wave 2D MxM Animation (Nx=25, Ny=25) (26 nodes)"),
        (6, "Wave 2D MxV Animation (Nx=25, Ny=25) (26 nodes)"),
        ("", ""),  # Empty line separator
        (0, "Back to Main Menu (0 or Enter)")
    ]
end

# Function to execute Annexes option - updated to handle empty input
function execute_annexes_option(option)
    paths = Dict(
        1 => "code/1_annexes/setup.jl",
        2 => "code/1_annexes/system_info.jl",
        3 => "code/1_annexes/tests_functions.jl"
    )
    if option == 0 || option === nothing
        println()  # Add newline before returning to menu
        return false
    elseif haskey(paths, option)
        println()  # Add newline after user input but before execution
        try
            if option == 2
                system_info()
            else
                include(paths[option])
            end
        catch e
            println("Error executing the option: $e")
        end
        println("\nPress Enter to go back to the menu...")
        readline()  # Wait for user input before returning
        println()  # Add newline after returning
        return true
    else
        println("Invalid option. Please try again.")
        println()
        return true
    end
end

# Function to execute Foundational concepts option - updated to handle empty input
function execute_foundational_concepts_option(option)
    paths = Dict(
        1 => "code/2_foundational_concepts/operation-to-data.jl"
    )
    if option == 0 || option === nothing
        println()  # Add newline before returning to menu
        return false
    elseif haskey(paths, option)
        println()  # Add newline after user input but before execution
        try
            include(paths[option])
        catch e
            println("Error executing the option: $e")
        end
        println("\nPress Enter to go back to the menu...")
        readline()  # Wait for user input before returning
        println()  # Add newline after returning
        return true
    else
        println("Invalid option. Please try again.")
        println()
        return true
    end
end

# Function to execute CPU options - updated to handle empty input
function execute_cpu_option(option)
    paths = Dict(
        1 => "code/3_cpu/GFLOPS.jl",
        2 => "code/3_cpu/tcpu_tmemory.jl",
        3 => "code/3_cpu/tmemory.jl"
    )
    if option == 0 || option === nothing
        println()  # Add newline before returning to menu
        return false
    elseif haskey(paths, option)
        println()  # Add newline after user input but before execution
        try
            include(paths[option])
        catch e
            println("Error executing the option: $e")
        end
        println("\nPress Enter to go back to the menu...")
        readline()  # Wait for user input before returning
        println()  # Add newline after returning
        return true
    else
        println("Invalid option. Please try again.")
        println()
        return true
    end
end

# Function to execute GPU options - updated to handle empty input
function execute_gpu_option(option)
    paths = Dict(
        1 => "code/4_gpu/GFLOPS_GPU.jl",
        2 => "code/4_gpu/tgpu_tmemory_GPU.jl"
    )
    if option == 0 || option === nothing
        println()  # Add newline before returning to menu
        return false
    elseif haskey(paths, option)
        println()  # Add newline after user input but before execution
        try
            include(paths[option])
        catch e
            println("Error executing the option: $e")
        end
        println("\nPress Enter to go back to the menu...")
        readline()  # Wait for user input before returning
        println()  # Add newline after returning
        return true
    else
        println("Invalid option. Please try again.")
        println()
        return true
    end
end

# Function to execute CPU vs GPU options - updated to handle empty input
function execute_cpu_vs_gpu_option(option)
    paths = Dict(
        1 => "code/5_cpu_vs_gpu/cpu_vs_gpu.jl",
    )
    if option == 0 || option === nothing
        println()  # Add newline before returning to menu
        return false
    elseif haskey(paths, option)
        println()  # Add newline after user input but before execution
        try
            include(paths[option])
        catch e
            println("Error executing the option: $e")
        end
        println("\nPress Enter to go back to the menu...")
        readline()  # Wait for user input before returning
        println()  # Add newline after returning
        return true
    else
        println("Invalid option. Please try again.")
        println()
        return true
    end
end

# Function to execute State Formulation options - updated to handle empty input
function execute_state_formulation_option(option)
    paths = Dict(
        1 => "code/6_state_formulation/version1/version1_1D.jl",
        2 => "code/6_state_formulation/version2/version2_1D.jl",
        3 => "code/6_state_formulation/version3/version3_1D.jl",
        4 => "code/6_state_formulation/version4/version4_1D.jl"
    )
    if option == 0 || option === nothing
        println()  # Add newline before returning to menu
        return false
    elseif haskey(paths, option)
        println()  # Add newline after user input but before execution
        try
            include(paths[option])
        catch e
            println("Error executing the option: $e")
        end
        println("\nPress Enter to go back to the menu...")
        readline()  # Wait for user input before returning
        println()  # Add newline after returning
        return true
    else
        println("Invalid option. Please try again.")
        println()
        return true
    end
end

# Function to execute Global Interpolation options - updated to handle empty input
function execute_global_interpolation_option(option)
    paths = Dict(
        1 => "code/7_global_interpolation/tests_functions.jl"
    )
    if option == 0 || option === nothing
        println()  # Add newline before returning to menu
        return false
    elseif haskey(paths, option)
        println()  # Add newline after user input but before execution
        try
            include(paths[option])
        catch e
            println("Error executing the option: $e")
        end
        println("\nPress Enter to go back to the menu...")
        readline()  # Wait for user input before returning
        println()  # Add newline after returning
        return true
    else
        println("Invalid option. Please try again.")
        println()
        return true
    end
end

# Function to execute Wave Equation Solvers options - updated with new structure and paths
function execute_wave_equation_solvers_option(option)
    paths = Dict(
        1 => "code/8_wave_equation_solvers/benchmark_laplacian.jl",
        2 => "code/8_wave_equation_solvers/optimized_versions/wave2D_MxM_schemes_optimized.jl",
        3 => "code/8_wave_equation_solvers/optimized_versions/wave2D_MxV_schemes_optimized.jl",
        4 => "code/8_wave_equation_solvers/animation_versions/wave1D.jl",
        5 => "code/8_wave_equation_solvers/animation_versions/wave2D_MxM.jl",
        6 => "code/8_wave_equation_solvers/animation_versions/wave2D_MxV.jl"
    )
    if option == 0 || option === nothing
        println()
        return false
    elseif haskey(paths, option)
        println()
        try
            include(paths[option])
        catch e
            println("Error executing the option: $e")
        end
        println("\nPress Enter to go back to the menu...")
        readline()
        println()
        return true
    else
        println("Invalid option. Please try again.")
        println()
        return true
    end
end

# Main loop - updated to handle empty input as option 0
function main()
    while true
        show_menu("Main Menu", get_main_menu_options(), 50, "Select an option (0-8): ")
        input = strip(readline())
        
        if input == ""
            option = 0
        else
            option = tryparse(Int, input)
            if isnothing(option)
                println("Please enter a valid number.")
                println()
                continue
            end
        end

        if option == 1  # Annexes submenu
            while true
                show_menu("Annexes Menu", get_annexes_menu_options(), 50, "Select an option (0-3): ")
                input = strip(readline())
                annexes_option = input == "" ? 0 : tryparse(Int, input)
                
                if isnothing(annexes_option)
                    println("Please enter a valid number.")
                    println()
                    continue
                end
                
                continue_annexes = execute_annexes_option(annexes_option)
                if !continue_annexes
                    println()  # Add newline before returning to main menu
                    break
                end
            end
        elseif option == 2  # Foundational Concepts submenu
            while true
                show_menu("Foundational Concepts", get_foundational_concepts_options(), 50, "Select an option (0-1): ")
                input = strip(readline())
                foundational_option = input == "" ? 0 : tryparse(Int, input)
                
                if isnothing(foundational_option)
                    println("Please enter a valid number.")
                    println()
                    continue
                end
                
                continue_foundational = execute_foundational_concepts_option(foundational_option)
                if !continue_foundational
                    println()  # Add newline before returning to main menu
                    break
                end
            end
        elseif option == 3  # CPU submenu
            while true
                show_menu("CPU Menu", get_cpu_menu_options(), 50, "Select an option (0-3): ")
                input = strip(readline())
                cpu_option = input == "" ? 0 : tryparse(Int, input)
                
                if isnothing(cpu_option)
                    println("Please enter a valid number.")
                    println()
                    continue
                end
                
                continue_cpu = execute_cpu_option(cpu_option)
                if !continue_cpu
                    println()  # Add newline before returning to main menu
                    break
                end
            end
        elseif option == 4  # GPU submenu
            while true
                show_menu("GPU Menu", get_gpu_menu_options(), 50, "Select an option (0-2): ")
                input = strip(readline())
                gpu_option = input == "" ? 0 : tryparse(Int, input)
                
                if isnothing(gpu_option)
                    println("Please enter a valid number.")
                    println()
                    continue
                end
                
                continue_gpu = execute_gpu_option(gpu_option)
                if !continue_gpu
                    println()  # Add newline before returning to main menu
                    break
                end
            end
        elseif option == 5  # CPU vs GPU submenu
            while true
                show_menu("CPU vs GPU Menu", get_cpu_vs_gpu_menu_options(), 50, "Select an option (0-1): ")
                input = strip(readline())
                cpu_vs_gpu_option = input == "" ? 0 : tryparse(Int, input)
                
                if isnothing(cpu_vs_gpu_option)
                    println("Please enter a valid number.")
                    println()
                    continue
                end
                
                continue_cpu_vs_gpu = execute_cpu_vs_gpu_option(cpu_vs_gpu_option)
                if !continue_cpu_vs_gpu
                    println()  # Add newline before returning to main menu
                    break
                end
            end
        elseif option == 6  # State Formulation submenu
            while true
                show_menu("State Formulation", get_state_formulation_options(), 50, "Select an option (0-4): ")
                input = strip(readline())
                state_option = input == "" ? 0 : tryparse(Int, input)
                
                if isnothing(state_option)
                    println("Please enter a valid number.")
                    println()
                    continue
                end
                
                continue_state = execute_state_formulation_option(state_option)
                if !continue_state
                    println()  # Add newline before returning to main menu
                    break
                end
            end
        elseif option == 7  # Global Interpolation submenu
            while true
                show_menu("Global Interpolation", get_global_interpolation_options(), 50, "Select an option (0-1): ")
                input = strip(readline())
                global_option = input == "" ? 0 : tryparse(Int, input)
                
                if isnothing(global_option)
                    println("Please enter a valid number.")
                    println()
                    continue
                end
                
                continue_global = execute_global_interpolation_option(global_option)
                if !continue_global
                    println()  # Add newline before returning to main menu
                    break
                end
            end
        elseif option == 8  # Wave Equation Solvers submenu
            while true
                show_menu("Wave Equation Solvers", get_wave_equation_solvers_options(), 60, "Select an option (0-7): ")
                input = strip(readline())
                wave_option = input == "" ? 0 : tryparse(Int, input)
                
                if isnothing(wave_option)
                    println("Please enter a valid number.")
                    println()
                    continue
                end
                
                continue_wave = execute_wave_equation_solvers_option(wave_option)
                if !continue_wave
                    println()  # Add newline before returning to main menu
                    break
                end
            end
        elseif option == 0
            println("\nExiting the program. Goodbye!")
            println()  # Add newline before exiting
            break
        else
            println("Invalid option. Please try again.")
            println()
        end
    end
end

# Run the program
main()
