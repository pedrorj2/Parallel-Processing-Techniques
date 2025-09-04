# File: 1_annexes/setup.jl

using Pkg
project_path = pwd()

println("Activating the project environment...")
Pkg.activate(project_path)

println("Checking for inconsistencies in the project...")
try
    Pkg.resolve()
    println("Inconsistencies resolved.")
catch e
    println("Error while resolving dependencies: ", e)
    println("Trying to fix dependencies...")
    try
        Pkg.instantiate()
        println("Dependencies fixed.")
    catch e_inner
        println("Error while fixing dependencies: ", e_inner)
        println("Consider manually inspecting Project.toml and Manifest.toml.")
        return
    end
end

println("Installing the packages according to [deps]...")
try
    Pkg.instantiate()
    println("Packages installed successfully.")
catch e
    println("Error while installing packages: ", e)
    println("Ensure that Project.toml is correctly configured.")
    return
end

println("Updating the packages according to [compat]...")
try
    Pkg.resolve()
    println("Packages updated successfully.")
catch e
    println("Error while updating packages: ", e)
end

println("Versions of the packages installed:")
Pkg.status()

println("Project environment ready.")
