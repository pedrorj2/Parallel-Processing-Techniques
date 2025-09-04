# File: utilities/nodes.jl
#
# Functions for generating different types of node distributions in [0,1].

using OffsetArrays

# -------------------------------------------------
# Equispaced nodes
# -------------------------------------------------

"""
    equispaced_nodes(N::Int)

Return N+1 equally spaced nodes in [0, 1], indexed from 0 to N.
"""
function equispaced_nodes(N::Int)
    x = collect(range(0.0, 1.0, length=N + 1))
    return OffsetArray(x, 0:N)
end

# -------------------------------------------------
# Chebyshev-Gauss nodes
# -------------------------------------------------

"""
    chebyshev_nodes(N::Int)

Return N+1 Chebyshev-Gauss nodes (zeros), mapped to [0,1], indexed from 0 to N.
"""
function chebyshev_nodes(N::Int)
    θ = [(2 * j + 1) * π / (2 * (N + 1)) for j in 0:N]
    x = cos.(θ)
    x_mapped = reverse(x .* 0.5 .+ 0.5)   # Map from [-1,1] to [0,1], ascending
    return OffsetArray(x_mapped, 0:N)
end

# -------------------------------------------------
# Chebyshev-Gauss-Lobatto nodes
# -------------------------------------------------

"""
    chebyshev_lobatto_nodes(N::Int)

Return N+1 Chebyshev-Gauss-Lobatto nodes (including endpoints), mapped to [0,1], indexed from 0 to N.
"""
function chebyshev_lobatto_nodes(N::Int)
    θ = [π * j / N for j in 0:N]
    x = cos.(θ)
    x_mapped = reverse(x .* 0.5 .+ 0.5)   # Map from [-1,1] to [0,1], ascending
    return OffsetArray(x_mapped, 0:N)
end