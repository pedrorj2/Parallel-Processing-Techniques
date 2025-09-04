# File: 7_global_interpolation/D0_matrix.jl

using OffsetArrays

include("lagrange.jl")

"""
    select_stencil(i, M, q) -> UnitRange{Int}

Select a stencil (subset of nodes) around index `i` for interpolation.
The stencil size is `q`, centered as much as possible around `i`,
but clamped to stay within 0 to M-1.

- `i`: Central index (0-based).
- `M`: Total number of nodes.
- `q`: Stencil size (number of nodes to include).

Returns a range `imin:imax` of indices.
"""
function select_stencil(i::Int, M::Int, q::Int)
    half = div(q, 2)
    imin = clamp(i - half, 0, M - q)
    imax = imin + q - 1
    return imin:imax
end

"""
    build_D0_matrix(xnodes, stencil_size, xinterp) -> OffsetArray{T, 2}

Build the D0 matrix for Lagrange interpolation evaluation.
This is the "interpolation matrix" where each row corresponds to an evaluation point in `xinterp`,
and columns to the basis weights for the nodes in `xnodes`.

- `xnodes`: OffsetVector of node points (0-based indexing).
- `stencil_size`: Size of the local stencil (subset of nodes) used for each interpolation.
- `xinterp`: Vector of points where to evaluate the interpolant (1-based).

Returns an N×M OffsetArray (0-based rows and columns), where N = length(xinterp), M = length(xnodes).
Only the stencil columns are non-zero for each row.
"""
function build_D0_matrix(xnodes::OffsetVector{T,Vector{T}},
                         stencil_size::Int,
                         xinterp::Vector{T}) where T<:Real
    M = length(xnodes)  # Number of nodes.
    N = length(xinterp)  # Number of evaluation points.
    D0 = OffsetArray(zeros(T, N, M), 0:N-1, 0:M-1)  # Initialize matrix.

    for i in 0:N-1
        x = xinterp[i + 1]  # Get current evaluation point (xinterp is 1-based).
        # Find the nearest node to x for centering the stencil.
        idx_nearest = findmin(abs.(xnodes .- x))[2]
        stencil_range = select_stencil(idx_nearest, M, stencil_size)
        # Create a sub-OffsetArray for the stencil nodes.
        xstencil = OffsetArray(collect(xnodes[stencil_range]), stencil_range)

        for j in stencil_range
            q = length(xstencil)  # Actual stencil size (may be adjusted at boundaries).
            # Compute L_j(x) using the recursive function (only need the value, not derivatives).
            lj, _, _ = iterative_lagrange(xstencil, j, x, q)
            D0[i, j] = lj  # Set the basis weight.
        end
    end

    return D0
end

# ------------------------------------------------------------------
# Nodal version
# ------------------------------------------------------------------
"""
    build_D0_matrix(xnodes, stencil_size) -> OffsetArray{T, 2}

Nodal version of build_D0_matrix: evaluates at the nodes themselves (xinterp = xnodes).
This calls the general version internally.

Returns an M×M OffsetArray, which should be the identity matrix if stencil_size >= M,
but uses local stencils for approximation.
"""
function build_D0_matrix(xnodes::OffsetVector{T,Vector{T}},
                         stencil_size::Int) where T<:Real
    build_D0_matrix(xnodes, stencil_size, collect(xnodes))  # Call the general version.
end