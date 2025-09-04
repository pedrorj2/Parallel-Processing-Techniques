# File: 7_global_interpolation/D2_matrix.jl

using OffsetArrays

include("lagrange.jl")

"""
    select_stencil(i, M, q) -> UnitRange{Int}

Select a stencil (subset of nodes) around index `i` for differentiation.
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
    build_D2_matrix(xnodes, stencil_size, T=Float64) -> OffsetArray{T, 2}

Build the D2 matrix for second-order finite difference approximation using Lagrange interpolation.
Each row corresponds to the weights for approximating the second derivative at a node.

- `xnodes`: OffsetVector of node points (0-based indexing).
- `stencil_size`: Size of the local stencil (subset of nodes) used for each row.
- `T`: Type of the output matrix (e.g., Float32 or Float64).

Returns an M×M OffsetArray (0-based), where M = length(xnodes).
Only the stencil columns are non-zero for each row.
"""
function build_D2_matrix(xnodes::OffsetVector{<:Real}, stencil_size::Int, ::Type{T}=Float64) where T<:Real
    M = length(xnodes)  # Number of nodes.
    D2 = OffsetArray(zeros(T, M, M), 0:M-1, 0:M-1)  # Initialize matrix with type T.

    for i in 0:M-1
        # Select stencil centered at i.
        stencil_range = select_stencil(i, M, stencil_size)
        # Create a sub-OffsetArray for the stencil nodes, converting to T.
        xstencil = OffsetArray(convert.(T, collect(xnodes[stencil_range])), stencil_range)

        for j in stencil_range
            q = length(xstencil)  # Actual stencil size.
            # Compute L''_j(x_i) using the recursive function (only need the second derivative).
            _, _, lpp = iterative_lagrange(xstencil, j, T(xnodes[i]), q)
            D2[i, j] = lpp  # Set the differentiation weight.
        end
    end

    return D2
end