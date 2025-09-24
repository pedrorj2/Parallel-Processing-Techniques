# File: 7_global_interpolation/lagrange.jl

using OffsetArrays

# -------------------------------------------------------
# Recursive calculation of the Lagrange basis polynomial L_j(x) and its first two derivatives L'_j(x), L''_j(x).
# This uses OffsetArray for 0-based indexing of nodes.
# The Lagrange basis L_j(x) is defined as the product over k ≠ j of (x - x_k) / (x_j - x_k),
# and this function computes it recursively along with derivatives.
# -------------------------------------------------------

function evalL(xnodes::OffsetVector{T,Vector{T}}, j::Int, x::T; order::Int=0) where T<:Real
    # @show typeof(xnodes), length(xnodes), j, x
    L, dL, ddL = lagrange_derivatives(xnodes, j, x)
    val = (order == 0 ? L : order == 1 ? dL : ddL)
    # @show val
    return val
end

function lagrange_derivatives(xnodes::OffsetVector{T,Vector{T}}, j::Int, x::T) where T<:Real
    return iterative_lagrange(xnodes, j, x, length(xnodes))
end

function iterative_lagrange(xnodes::OffsetVector{T,Vector{T}}, j::Int, x::T, q::Int) where T<:Real
    L, dL, ddL = T(1.0), T(0.0), T(0.0)
    idxs = axes(xnodes, 1)
    for idx in idxs
        if idx == j
            continue
        end
        denom = xnodes[j] - xnodes[idx]
        factor = (x - xnodes[idx]) / denom
        dfactor = T(1.0) / denom
        ddfactor = T(0.0)
        new_L = L * factor
        new_dL = dL * factor + L * dfactor
        new_ddL = ddL * factor + 2 * dL * dfactor + L * ddfactor
        L, dL, ddL = new_L, new_dL, new_ddL
    end
    return (L, dL, ddL)
end
