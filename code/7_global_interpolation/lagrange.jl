# File: 7_global_interpolation/lagrange.jl

using OffsetArrays

# -------------------------------------------------------
# Recursive calculation of the Lagrange basis polynomial L_j(x) and its first two derivatives L'_j(x), L''_j(x).
# This uses OffsetArray for 0-based indexing of nodes.
# The Lagrange basis L_j(x) is defined as the product over k ≠ j of (x - x_k) / (x_j - x_k),
# and this function computes it recursively along with derivatives.
# -------------------------------------------------------

"""
    lagrange_derivatives(xnodes, j, x) -> (L_j, L'_j, L''_j)

Compute the value of the j-th Lagrange basis polynomial and its first two derivatives at point x,
given the node points in `xnodes` (an OffsetVector for 0-based indexing).

- `xnodes`: OffsetVector of node points.
- `j`: Index of the basis polynomial (0-based).
- `x`: Evaluation point.

Returns a tuple: (L_j(x), L'_j(x), L''_j(x)).
"""
function lagrange_derivatives(xnodes::OffsetVector{T,Vector{T}}, j::Int, x::T) where T<:Real
    # Start recursion with q equal to the number of nodes (full product).
    return iterative_lagrange(xnodes, j, x, length(xnodes))
end


"""
    iterative_lagrange(xnodes, j, x, q) -> (L, dL, ddL)

Internal iterative function to build the Lagrange polynomial and derivatives step by step.
The function initializes the product to 1 (with derivatives 0) and iteratively multiplies factors one at a time, skipping the j-th node, to improve performance in constructing derivative matrices.

- `xnodes`: OffsetVector of node points.
- `j`: Index of the basis polynomial (0-based).
- `x`: Evaluation point.
- `q`: Total number of terms to include in the product (number of nodes considered).

The loop processes nodes in order, accumulating the product and its derivatives using the product rule.
"""
function iterative_lagrange(xnodes::OffsetVector{T,Vector{T}}, j::Int, x::T, q::Int) where T<:Real
    L, dL, ddL = T(1.0), T(0.0), T(0.0)
    keys = collect(axes(xnodes, 1))
    for k in 1:q
        idx = keys[k]
        if idx == j
            continue  # Factor=1, no cambio
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

"""
    evalL(xnodes, j, x; order=0) -> value

Wrapper function to evaluate the j-th Lagrange basis or its derivative at x.

- `order`: 0 for L_j(x), 1 for L'_j(x), 2 for L''_j(x). Defaults to 0.
"""
function evalL(xnodes::OffsetVector{T,Vector{T}}, j::Int, x::T; order::Int=0) where T<:Real
    L, dL, ddL = lagrange_derivatives(xnodes, j, x)
    return order == 0 ? L : order == 1 ? dL : ddL
end