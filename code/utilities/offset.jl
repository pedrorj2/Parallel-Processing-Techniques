# File: utilities/offset.jl
#
# Custom matrix and vector multiplication for OffsetArray types.

using OffsetArrays
import Base: *
import OffsetArrays: no_offset_view

# -------------------------------------------------
# Matrix-vector multiplication (OffsetArray and Array)
# -------------------------------------------------

"""
    *(A::OffsetArray{T,2}, x::AbstractVector{T}) where T<:Number

Matrix-vector multiplication for a 2D OffsetArray and a vector:
1. Checks that the number of columns in A matches the length of x.
2. Removes offsets from A and x.
3. Performs standard multiplication.
4. Returns an Array (Vector) with the resulting values.
"""
function *(A::OffsetArray{T,2}, x::AbstractVector{T}) where T<:Number
    @assert size(A, 2) == length(x) "Incompatible dimensions: size(A,2)=$(size(A,2)) ≠ length(x)=$(length(x))"
    A_std = no_offset_view(A)
    x_std = no_offset_view(x)
    return A_std * x_std
end

# -------------------------------------------------
# Matrix-matrix multiplication (OffsetArray and Array)
# -------------------------------------------------

"""
    *(A::OffsetArray{T,2}, B::AbstractMatrix{T}) where T<:Number

Matrix-matrix multiplication for a 2D OffsetArray and a matrix:
1. Checks that the number of columns in A matches the number of rows in B.
2. Removes offsets from A and converts B to an Array.
3. Performs standard multiplication.
4. Returns an OffsetArray with the original axes of A and B.
"""
function *(A::OffsetArray{T,2}, B::AbstractMatrix{T}) where T<:Number
    @assert size(A, 2) == size(B, 1) "Incompatible dimensions: size(A,2)=$(size(A,2)) ≠ size(B,1)=$(size(B,1))"
    A_std = no_offset_view(A)
    B_std = Array(B)
    C_std = A_std * B_std
    return OffsetArray(C_std, axes(A, 1), axes(B, 2))
end

# -------------------------------------------------
# OffsetArray matrix-vector multiplication
# -------------------------------------------------

"""
    *(A::OffsetArray{T,2}, x::OffsetArray{T,1}) where T<:Number

Matrix-vector multiplication for a 2D OffsetArray and a 1D OffsetArray:
1. Checks that the number of columns in A matches the length of x.
2. Removes offsets from A and x.
3. Performs standard multiplication.
4. Returns an OffsetArray with the original axes of A.
"""
function *(A::OffsetArray{T,2}, x::OffsetArray{T,1}) where T<:Number
    @assert size(A, 2) == length(x) "Incompatible dimensions: size(A,2)=$(size(A,2)) ≠ length(x)=$(length(x))"
    A_std = no_offset_view(A)
    x_std = no_offset_view(x)
    result_std = A_std * x_std
    return OffsetArray(result_std, axes(A, 1))
end

# -------------------------------------------------
# OffsetArray matrix-matrix multiplication
# -------------------------------------------------

"""
    *(A::OffsetArray{T,2}, x::OffsetArray{T,2}) where T<:Number

Matrix-matrix multiplication for two 2D OffsetArrays:
1. Checks that the number of columns in A matches the number of rows in x.
2. Removes offsets from A and x.
3. Performs standard multiplication.
4. Returns an OffsetArray with the original axes of A and x.
"""
function *(A::OffsetArray{T,2}, x::OffsetArray{T,2}) where T<:Number
    @assert size(A, 2) == size(x, 1) "Incompatible dimensions: size(A,2)=$(size(A,2)) ≠ size(x,1)=$(size(x,1))"
    A_std = no_offset_view(A)
    x_std = no_offset_view(x)
    result_std = A_std * x_std
    return OffsetArray(result_std, axes(A, 1), axes(x, 2))
end