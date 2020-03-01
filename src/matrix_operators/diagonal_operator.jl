# This operator helps interface with the DerivativeOperators and enable diagonalization
# Example use case
#==
The Kolmogorov forward equation (in 1D) is given by
∂u(x,t)/∂t = -∂/∂x[μ(x, t) u(x,t)] + (1/2) ∂²/∂x²[σ(x, t)²u(x, t)]
for the SDE
dXₜ = μ(Xₜ, t) dt + σ(Xₜ, dt) dWₜ.

We want to rewrite this as
∂u(x,t)/∂t = A u(x,t),
and one way to write out A (assuming finite differences as the discretization) is

A = -L₁ * diag(μ(x⃗, t)) + (1/2) * L₂ * diag(σ(x, t)²),

where L₁ is the finite-difference approximation of a first derivative at the grid points x⃗,
diag(c⃗) creates a diagonal matrix with c⃗ on the diagonal, and L₂ is the finite-difference
approximation of a second derivative at the grid points x⃗.

Ideally, we'd be able to write A without doing any concretizations. Note that this is
fundamentally different from multiplying the finite difference operators by a vector
of coefficients.

Note that this automatically generalizes to multiple dimensions, assuming
the operators are intended to act upon vec(u).
==#
mutable struct DiagonalOperator{T} <: AbstractDiffEqLinearOperator{T}
    diagonal::Vector{T}
end

mutable struct IdentityOperator{T} <: AbstractDiffEqLinearOperator{T}
    n::Int
end

function IdentityOperator(n::Int, type = Float64)
    return IdentityOperator{type}(n)
end

function get_diagonal(D::DiagonalOperator)
    return D.diagonal
end

function get_n(D::IdentityOperator)
    return D.n
end

Base.size(D::DiagonalOperator) = (length(D.diagonal), length(D.diagonal))
Base.size(D::IdentityOperator) = (D.n, D.n)
function Base.size(D::DiagonalOperator, arraysize::Int)
    if arraysize > 2
        return 1
    elseif arraysize > 0
        return length(D.diagonal)
    else
        error("arraysize: dimension out of range")
    end
end
function Base.size(D::IdentityOperator, arraysize::Int)
    if arraysize > 2
        return 1
    elseif arraysize > 0
        return D.n
    else
        error("arraysize: dimension out of range")
    end
end
Base.length(D::DiagonalOperator) = length(D.diagonal)^2
Base.length(D::IdentityOperator) = D.n^2

# Multiplication
function *(D::DiagonalOperator{T}, M::AbstractVecOrMat{T}) where {T <: Real}
    x_temp = similar(M)
    LinearAlgebra.mul!(x_temp, D, M)
    return x_temp
end

*(c::Number, D::DiagonalOperator{T}) where {T <: Real} = DiagonalOperator(c * D.diagonal)
*(c::AbstractVector{S}, D::DiagonalOperator{T}) where {S <: Number, T <: Real} = DiagonalOperator(c .* D.diagonal)

function *(D::IdentityOperator{T}, M::AbstractVecOrMat{T}) where {T <: Real}
    if D.n == size(M, 1)
        return M
    else
        throw(DimensionMismatch("IdentityOperator has dimensions $(D.n), but vector/matrix M has dimensions $(size(M))"))
    end
end

*(c::Number, D::IdentityOperator{T}) where {T <: Real} = DiagonalOperator(c * ones(eltype(D), D.n))
*(c::AbstractVector{S}, D::IdentityOperator{T}) where {S <: Number, T <: Real} = DiagonalOperator(c)


function LinearAlgebra.mul!(x_temp::AbstractArray{T}, D::DiagonalOperator{T}, M::AbstractVecOrMat{T}) where {T}
    x_temp[:] = D.diagonal .* M
end
