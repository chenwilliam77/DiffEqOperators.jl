# Concretizations of DiagonalOperator
function Array(D::DiagonalOperator)
    return diagm(0 => get_diagonal(D))
end

convert(::Type{AbstractMatrix}, D::DiagonalOperator) = Array(D)

function SparseArrays.sparse(D::DiagonalOperator)
    return spdiagm(0 => get_diagonal(D))
end

function LinearAlgebra.Diagonal(D::DiagonalOperator)
    return Diagonal(get_diagonal(D))
end

# Concretizations of IdentityOperator
function Array(D::IdentityOperator)
    return Matrix{eltype(D)}(I, get_n(D), get_n(D))
end

convert(::Type{AbstractMatrix}, D::IdentityOperator) = Array(D)

function SparseArrays.sparse(D::IdentityOperator)
    return SparseMatrixCSC{eltype(D)}(I, get_n(D), get_n(D))
end

function LinearAlgebra.Diagonal(D::IdentityOperator)
    return Diagonal(ones(eltype(D), get_n(D)))
end
