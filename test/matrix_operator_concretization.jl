using Test, DiffEqOperators, Random, LinearAlgebra, SparseArrays

# DiagonalOperator concretization
@testset "DiagonalOperator concretization" begin
    Random.seed!(1793)
    D = DiagonalOperator(rand(10))
    @test Array(D) ≈ diagm(0 => D.diagonal)
    @test sparse(D) ≈ spdiagm(0 => D.diagonal)
    @test Diagonal(D) ≈ Diagonal(D.diagonal)
end

# IdentityOperator concretization
@testset "IdentityOperator concretization" begin
    Random.seed!(1793)
    D = IdentityOperator(10)
    Dint = IdentityOperator(10, Int)
    @test Array(D) ≈ diagm(0 => ones(10))
    @test sparse(D) ≈ spdiagm(0 => ones(10))
    @test Diagonal(D) ≈ Diagonal(ones(10))
    @test Array(Dint) ≈ Matrix{Int}(I, 10, 10)
    @test sparse(Dint) ≈ SparseMatrixCSC{Int}(I, 10, 10)
    @test Diagonal(Dint) ≈ Diagonal(ones(Int, 10))
end
