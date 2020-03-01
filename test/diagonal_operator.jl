using Test, DiffEqOperators, Random, LinearAlgebra

@testset "DiagonalOperator and IdentityOperator core functions" begin
    Random.seed!(1793)
    D = DiagonalOperator(rand(10))
    iden = DiagonalOperator(ones(10))
    iop = IdentityOperator(10)
    u = rand(10)
    M = rand(10, 10)
    AA = rand(9, 10)
    c1 = 2.3
    c2 = rand(10)

    # DiagonalOperator
    @test D.diagonal == DiffEqOperators.get_diagonal(D)
    @test size(D) == (10, 10)
    @test size(D, 1) == 10
    @test size(D, 2) == 10
    @test size(D, 100) == 1
    @test length(D) == 100
    @test iden * u ≈ u
    @test D * u ≈ Diagonal(D.diagonal) * u
    @test iden * M ≈ M
    @test D * M ≈ D.diagonal .* M
    @test DiffEqOperators.get_diagonal(c1 * D) ≈ c1 * D.diagonal
    @test DiffEqOperators.get_diagonal(c2 * D) ≈ c2 .* D.diagonal
    @test_throws DimensionMismatch D * AA
    @test_throws ErrorException size(D, 0)
    @test_throws ErrorException size(D, -100)

    # IdentityOperator
    @test DiffEqOperators.get_n(iop) == 10
    @test DiffEqOperators.eltype(iop) == Float64
    @test size(iop) == (10, 10)
    @test size(iop, 1) == 10
    @test size(iop, 2) == 10
    @test size(iop, 100) == 1
    @test length(iop) == 100
    @test iop * u == u
    @test iop * M == M
    @test DiffEqOperators.get_diagonal(c1 * iop) ≈ c1 * ones(iop.n)
    @test DiffEqOperators.get_diagonal(c2 * iop) ≈ c2
    @test_throws DimensionMismatch iop * AA
    @test_throws ErrorException size(iop, 0)
    @test_throws ErrorException size(iop, -100)
end

@testset "DiagonalOperator Composition with DerivativeOperators" begin
    Random.seed!(1793)
    D = DiagonalOperator(rand(10))
    Iop = IdentityOperator(10)
    u = ones(10)
    d = D.diagonal
    L = CenteredDifference(2, 2, 1., 10)
    c1 = 2.3
    c2 = rand(10)

    @test L ∘ D * u ≈ L * d
    @test L ∘ D * u ≈ (L ∘ D) * u
    @test L ∘ Iop * u ≈ zeros(8)
    @test L ∘ Iop * u ≈ (L ∘ Iop) * u
    @test c1 * L ∘ D * u ≈ c1 * L * d
    @test c1 * L ∘ D * u ≈ (c1 * L ∘ D) * u ≈ ((c1 * L) ∘ D) * u
    @test c1 * L ∘ Iop * u ≈ zeros(8)
    @test c1 * L ∘ Iop * u ≈ (c1 * L ∘ Iop) * u ≈ ((c1 * L) ∘ Iop) * u
    @test c2 * L ∘ D * u ≈ c2 * L * d
    @test c2 * L ∘ D * u ≈ (c2 * L ∘ D) * u ≈ ((c2 * L) ∘ D) * u
    @test c2 * L ∘ Iop * u ≈ zeros(8)
    @test c2 * L ∘ Iop * u ≈ (c2 * L ∘ Iop) * u ≈ ((c2 * L) ∘ Iop) * u
end

nothing
