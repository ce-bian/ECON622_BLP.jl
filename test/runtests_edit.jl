using Test

@testset "integrator" begin
    # include("../src/integrate_edit.jl") # for interactive execution
    using Distributions

    dimx = 3
    A = rand(dimx,dimx)
    Σ = A*A'
    dx = MvNormal(zeros(dimx), Σ)
    ∫a = Integrate_edit.AdaptiveIntegrator(dx, options=(rtol=1e-6,initdiv=3))        
    V = ∫a(x->x*x')
    @test isapprox(V, Σ, rtol=1e-5)
    
    val = ∫a(f)
    for N ∈ [1_000, 10_000, 100_000]
        ∫mc = Integrate_edit.MonteCarloIntegrator(dx, N)
        ∫qmc = Integrate_edit.QuasiMonteCarloIntegrator(dx,N)        
        @test isapprox(∫mc(x->x*x'), Σ, rtol=10/sqrt(N))
        @test isapprox(∫qmc(x->x*x'), Σ, rtol=10/sqrt(N))

        f(x) = exp(x[1])/sum(exp.(x))        
        @test isapprox(∫mc(f),val,rtol=1/sqrt(N))
        @test isapprox(∫qmc(f),val,rtol=1/sqrt(N))
    end
end

@testset "share=δ⁻¹" begin
    # include("../src/blp.jl") 
    using LinearAlgebra
    J = 4
    dimx=2
    dx = MvNormal(dimx, 1.0)
    Σ = [1 0.5; 0.5 1]
    N = 1_000
    ∫ = BLP.Integrate_edit.QuasiMonteCarloIntegrator(dx, N)
    X = [(-1.).^(1:J) 1:J]
    δ = collect((1:J)./J)
    s = BLP.share(δ,Σ,X,∫) 
    d = BLP.delta(s, Σ, X, ∫)
    @test d ≈ δ

    J = 10
    dimx = 4
    X = rand(J, dimx)
    dx = MvNormal(dimx, 1.0)
    Σ = I + ones(dimx,dimx)
    ∫ = BLP.Integrate_edit.QuasiMonteCarloIntegrator(dx, N)
    δ = 1*rand(J)
    s = BLP.share(δ,Σ,X,∫) 
    d = BLP.delta(s, Σ, X, ∫)
    @test isapprox(d, δ, rtol=1e-6)
    
end


# update version
@testset "QuadratureIntegrator" begin
    dimx = 3
    A = rand(dimx, dimx)
    Σ = A * A'
    dx = MvNormal(zeros(dimx), Σ)

    f(x) = exp(x[1]) / sum(exp.(x))

    # Testing the AdaptiveIntegrator
    ∫a = Integrate_edit.AdaptiveIntegrator(dx, options=(rtol=1e-6, initdiv=3))        
    V = ∫a(x -> x * x')
    @test isapprox(V, Σ, rtol=1e-5)

    val = ∫a(f)
    
    for N in [1_000, 10_000, 100_000]
 
        ∫q = Integrate_edite.QuadratureIntegrator(dx, ndraw=N)
        @test isapprox(∫q(x -> x * x'), Σ, rtol=1e-5)
        @test isapprox(∫q(f), val, rtol=1e-5)

        # Testing the SparseGridQuadratureIntegrator with various order
        for order in [3, 5, 7]
            ∫sgq = Integrate_edit.SparseGridQuadratureIntegrator(dx, order=order)
            @test isapprox(∫sgq(x -> x * x'), Σ, rtol=1e-5)
            @test isapprox(∫sgq(f), val, rtol=1e-5)
        end
    end
end


@testset "delta function" begin
    s = [0.2, 0.3, 0.5]  
    Σ = [1 0.5 0; 0.5 1 0; 0 0 1]  
    x = [1, 2, 3] 
    N = 1_000
    ∫ = Integrate_edit.QuasiMonteCarloIntegrator(MvNormal(3, 1.0), N)  

    @testset "test1" begin
        s = [0.2, 0.3, 0.5]
        Σ = [1 0.5 0; 0.5 1 0; 0 0 1]
        x = [1, 2, 3]
        N = 1_000
        ∫ = Integrate_edit.QuasiMonteCarloIntegrator(MvNormal(3, 1.0), N)
        
        δ = Integrate_edit.delta(s, Σ, x, ∫)
        
    end

   # below is from ChatGPT...
    # A scenario where delta should fail and emit a warning
    @testset "test2" begin
        s_faulty = [0.2, 0.3, 0.6]  # does not sum to 1
        
        # Expect a warning to be emitted, which means that delta should NOT successfully compute a solution
        @test_logs (:warn, r"Possible problem in delta\(s, \.\.\.\)\n.*") YourModule.delta(s_faulty, Σ, x, ∫)
        
        # Alternatively, if delta throws an error on failure, you might want to check that this happens:
        @test_throws YourExpectedExceptionType YourModule.delta(s_faulty, Σ, x, ∫)
        # Replace `YourExpectedExceptionType` with the type of error you expect, e.g., DomainError
    end
end
