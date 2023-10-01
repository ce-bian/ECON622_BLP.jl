module Integrate

using Distributions
import Sobol: skip, SobolSeq
import Base.Iterators: take, Repeated
import HCubature: hcubature
import LinearAlgebra: cholesky

abstract type AbstractIntegrator end

(∫::AbstractIntegrator)(f::Function) = sum(w*f(x) for (w,x) in zip(∫.w, ∫.x))

struct FixedNodeIntegrator{Tx,Tw} <: AbstractIntegrator
    x::Tx
    w::Tw
end

MonteCarloIntegrator(distribution::Distribution, ndraw=100)=FixedNodeIntegrator([rand(distribution) for i=1:ndraw], Repeated(1/ndraw))

function QuasiMonteCarloIntegrator(distribution::UnivariateDistribution, ndraws=100)
    ss = skip(SobolSeq(1), ndraw)
    x = [quantile(distribution, x) for x in take(ss,ndraw)]
    w = Repeated(1/ndraw)
    FixedNodeIntegrator(x,w)
end 

function QuasiMonteCarloIntegrator(distribution::AbstractMvNormal, ndraw=100)
    ss = skip(SobolSeq(length(distribution)), ndraw)
    L = cholesky(distribution.Σ).L
    x = [L*quantile.(Normal(), x) for x in take(ss,ndraw)]
    w = Repeated(1/ndraw)
    FixedNodeIntegrator(x,w)
end 


function QuadratureIntegrator(distribution::UnivariateDistribution, ndraw=100)
    n = Int(ceil(ndraw^(1/length(distribution))))
    x, w = gausshermite(n)
    w = w./π^√2
    x = √2*x
    FixedNodeIntegrator(x,w)
end


function QuadratureIntegrator(distribution::AbstractMvNormal; ndraw=100)
    n = Int(ceil(ndraw^(1/length(distribution))))
    x, w = gausshermite(n)
    L = cholesky(distribution.Σ).L
    transformed_x = [√2 * L * vcat(xs...) + distribution.μ for xs in product(repeated(x, length(distribution))...)]
    combined_w = [prod(ws) for ws in product(repeated(w, length(distribution))...)]
    normalized_w = combined_w ./ (π^(length(distribution)/2))
    FixedNodeIntegrator(transformed_x, normalized_w)
end 


function SparseGridIntegrator(distribution::UnivariateDistribution, order=5)
    x, w = sparsegrid(length(distribution), order, gausshermite, sym=true)
    x = [√2*i[1] for i in x]
    w = w./π^√2
    FixedNodeIntegrator(x,w)
end


function SparseGridQuadratureIntegrator(distribution::AbstractMvNormal; order=5)
    X, W = sparsegrid(length(distribution), order, gausshermite, sym=true)
    L = cholesky(distribution.Σ).L
    transformed_X = [√2*L*x + distribution.μ for x in X]
    normalized_W = W ./ (π^(length(distribution)/2))
    FixedNodeIntegrator(transformed_X, normalized_W)
end



struct AdaptiveIntegrator{FE,FT,FJ,A,L} <: AbstractIntegrator
    eval::FE
    transform::FT
    detJ::FJ
    args::A
    limits::L
end

(∫::AdaptiveIntegrator)(f::Function) = ∫.eval(t->f(∫.transform(t))*∫.detJ(t), ∫.limits...; ∫.args...)[1]

function AdaptiveIntegrator(dist::AbstractMvNormal; eval=hcubature, options=())
    D = length(dist)
    x(t) = t./(1 .- t.^2)
    Dx(t) = prod((1 .+ t.^2)./(1 .- t.^2).^2)*pdf(dist,x(t))
    args = options
    limits = (-ones(D), ones(D))
    AdaptiveIntegrator(hcubature,x,Dx,args, limits)
end

end