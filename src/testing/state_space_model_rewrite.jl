using LinearAlgebra
using Distributions


abstract type AbstractSSM end
abstract type ModelParameters end

struct StateSpaceModel <: AbstractSSM
    # define general structure using functions
    transition::Function
    observation::Function
    
    # need the dims for initializing states...maybe...
    dim_x::Int64
    dim_y::Int64
end


# for now assume x0 and Σ0 are known
struct LinearGaussian <: ModelParameters

    A::Union{Float64,Matrix{Float64}}
    B::Union{Float64,Matrix{Float64}}

    Q::Union{Float64,Matrix{Float64}}
    R::Union{Float64,Matrix{Float64}}

    # implicitly defined by the constructor
    dim_x::Int64
    dim_y::Int64

    function LinearGaussian(A,B,Q,R)
        # determine the dimensions
        dim_x,dim_y = size(A,1),size(B,1)

        @assert dim_x == size(A,2) "A is not a square matrix"
        @assert dim_y == size(B,2) "B is not a square matrix"

        @assert size(Q,1) == size(Q,2) "Q is not a square matrix"
        @assert size(R,1) == size(R,2) "R is not a square matrix"

        @assert dim_x == size(Q,1) "A,Q dimension mismatch"
        @assert dim_y == size(R,1) "B,R dimension mismatch"

        @assert issymmetric(Q) "Q is not symmetric"
        @assert issymmetric(R) "R is not symmetric"

        @assert isposdef(Q) "Q is not positive definite"
        @assert isposdef(R) "R is not positive definite"

        # construct the new object
        new(A,B,Q,R,dim_x,dim_y)
    end
end


function StateSpaceModel(params::LinearGaussian)
    # import parameters
    A,B = params.A,params.B
    Q,R = params.Q,params.R

    dim_x = params.dim_x
    dim_y = params.dim_y

    # depending on the type of input set the kernel
    Kx = (dim_x == 1) ? Normal : MvNormal
    Ky = (dim_y == 1) ? Normal : MvNormal

    f(xt) = Kx(A*xt,sqrt(Q))
    g(xt) = Ky(B*xt,sqrt(R))

    return StateSpaceModel(f,g,dim_x,dim_y)
end


function simulate(model::StateSpaceModel,T::Int64)
    y = (model.dim_y == 1) ? 0.0 : zeros(Float64,model.dim_y)
    y = fill(y,T)
    
    x = (model.dim_x == 1) ? 0.0 : zeros(Float64,model.dim_x)
    x = fill(x,T)

    for t in 1:(T-1)
        y[t] = rand(model.observation(x[t]))
        x[t+1] = rand(model.transition(x[t]))
    end

    y[T] = rand(model.observation(x[T]))

    return (x,y)
end

# test_params = LinearGaussian(1.0,1.0,1.0,1.0)
# test_model  = StateSpaceModel(test_params)

# simulate(test_model,100)


const ParticleType = Union{Float, Vector{Float}}

mutable struct Particles{T <: ParticleType}
    x::Vector{T}
    logw::Vector{Float64}
end

function Particles(N::Int64,dims::Int64=1)
    particles   = (dims == 1) ? 0.0 : zeros(Float64,dims)
    log_weights = fill(-1*log(N),N)

    return Particles(particles,log_weights)
end

function mean(p::Particles)
    # convert the vector of vectors to a matrix
    k = length(p.x[1])
    x = reduce(hcat,p.x)
    max_x = maximum(x,dims=2)
    w = exp.(p.logw.-maximum(p.logw))

    μ = [exp(max_x[i])*sum(w.*exp.(x[i,:].-max_x[i]))/sum(w) for i in 1:k]
    
    return μ
end


mutable struct ParticleSet
    p::Vector{Particles}
end

function ParticleSet(N::Int64,dim::Int64,T::Int64)
    particle_set = [Particles(N,dim) for _ in 1:T]

    return ParticleSet(particle_set)
end

function mean(ps::ParticleSet)
    return [mean(p) for p in ps.p]
end


function ESS(logx::Vector{Float64})
    max_logx = maximum(logx)
    x_scaled = exp.(logx.-max_logx)

    ESS = sum(x_scaled)^2 / sum(x_scaled.^2)
    if ESS == NaN; ESS = 0.0 end

    return ESS
end

function resample(p::Particles,B::Number=Inf)::Tuple{Particles,Float64}
    ess = ESS(p.logw)
    N = length(p.x)

    if ess < B*N
        w = exp.(p.logw.-maximum(p.logw))
        w = w/sum(w)
        κ = wsample(1:N,w,N)
        
        x = p.x[κ]
        w = fill(-1*log(N),N)

        return (Particles(x,w),ess)
    else
        return (p,ess)
    end
end


function bootstrapFilter(
        N::Int64,
        y::Vector{Union{Float64,Vector{Float64}}},
        prior::StateSpaceModel,
        B::Float64 = 0.5,
        proposal::StateSpaceModel = nothing
    )
    T = length(y)

    if proposal === nothing
        proposal = prior
    end

    # initialize algorithm
    ps  = ParticleSet(N,prior.dim_x,T)
    ess = zeros(Float64,T)

    # in the case where x0 is located at the origin
    x0 = (prior.dim_x == 1) ? 0.0 : zeros(Float64,prior.dim_x)
    p0 = Particles(rand(proposal.transition(x0),N),fill(-1*log(N),N))

    ps.p[1],ess[1] = resample(p0,B)

    for t in 2:T
        xt = rand.(proposal.transition.(ps.p[t-1].x))
        w = logpdf.(prior.observation.(xt),y[t])

        if !(proposal === nothing)
            w += logpdf.(prior.transition.(ps.p[t-1].x),xt)
            w -= logpdf.(proposal.transition.(ps.p[t-1].x),xt)
        end

        #w += ps.p[t-1].logw
        scaled_w = exp.(w.-maximum(w))
        normed_w = scaled_w/sum(scaled_w)
        
        #xt = [xt[:,i] for i in 1:size(xt,2)]
        ps.p[t],ess[t] = resample(Particles(xt,normed_w),B)
    end
    return (ps,ess)
end
