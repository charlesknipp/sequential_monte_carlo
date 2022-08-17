include("reworked_particles.jl")

#=
    define an object type for this... no way you want to keep storing all that 
    information globally
=#

mutable struct SMC²{SSM}
    N::Int64
    M::Int64

    θ::Vector{Vector{Float64}}
    ω::Vector{Float64}

    x::Vector{Vector{Float64}}  # later shoule be Vector{Vector{XT}}
    w::Vector{Vector{Float64}}

    ess::Float64
    ess_min::Float64
    chain::Int64

    model::SSM
    prior::Sampleable

    rng::AbstractRNG
end

# define outer constructor
function SMC²(
        rng::AbstractRNG,
        N::Int64,M::Int64,
        y::Float64,
        model::SSM,
        prior::Sampleable,
        chain::Int64,
        ess_threshold::Float64
    ) where SSM

    ## initialize parameter particles
    θ = map(x->rand(rng,prior),1:M)
    ω = zeros(Float64,M)

    ## initialize state particles
    x = fill(zeros(Float64,N),M)
    w = similar(x)

    ## set the particle weights at t = 0
    for m in 1:M
        x[m],w[m],ω[m] = bootstrap_filter(rng,N,y,model(θ[m]))
    end

    _,ω,ess = normalize(ω)
    ess_min = ess_threshold*M

    return SMC²(N,M,θ,ω,x,w,ess,ess_min,chain,model,prior,rng)
end


function update!(smc²::SMC²,y::Float64)
    ## alias variables
    x,w = smc².x,smc².w
    θ,ω = smc².θ,smc².ω
    logω = copy(log.(ω))

    for m in 1:M
        likelihood,w[m],_ = bootstrap_filter!(smc².rng,x,w,y,smc².model(θ[m]))
        logω[m] += likelihood
    end

    _,ω,smc².ess = normalize(logω)

    ## TODO: rejuvenation step
end

prior = product_distribution([
    TruncatedNormal(0,1,-1,1),
    LogNormal(),
    LogNormal()
])

mod_func(θ) = StateSpaceModel(
    LinearGaussian(θ[1],1.0,θ[2],θ[3],0.0),
    (1,1)
)