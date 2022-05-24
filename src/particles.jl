#=
    Particles is a parametric object type which stores a vector x of type XT,
    the particles, and weights each one by w. Unlike last time, Particles 
    encapsulate more information about the state of the filter at a given time; 
    intuitively the weights are evalutaed at a time t, and thus it makes sense 
    to convery that info as part of the particle set.

    Particles{XT,WT}
        x = vector of particles at time t
        xprev = vector of particles at t-1

        logw = vector of log weights
        w = vector of normalized weights
        maxw = maximum value of logw, used to normalize weights

        a = vector of indicies of the ancestor particles
        t = time index of the current state
=#

export Particles,reset_weights!,ESS,normalize!

struct Particles{XT,WT<:Real}
    x::Vector{XT}
    xprev::Vector{XT}

    logw::Vector{WT}
    w::Vector{WT}
    maxw::Base.RefValue{WT}

    a::Vector{Int}
    t::Base.RefValue{Int}
end

# default constructor of a particle set
Particles(N::Integer) = Particles(
    [zeros(N)],
    [zeros(N)],
    fill(-log(N),N),
    fill(1/N,N),
    Ref(0.),
    collect(1:N),
    Ref(1)
)

Base.length(p::Particles) = length(p.x)

# equation (19) of Tsay and Lopes
ESS(p::Particles) = 1.0/sum(abs2,p.w)

function reset_weights!(p)
    N = length(p)
    fill!(p.logw,log(1/N))
    fill!(p.w,1/N)
    p.maxw[] = 0
end

# normalizes weights and updates particle cloud (too complicated)
function normalize!(logw,w,maxw=Ref(zero(eltype(logw))))::eltype(logw)
    offset,maxind = findmax(logw)
    logw .-= offset

    # normalize new weights
    LoopVectorization.vmap!(exp,w,logw)
    sumw   = sum_all_but(w,maxind)
    w    .*= 1/(sumw+1)
    logw .-= log1p(sumw)

    # adjusted maximum log weight
    maxw[] += offset

    return log1p(sumw) + maxw[] - log(length(logw))
end

@inline normalize!(p) = normalize!(p.logw,p.w,p.maxw)