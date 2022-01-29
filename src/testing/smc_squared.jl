mutable struct SMC²
    θ::Particles
    X::Particles

    model
    y::AbstractVector

    t::Int64

    function SMC²(N::Int64,M::Int64,y,model,θ0::AbstractVector)
        X = ParticleSet([Particles()])
        new()
    end
end