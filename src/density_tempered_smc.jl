function gridSearch()

end

function 

function densityTemperedSMC()
    # initialize particles
    θ = rand(prior(θ0,Matrix{Float64}(I,k,k)),M)
    θ = Particles([θ[:,m] for m in 1:M])

    pθ = zeros(Float64,M)

    # parallelize this
    for m in 1:M
        Θm = StateSpaceModel(model(θ.x[m]...))
        Xm = bootstrapFilter(N,y,Θm)
        pθ[m] = Xm.logμ
    end
end