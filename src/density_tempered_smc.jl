# this one needs quite a bit of work, but that's okay since I already have a
# working version of the old code

function densityTemperedSMC(
        N::Int64,
        M::Int64,
        y::Vector{Float64},
        θ0::Vector{Float64},
        model=LinearGaussian
    )
    T = length(y)
    k = length(θ0)

    lb = [-1.0,-1.0,0.0,0.0]
    ub = [1.0,1.0,2.0,2.0]
    Σ0 = Matrix{Float64}(I,k,k)

    θ = randTruncatedMvNormal(N,θ0,Σ0,lb,ub)
    θ = Particles([θ[:,i] for i in 1:N],fill(-1*log(N),N))

    ## try catch this later on to make sure the model is well defined
    # model = eval(Meta.parse(model))

    for i in 1:N
        # see equation (4) from Duan & Fulop
        mod_i = StateSpaceModel(model(θ.x...))
        
        # find the likelihood p_hat as Duan & Fulop call it

        # weights evenly for l=1
        S[1][i] = -log(N)
    end
end
