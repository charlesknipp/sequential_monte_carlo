export forecast,density_forecast,wlr_test

#=
Forecasting is relatively complex and there are a few scenarios I have to iron
out. For brevity, refer to the following list for concerns/problems/quirks:
    - should I return the observation or the state variable?
    - how should I handle samplers which have a cloud of parameterizations?
    - should I defined in-sample vs out-of-sample as different operations?
    - I'm still unclear about the rolling window operation

I currently only report the forecasted measurement, since that is univariate
across models (for now) which results in some consistency. I also only report
sample statistics instead of full densities, which is an easy switch to make.
As a result, I also output the summary stats of each parameterization over an
IBIS or SMC object.
=#

# for univariate linear models, extend the Kalman filter for floats
function forecast(
        model::LinearModel,
        xt::XT,
        Σt::ΣT,
        j::Int64
    ) where {XT,ΣT}
    A,Q = model.A,model.Q
    B,R = model.B,model.R

    # perform the prediction step for j periods ahead
    for _ in 1:j
        xt = A*xt
        Σt = A*Σt*A' + Q
    end

    # get the predicted measurement
    yt = B*xt
    Σy = B*Σt*B' + R

    return yt[1],Σy[1,1]
end

"""
forecast(ibis,N,j)

this function performs j period ahead forecasts for a generalized IBIS object
and returns the distribution of the measurement variable.
"""
function forecast(ibis::IBIS,j::Int64,N::Int64=ibis.M)
    idx = sample(1:ibis.M,Weights(ibis.ω),N)
    i   = 0

    ys = zeros(N)
    Σs = zeros(N)

    for n in idx
        y,Σ = forecast(ibis.model(ibis.θ[n]),ibis.x[n],ibis.Σ[n],j)
        i  += 1
        
        ys[i] = y
        Σs[i] = Σ
    end

    # return expexted values for now since technically resampled
    return mean(ys),mean(Σs)
end


# for non-linear models, borrow from the bootstrap filter
function forecast(
        rng::AbstractRNG,
        model::StateSpaceModel,
        xs::Vector{XT},
        j::Int64
    ) where XT

    xp = deepcopy(xs)
    N  = length(xs)
    ys = zeros(N)

    for _ in 1:j, n in 1:N
        xp[n] = rand(rng,transition(model,xp[n]))
        ys[n] = rand(rng,observation(model,xp[n]))
    end

    return ys
end

# write a default constructor for global rng
function forecast(
        model::StateSpaceModel,
        xs::Vector{XT},
        j::Int64
    ) where XT
    return forecast(Random.GLOBAL_RNG,model,xs,j)
end

"""
forecast(smc,N,j)

this function performs j period ahead forecasts for a generalized SMC object
and returns the distribution of the measurement variable.
"""
function forecast(smc::SMC,j::Int64)
    idx = sample(1:smc.M,Weights(smc.ω),smc.M)
    i   = 0

    # preallocate the particle cloud
    ys = zeros(smc.M)

    # calculate for all parameters and take weighted sum over the entire trajectory
    for m in idx
        yn = forecast(smc.model(smc.θ[m]),smc.x[m],j)
        i += 1
        
        ys[i] = smc.w[m]'*yn
    end

    return ys
end


## TEST STATISTICS ############################################################

#=
    if all else fails, pick a model with parameters using y[(t-m):t] and fore-
    cast n steps ahead; this is the so called "empirical bayes" method
=#

# one step ahead density forecasts
function density_forecast(
        ibis::IBIS,
        y::Vector{Float64},
        m::Int64
    )
    # initialize the denisty forecast
    likelihood = 0.0

    # integrate out the parameter uncertainty
    for i in 1:ibis.M
        model = ibis.model(ibis.θ[i])
        μt,σt = forecast(model,ibis.x[i],ibis.Σ[i],1)
        
        forecast_dist = Normal(μt,sqrt(σt))
        likelihood += log(ibis.ω[i]*pdf(forecast_dist,y[m+1]))
    end

    return likelihood
end

# t steps ahead...
function density_forecast(
        ibis::IBIS,
        y::Vector{Float64},
        m::Int64,
        t::Int64
    )
    # initialize the denisty forecast
    likelihood = 0.0

    # integrate out the parameter uncertainty
    for i in 1:ibis.M
        model = ibis.model(ibis.θ[i])
        μt,σt = forecast(model,ibis.x[i],ibis.Σ[i],t)
        
        forecast_dist = Normal(μt,sqrt(σt))
        likelihood += log(ibis.ω[i]*pdf(forecast_dist,y[m+t]))
    end

    return likelihood
end

# work in progress...
function wlr_test(
        ibis_1::IBIS,
        ibis_2::IBIS,
        y::Vector{Float64},
        m::Int64,
        n::Int64
    )
    wlr = zeros(n)
    
    # calculate the wlr
    for t in 1:n
        logf = density_forecast(ibis_1,y,m,t)
        logg = density_forecast(ibis_2,y,m,t)
        wlr[t] = logf-logg
    end

    # obtain the heteroskedasticity and autocorrelation consistent estimator
    newey_west = (2/n)*sum(wlr[t]*wlr[1] for t in 1:n)
    σn² = (1/n)*sum(abs2,wlr) + newey_west

    # calculate the test statistic (using equation 4)
    return mean(wlr)/sqrt(σn²/n)
end
