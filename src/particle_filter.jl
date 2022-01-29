export bootstrapFilter,kalmanFilter

# this works, but could honestly perform a little better
function bootstrapFilter(
        N::Int64,
        y::Vector{Float64},
        prior::StateSpaceModel,
        B::Number = 0.5
    )
    T = length(y)

    # initialize algorithm
    ps = ParticleSet(N,prior.dim_x,T)

    # in the case where x0 is located at the origin
    x0 = (prior.dim_x == 1) ? 0.0 : zeros(Float64,prior.dim_x)
    p0 = Particles(rand(prior.transition(x0),N))

    ps.p[1] = resample(p0,B)

    for t in 2:T
        xt = rand.(prior.transition.(ps.p[t-1].x))
        wt = logpdf.(prior.observation.(xt),y[t])

        ## for higher dimensions of x
        # xt = [xt[:,i] for i in 1:size(xt,2)]
        wt += ps.p[t-1].logw
        ps.p[t] = resample(Particles(xt,wt),B)
    end

    return ps
end

# this might work, but I'm not sure and not inclined to benchmark
function auxiliaryParticleFilter(
        N::Int64,
        y::Vector{Float64},
        prior::StateSpaceModel,
        auxiliary::Function = mean,
        B::Number = 0.5,
        proposal = nothing
    )
    T = length(y)

    if proposal === nothing
        proposal = prior
    end

    # aux must be a function of a distribution
    aux(x) = auxiliary(prior.transition(x))

    # initialize algorithm
    ps  = ParticleSet(N,prior.dim_x,T)

    # in the case where x0 is located at the origin
    x0 = (prior.dim_x == 1) ? 0.0 : zeros(Float64,prior.dim_x)
    p0 = Particles(rand(proposal.transition(x0),N))

    ps.p[1] = resample(p0,B)

    for t in 2:T
        # first stage weights
        wt1 = logpdf.(prior.observation(aux.(ps.p[t-1].x)))
        xt1 = resample(reweight(ps.p[t-1].x,wt1),B)

        xt2 = rand.(proposal.transition.(xt1))
        wt2 = logpdf.(prior.observation.(xt1),y[t])

        # simplify calculations if the proposal is not provided
        if !(proposal === nothing)
            wt2 += logpdf.(prior.transition.(ps.p[t-1].x),xt)
            wt2 -= logpdf.(proposal.transition.(ps.p[t-1].x),xt)
        end

        ## for higher dimensions of x
        # xt = [xt[:,i] for i in 1:size(xt,2)]
        wt2 += ps.p[t-1].logw
        ps.p[t] = resample(Particles(xt2,wt2),B)
    end

    return ps
end


"""
    kalmanFilter(y,model)

Let x be normally a distributed AR process and y be a function of x.
Suppose we observe y, thus we use a Kalman filter to predict x.
More specifically, predict x using the following DLM...

* y[t]   = B*x[t] + δ   s.t. δ ~ N(0,R)
* x[t+1] = A*x[t] + ϵ   s.t. ϵ ~ N(0,Q)

```
model = NDLM(1.0,1.0,1.0,1.0)
x,y = simulate(100,model)
```
```julia-repl
> kalmanFilter(y,model)
100×3 Array{Float64,2}:
  -0.47693    0.00000    0.47694
  -0.41495    0.09382    0.60258
  -1.17056   -0.65821   -0.14586
  -0.94360   -0.43084    0.08192
   ⋮
  -1.49669   -0.98388   -0.47107
  -2.06760   -1.55479   -1.04198
  -0.84181   -0.32899    0.18382
```

Note: this model only works for 1 dimensional arrays, or vectors of x,
but has the capability to expand to multiple dimensions.

"""
function kalmanFilter(y::Vector{Float64},model::LinearGaussian)
    T = length(y)

    x = zeros(Float64,T+1,model.dim_x)
    Σ = zeros(Float64,T+1,model.dim_x)
    
    qs = zeros(Float64,T,3)

    x[1] = 0.0
    Σ[1] = 1.0

    for t in 1:T
        # calculate the Kalman gain
        Σxy = Σ[t]*(model.B)'
        Σyy = (model.B)*Σ[t]*(model.B)'+(model.R)
        gain = Σxy*inv(Σyy)
        
        # calculate minimally variant x[t+1] and propogate
        xf = x[t] + gain*(y[t]-(model.B)*x[t])
        x[t+1] = (model.A)*xf
        
        # calculate covariance and propogate
        Σf = Σ[t] - gain*(model.B)*Σ[t]
        Σ[t+1] = (model.A)*Σf*(model.A)' + model.Q

        qs[t,:] = quantile(Normal(xf,sqrt(Σf)),[.25,.50,.75])
    end

    return qs
end
