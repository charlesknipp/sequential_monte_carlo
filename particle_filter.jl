using LinearAlgebra,Distributions,Statistics


"""
    boostrapFilter(n,y,model)

Let x be normally a distributed AR process and y be a function of x.
Suppose we observe y, so use a particle filter to predict x.
More specifically, predict x using the following DLM...

* y[t]   = B*x[t] + δ   s.t. δ ~ N(0,R)
* x[t+1] = A*x[t] + ε   s.t. ε ~ N(0,Q)
* x[1] ~ N(x0,Σ0)

```
model = NDLM(1.0,1.0,1.0,1.0)
x,y = simulate(100,model)
```
```julia-repl
> boostrapFilter(1000,y,model)[3]
100×3 Array{Float64,2}:
  -0.59210    0.00624    0.62103
  -1.17245   -0.71156   -0.20154
   0.28959    0.80968    1.42300
   ⋮
   0.94486    1.53216    2.08224
  -1.02793   -0.52479   -0.06452
  -0.95006   -0.37365    0.10438
```
"""
function bootstrapFilter(n::Int64,y::Vector{Float64},model::NDLM)
    T = length(y)

    xs = zeros(Float64,T,n)
    qs = zeros(Float64,T,3)
    Z  = zeros(Float64,T)

    x = rand(Normal(model.x0,sqrt(model.Σ0)),n)

    for t in 1:T
        # propogate forward in time
        x = (model.A)*x + rand(Normal(0,sqrt(model.Q)),n)
        
        # calculate likelihood that y fits the simulated distribution
        d  = Normal.((model.B)*x,sqrt(model.R))
        wt = logpdf.(d,y[t])

        # normalize weights and resample
        w = exp.(wt.-maximum(wt))
        w = w/sum(w)
        κ = wsample(1:n,w,n)
        x = x[κ]

        # store the normalizing constant and sample
        Z[t] = mean(exp.(wt))
        xs[t,:] = x
        qs[t,:] = quantile(x,[.25,.50,.75])
    end

    return Z,xs,qs
end


"""
    auxiliaryParticleFilter(n,y,model)

Let x be normally a distributed AR process and y be a function of x.
Suppose we observe y, so use a particle filter to predict x.
More specifically, predict x using the following DLM...

* y[t]   = B*x[t] + δ   s.t. δ ~ N(0,R)
* x[t+1] = A*x[t] + ε   s.t. ε ~ N(0,Q)
* x[1] ~ N(x0,Σ0)

```
model = NDLM(1.0,1.0,1.0,1.0)
x,y = simulate(100,model)
```
```julia-repl
> auxiliaryParticleFilter(1000,y,model)[3]
100×3 Array{Float64,2}:
  -0.59210    0.00624    0.62103
  -1.17245   -0.71156   -0.20154
   0.28959    0.80968    1.42300
   ⋮
   0.94486    1.53216    2.08224
  -1.02793   -0.52479   -0.06452
  -0.95006   -0.37365    0.10438
```
"""
function auxiliaryParticleFilter(n::Int64,y::Vector{Float64},model::NDLM)
    T = length(y)

    xs = zeros(Float64,T,n)
    qs = zeros(Float64,T,3)
    Z  = zeros(Float64,T)

    x = rand(Normal(model.x0,sqrt(model.Σ0)),n)

    for t in 1:T
        # first stage weight
        w1 = logpdf.(Normal.((model.B*model.A)*x,sqrt(model.R)),y[t])
        w1 = exp.(w1.-maximum(w1))
        w1 = w1/sum(w1)
        
        # resample and propogate forward
        κ = wsample(1:n,w1,n)
        x1 = (model.A)*x[κ] + rand(Normal(0,sqrt(model.Q)),n)

        # calculate second stage weights
        w2 = logpdf.(Normal.(model.B*x1,sqrt(model.R)),y[t])
        w2 = exp.(w2.-maximum(w2))./w1[κ]
        w2 = w2/sum(w2)

        # resample
        x = wsample(x1,w2,n)

        # store the normalizing constant and sample
        Z[t] = mean(exp.(w2))
        xs[t,:] = x
        qs[t,:] = quantile(x,[.25,.50,.75])
    end

    return Z,xs,qs
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
> kalmanFilter(y,0.0,1.0,1,1,1.0,1.0)
100×3 Matrix{Float64}:
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
function kalmanFilter(y::Vector{Float64},model::NDLM)
    T = length(y)

    x = zeros(Float64,T+1,1)
    Σ = zeros(Float64,T+1,1)
    
    qs = zeros(Float64,T,3)

    x[1] = model.x0
    Σ[1]  = model.Σ0

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