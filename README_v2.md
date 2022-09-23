# SequentialMonteCarlo.jl

SequentialMonteCarlo.jl presents a set of particle filters and Sequential Monte Carlo algorithms for both latent state and model parameter inference.

### State Space Models

To begin using this module, we must first properly establish the construction of state space models (SSMs) under my framework. Typical model construction requires a small set of constructors for: (1) an initial distribution, (2) a transition density, and (3) an observation density.

To construct a univariate linear Gaussian SSM, we can define the following constructor, where `θ::Vector{Float64}` represents a vector of parameters.

```julia
lg_mod(θ) = StateSpaceModel(
    LinearGaussian(θ[1],1.0,θ[2],θ[3],0.0),
    (1,1)
)
```

Calling `lg_mod(⋅)` constructs a `StateSpaceModel` object, which can be called to simulate data, or to be used in a particle filter. As long as the model in question has the required methods defined. For demonstration define `lg_example` and simulate 100 periods.

```julia
lg_example = lg_mod([0.5,0.9,0.8])
_,y = simulate(lg_example,100)
```

This construction weighs heavily on multiple dispatch, which is a nice feature of Julia but also requires more work from the user. As such, a better method is in the works.

### Particle Filters

Particle filters are primarily used for online inference for latent states, which is ideally defined as a mutating operation given a set of particles `x` and weights `w`.

Suppose we observe a vector `y` simulated from model `lg_example`. To construct a bootstrap filter with 1024 particles and capture summary statistics at each time period we do the following:

```julia
# preallocate quantile vector and log likelihood
xq = fill(zeros(Float64,3),length(y))
logZ = 0.0

# initialize bootstrap filter at time t=1
x,w,logμ = bootstrap_filter(1024,y[1],lg_example)

xq[1] = quantile(x,[0.25,0.5,0.75])
logZ += logμ

# subsequent iterations of the bootstrap filter
for t in 2:length(y)
    # run filter and print ess
    logμ,w,ess = bootstrap_filter!(x,w,y[t],lg_example)
    @printf("t = %4d\tess = %4.3f",t,ess)

    # update summary statistics
    xq[t] = quantile(x,[0.25,0.5,0.75])
    logZ += logμ
end
```

It should be noted that `x` and `w` are both operated in-place and thus update with each call of `bootstrap_filter!()`. Furthermore, to avoid such long winded code, one can instead call the function `log_likelihood()` to run a particle filter over all periods of a model and return the log likelihood (or `logZ` as its referred to above).

```julia
# for a bootstrap filter leave the proposal argument empty
logZ = log_likelihood(1024,y,lg_example)
```

The construction of these particle filters is not perfect on its own, since it is meant specifically to perform joint estimation. However, they are fully functional and meant to be flexible enough when called in high volumes. Something that `LowLevelParticleFilters.jl` actually fails to do (more on this in issue...).

### Joint Inference

Suppose a given state space model has an uncertain construction such that the model parameters are unobserved. Now the problem becomes twofold: what can we infer about the latent states as well as the parameters?

To solve this problem, I introduce two main algorithms: [SMC²](https://arxiv.org/pdf/1101.1528.pdf) and [density tempered SMC](https://www.tandfonline.com/doi/pdf/10.1080/07350015.2014.940081).

Running these algorithms requires a prior for the parameter space as well as a model constructor as we defined above with `lg_mod()`. Below we define `lg_prior` such that `length(lg_prior) == length(θ)` where `θ` is the input to the function `lg_mod`.

```julia
# model constructor
lg_mod(θ) = StateSpaceModel(
    LinearGaussian(θ[1],1.0,θ[2],θ[3],0.0),
    (1,1)
)

# prior definition
lg_prior = product_distribution([
    TruncatedNormal(0,1,-1,1),
    LogNormal(),
    LogNormal()
])
```

Upon declaration of the prior and model constructor, we define a generic SMC sampler with 513 parameter particles, 1024 state particles, and ESS threshold of 0.5, and 3 MCMC steps. To demonstrate, we can run density tempered SMC like so...

```julia
lg_dt_smc = SMC(512,1024,lg_mod,lg_prior,3,0.5)
density_tempered(lg_dt_smc,y)
```

For an online algorithm like SMC², we treat it similarly to the particle filter by using mutating functions to change properties of `lg_smc²` as time progresses.

```julia
lg_smc² = SMC(512,1024,lg_mod,lg_prior,3,0.5)
smc²(lg_smc²,y)

for t in 2:T
    smc²!(lg_smc²,y,t)
end
```
