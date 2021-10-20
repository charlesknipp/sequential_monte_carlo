# Sequential Monte Carlo Methods

This module is intended for personal development and understanding of sequential monte carlo techniques and algorithms. Supported models include only Gaussian dynamic linear models, but support for more is feasible and likely will be implemented in future iterations of this module.

## Creating a model

Built into the module, I define a class called `NDLM` which describes the parameters of a Normal dynamic linear model (NDLM for short, hence the name). This is by design since certain algorithms test the fit of estimated model parameters against a known vector of noise typically denoted as `y` throughout this program. To create a model, and simulate `T` periods of data, define the model and plug it in to the `simulate()` function defined for `NDLM` object types.

```julia
> sim = NDLM(A=0.8,B=1.0,Q=1.0,R=1.0)
> x,y = simulate(T=200,model=sim)
```

Note, `x0` and `Σ0` are set around the origin with unit deviation, but these can be inputs to an `NDLM` as well.

## Filters

If the name doesn't give it away, a large part of this module is filtering. Which can be broken down into two categories: observed parameters or unobserved parameters. For the first category there are currently 3 working filters: a Kalman Filter, a bootstrap filter, and an auxiliary particle filter. The other category currently only has one filter: the density tempered marginalized sequential monte carlo sampler based on the paper of the same name by J. Duan and A. Fulop.

### Kalman Filter

To demonstrate the use of the Kalman filter, let us use the model we simulated above along with the function `kalmanFilter()` to output the quantiles of `x` at each period.

```julia
> kalmanFilter(y,sim)
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

### Bootstrap Filter

The Bootstrap Filter is defined by the function `boostrapFilter()` which outputs a tuple of 3 items: a vector of the normalizing constants `Z`, the particle cloud generated at all periods `xs`, and the quantiles of the particle cloud at each period. Again we use the simulations from above to output the quantiles.

```julia
> boostrapFilter(1000,y,sim)[3]
100×3 Array{Float64,2}:
  -0.59210    0.00624    0.62103
  -1.17245   -0.71156   -0.20154
   0.28959    0.80968    1.42300
   ⋮
   0.94486    1.53216    2.08224
  -1.02793   -0.52479   -0.06452
  -0.95006   -0.37365    0.10438
```

### Auxiliary Particle Filter

The Auxiliary Particle Filter is defined by the function `auxiliaryParticleFilter()`, and the process is identicle to the previous filter.

```julia
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

### Density Tempered Marginalized Sequential Monte Carlo

This work in progress algorithm was inspired by the paper of the same name by J. Duan and A. Fulop, and is implemented as the function `densityTemperedSMC()`. As mentioned this is a work in progress and while it compiles, the parameters that it outputs are only in rough neighborhoods of where they are expected to be.
