## Density Tempered Particle Filter

Based on the algorithm presented in [(Duan & Fulop, 2013)](https://www.tandfonline.com/doi/pdf/10.1080/07350015.2014.940081), the density tempered particle filter estimates the joint posterior of latent states and model parameters by way of a tempering scheme.

Consider a state space model with latent states `x[t]` and observations `y[t]`; let vector `θ` parameterize the transition density `f(x[t]|θ)` and the observation density `g(x[t]|θ)`. For parameter estimation, let `prior` represent the prior of `θ` with a tempering scheme defined by a sequence of `ξ`s. For the filters themselves, let `M` represent the number of `θ` particles and `N` be the number of state particles.


### Algorithm

Accompanying this algorithm are a selection of helper functions necessary to understanding the particle filter. The first is a simple calculation of the effective sample size (ESS) specifically using the weighting scheme from [(Duan & Fulop, 2013)](https://www.tandfonline.com/doi/pdf/10.1080/07350015.2014.940081).

```julia
function ess(Δξ,Z)
    s = [Z[i]^(Δξ) for i in 1:M]
    return sum(s)^2 / sum([s[i]^2 for i in 1:M])
end
```

Subsequently, we must define a sequential Monte Carlo method to calculate the marginal densities `Z`, as noted in the algorithm. Specifically, we employ a bootstrap filter for its simplicity and effectiveness.

```julia
function bootstrap_filter(N,y,θ)
    xt = rand(μ(θ),N)
    
    for t in 1:T
        # propagate
        for i in 1:N
            xt[i] = rand(f(xt[i],θ))
            w[i]  = pdf(g(xt[i],θ),y[t])
        end

        # normalize weights
        Z[t] = mean(w)
        w   /= sum(w)

        # resample
        a  = wsample(1:N,w,N)
        xt = xt[a]
    end

    return reduce(*,Z)
end
```

Given the prerequisite functions, we have sufficient background to define the main algorithm. Following is a simplified, albeit honest, pseudocode of the filter. For brevity sake, certain details are omitted; however the following description is an accurate representation of [(Duan & Fulop, 2013)](https://www.tandfonline.com/doi/pdf/10.1080/07350015.2014.940081)s intended proces.

```julia
# (2.2.1) initialize the particle set
for i in 1:M
    θ[i] = rand(prior)
    Z[i] = bootstrap_filter(N,y,θ[i])
    S[i] = 1/M
end

# begin the tempering sequence at 0
ξ = 0.0

while ξ < 1
    # (2.2.2) find optimal ξ and reweight
    Δξ = optimize(Δξ -> ess(Δξ,Z)-(M/2))
    ξ += Δξ

    S  = [Z[i]^(Δξ) for i in 1:M]
    S /= sum(S)

    if ess(Δξ) < M/2
        # (2.2.3) resample particles
        a = wsample(1:M,S,M)
        θ = θ[a]
        Z = Z[a]

        # (2.2.4) moving the particles
        Σ = cov(θ')

        for _ in 1:mcmc_steps
            for i in 1:M
                prop_θ = rand(MvNormal(θ[i],Σ))
                prop_Z = bootstrap_filter(N,y,prop_θ)

                α  = (prop_Z^ξ)*pdf(prior,prop_θ)
                α /= (Z[i]^ξ)*pdf(prior,θ[i])

                if rand() ≤ minimum([1.0,α])
                    θ[i] = prop_θ
                    Z[i] = prop_Z
                end
            end
        end
    end
end
```

It should be noted that the MCMC kernel is a random walk from the currently observed particle `θ[i]` with a given covariance `Σ` determined after each new choice of `ξ`. As such, the jump kernel has a symmetric distribution in which `h(θ[i]|prop_θ)` is actually equal to its counterpart `h(prop_θ|θ[i])`, thus eliminating it from the acceptance ratio `α`.

An additional point of interest are the particle weights `S`. In (Duan & Fulop, 2013)](https://www.tandfonline.com/doi/pdf/10.1080/07350015.2014.940081) weights are carried over from the previous iteration as follows:

```julia
S  = [S[i]*(Z[i]^Δξ) for i in 1:M]
S /= sum(S)
```

And reset to `1/M` following each resmapling step. It is clear to see this is redundant since resampling occurs at every step; maximal selection of `ξ` is performed via a root finding algorithm, and thus will always result in an ESS which requires resampling. The resulting particles are equally weighted and make no difference in the calculation of the new weights.

## SMC²

Based on the algorithm presented in [Chopin, 2012](https://arxiv.org/pdf/1101.1528.pdf), SMC² performs online estimation of the joint posterior for latent states and model paramters; the process parallels methods akin to particle Markov Chain Monte Carlo (PMCMC) and iterated batch importance sampling (IBIS).

Consider a state space model with latent states `x[t]` and observations `y[t]`; let vector `θ` parameterize the transition density `f(x[t]|θ)` and the observation density `g(x[t]|θ)`. For parameter estimation, let `prior` represent the prior of `θ`. For the filters themselves, let `M` represent the number of `θ` particles and `N` be the number of state particles.

### Algorithm

...
