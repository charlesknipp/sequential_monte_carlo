#### based on Pawel's original R code
#### not supposed to be functional, just as inspiration

function mean_exp(x, log_weights, scaled = false) 
    max_x = maximum(x)
    w = exp.(log_weights.-maximum(log_weights))
    mean_exp_scaled = sum(w.*exp(x.-max_x))/sum(w)

    if (scaled == false)
        res_out = mean_exp_scaled * exp(max_x)
    else
        res_out = mean_exp_scaled
    end

    return res_out
end

function Sigma_exp(x, log_weights)
    # max_x = maximum(x)
    w = exp.(log_weights.-maximum(log_weights))
    Sigma_x = cor(x,Weights(w))

    return Sigma_x
end


function ESS_calc(logx)
    max_logx = maximum(logx)
    x_scaled = exp.(logx.-max_logx)

    ESS = sum(x_scaled)^2 / sum(x_scaled.^2)
    if ESS == NaN; ESS = 0.0 end

    return ESS
end

function ESS_calc_mat(logx)
    max_logx = maximum(logx,dims=1)
    x_scaled = exp.(logx.-max_logx)

    ESS = sum(x_scaled,dims=1).^2 / sum(x_scaled.^2,dims=1)
    if ESS == NaN; ESS = 0.0 end

    return ESS
end


# draw from proposal denisty of transition to x_prop
function r_trunc_mult_norm(mean_par, cov_par, lowerbnd, upperbnd, fixpar = NULL)
    Sigma_list = map(
        j_par -> begin
            Sig_mix = cov_par[-j_par,-j_par] \ cov_par[-j_par, j_par]
            cov_par_cond = cov_par[j_par,j_par] - cov_par[j_par, -j_par]*Sig_mix

            return list(Sig_mix = Sig_mix, sigma_par_cond = sqrt(cov_par_cond))
        end,
        1:length(mean_par)
    )

    #initialize the algorithm
    x = mean_par
    for j_par in 1:length(mean_par)
        if (!(j_par %in% fixpar))
            mean_par_cond  = mean_par[j_par] + t(Sigma_list[[j_par]]$Sig_mix) * (x[-j_par] - mean_par[-j_par])

            lowerbnd_scaled = (lowerbnd[[j_par]] - mean_par_cond) / Sigma_list[[j_par]]$sigma_par_cond
            upperbnd_scaled = (upperbnd[[j_par]] - mean_par_cond) / Sigma_list[[j_par]]$sigma_par_cond

            x[j_par] = qnorm(runif(1, min = pnorm(lowerbnd_scaled), max = pnorm(upperbnd_scaled))) * Sigma_list[[j_par]]$sigma_par_cond + mean_par_cond
        end
    end

    return x
end

# evaluate proposal denisty of transition to x_prop
function D_trunc_mult_norm(x_prop, mean_par, cov_par, lowerbnd, upperbnd, fixpar = NULL)
    Sigma_list = map(
        j_par -> begin
            Sig_mix = cov_par[-j_par, -j_par] \ cov_par[-j_par, j_par]
            cov_par_cond = cov_par[j_par,j_par] - cov_par[j_par, -j_par]*Sig_mix

            return list(Sig_mix = Sig_mix, sigma_par_cond = sqrt(cov_par_cond))
        end,
        1:length(mean_par)
    )

    #initialize the algorithm
    loglik = 0
    x_current = mean_par
    for j_par in 1:length(mean_par)
        if !(j_par %in% fixpar)
            mean_par_cond  = mean_par[j_par] + t(Sigma_list[[j_par]]$Sig_mix)*(x_current[-j_par] - mean_par[-j_par])
            lowerbnd_in = lowerbnd[[j_par]]
            upperbnd_in = upperbnd[[j_par]]
            loglik = loglik + dnorm(x_prop[j_par], mean = mean_par_cond, sd = Sigma_list[[j_par]]$sigma_par_cond, log = T) - log(pnorm(upperbnd_in) - pnorm(lowerbnd_in))
            x_current[j_par] = x_prop[j_par]
        end
    end

    return loglik
end
