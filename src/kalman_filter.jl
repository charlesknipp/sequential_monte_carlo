export kalman_filter,log_likelihood

function kalman_filter(
        model::LinearModel{AT,BT,QT,RT,XT,ΣT},
        xt::XT,
        Σt::ΣT,
        yt::Float64
    ) where {AT,BT,QT,RT,XT<:AbstractVector,ΣT<:AbstractArray}
    A,Q = model.A,model.Q
    B,R = model.B,model.R

    # predict
    xt = A*xt
    Σt = A*Σt*A' + Q
    
    σt  = B*Σt*B' + R
    Δyt = yt .- B*xt

    # update
    xt = xt + (Σt*B')*inv(σt)*Δyt
    Σt = Σt - (Σt*B')*inv(σt)*(B*Σt')

    # likelihood
    log_prob = log(2π) + logdet(σt) + (Δyt'*inv(σt)*Δyt)

    return xt,Σt,-0.5*log_prob
end

function kalman_filter(
        model::LinearModel{AT,BT,QT,RT,Float64,Float64},
        xt::Float64,
        Σt::Float64,
        yt::Float64
    ) where {AT,BT,QT,RT}
    A,Q = model.A,model.Q
    B,R = model.B,model.R

    # predict
    xt = A*xt
    Σt = (A^2)*Σt + Q
    
    σt  = (B^2)*Σt + R
    Δyt = yt - B*xt

    # update
    xt = xt + (Σt*B)*inv(σt)*Δyt
    Σt = Σt - ((Σt*B)^2)*inv(σt)

    # likelihood
    log_prob = log(2π) + logdet(σt) + (Δyt/(σt)*Δyt)

    return xt,Σt,-0.5*log_prob
end

function log_likelihood(
        y::Vector{Float64},
        model::LinearModel
    )

    xt = model.x0
    σt = model.σ0
    logZ = 0.0

    for t in 1:length(y)
        xt,σt,logμ = kalman_filter(model,xt,σt,y[t])
        logZ  += logμ
    end

    return xt,σt,logZ
end
