makeRadauIntegrator(N::Int64, tol::Float64, de_object::T_object) where {T_object} = RadauIntegrator{T_object, N, 3}(tol, de_object)

put_real_eigenvalue_first(λ, T) = put_real_eigenvalue_first(λ, T, 0)
function put_real_eigenvalue_first(λ::Vector{Complex{BigFloat}}, T::Matrix{Complex{BigFloat}}, n_attempt::Int64)
    i_real = findfirst(λ .== real.(λ))
    if i_real == nothing
        return λ, T
    elseif i_real == 1
        return λ, T
    else
        n = length(λ)
        perm_mat = diagm(1=>ones(n-1), -2=>ones(1))
        λ_mat = diagm(0=>λ)
        λ_perm_order = Int64.(collect(1:n)' * perm_mat)[:]
        λ_perm = λ[λ_perm_order]
        T_perm = T * perm_mat
        (n < n_attempt) && error("too many attempts")
        return put_real_eigenvalue_first(λ_perm, T_perm, n_attempt + 1)
    end
end
