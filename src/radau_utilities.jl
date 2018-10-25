
function makeRadauIntegrator(N::Int64, tol::Float64, de_object::T_object, NC::Int64) where {T_object}
    return RadauIntegrator{T_object, N, 3, NC}(tol, de_object)
end

function makeRadauIntegrator(x0::Vector{Float64}, tol::Float64, de_object::T_object, NC::Int64) where {T_object}
    return makeRadauIntegrator(length(x0), tol, de_object, NC)
end

put_real_eigenvalue_first(λ, T) = put_real_eigenvalue_first(λ, T, 0)
function put_real_eigenvalue_first(λ::Vector{Complex{BigFloat}}, T::Matrix{Complex{BigFloat}}, n_attempt::Int64)
    i_real = findfirst(λ .== real.(λ))
    if i_real == nothing
        return λ, T
    elseif i_real == 1
        return λ, T
    else
        n = length(λ)
        perm_mat = diagm(1=>ones(n-1), (1-n)=>ones(1))
        λ_mat = diagm(0=>λ)
        λ_perm_order = Int64.(collect(1:n)' * perm_mat)[:]
        λ_perm = λ[λ_perm_order]
        T_perm = T * perm_mat
        (n < n_attempt) && error("too many attempts")
        return put_real_eigenvalue_first(λ_perm, T_perm, n_attempt + 1)
    end
end

function get_X_final(rr::RadauIntegrator{T_object, N, n_stage_max}, table::RadauTable{n_stage}) where {n_stage, n_stage_max, N, T_object}
    return rr.ct.X_stage[n_stage] * 1.0
end

get_exponent(table::RadauTable{n_stage}) where {n_stage} = 1 / (1 + n_stage)

@inline function get_table_from_current_s(rr::RadauIntegrator{T_object, N, n_stage_max}) where {n_stage_max, N, T_object}
    return rr.table[rr.order.s]
end
