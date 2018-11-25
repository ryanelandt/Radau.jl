
function calcJacobian!(rr::RadauIntegrator{T_object,N,n_rule_max,NC}, x0::Vector{Float64}, t::Float64) where {n_rule_max, N, T_object, NC}
    N_Loop, n_rem = divrem(N,NC)
    (n_rem != 0) && (N_Loop += 1)
    i_now = 1:NC
    for k = 1:N_Loop
        i_clamp = i_now[1]:min(i_now[end],N)
        seed_indices!(rr.cv.x_dual, x0, i_clamp, rr.cv.seed)
        rr.de_object.de(rr.cv.xx_dual, rr.cv.x_dual, rr.de_object, t)
        write_indices!(rr, i_clamp)
        i_now = i_now .+ NC
    end
    return nothing
end

function seed_indices!(duals::Vector{ForwardDiff.Dual{T,V,N}}, x::Vector{Float64}, index::UnitRange{Int64}, seed::NTuple{N,ForwardDiff.Partials{N,V}}) where {T,V,N}
    duals .= x
    i = 1
    for k = index
        duals[k] = ForwardDiff.Dual{T,V,N}(x[k], seed[i])
        i += 1
    end
    return nothing
end

function write_indices!(rr::RadauIntegrator{T_object,N,n_rule_max,NC}, index::UnitRange{Int64}) where {n_rule_max,N,T_object,NC}
    for i = 1:N
        xx_dual_i = rr.cv.xx_dual[i]
        rr.cv.xx_0[i] = ForwardDiff.value(xx_dual_i)
        the_partials = ForwardDiff.partials(xx_dual_i)
        k = 1
        for j = index  # for each x
            rr.cv.neg_J[i, j] = -the_partials[k]  ### the first dual is the partial of the first element wrt all partials ###
            k += 1
        end
    end
    return nothing
end

function zeroFill!(tup_vec_in::NTuple{N, Vector{T}},  table::RadauTable{n_stage}) where {N, T, n_stage}
    for i = 1:n_stage
        fill!(tup_vec_in[i], zero(T))
    end
    return nothing
end

function initialize_X!(rr::RadauIntegrator{T_object, N, n_rule_max}, table::RadauTable{n_stage}, x0::Vector{Float64}) where {n_stage, n_rule_max, N, T_object}
    if rr.dense.is_has_X
        initialize_X_with_interp!(rr, table)
    else
        initialize_X_with_X0!(rr, table, x0)
    end
end

function initialize_X_with_X0!(rr::RadauIntegrator{T_object, N, n_rule_max}, table::RadauTable{n_stage}, x0::Vector{Float64}) where {n_stage, n_rule_max, N, T_object}
    for i = 1:n_stage
        LinearAlgebra.BLAS.blascopy!(N, x0, 1, rr.ct.X_stage[i], 1)
    end
    return nothing
end

function updateFX!(rr::RadauIntegrator{T_object, N, n_rule_max}, table::RadauTable{n_stage}, x0::Vector{Float64}, t::Float64) where {n_stage, n_rule_max, N, T_object}
    for i = 1:n_stage
        time_stage = table.c[i] * rr.step.h + t
        rr.de_object.de(rr.ct.F_X_stage[i], rr.ct.X_stage[i], rr.de_object, time_stage)
    end
    return nothing
end
function calcEw!(rr::RadauIntegrator{T_object, N, n_rule_max}, table::RadauTable{n_stage}, x0::Vector{Float64}) where {n_stage, n_rule_max, N, T_object}
    residual = 0.0
    for i = 1:n_stage
        # rr.cv.store_float .= rr.ct.X_stage[i] - x0 - rr.step.h * sum( rr.A[i, j] * rr.ct.F_X_stage[j])
        LinearAlgebra.BLAS.blascopy!(N, rr.ct.X_stage[i], 1, rr.cv.store_float, 1)
        LinearAlgebra.BLAS.axpy!(-1.0, x0, rr.cv.store_float)
        for j = 1:n_stage
            coefficient = -rr.step.h * table.A[i, j]
            # rr.cv.store_float .-= (rr.h * rr.A[i, j]) * rr.ct.F_X_stage[j]
            LinearAlgebra.BLAS.axpy!(coefficient, rr.ct.F_X_stage[j], rr.cv.store_float)
        end
        residual += dot(rr.cv.store_float, rr.cv.store_float)
        for j = 1:n_stage
            # rr.ct.Ew_stage[j] .+= (rr.step.h⁻¹ * rr.λ[j] * rr.T⁻¹[j, i]) * rr.cv.store_float
            coefficient = rr.step.h⁻¹[1] * table.λ[j] * table.T⁻¹[j, i]
            LinearAlgebra.BLAS.axpy!(coefficient, rr.cv.store_float, rr.ct.Ew_stage[j])
        end
    end
    return residual
end
function updateInvC!(rr::RadauIntegrator{T_object, N, n_rule_max}, table::RadauTable{n_stage}) where {n_stage, n_rule_max, N, T_object}
    for i = 1:n_stage
        rr.ct.inv_C_stage[i] .= rr.cv.neg_J
        for k = 1:N
            rr.ct.inv_C_stage[i][k, k] += rr.step.h⁻¹[1] * table.λ[i]
        end
        ### NOTE: Negative sign taken care of when update X_stage ###
        _, ipiv, info = LinearAlgebra.LAPACK.getrf!(rr.ct.inv_C_stage[i])  # rr.cv.store_complex .= - inv(rr.cv.C) * rr.ct.Ew_stage[i]
        LinearAlgebra.LAPACK.getri!(rr.ct.inv_C_stage[i], ipiv)
    end
    return nothing
end
function updateStageX!(rr::RadauIntegrator{T_object, N, n_rule_max},  table::RadauTable{n_stage}) where {n_stage, n_rule_max, N, T_object}
    for i = 1:n_stage
        ### NOTE: Negative sign taken care of when update X_stage ###
        # rr.cv.store_complex .= rr.ct.inv_C_stage[i] * rr.ct.Ew_stage[i]
        LinearAlgebra.BLAS.gemv!('N', one(ComplexF64), rr.ct.inv_C_stage[i], rr.ct.Ew_stage[i], zero(ComplexF64), rr.cv.store_complex)
        for j = 1:n_stage
            # rr.ct.delta_Z_stage[j] .+= rr.T[j, i] * rr.cv.store_complex
            LinearAlgebra.BLAS.axpy!(table.T[j, i], rr.cv.store_complex, rr.ct.delta_Z_stage[j])
        end
    end
    for i = 1:n_stage  # Update X_stage
        # LinearAlgebra.BLAS.axpy!(1.0, real.(rr.ct.delta_Z_stage[i]), rr.ct.X_stage[i])
        rr.ct.X_stage[i] .-= real.(rr.ct.delta_Z_stage[i])
    end
    return nothing
end
