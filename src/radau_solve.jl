function solveRadau(rr::RadauIntegrator{T_object, N, n_stage_max}, x0::Vector{Float64}, h::Float64, n_order::Int64, is_recalc_jac::Bool=true) where {n_stage_max, N, T_object}
    rr.step.h = h
    rr.step.h⁻¹ = 1 / h
    
    table = rr.table[n_order]
    if is_recalc_jac
        calcJacobian!(rr, table, x0)
    end
    updateInvC!(rr, table)
    initializeX!(rr, table, x0)
    residual_prev = Inf
    k_iter_max = 15
    for k = 1:k_iter_max
        zeroFill!(rr.ct.Ew_stage, table)
        zeroFill!(rr.ct.delta_Z_stage, table)
        updateFX!(rr, table, x0)
        residual__ = calcEw!(rr, table, x0)
        updateStageX!(rr, table )
        if residual__ < rr.step.tol_newton
            return true, k, residual__, rr.ct.X_stage[n_order]
        elseif k == k_iter_max
            return false, k, residual__, rr.ct.X_stage[n_order]
        end
        if residual_prev < residual__  # residual gets worse
            return false, k, residual__, rr.ct.X_stage[n_order]
        end
        residual_prev = residual__
    end
end
