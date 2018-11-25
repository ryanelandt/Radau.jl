
function solveRadau(rr::RadauIntegrator{T_object,NX,NC,NR,NSM}, x0::Vector{Float64}, t::Float64=0.0) where {T_object,NX,NC,NR,NSM}
    table = rr.table[rr.rule.s]
    calcJacobian!(rr, x0, t)
    return solveRadau_inner(rr, x0, table, t)
end

function solveRadau_inner(rr::RadauIntegrator{T_object,NX,NC,NR,NSM}, x0::Vector{Float64},
        table::RadauTable{NS}, t::Float64=0.0) where {T_object,NX,NC,NR,NSM,NS}

    is_converge = simple_newton(rr, x0, table, t)
    update_x_err_norm!(rr, table, x0)
    h_new = calc_h_new(rr, table, x0, is_converge)
    update_h!(rr, h_new)  # TODO: move this and the above line into one function
    update_rule!(rr, is_converge)
    if is_converge
        x_final = get_X_final(rr, table)
        h_taken = rr.step.hᵏ⁻¹
        return rr.step.hᵏ⁻¹, x_final, t + h_taken  # step actually taken, x_final
    else
        if rr.step.h < rr.step.h_min
            error("time step is too small, something is wrong")
        else
            table = get_table_from_current_s(rr)  # need to get new table because s may have changed
            return solveRadau_inner(rr, x0, table)
        end
    end
end

function simple_newton(rr::RadauIntegrator{T_object,NX,NC,NR,NSM}, x0::Vector{Float64}, table::RadauTable{NS},
        t::Float64=0.0) where {T_object,NX,NC,NR,NSM,NS}

    # TODO: get rid of residual__ and replace with rr.rule.θ

    updateInvC!(rr, table)
    initialize_X_with_X0!(rr, table, x0)

    residual_prev = Inf
    k_iter_max = rr.rule.k_iter_max
    res_vec = SVector{3,Float64}(Inf,Inf,Inf)
    for k_iter = 1:k_iter_max
        rr.rule.k_iter = k_iter
        zeroFill!(rr.ct.Ew_stage, table)
        zeroFill!(rr.ct.delta_Z_stage, table)
        updateFX!(rr, table, x0, t)
        residual__ = calcEw!(rr, table, x0)
        updateStageX!(rr, table)
        if residual__ < rr.step.tol_newton
            update_dense_successful!(rr, table)
            return true
        end

        if k_iter != 1
            rr.rule.θᵏ⁻¹ = rr.rule.θ
            rr.rule.θ = sqrt(residual__)
            rr.rule.Ψ_k = sqrt(rr.rule.θᵏ⁻¹ * rr.rule.θ)
        else
            rr.rule.θ = sqrt(residual__)
            rr.rule.Ψ_k = rr.rule.θ
        end
        residual_prev = residual__  # update residual
        res_vec = SVector{3,Float64}(residual__, res_vec[1], res_vec[2])
        if res_vec[3] < res_vec[2] < res_vec[1]  # two consequitive issues
            update_dense_unsuccessful!(rr)
            return false
        end
    end
    update_dense_unsuccessful!(rr)
    return false  # exceeded iteration limit
end
