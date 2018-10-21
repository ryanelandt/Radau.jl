
function solveRadau(rr::RadauIntegrator{T_object, N, n_stage_max}, x0::Vector{Float64}) where {n_stage_max, N, T_object}
    table = rr.table[rr.order.s]
    calcJacobian!(rr, x0)
    return solveRadau_inner(rr, x0, table)
end

function solveRadau_inner(rr::RadauIntegrator{T_object, N, n_stage_max}, x0::Vector{Float64}, table::RadauTable{n_stage}) where {n_stage_max, N, T_object, n_stage}
    is_converge = simple_newton(rr, x0, table)
    update_x_err_norm!(rr, table, x0)
    h_new = calc_h_new(rr, table, x0, is_converge)
    update_h!(rr, h_new)  # TODO: move this and the above line into one function
    update_order!(rr, is_converge)
    if is_converge
        x_final = get_X_final(rr, table)
        return rr.step.hᵏ⁻¹, x_final  # step actually taken, x_final
    else
        if rr.step.h < rr.step.h_min
            error("time step is too small, something is wrong")
        else
            table = get_table_from_current_s(rr)  # need to get new table because s may have changed
            return solveRadau_inner(rr, x0, table)
        end
    end
end

function simple_newton(rr::RadauIntegrator{T_object, N, n_stage_max}, x0::Vector{Float64}, table::RadauTable{n_stage}) where {n_stage_max, N, T_object, n_stage}
    # TODO: get rid of residual__ and replace with rr.order.θ

    updateInvC!(rr, table)
    # initialize_X!(rr, table, x0)  # this doesn't seem to help at all
    initialize_X_with_X0!(rr, table, x0)

    residual_prev = Inf
    k_iter_max = rr.order.k_iter_max
    res_vec = SVector{3,Float64}(Inf,Inf,Inf)
    for k_iter = 1:k_iter_max
        rr.order.k_iter = k_iter
        zeroFill!(rr.ct.Ew_stage, table)
        zeroFill!(rr.ct.delta_Z_stage, table)
        updateFX!(rr, table, x0)
        residual__ = calcEw!(rr, table, x0)
        updateStageX!(rr, table)
        if residual__ < rr.step.tol_newton
            update_dense_successful!(rr, table)
            return true
        end

        if k_iter != 1
            rr.order.θᵏ⁻¹ = rr.order.θ
            rr.order.θ = sqrt(residual__)
            rr.order.Ψ_k = sqrt(rr.order.θᵏ⁻¹ * rr.order.θ)
        else
            rr.order.θ = sqrt(residual__)
            rr.order.Ψ_k = rr.order.θ
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
