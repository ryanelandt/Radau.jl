
function calc_x̂_minus_x(rr::RadauIntegrator{T_object, N, n_stage_max}, table::RadauTable{n_stage}, x0::Vector{Float64}) where {n_stage, n_stage_max, N, T_object}
    # Implements Eq. 8.17

    h = rr.step.h
    x̂_minus_x = zeros(Float64, length(x0))
    LinearAlgebra.BLAS.axpy!(table.b̂_0 * h, rr.cv.xx_0, x̂_minus_x)  # x̂_minus_x = (b̂_0 * h) * ẋ_0
    for k = 1:n_stage
        b̂_minus_b = table.b̂[k] - table.A[n_stage, k]
        LinearAlgebra.BLAS.axpy!(b̂_minus_b * h, rr.ct.F_X_stage[k], x̂_minus_x)  # x̂_minus_x += b̂_minus_b * F_X_stage[k]
    end
    return x̂_minus_x
end

function calc_x_err_norm(rr::RadauIntegrator{TO, N, n_stage_max}, table::RadauTable{n_stage}, x0::Vector{Float64},
    x_err::Vector{Float64}) where {n_stage, n_stage_max, N, TO}
    # Implements Eq 8.21 in Solving Ordinary Differential Equations II (Stiff and Differential-Algebraic Problems)

    x_final = get_X_final(rr, table)
    term_sigma = 0.0
    for k = 1:n_stage
        sc_k = rr.step.tol_a + max(abs(x_final[k]), abs(x0[k])) * rr.step.tol_r
        term_sigma += (x_err[k] / sc_k)^2
    end
    return sqrt(term_sigma / N)
end

function update_x_err_norm!(rr::RadauIntegrator{TO, N, n_stage_max}, table::RadauTable{n_stage}, x0::Vector{Float64}) where {n_stage, n_stage_max, N, TO}
    rr.step.x_err_norm = rr.step.x_err_normᵏ⁺¹
    x̂_minus_x = calc_x̂_minus_x(rr, table, x0)
    x_err = rr.ct.inv_C_stage[1] * x̂_minus_x  # Eq. 8.19
    x_err = real.(x_err)
    rr.step.x_err_normᵏ⁺¹ = calc_x_err_norm(rr, table, x0, x_err)  # see definition in middle of pg. 124
    return nothing
end

function calc_h_new_estimate_1(rr::RadauIntegrator{TO, N, n_stage_max}, table::RadauTable{n_stage}) where {n_stage, n_stage_max, N, TO}
    #  Implements first half of part of 8.25

    k_iter = rr.order.k_iter
    two_k_max_iter = 2 * rr.order.k_iter_max

    fac = 0.9 * (two_k_max_iter + 1) / (two_k_max_iter + k_iter)
    term_1 = (1 / rr.step.x_err_normᵏ⁺¹)^get_exponent(table)
    return fac * rr.step.h * term_1
end

function predictive_correction(rr::RadauIntegrator{TO, N, n_stage_max}, table::RadauTable{n_stage}) where {n_stage, n_stage_max, N, TO}
    # Implements second half of part of 8.25

    # an x_err_norm can be 0.0 if the equation is integratable EXACTLY
    if 0.0 == rr.step.x_err_normᵏ⁺¹  # things went well let's keep it that way
        return 1.0
    elseif rr.step.x_err_norm == 0.0  # things got much much worse let's drastically reduce step size in anticipation
        return 0.1
    else
        ratio_h = rr.step.h / rr.step.hᵏ⁻¹
        ratio_x_err_norm = rr.step.x_err_norm / rr.step.x_err_normᵏ⁺¹
        return ratio_h * (ratio_x_err_norm)^get_exponent(table)
    end
end

function calc_h_new(rr::RadauIntegrator{TO, N, n_stage_max}, table::RadauTable{n_stage}, x0::Vector{Float64}, is_converge::Bool) where {n_stage, n_stage_max, N, TO}
    if is_converge
        h_est¹ = calc_h_new_estimate_1(rr, table)
        if rr.step.is_has_prev_step
            h_new = h_est¹ * min(1.0, predictive_correction(rr, table))  # conservative == good
        else
            rr.step.is_has_prev_step = true
            h_new = h_est¹
        end
        return min(h_new, 2 * rr.step.h)  # otherwise it will try to use numbers O(10.0-50.0) next time
    else  # failure
        rr.step.is_has_prev_step = false
        return rr.step.h * 0.1  # drastically decrease time-step and return
    end
end

function update_h!(rr::RadauIntegrator{TO, N, n_stage_max}, h_new::Float64) where {n_stage_max, N, TO}
    (0.0 < h_new < Inf) || error("unacceptable h: $h_new")
    rr.step.hᵏ⁻¹ = rr.step.h
    rr.step.h    = h_new
    rr.step.h⁻¹  = 1 / h_new
    return nothing
end

function update_order!(rr::RadauIntegrator{T_object, N, n_stage_max}, is_converge::Bool) where {n_stage_max, N, T_object}
    # Implements stategy on page 14 of "Stiff differential equations solved by Radau methods (Hairer)"

    cooldown_reset = 10
    s = rr.order.s
    if is_converge
        rr.order.n_increase_cooldown -= 1
        cool = rr.order.n_increase_cooldown
        Ψ_k = rr.order.Ψ_k
        if cool < 1  # has the cooldown expired
            if s == 1  # Radau1
                if Ψ_k < 0.1  # convergence isn't terrible
                    rr.order.s = 3  # Radau5
                end
            else
                # no higher order implemented yet so nothing to do yet
            end
        end
    else
        if s != 1  # not Radau1
            rr.order.s -= 2  # drop 2 stages
        end
        rr.order.n_increase_cooldown = cooldown_reset
    end
    (1 <= rr.order.s ) || error("something is wrong")
    return nothing
end
