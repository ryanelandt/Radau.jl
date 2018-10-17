
function make_vandermonte_real(c::Vector{T}) where {T}
    n = length(c)
    V = zeros(T, n, n)
    for k = 1:n
        for kk = 1:n
            V[k, kk] = c[kk] ^ (k-1)
        end
    end
    return V
end

function find_real_eigenvalue(A::Matrix{BigFloat})
    n, _ = size(A)
    isodd(n) || error("A needs to be odd sized")
    ev = eigvals(A)
    min_val, min_ind = findmin(abs.(imag.(ev)))
    (min_val != 0.0) && error("something is wrong")
    return real.(ev[min_ind])
end

function calc_b_hat(A, c)
    gamma_0 = find_real_eigenvalue(A)
    n, _ = size(A)
    rhs = BigFloat.(collect(1 .// (1:n)))
    rhs[1] = 1 - gamma_0
    V = make_vandermonte_real(c)
    return vcat([gamma_0], V \ rhs)
end

function radau_butcher_table_plus(s::Int64)
    # ABC returns coefficients for S-stage Gauss-Legendre methods. (From RKGL, DiffMan package v.2 by K. Engo, A. Marthinsen & H. Munthe-Kaas.)
    P_Radau = output_p_Radau(s)
    c = roots(BigFloat.(coeffs(P_Radau)))
    c = sort(real.(c))
    V = make_vandermonte_real(c)'
    J = diagm(0 => 1 .// (1:s) )
    A = diagm(0 => c) * V * J / V
    b = (ones(BigFloat, 1, s) * J / V)[:]
    bi = get_dense_radau(J, V, c)
    if isodd(s)
        b̂ = calc_b_hat(A, c)
    else
        b̂ = zeros(BigFloat, s + 1) .+ NaN
    end
    return A, b, c, bi, b̂
end

function make_legendre_poly_by_order(n::Int64)
    P_x2_minus_1 = Poly([-1, 0, 1])
    return (1 // (2^n * factorial(n))) * polyder(P_x2_minus_1^n, n)
end

function output_p_Radau(n::Int64)
    Pn = make_legendre_poly_by_order(n)
    Pn_m = make_legendre_poly_by_order(n-1)
    P_2x_minus_1 = Poly([-1, 2])
    P_Radau = polyval(Pn, P_2x_minus_1) - polyval(Pn_m, P_2x_minus_1)
    return P_Radau
end

function get_dense_radau(J, V, c)
    s = length(c)
    interp_poly = (J / V)' .* ((1 ./ c) * collect(1:s)')
    return interp_poly
end
