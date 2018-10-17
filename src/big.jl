
function big_eigen(M::Matrix{BigFloat})
    n, n2 = size(M)
    (n == n2) || error("matrix must be square")
    M_ = deepcopy(M)
    λ = eigvals(M_)
    T = zeros(Complex{BigFloat}, n, n)
    for k = 1:n
        sss = svd(M - I*λ[k])  # I'm aware this is massivly ineffecient
        T[:, k] = sss.Vt[end, :]'   # this does conjugate transpose
    end
    λ, T = put_real_eigenvalue_first(λ, T)
    return λ, T
end
