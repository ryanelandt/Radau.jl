struct RadauVectorCache{n_stage, N}
    seed::NTuple{N, ForwardDiff.Partials{N, Float64}}
    store_float::Vector{Float64}
    store_complex::Vector{ComplexF64}
    neg_J::Matrix{Float64}
    x_dual::Vector{ForwardDiff.Dual{Nothing,Float64,N}}
    xx_dual::Vector{ForwardDiff.Dual{Nothing,Float64,N}}
    xx_0::Vector{Float64}
    function RadauVectorCache{n_stage_, N}() where {n_stage_, N}
        seed_  = ForwardDiff.construct_seeds(ForwardDiff.Partials{N,Float64})
        store_float_ = Vector{Float64}(undef, N)
        store_complex_ = Vector{ComplexF64}(undef, N)
        neg_J_ = Matrix{Float64}(undef, N, N)
        x_dual_ = Vector{ForwardDiff.Dual{Nothing,Float64,N}}(undef, N)
        xx_dual_ = Vector{ForwardDiff.Dual{Nothing,Float64,N}}(undef, N)
        xx_0_ = Vector{Float64}(undef, N)
        return new(seed_, store_float_, store_complex_, neg_J_, x_dual_, xx_dual_, xx_0_)
    end
end

struct RadauCacheTuple{n_stage, N}
    Ew_stage::NTuple{n_stage, Vector{ComplexF64}}
    delta_Z_stage::NTuple{n_stage, Vector{ComplexF64}}
    X_stage::NTuple{n_stage, Vector{Float64}}
    F_X_stage::NTuple{n_stage, Vector{Float64}}
    inv_C_stage::NTuple{n_stage, Matrix{ComplexF64}}
    function RadauCacheTuple{n_stage_, N}() where {n_stage_, N}
        Ew_stage_ = Tuple(Vector{ComplexF64}(undef, N) for _ = 1:n_stage_)
        delta_Z_stage_ = Tuple(Vector{ComplexF64}(undef, N) for _ = 1:n_stage_)
        X_stage_ = Tuple(Vector{Float64}(undef, N) for _ = 1:n_stage_)
        F_X_stage_ = Tuple(Vector{Float64}(undef, N) for _ = 1:n_stage_)
        inv_C_stage_ = Tuple(Matrix{ComplexF64}(undef, N, N) for _ = 1:n_stage_)
        return new(Ew_stage_, delta_Z_stage_, X_stage_, F_X_stage_, inv_C_stage_)
    end
end

struct RadauTable{n_stage}
    A::Matrix{Float64}
    c::NTuple{n_stage,Float64}
    inv_A::Matrix{Float64}
    λ::Vector{ComplexF64}
    T::Matrix{ComplexF64}
    inv_T::Matrix{ComplexF64}
    bi::Matrix{Float64}
    b̂::Vector{Float64}
    b̂_0::Float64
    function RadauTable{n_stage_}() where {n_stage_}
        (0 <= n_stage_) || error("radau_rule stage number needs to be positive, but is $radau_rule")
        (n_stage_ <= 7) || error("are you sure that you want a RadauIIA rule with $radau_rule stages (order $(2*radau_rule - 1))?")

        A, b, c, bi, b̂ = radau_butcher_table_plus(n_stage_)
        inv_A = inv(A)
        λ, T = big_eigen(inv_A)
        inv_T = inv(T)

        A = Float64.(A)
        c = Float64.(c)
        inv_A = Float64.(inv_A)
        λ = Complex{Float64}.(λ)
        T = Complex{Float64}.(T)
        inv_T = Complex{Float64}.(inv_T)

        c = Tuple(c)

        return new(A, c, inv_A, λ, T, inv_T, bi, b̂[2:end], b̂[1])
    end
end

mutable struct RadauStep
    h::Float64
    h⁻¹::Float64
    tol_a::Float64
    tol_r::Float64
    tol_newton::Float64
    is_has_prev_step::Bool
    hᵏ⁻¹::Float64
    x_err_norm::Float64
    x_err_normᵏ⁺¹::Float64
    h_max::Float64
    h_min::Float64
    function RadauStep(;h=1.0e-4, tol_a=1.0e-4, tol_r=1.0e-4, tol_newton=1.0e-16)
        return new(h, 1 / h, tol_a, tol_r, tol_newton, false, -9999.0, -9999.0, -9999.0, 0.01, 1.0e-8)
    end
end


mutable struct RadauOrder
    s::Int64
    n_increase_cooldown::Int64
    θ::Float64
    θᵏ⁻¹::Float64
    k_iter::Int64
    k_iter_max::Int64
    Ψ_k::Float64
    function RadauOrder()
        return new(1, 10, -9999.0, -9999.0, -9999, 15, 9999.0)
    end
end

struct RadauIntegrator{T_object, N, n_stage_max}
    table::NTuple{n_stage_max, RadauTable}
    step::RadauStep
    order::RadauOrder
    cv::RadauVectorCache{n_stage_max, N}
    ct::RadauCacheTuple{n_stage_max, N}
    de_object::T_object
    function RadauIntegrator{T_object_, N, n_stage_max_}(tol::Float64, de_object_::T_object_) where {T_object_, N, n_stage_max_}
        table_ = Tuple([RadauTable{k}() for k = 1:3])
        cv_ = RadauVectorCache{n_stage_max_, N}()
        ct_ = RadauCacheTuple{n_stage_max_, N}()
        radau_step = RadauStep(tol_newton=tol)
        radau_order = RadauOrder()
        return new(table_, radau_step, radau_order, cv_, ct_, de_object_)
    end
end
