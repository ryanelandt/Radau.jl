struct RadauVectorCache{n_stage, N}
    seed::NTuple{N, ForwardDiff.Partials{N, Float64}}
    store_float::Vector{Float64}
    store_complex::Vector{ComplexF64}
    neg_J::Matrix{Float64}
    x_dual::Vector{ForwardDiff.Dual{Float64,Float64,N}}
    xx_dual::Vector{ForwardDiff.Dual{Float64,Float64,N}}
    xx_0::Vector{Float64}
    function RadauVectorCache{n_stage_, N}() where {n_stage_, N}
        seed_  = ForwardDiff.construct_seeds(ForwardDiff.Partials{N,Float64})
        store_float_ = Vector{Float64}(undef, N)
        store_complex_ = Vector{ComplexF64}(undef, N)
        neg_J_ = Matrix{Float64}(undef, N, N)
        x_dual_ = Vector{ForwardDiff.Dual{Float64,Float64,N}}(undef, N)
        xx_dual_ = Vector{ForwardDiff.Dual{Float64,Float64,N}}(undef, N)
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
    lambda::Vector{ComplexF64}
    T::Matrix{ComplexF64}
    inv_T::Matrix{ComplexF64}
    function RadauTable{n_stage_}() where {n_stage_}
        function outputRadauRuleMatrix(radau_rule::Int64)
            if radau_rule == 1
                A = SMatrix{radau_rule, radau_rule, Float64, radau_rule * radau_rule}(1)
                c = (1.0, )
            elseif radau_rule == 2
                v1 = SVector{2, Float64}(5/12, -1/12)
                v2 = SVector{2, Float64}(3/4, 1/4)
                A = Matrix{Float64}(I,radau_rule,radau_rule)
                A[1, :] = v1
                A[2, :] = v2
                A = SMatrix{radau_rule, radau_rule, Float64, radau_rule * radau_rule}(A)
                c = (sum(v1), sum(v2))
            elseif radau_rule == 3
                v1 = SVector{3, Float64}(11/45 - 7*sqrt(6)/360, 37/225 - 169 * sqrt(6) / 1800, -2/225 + sqrt(6) / 75)
                v2 = SVector{3, Float64}(37/225 + 169*sqrt(6) / 1800, 11/45 + 7 * sqrt(6) / 360, -2/225 - sqrt(6)/75)
                v3 = SVector{3, Float64}(4/9 - sqrt(6)/36, 4/9 + sqrt(6)/36, 1/9)
                A = Matrix{Float64}(I,radau_rule,radau_rule)
                A[1, :] = v1
                A[2, :] = v2
                A[3, :] = v3
                A = SMatrix{radau_rule, radau_rule, Float64, radau_rule * radau_rule}(A...)
                c = (sum(v1), sum(v2), sum(v3))
            end
            return A, c
        end

        A_mat, c_ = outputRadauRuleMatrix(n_stage_)
        A_ = Matrix(A_mat)
        inv_A_ = inv(A_)
        lambda_, T_ = eigen(inv_A_)
        inv_T_ = inv(T_)
        return new(A_, c_, inv_A_, lambda_, T_, inv_T_)
    end
end

struct RadauIntegrator{T_object, N, n_stage_max}
    table::NTuple{n_stage_max, RadauTable}
    h::MVector{1, Float64}
    inv_h::MVector{1, Float64}
    tol::Float64
    cv::RadauVectorCache{n_stage_max, N}
    ct::RadauCacheTuple{n_stage_max, N}
    de_object::T_object
    function RadauIntegrator{T_object_, N, n_stage_max_}(tol::Float64, de_object_::T_object_) where {T_object_, N, n_stage_max_}
        table_ = Tuple([RadauTable{k}() for k = 1:3])
        h_mv = MVector{1, Float64}(NaN)
        inv_h_mv = MVector{1, Float64}(NaN)
        cv_ = RadauVectorCache{n_stage_max_, N}()
        ct_ = RadauCacheTuple{n_stage_max_, N}()
        return new(table_, h_mv, inv_h_mv, tol, cv_, ct_, de_object_)
    end
end
