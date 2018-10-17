struct SimpleStiffSystem
    λ::Float64
    de::Function
    function SimpleStiffSystem(de::Function)
        return new(1.0, de)
    end
end

function stiff_de!(xx::Vector{T}, x::Vector{T}, s::SimpleStiffSystem) where {T}
    xx .= -s.λ * x
    return nothing
end

a0 = SimpleStiffSystem(stiff_de!)

nn = 4
xx0 = zeros(Float64, nn) .+ 1.0
rr_ = makeRadauIntegrator(nn, 1.0e-16, a0)

n_stage = 3
t_final = 0.2
is_converge, k_iter, res, x_final = solveRadau(rr_, xx0, t_final, n_stage)
x_ana = exp(-t_final)


@testset "basic_test" begin
    @test x_ana ≈ x_final[1]
    @test is_converge
    @test k_iter < 15
end
