
struct SimpleStiffSystem
    λ::Float64
    de::Function
    function SimpleStiffSystem(de::Function)
        return new(1.0, de)
    end
end

function stiff_de!(xx::Vector{T}, x::Vector{T}, s::SimpleStiffSystem, t::Float64=0.0) where {T}
    xx .= -s.λ * x
    return nothing
end

my_stiff_struct = SimpleStiffSystem(stiff_de!)

x0 = zeros(Float64, 4) .+ 1.0
rr = makeRadauIntegrator(x0, 1.0e-16, my_stiff_struct, 2)
rr.rule.s = 2
t_final = 0.2
update_h!(rr, t_final)

h, x_final = solveRadau(rr, x0)
x_ana = exp(-t_final)

@testset "basic_test" begin
    @test x_ana ≈ x_final[1]
end
