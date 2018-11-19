
struct SimpleTimeDependentSystem
    de::Function
    function SimpleTimeDependentSystem(de::Function)
        return new(de)
    end
end

function stiff_de!(xx::Vector{T}, x::Vector{T}, s::SimpleTimeDependentSystem, t::Float64=0.0) where {T}
    xx .= t
    return nothing
end

my_stiff_struct = SimpleTimeDependentSystem(stiff_de!)

x0 = zeros(Float64, 1) .+ 0.0
rr = makeRadauIntegrator(x0, 1.0e-16, my_stiff_struct, 2)
rr.order.s = 3
t_final = 1.0
update_h!(rr, t_final)
h, x_final, t_f = solveRadau(rr, x0, 0.0)

@testset "time dependent" begin
    @test x_final[1] ≈ 0.5
end
