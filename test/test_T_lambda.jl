
using Test

@testset "λ T" begin
    for k = 1:7
        table = Radau.RadauTable{k}()
        @test table.T * diagm(0=>table.λ) * table.inv_T ≈ table.inv_A
        if isodd(k)
            @test table.λ[1] == real.(table.λ[1])
        end
    end
end
