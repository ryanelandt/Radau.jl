
function hard_coded_radau_3()
    s6 = sqrt(BigFloat(6))

    # https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods#Radau_IIA_methods
    a11 = 11 // 45 - s6 * (7 // 360)
    a12 = 37 // 225 - s6 * (169 // 1800)
    a13 = -2 // 225 + s6 * (1 // 75)
    a21 = 37 // 225 + s6 * (169 // 1800)
    a22 = 11 // 45 + s6 * (7 // 360)
    a23 = -2 // 225 - s6 * (1 // 75)
    a31 = 4 // 9 - s6 * (1 // 36)
    a32 = 4 // 9 + s6 * ( 1 // 36)
    a33 = 1 // 9
    A = [a11 a12 a13; a21 a22 a23; a31 a32 a33]

    b = A[end, :]

    c1 = 2 // 5 - s6 * (1 // 10)
    c2 = 2 // 5 + s6 * (1 // 10)
    c3 = 1
    c = [c1, c2, c3]

    # https://github.com/scipy/scipy/blob/14142ff70d84a6ce74044a27919c850e893648f7/scipy/integrate/_ivp/radau.py#L37
    bi = [ 13//3 + s6 *(7//3) -23//3 - s6 * (22 // 3)  10//3 + 5 * s6;
    13//3 - s6 * (7//3)  -23//3 + s6 * (22 // 3)  10//3 - 5 * s6;
    1//3 -8//3 10//3]

    # Practical Implementation of High-Order Multiple Precision Fully Implicit
    # Runge-Kutta Methods with Step Size Control Using Embedded Formula
    # (Tomonori Kouya)
    b̂ = b + [-(2 + 3 * s6)/6, -(2-3*s6)/6, -1//3] * find_real_eigenvalue(A)

    return A, b, c, bi, b̂
end

function hard_coded_radau_2()
    a11 = 5 // 12
    a12 = -1 // 12
    a21 = 3 // 4
    a22 = 1 // 4
    A = [a11 a12; a21 a22]

    B = A[end, :]

    C = [1 // 3,   1]

    return A, B, C
end


big_eps = eps(BigFloat(1))

n = 2
A, b, c, d, bi = radau_butcher_table_plus(n)
A2, B2, C2 = hard_coded_radau_2()

@testset "Radau Butcher Table 2" begin
    tol = 100 * big_eps * 2^2
    @test sum(abs.(A2 - A)) < tol
    @test sum(abs.(C2 - c)) < tol
    @test sum(abs.(B2 - b)) < tol
end

n = 3
A, b, c, d, bi = radau_butcher_table_plus(n)
A3, B3, C3, D3, bi_3 = hard_coded_radau_3()

@testset "Radau Butcher Table 3" begin
    tol = 100 * big_eps * 3^2
    @test sum(abs.(A3 - A)) < tol
    @test sum(abs.(B3 - b)) < tol
    @test sum(abs.(C3 - c)) < tol
    @test sum(abs.(D3 - d)) < tol
    @test sum(abs.(bi_3 - bi[2:4])) < tol
    @test abs(sum(bi) - 1) < tol
    @test bi[1] == find_real_eigenvalue(A)
end
