module Radau

using ForwardDiff
using StaticArrays
using LinearAlgebra
using Polynomials: Poly, polyval, polyder, coeffs
using PolynomialRoots: roots
using GenericLinearAlgebra


include("generate_butcher_table.jl")
include("radau_struct.jl")
include("radau_functions.jl")
include("radau_utilities.jl")
include("radau_solve.jl")

export
    # generate_butcher_table.jl
    find_real_eigenvalue,
    radau_butcher_table_plus,

    # radau_struct.jl
    RadauIntegrator,

    # radau_utilities.jl
    makeRadauIntegrator,

    # radau_solve.jl
    solveRadau
end
