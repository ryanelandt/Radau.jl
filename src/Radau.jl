module Radau

using ForwardDiff
using StaticArrays
using LinearAlgebra
using Polynomials: Poly, polyval, polyder, coeffs
using PolynomialRoots: roots
using GenericLinearAlgebra: eigvals
using GenericSVD: svd


include("big.jl")
include("generate_butcher_table.jl")
include("radau_struct.jl")
include("radau_functions.jl")
include("radau_utilities.jl")
include("radau_solve.jl")

export
    # big.jl
    big_eigen,

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
