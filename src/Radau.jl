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
include("adaptive.jl")

export
    # big.jl
    big_eigen,

    # generate_butcher_table.jl
    find_real_eigenvalue,
    radau_butcher_table_plus,

    # radau_struct.jl
    RadauIntegrator,
    RadauTable,
    RadauStep,
    RadauOrder,

    # radau_functions.jl
    calcJacobian!,
    updateInvC!,
    initializeX!,
    zeroFill!,
    updateFX!,
    calcEw!,
    updateStageX!,

    # radau_utilities.jl
    makeRadauIntegrator,
    put_real_eigenvalue_first,
    get_X_final,
    get_exponent,

    # radau_solve.jl
    solveRadau,

    # adaptive.jl
    calc_x̂_minus_x,
    calc_x_err_norm,
    update_x_err_norm!,
    calc_h_new,
    update_h!,
    calc_h_new_estimate_1,
    predictive_correction

end
