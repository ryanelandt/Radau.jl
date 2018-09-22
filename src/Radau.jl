module Radau

using ForwardDiff
using StaticArrays
using LinearAlgebra

include("radau_struct.jl")
include("radau_functions.jl")

export
    # radau_struct.jl
    RadauIntegrator,
    # RadauTable,

    # # radau_functions.jl
    solveRadau,
    # calcJacobian!,
    # updateInvC!,
    # initializeX!,
    # zeroFill!,
    # updateFX!,
    # calcEw!,
    # updateStageX!,
    # solveRadauDefault,

    # radau_utilities.jl
    makeRadauIntegrator


end
