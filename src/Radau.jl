module Radau

using ForwardDiff
using StaticArrays
using LinearAlgebra


include("radau_struct.jl")
include("radau_functions.jl")
include("radau_utilities.jl")
include("radau_solve.jl")


export
    # radau_struct.jl
    RadauIntegrator,

    # radau_utilities.jl
    makeRadauIntegrator,

    # radau_solve.jl
    solveRadau
end
