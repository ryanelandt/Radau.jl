module Radau

using ForwardDiff
using StaticArrays
using LinearAlgebra

include("radau_struct.jl")
include("radau_functions.jl")

export
# radau_struct.jl
RadauIntegrator,
makeRadauIntegrator,

# radau_functions.jl
calcJacobian!,
updateInvC!,
initializeX!,
zeroFill!,
updateFX!,
calcEw!,
updateStageX!,
solveRadauDefault,
RadauTable,
solveRadauFast

end
