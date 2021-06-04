module Surprise
# package code goes here

using Persa
using PyCall

const surprise = PyNULL()

function __init__()
    copy!(surprise, pyimport("surprise"))
end

abstract type SurpriseModel{T} <: Persa.Model{T} end

function Persa.train!(model::SurpriseModel, ds::Persa.Dataset)
    ds_surprise  = transform(ds)
  
    model.object.fit(ds_surprise)
  
    return nothing
end

include("utils.jl")
include("memorybased.jl")

end # module
