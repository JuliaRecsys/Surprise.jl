using PyCall

const surprise = PyNULL()

function __init__()
    copy!(surprise, pyimport("surprise"))
end

abstract type SurpriseModel{T} <: Persa.Model{T} end

abstract type KNNAbstract{T} <: SurpriseModel{T} end

mutable struct KNNBasic{T} <: KNNAbstract{T}
    object::PyObject
    k::Int
    min_k::Int
    preference::Persa.Preference{T}
    users::Int
    items::Int
end

"""
    KNNBasic(dataset::Persa.CFDatasetAbstract; k = 40, min_k = 1)

A basic KNN algorithm.

# Arguments
- `k::Int = 40`: Maximum Number of neighbors used.
- `min_k::Int = 1`: Minimum number of neighbors used.
"""
function KNNBasic(dataset::Persa.Dataset; k = 40, min_k = 1)
    return KNNBasic(
        surprise.KNNBasic(k = k, min_k = min_k),
        k,
        min_k,
        dataset.preference,
        Persa.users(dataset),
        Persa.items(dataset),
    )
end

function Persa.predict(model::SurpriseModel, user::Int, item::Int)
    uid, vid = rawtoid(model.object, user, item)

    if isnan(uid) || isnan(vid)
        return NaN
    end

    return model.object.estimate(uid, vid)
end

function Persa.predict(model::KNNAbstract, user::Int, item::Int)
    uid, vid = rawtoid(model.object, user, item)

    if isnan(uid) || isnan(vid)
        return missing
    end

    return model.object.estimate(uid, vid)[1]
end

rawtoid(object::PyObject, user::Int, item::Int) = (raw2inneruser(object, user), raw2inneritem(object, item))

raw2inneruser(object::PyObject, user::Int) = object.trainset.to_inner_uid(user)
raw2inneritem(object::PyObject, item::Int) = object.trainset.to_inner_iid(item)

function transform(dataset::Persa.Dataset)
    users = [1:Persa.users(dataset)...]
    items = [1:Persa.items(dataset)...]
    
    u = Dict{Int, Array{Tuple}}()
    v = Dict{Int, Array{Tuple}}()
    u_raw_inner = Dict{Int, Int}()
    v_raw_inner = Dict{Int, Int}()
    
    for i=1:length(users)
        u[i - 1] = Tuple[]
        u_raw_inner[users[i]] = i - 1
    end
    
    for i=1:length(items)
        v[i - 1] = Tuple[]
        v_raw_inner[items[i]] = i - 1
    end
    
    for (user, item, rating) in dataset
        Base.push!(u[u_raw_inner[user]], (v_raw_inner[item], Float64(Persa.value(rating))))
        Base.push!(v[v_raw_inner[item]], (u_raw_inner[user], Float64(Persa.value(rating))))
    end
    
    return surprise.Trainset(u, v, length(users), length(items), length(dataset), (minimum(dataset.preference), maximum(dataset.preference)), u_raw_inner, v_raw_inner)
end

function Persa.train!(model::SurpriseModel, ds::Persa.Dataset)
  ds_surprise  = transform(ds)

  model.object.fit(ds_surprise)

  return nothing
end
