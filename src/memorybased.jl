abstract type KNNAbstract{T} <: SurpriseModel{T} end

struct KNNBasic{T} <: KNNAbstract{T}
    object::PyObject
    k::Int
    min_k::Int
    preference::Persa.Preference{T}
    users::Int
    items::Int
end

"""
    KNNBasic(dataset::Persa.Dataset; k = 40, min_k = 1)

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

struct KNNBaseline{T} <: KNNAbstract{T}
    object::PyObject
    k::Int
    min_k::Int
    preference::Persa.Preference{T}
    users::Int
    items::Int
end

"""
    KNNBaseline(dataset::Persa.Dataset; k = 40, min_k = 1)
A basic KNN algorithm but using a baseline factor.
# Arguments
- `k::Int = 40`: Maximum Number of neighbors used.
- `min_k::Int = 1`: Minimum number of neighbors used.
"""
function KNNBaseline(dataset::Persa.Dataset; k = 40, min_k = 1)
    return KNNBaseline(
        surprise.KNNBaseline(k = k, min_k = min_k),
        k,
        min_k,
        dataset.preference,
        Persa.users(dataset),
        Persa.items(dataset),
    )
end

struct KNNWithMeans{T} <: KNNAbstract{T}
    object::PyObject
    k::Int
    min_k::Int
    preference::Persa.Preference{T}
    users::Int
    items::Int
end

"""
    KNNWithMeans(dataset::Persa.Dataset; k = 40, min_k = 1)
A basic KNN algorithm but using user or item mean.
# Arguments
- `k::Int = 40`: Maximum Number of neighbors used.
- `min_k::Int = 1`: Minimum number of neighbors used.
"""
function KNNWithMeans(dataset::Persa.Dataset; k::Int = 40, min_k::Int = 1)
    sim_options = Dict()
    sim_options["name"] = "cosine"
    return KNNWithMeans(
        surprise.KNNWithMeans(k = k, min_k = min_k, sim_options = sim_options),
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
