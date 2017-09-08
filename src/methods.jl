using PyCall
@pyimport surprise

abstract type SurpriseModel <: Persa.CFModel end

abstract type KNNAbstract <: SurpriseModel end

mutable struct KNNBasic <: KNNAbstract
  object::PyObject
  preferences::Persa.RatingPreferences
  k::Int
  min_k::Int
end

"""
    KNNBasic(dataset::Persa.CFDatasetAbstract; k = 40, min_k = 1)

A basic KNN algorithm.

# Arguments
- `k::Int = 40`: Maximum Number of neighbors used.
- `min_k::Int = 1`: Minimum number of neighbors used.
"""
function KNNBasic(dataset::Persa.CFDatasetAbstract; k = 40, min_k = 1)
  return KNNBasic(surprise.KNNBasic(k = k, min_k = min_k), dataset.preferences, k , min_k);
end

mutable struct KNNBaseline <: KNNAbstract
  object::PyObject
  preferences::Persa.RatingPreferences
  k::Int
  min_k::Int
end

"""
    KNNBaseline(dataset::Persa.CFDatasetAbstract; k = 40, min_k = 1)

A basic KNN algorithm but using a baseline factor.

# Arguments
- `k::Int = 40`: Maximum Number of neighbors used.
- `min_k::Int = 1`: Minimum number of neighbors used.
"""
function KNNBaseline(dataset::Persa.CFDatasetAbstract; k = 40, min_k = 1)
  return KNNBaseline(surprise.KNNBaseline(k = k, min_k = min_k), dataset.preferences, k , min_k);
end

mutable struct KNNWithMeans <: KNNAbstract
  object::PyObject
  preferences::Persa.RatingPreferences{Float64}
  k::Int
  min_k::Int
end

"""
    KNNWithMeans(dataset::Persa.CFDatasetAbstract; k = 40, min_k = 1)

A basic KNN algorithm but using user or item mean.

# Arguments
- `k::Int = 40`: Maximum Number of neighbors used.
- `min_k::Int = 1`: Minimum number of neighbors used.
"""
function KNNWithMeans(dataset; k::Int = 40, min_k::Int = 1)
  sim_options = Dict()
  sim_options["name"] = "cosine"
  return KNNWithMeans(surprise.KNNWithMeans(k = k, min_k = min_k, sim_options=sim_options), dataset.preferences, k, min_k);
end


mutable struct SlopeOne <: SurpriseModel
  object::PyObject
  preferences::Persa.RatingPreferences
end

"""
    SlopeOne(dataset::Persa.CFDatasetAbstract)

SlopeOne algorithm.
"""
function SlopeOne(dataset::Persa.CFDatasetAbstract)
  return SlopeOne(surprise.SlopeOne(), dataset.preferences);
end

mutable struct SVD <: SurpriseModel
  object::PyObject
  preferences::Persa.RatingPreferences
  features::Int
  n_epochs::Int
  lrate::Float64
  lambda::Float64
end

function SVD(dataset::Persa.CFDatasetAbstract, biased::Bool; features = 100, n_epochs = 20, lrate = 0.005, lambda = 0.02)
  return SVD(surprise.SVD(biased = biased, n_factors = features, n_epochs = n_epochs, lr_all = lrate, reg_all = lambda), dataset.preferences, features, n_epochs, lrate, lambda);
end

"""
    IRSVD(dataset::Persa.CFDatasetAbstract; features = 100, n_epochs = 20, lrate = 0.005, lambda = 0.02)

Improved Regulared SVD. Extension of RSVD algorithm adding the user and item
bias.

# Arguments
- `features::Int = 100`: Number of factors.
- `n_epochs::Int = 20`: The number of iteration of the SGD algorithm.
- `lrate::Float = 0.005`: Learning rate parameter.
- `lambda::Int = 0.02`: Regularization parameter.
"""
function IRSVD(dataset::Persa.CFDatasetAbstract; features = 100, n_epochs = 20, lrate = 0.005, lambda = 0.02)
  return SVD(dataset, true; features = features, n_epochs = n_epochs, lrate = lrate, lambda = lambda);
end

"""
    RSVD(dataset::Persa.CFDatasetAbstract; features = 100, n_epochs = 20, lrate = 0.005, lambda = 0.02)

Regulared SVD. The algorithm is also known as SVD by Funk.

# Arguments
- `features::Int = 100`: Number of factors.
- `n_epochs::Int = 20`: The number of iteration of the SGD algorithm.
- `lrate::Float = 0.005`: Learning rate parameter.
- `lambda::Int = 0.02`: Regularization parameter.
"""
function RSVD(dataset::Persa.CFDatasetAbstract; features = 100, n_epochs = 20, lrate = 0.005, lambda = 0.02)
  return SVD(dataset, false; features = features, n_epochs = n_epochs, lrate = lrate, lambda = lambda);
end

mutable struct NMF <: SurpriseModel
  object::PyObject
  preferences::Persa.RatingPreferences
  features::Int
  n_epochs::Int
  biased::Bool
end

"""
    NMF(dataset::Persa.CFDatasetAbstract; features = 100, n_epochs = 20, lrate = 0.005, lambda = 0.02)

Collaborative Filtering based on Non-negative Matrix Factorization.

# Arguments
- `features::Int = 15`: Number of factors.
- `n_epochs::Int = 50`: The number of iteration of the SGD algorithm.
- `biased::Bool = false`: If the method use biases.
"""
function NMF(dataset::Persa.CFDatasetAbstract; features::Int = 15, n_epochs::Int = 50, biased::Bool = false)
  return NMF(surprise.NMF(n_factors = features, n_epochs = n_epochs, biased = biased), dataset.preferences, features, n_epochs, biased);
end

function Persa.predict(model::SurpriseModel, user::Int, item::Int)
    uid, vid = rawtoid(model.object, user, item)

    if isnan(uid) || isnan(vid)
        return NaN
    end

    return Persa.correct(model.object[:estimate](uid, vid), model.preferences)
end

function Persa.predict(model::KNNAbstract, user::Int, item::Int)
    uid, vid = rawtoid(model.object, user, item)

    if isnan(uid) || isnan(vid)
        return NaN
    end

    return Persa.correct(model.object[:estimate](uid, vid)[1], model.preferences)
end

function Persa.canpredict(model::KNNAbstract, user::Int, item::Int)
    uid, vid = rawtoid(model.object, user, item)

    if isnan(uid) || isnan(vid)
        return false
    end

    try
        return model.object[:estimate](uid, vid)[2]["actual_k"] >= model.min_k ? true : false
    catch
        return false
    end
end

function Persa.canpredict(model::SurpriseModel, user::Int, item::Int)
    uid, vid = rawtoid(model.object, user, item)

    if isnan(uid) || isnan(vid)
        return false
    end

    return true
end

rawtoid(object::PyObject, user::Int, item::Int) = (raw2inneruser(object, user), raw2inneritem(object, item))

function raw2inneruser(object::PyObject, user::Int)
    try
        return object[:trainset][:to_inner_uid](user)
    catch
        return NaN
    end
end

function raw2inneritem(object::PyObject, item::Int)
    try
        return object[:trainset][:to_inner_iid](item)
    catch
        return NaN
    end
end

function transform(ds::Persa.CFDatasetAbstract)
    users = sort(unique(ds.file[:user]))
    items = sort(unique(ds.file[:item]))

    u = Dict{Int, Array{Tuple}}()
    v = Dict{Int, Array{Tuple}}()
    u_raw_inner = Dict{Int, Int}()
    v_raw_inner = Dict{Int, Int}()

    for i=1:length(users)
      u[i - 1] = Array{Tuple}(0)
      u_raw_inner[users[i]] = i - 1
    end

    for i=1:length(items)
        v[i - 1] = Array{Tuple}(0)
        v_raw_inner[items[i]] = i - 1
    end

    for (user, item, rating) in ds
        Base.push!(u[u_raw_inner[user]], (v_raw_inner[item], Float64(rating)))
        Base.push!(v[v_raw_inner[item]], (u_raw_inner[user], Float64(rating)))
    end

    return surprise.Trainset(u, v, length(users), length(items), length(ds), (ds.preferences.min, ds.preferences.max), 0, u_raw_inner, v_raw_inner)
end

function Persa.train!(model::SurpriseModel, ds::Persa.CFDatasetAbstract)
  ds_surprise  = transform(ds)

  return model.object[:train](ds_surprise)
end
