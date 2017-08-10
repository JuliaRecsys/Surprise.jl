using PyCall
@pyimport surprise

abstract type SurpriseModel <: Persa.CFModel

end

mutable struct KNNBasic <: SurpriseModel
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

mutable struct KNNBaseline <: SurpriseModel
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

mutable struct KNNWithMeans <: SurpriseModel
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
  println(dataset)
  println(dataset.preferences)
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

Persa.predict(model::SurpriseModel, user::Int, item::Int) = Persa.correct(model.object[:estimate](user - 1, item - 1), model.preferences)

Persa.predict(model::KNNBasic, user::Int, item::Int) = Persa.correct(model.object[:estimate](user - 1, item - 1)[1], model.preferences)

Persa.predict(model::KNNBaseline, user::Int, item::Int) = Persa.correct(model.object[:estimate](user - 1, item - 1)[1], model.preferences)

Persa.predict(model::KNNWithMeans, user::Int, item::Int) = Persa.correct(model.object[:estimate](user - 1, item - 1)[1], model.preferences)

function Persa.canpredict(model::KNNBasic, user::Int, item::Int)
  try
    return model.object[:estimate](user - 1, item - 1)[2]["actual_k"] >= model.min_k ? true : false
  catch
    return false
  end
end

function Persa.canpredict(model::KNNBaseline, user::Int, item::Int)
  try
    return model.object[:estimate](user - 1, item - 1)[2]["actual_k"] >= model.min_k ? true : false
  catch
    return false
  end
end

function Persa.canpredict(model::KNNWithMeans, user::Int, item::Int)
  try
    return model.object[:estimate](user - 1, item - 1)[2]["actual_k"] >= model.min_k ? true : false
  catch
    return false
  end
end

function Persa.canpredict(model::SurpriseModel, user::Int, item::Int)
  try
    model.object[:estimate](user - 1, item - 1)
    return true
  catch
    return false
  end
end

function transform(ds::Persa.CFDatasetAbstract)
  u = Dict()
  v = Dict()
  uu = Dict()
  vv = Dict()
  for i=1:ds.users
    u[i - 1] = Array{Tuple}(0)
    uu[i - 1] = i
  end

  for i=1:ds.items
    v[i - 1] = Array{Tuple}(0)
    vv[i - 1] = i
  end

  for i=1:length(ds)
    Base.push!(u[ds.file[i,1] - 1], (ds.file[i, 2] - 1, ds.file[i, 3]))
    Base.push!(v[ds.file[i,2] - 1],(ds.file[i, 1] - 1, ds.file[i, 3]))
  end

  return surprise.Trainset(u, v, ds.users, ds.items, length(ds), (ds.preferences.min, ds.preferences.max), 0, uu, vv)
end

function Persa.train!(model::SurpriseModel, ds::Persa.CFDatasetAbstract)
  ds_surprise = transform(ds)

  return model.object[:train](ds_surprise)
end
