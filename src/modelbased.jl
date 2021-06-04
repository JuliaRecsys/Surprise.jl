struct SVD{T} <: SurpriseModel{T}
    object::PyObject
    features::Int
    n_epochs::Int
    lrate::Float64
    lambda::Float64
    preference::Persa.Preference{T}
    users::Int
    items::Int
end

function SVD(
    dataset::Persa.Dataset,
    biased::Bool;
    features = 100,
    n_epochs = 20,
    lrate = 0.005,
    lambda = 0.02,
)
    return SVD(
        surprise.SVD(
            biased = biased,
            n_factors = features,
            n_epochs = n_epochs,
            lr_all = lrate,
            reg_all = lambda,
        ),
        features,
        n_epochs,
        lrate,
        lambda,
        dataset.preference,
        Persa.users(dataset),
        Persa.items(dataset)
    )
end

"""
    IRSVD(dataset::Persa.Dataset; features = 100, n_epochs = 20, lrate = 0.005, lambda = 0.02)
Improved Regulared SVD. Extension of RSVD algorithm adding the user and item
bias.
# Arguments
- `features::Int = 100`: Number of factors.
- `n_epochs::Int = 20`: The number of iteration of the SGD algorithm.
- `lrate::Float = 0.005`: Learning rate parameter.
- `lambda::Int = 0.02`: Regularization parameter.
"""
function IRSVD(
    dataset::Persa.Dataset;
    features = 100,
    n_epochs = 20,
    lrate = 0.005,
    lambda = 0.02,
)
    return SVD(
        dataset,
        true;
        features = features,
        n_epochs = n_epochs,
        lrate = lrate,
        lambda = lambda,
    )
end

"""
    RSVD(dataset::Persa.Dataset; features = 100, n_epochs = 20, lrate = 0.005, lambda = 0.02)
Regulared SVD. The algorithm is also known as SVD by Funk.
# Arguments
- `features::Int = 100`: Number of factors.
- `n_epochs::Int = 20`: The number of iteration of the SGD algorithm.
- `lrate::Float = 0.005`: Learning rate parameter.
- `lambda::Int = 0.02`: Regularization parameter.
"""
function RSVD(
    dataset::Persa.Dataset;
    features = 100,
    n_epochs = 20,
    lrate = 0.005,
    lambda = 0.02,
)
    return SVD(
        dataset,
        false;
        features = features,
        n_epochs = n_epochs,
        lrate = lrate,
        lambda = lambda,
    )
end

struct NMF{T} <: SurpriseModel{T}
    object::PyObject
    features::Int
    n_epochs::Int
    biased::Bool
    preference::Persa.Preference{T}
    users::Int
    items::Int
end

"""
    NMF(dataset::Persa.Dataset; features = 100, n_epochs = 20, lrate = 0.005, lambda = 0.02)
Collaborative Filtering based on Non-negative Matrix Factorization.
# Arguments
- `features::Int = 15`: Number of factors.
- `n_epochs::Int = 50`: The number of iteration of the SGD algorithm.
- `biased::Bool = false`: If the method use biases.
"""
function NMF(
    dataset::Persa.Dataset;
    features::Int = 15,
    n_epochs::Int = 50,
    biased::Bool = false,
)
    return NMF(
        surprise.NMF(n_factors = features, n_epochs = n_epochs, biased = biased),
        features,
        n_epochs,
        biased,
        dataset.preference,
        Persa.users(dataset),
        Persa.items(dataset)
    )
end
