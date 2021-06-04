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


