using PyCall

println("Running build.jl for the OdsIO package.")

# Change that to whatever packages you need.
const PACKAGES = ["numpy", "scikit-surprise"]

# Use eventual proxy info
proxy_arg=String[]
if haskey(ENV, "http_proxy")
    push!(proxy_arg, "--proxy")
    push!(proxy_arg, ENV["http_proxy"])
end

# Import pip
try
    @pyimport pip
catch ee
    typeof(ee) <: PyCall.PyError || rethrow(ee)
    error("""
Python Pip not installed
Please either:
- Install pip
- Rebuild Surprise.jl via `Pkg.build("Surprise")` in the julia REPL
""")
end

try
    @pyimport surprise
catch
    println("Installing required python packages using pip")
    run(`$(PyCall.python) $(proxy_arg) -m pip install --user $(PACKAGES)`)
end
