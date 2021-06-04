using PyCall: pyimport_conda
using Conda

function installpypackage()
	try
		pyimport_conda("sklearn", "scikit-surprise")
	catch
		try
			Conda.add("scikit-surprise", channel="conda-forge")
		catch
			println("scikit-learn failed to install")
		end
	end
end

installpypackage()