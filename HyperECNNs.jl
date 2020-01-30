module HyperECNNs

using LinearAlgebra
using Quaternions
using Random
rng = MersenneTwister(1234);

### Bilinear form used in the paper!

function LambdaInner(U,x,Params)
    w = zeros(size(U,3),1)  
    for i = 1:size(U,2)
        w = w + Params[i]*(U[:,i,:]'*x[:,i])
    end        
    return w
end

### Some activation functions

function csign(x,K)
    z = x[:,1]+x[:,2]*im
    phase_quanta = (round.(K*(2*pi.+angle.(z))./(2*pi))).%K
    z = exp.(2.0*pi*phase_quanta*im/K)
    return hcat(real.(z),imag.(z))
end

function twincsign(x, Params = 16)
    c1 = csign(x[:,1:2],Params)
    c2 = csign(x[:,3:4],Params)
    return hcat(c1,c2)
end

function SplitSign(a, Params = nothing)
    x = zeros(size(a))
    for i = 1:size(a,3)
        x[:,:,i] = sign.(a[:,:,i])
    end
    return x
end


##########################################################################
### Generic Hypercomplex-Valued Networks
##########################################################################

function Sync(BilinearForm, BilinearFormParams, ActFunction, ActFunctionParams, U, xinput, alpha = 1, beta = 0, it_max = 1.e3)
    ### Exponential Hypercomplex RCNN
    Name = "Hypercomplex ECNN (Synchronous)"

    N = size(U,1)
    hyperN = size(U,2)
    tau = 1.e-6
    
    # Initialization
    x = copy(xinput)
    xold = copy(x)
    it = 0
    Error = 1+tau      
    # Compute the weights;
    w = exp.(alpha*BilinearForm(U,x,BilinearFormParams).+beta);

    # Compute the energy
    Energy = zeros(1)
    Energy[1]=-sum(w)/alpha
    a = zeros(N,hyperN)
    while (Error>tau)&&(it<it_max)
        it = it+1
        
        # Compute the quaternion-valued activation potentials;
        for j = 1:hyperN
            a[:,j] = U[:,j,:]*w
        end

        # Compute the next state;
        x = ActFunction(a, ActFunctionParams)

        # Compute the weights;
        w = exp.(alpha*BilinearForm(U,x,BilinearFormParams).+beta);

        append!(Energy,-sum(w)/alpha)
        Error = norm(x-xold)
        xold = copy(x)
    end
    if it_max<=it
        println(Name," failed to converge in ",it_max," iterations.") 
    end
    return x, Energy
end

function Seq(BilinearForm, BilinearFormParams, ActFunction, ActFunctionParams, U, xinput, alpha = 1, beta = 0, it_max = 1.e3)
    ### Exponential Hypercomplex RCNN
    Name = "Hypercomplex ECNN (Asynchronous)"

    N = size(U,1)
    hyperN = size(U,2)
    tau = 1.e-6
    
    # Initialization
    x = copy(xinput)
    xold = copy(x)
    it = 0
    Error = 1+tau       
    Energy = zeros(1)
    EnergyAux = zeros(N)
    # Compute the weights;
    w = exp.(alpha*BilinearForm(U,x,BilinearFormParams).+beta);
    Energy[1]=-sum(w)/alpha
    a = zeros(1,hyperN)
    while (Error>tau)&&(it<it_max)
        it = it+1
        ind = randperm(rng, N)
        for i = 1:N
            # Compute the quaternion-valued activation potentials;
            for j = 1:hyperN
                a[1,j] = dot(U[ind[i],j,:],w)
            end

            # Compute the next state;
            x[ind[i],:] = ActFunction(a, ActFunctionParams)

            # Compute the weights;
            w = w.*exp.(alpha*BilinearForm(U[[ind[i],ind[i]],:,:],vcat(-xold[ind[i],:]',x[ind[i],:]'),BilinearFormParams));
            EnergyAux[i]=-sum(w)/alpha
        end
        append!(Energy,EnergyAux)
        Error = norm(x-xold)
        xold = copy(x)
    end
    if it_max<=it
        println(Name," failed to converge in ",it_max," iterations.") 
    end
    return x, Energy
end

function Seq_slow(BilinearForm, BilinearFormParams, ActFunction, ActFunctionParams, U, xinput, alpha = 1, beta = 0, it_max = 1.e3)
    ### Exponential Hypercomplex RCNN
    Name = "Hypercomplex ECNN (Asynchronous)"

    N = size(U,1)
    hyperN = size(U,2)
    tau = 1.e-6
    
    # Initialization
    x = copy(xinput)
    xold = copy(x)
    it = 0
    Error = 1+tau       
    Energy = zeros(1)
    EnergyAux = zeros(N)
    # Compute the weights;
    w = exp.(alpha*BilinearForm(U,x,BilinearFormParams).+beta);
    Energy[1]=-sum(w)/alpha
    a = zeros(1,hyperN)
    while (Error>tau)&&(it<it_max)
        it = it+1
        ind = randperm(rng, N)
        for i = 1:N
            # Compute the quaternion-valued activation potentials;
            for j = 1:hyperN
                a[1,j] = dot(U[ind[i],j,:],w)
            end

            # Compute the next state;
            x[ind[i],:] = ActFunction(a, ActFunctionParams)

            # Compute the weights;
            w = exp.(alpha*BilinearForm(U,x,BilinearFormParams).+beta);
            EnergyAux[i]=-sum(w)/alpha
        end
        append!(Energy,EnergyAux)
        Error = norm(x-xold)
        xold = copy(x)
    end
    if it_max<=it
        println(Name," failed to converge in ",it_max," iterations.") 
    end
    return x, Energy
end


##########################################################################
### Unit Quaternion Networks
##########################################################################
function UnitQ_Sync(U, xinput, alpha = 1, beta=0, it_max = 1.e3)
    ### Exponential Unit Quaternion RCNN
    Name = "Unit Quaternion ECNN (Synchronous)"

    N = size(U,1)
    tau = 1.e-6
    
    # Initialization
    x = copy(xinput)
    xold = copy(x)
    it = 0
    Error = 1+tau
    
    # Compute the weights
    w = exp.(alpha*Array{Float64}(real((U'*x))).+beta);
    
    # Compute the energy
    Energy = zeros(Float64,(1,))
    Energy[1] = -sum(w)/alpha
    while (Error>tau)&&(it<it_max)
        it = it+1
        
        # Compute the next state
        a = U*w
        x = a ./ abs.(a)
        
        # Compute the weights and the energy
        w = exp.(alpha*Array{Float64}(real((U'*x))).+beta);
        append!(Energy,-sum(w)/alpha) 
        
        Error = norm(x-xold)
        xold = copy(x)
    end
    if it_max<=it
        println(Name," failed to converge in ",it_max," iterations.")
    #else
    #    println(Name," converged in ",it," iterations using synchronous update.")    
    end
    return x, Energy
end

function UnitQ_Seq(U, xinput, alpha = 1, beta=0, it_max = 1.e3)
    ### Exponential Unit Quaternion RCNN
    Name = "Unit Quaternion ECNN (Asynchronous)"

    N = size(U,1)
    tau = 1.e-6

    # Initialization
    x = copy(xinput)
    xold = copy(x)
    it = 0
    Error = 1+tau       
    
    # Compute the weights;
    w = exp.(alpha*Array{Float64}(real((U'*x))).+beta);
    
    # Compute the energy
    Energy = zeros(Float64,(1,))
    EnergyAux = zeros(N)
    Energy[1]=-sum(w)/alpha
    while (Error>tau)&&(it<it_max)
        it = it+1
        ind = randperm(rng, N)
        for i = 1:N
            # Compute the quaternion-valued activation potentials;
            a = dot(conj(U[ind[i],:]),w)

            # Compute the next state;
            x[ind[i]] = a/abs(a)

            # Compute the weights;
            w = exp.(alpha*Array{Float64}(real((U'*x))).+beta);
            EnergyAux[i]=-sum(w)/alpha
        end
        append!(Energy,EnergyAux)
        Error = norm(x-xold)
        xold = copy(x)
    end
    if it_max<=it
        println(Name," failed to converge in ",it_max," iterations.") 
    end
    return x, Energy
end

end