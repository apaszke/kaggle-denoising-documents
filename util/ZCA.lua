local ZCA = torch.class('ZCA')

function ZCA:__init(eps)
    self.eps = eps or 10e-5
end

function ZCA:fit(X)
    self.mean = torch.mean(X, 1)
    self.substractMean(X, self.mean)
    sigma = X:t() * X / X:size(1)
    U, S, V = torch.svd(sigma, 'S')
--     S, U = torch.symeig(sigma,'V')
    sigma = nil
    collectgarbage()
    self.components = U * torch.diag(torch.sqrt(S + self.eps):pow(-1)) * U:t()
end

function ZCA:transform(X)
    ZCA.substractMean(X, self.mean)
    return X * self.components:t()
end

function ZCA.substractMean(X, mean)
    for i = 1,X:size(1) do
        X[i]:add(-1, mean)
    end
end
