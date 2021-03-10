function [X_norm, mu, sigma] = featureNormalize(X)

mu = mean(X);         % Mean values of each feature
sigma = std(X);       % Standard deviation, or range, of each feature

t = ones(length(X), 1);
X_norm = (X - (t * mu)) ./ (t * sigma); % Vectorized

end
