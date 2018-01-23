function [ COEFF, SCORE, LATENT ] = PCA( X )
	mean_X = mean(X);
	[n_X, m_X] = size(X);
	X_mean = X - ones(n_X,1) * mean_X;
	Cov_T = X_mean' * X_mean / (m_X - 1);

	[U, S, V] = svd(Cov_T);
	[n_S, m_S] = size(S);
	sum_all = 0;
	sum2k = 0;
	k = 0;
	div = 0;
	for i = 1:n_S
		sum_all = sum_all + S(i,i);
	end
	while div < 0.95
		k = k + 1;
		sum2k = sum2k + S(k,k);
		div = sum2k / sum_all;
	end
	U_k = U(:,1:k);
	S_k = S(1:k,1:k)' * S(1:k,1:k); 

	COEFF = X_mean * U_k;			% n * k
	SCORE = COEFF' * X_mean;
	LATENT = S_k;

end