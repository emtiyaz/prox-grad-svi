function [post nlZ dnlZ] = infKL_sprox(hyp, mean, cov, lik, x, y)
% PG-SVI

if hyp.is_cached==1
	global cache_post;
	global cache_nlz;
	global cache_idx;
	
	post=cache_post(cache_idx);
	nlZ=cache_nlz(cache_idx);
	if nargout>2
		warning('to be implemented\n');
		dnlZ = NaN;
	end
	return 
end

snu2=hyp.snu2;
n=size(x,1);

% GP prior
K = feval(cov{:}, hyp.cov, x);                  % evaluate the covariance matrix
m = feval(mean{:}, hyp.mean, x);                      % evaluate the mean vector
K=snu2*eye(n)+K;

lik_name = func2str(lik{1});
mini_batch_size=hyp.mini_batch_size;
assert (mini_batch_size>0)
mini_batch_num=ceil(n/mini_batch_size);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%init value
post_m=hyp.init_m;%k=0
post_m2=hyp.init_m;%k=0
tm1 = zeros(n,1);%k=-1
tm2 = zeros(n,1);%k=-1
post_v=diag(hyp.init_V);%k=0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

iter = 0;%iteration
pass=0;%pass
max_pass=hyp.max_pass;
beta = hyp.learning_rate;
r = 1/(beta+1);
index=1:n;
while pass<max_pass
	if mini_batch_size<n
		index=randperm(n);
	end
	offset=0;
	mini_batch_counter=0;
	pass=pass+1;
	while mini_batch_counter<mini_batch_num
		mini_batch_counter=mini_batch_counter+1;
		iter=iter+1;
		%mini batch
		tmp_idx = mini_batch_counter*mini_batch_size;
		idx=index( (tmp_idx-mini_batch_size+1):min(tmp_idx,n) );

		weight=double(n)/size(x(idx,:),1);

		if hyp.stochastic_approx==1
			[ll, gf, gv] = sampling_E(y(idx), post_m(idx), post_v(idx), lik, hyp.sample_size, hyp.lik);
		else
			switch lik_name
			case {'laplace','likLaplace','poisson','bernoulli_logit','likLogistic'}
				[ll, gf, gv] = E_log_p(lik_name, y(idx), post_m(idx), post_v(idx), hyp.lik);
			otherwise	 
				[ll,gf,d2f,gv] = likKL(post_v(idx), lik, hyp.lik, y(idx), post_m(idx));
			end
		end

		tm1 = r.*tm1; tm2 = r.*tm2;
		tm1(idx) = tm1(idx) + (1-r).*(weight*( gf-2*(gv.*post_m(idx)) ) );
		tm2(idx) = tm2(idx) + (1-r).*((weight*-2)*gv);

		sW = sqrt(abs(tm2)) .* sign(tm2);
		L = chol(eye(n)+sW*sW'.*K); %L = chol(sW*K*sW + eye(n)); 
		T = L'\(repmat(sW,1,n).*K); %T  = L'\(sW*K);
		post_v = diag(K) - sum(T.*T,1)'; % v = diag(inv(inv(K)+diag(W))); %v^{k+1}
		pseudo_y = K*tm1;
		post_m = m + pseudo_y - K*(sW.*(L\(L'\(sW.*pseudo_y))));
		

		% pseudo observation
		pseudo_y2 = m + K(:,idx)*(weight*gf) - post_m2;
		%use this following line if we approximate r^{k} .* tW.^{k-1} by tW.{k}
		post_m2 = post_m2 + (1-r).*(pseudo_y2 - K*(sW.*(L\(L'\(sW.*pseudo_y2)))));%m^{k+1}

		%fprintf('debug m=%.5f %.5f\n', post_m(7), post_m2(7));

		if isfield(hyp,'save_iter') && hyp.save_iter==1
			global cache_nlz_iter
			global cache_iter

			alpha=K\(post_m-m);
			nlZ2=compute_nlz(lik, hyp, sW, K, m, alpha, post_m, y);
			cache_iter=[cache_iter; iter];
			cache_nlz_iter=[cache_nlz_iter; nlZ2];
		end

	end
	alpha=K\(post_m-m);
	nlZ=compute_nlz(lik, hyp, sW, K, m, alpha, post_m, y);

	if isfield(hyp,'save_iter') && hyp.save_iter==1
		if pass==1
			global num_iters_at_pass;
			num_iters_at_pass=iter;
		end
		fprintf('pass:%d) at %d iter %.4f %.4f\n', pass, iter, nlZ, nlZ2);
	else
		fprintf('pass:%d) %.4f\n', pass, nlZ);
	end

	if hyp.is_save==1
		global cache_post;
		global cache_nlz;

		post.sW = sW;                                             % return argument
		post.alpha = alpha;
		post.L = L;      

		cache_post=[cache_post; post];
		cache_nlz=[cache_nlz; nlZ];
	end
end

alpha=K\(post_m-m);
post.sW = sW;                                             % return argument
post.alpha = alpha;
post.L = L;                                              % L'*L=B=eye(n)+sW*K*sW

nlZ=compute_nlz(lik, hyp, sW, K, m, alpha, post_m, y);
fprintf('final: %.4f\n', nlZ);

if nargout>2
  warning('to be implemented\n');
  dnlZ = NaN;
end
