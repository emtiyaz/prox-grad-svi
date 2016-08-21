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
		pseudo_y = K*tm1+m;
		post_m = pseudo_y - K*(sW.*(L\(L'\(sW.*pseudo_y))));
		

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
	alpha2=(diag(abs(tm2))*K+eye(n))\(tm1-abs(tm2).*m);
	fprintf('ddd %.4f %.4f\n',alpha(4),alpha2(4));

	nlZ=compute_nlz(lik, hyp, sW, K, m, alpha, post_m, y);

	assert( all(tm2)>0 );
	pseudo_y3=tm1./tm2; 
	lp=sum((pseudo_y3.*pseudo_y3.*tm2) - log(tm2))/2;
	T2 = L'\(sW.*pseudo_y3);
	D=L'\(diag(sW));
	nlZ_exp=lp-0.5*T2'*T2+sum(log(diag(D)));


	%tnu = alpha + tm2.*post_m; ttau = tm2; 
	%tnu = tm1; ttau = tm2; 
	tnu = alpha + tm2.*(K*alpha); ttau = tm2; 
	%tnu = alpha + tm2.*post_m; ttau = tm2; 
	[~,~,~,~,nlZ_ep] = epComputeParams(K,y,ttau,tnu,lik,hyp,m,'infEP');

	if nlZ_ep < 82
	fprintf('\npass:%d) exp:%.4f %.4f\n', pass, nlZ_exp, nlZ_ep);
	end


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



% log(det(A)) for det(A)>0 using the LU decomposition of A
function y = logdet(A)
[L,U] = lu(A); u = diag(U); 
if prod(sign(u))~=det(L), error('det(A)<=0'), end 
y = sum(log(abs(u)));





function [Sigma,mu,L,alpha,nlZ] = epComputeParams(K,y,ttau,tnu,lik,hyp,m,inf)
% function to compute the parameters of the Gaussian approximation, Sigma and
% mu, and the negative log marginal likelihood, nlZ, from the current site
% parameters, ttau and tnu. Also returns L (useful for predictions).
  n = length(y);                                      % number of training cases
  sW = sqrt(ttau);                                        % compute Sigma and mu
  L = chol(eye(n)+sW*sW'.*K);                            % L'*L=B=eye(n)+sW*K*sW
  V = L'\(repmat(sW,1,n).*K);
  Sigma = K - V'*V;
  alpha = tnu-sW.*solve_chol(L,sW.*(K*tnu+m));
  mu = K*alpha+m; v = diag(Sigma);

  tau_n = 1./diag(Sigma)-ttau;             % compute the log marginal likelihood
  nu_n  = mu./diag(Sigma)-tnu;                    % vectors of cavity parameters
  lZ = feval(lik{:}, hyp.lik, y, nu_n./tau_n, 1./tau_n, inf);
  p = tnu-m.*ttau; q = nu_n-m.*tau_n;                        % auxiliary vectors
  nlZ = sum(log(diag(L))) - sum(lZ) - p'*Sigma*p/2 + (v'*p.^2)/2 ...
      - q'*((ttau./tau_n.*q-2*p).*v)/2 - sum(log(1+ttau./tau_n))/2;
