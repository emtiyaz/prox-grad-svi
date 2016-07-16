clear all
% This code compares the batch proximal-gradient (PG) method with EP
% For PG, we test 3 methods to approximate E[log p(y|f)]
% 'gauss_hermite', 'piecewise', 'monte_carlo'.
% For gauss_hermite we use likKL.m from GPML toolbox.
% For monte_carlo default is to increase sample size linearly
% but a fixed sample size can also be specificied (see infKL_PG.m)

% Choose a dataset and an approximation method for E(log p(y|f))
data_name = 'usps_3vs5';%, 'usps_3vs5', 'sonar', 'housing'
hyp.approx_method = 'monte_carlo'; % 'gauss_hermite', 'piecewise', 'monte_carlo'

switch data_name
case 'ionosphere'
  switch hyp.approx_method
  case 'gauss_hermite'; hyp.step_size = 1;
  case 'piecewise'; hyp.step_size = 1; 
  case 'monte_carlo'; hyp.step_size = 1; 
    hyp.test_convergence = 0; hyp.compute_marglik = 0;
  end
  % numer of iterations
  hyp.max_iters = 30; hyp.verbose = 1;
  % likelihood function
  lik_func = {@likLogistic}; hyp.lik = [];
  ell = 1; sf = 2.5; seed = 147;

case 'usps_3vs5' 
  switch hyp.approx_method
  case 'gauss_hermite'; hyp.step_size = .5;
  case 'piecewise'; hyp.step_size = 1; 
  case 'monte_carlo'; hyp.step_size = .5; 
    hyp.test_convergence = 0; hyp.compute_marglik = 0;
  end
  % numer of iterations
  hyp.max_iters = 100; hyp.verbose = 1;
  % likelihood function
  lik_func = {@likLogistic}; hyp.lik = [];
  ell = 2.5; sf = 5; seed = 147;
   
case 'sonar'
  switch hyp.approx_method
  case 'gauss_hermite'; hyp.step_size = .8;
  case 'piecewise'; hyp.step_size = .8; 
  case 'monte_carlo'; hyp.step_size = .1; 
    hyp.test_convergence = 0; hyp.compute_marglik = 0;
  end
  % numer of iterations
  hyp.max_iters = 300; hyp.verbose = 1;
  % likelihood function
  lik_func = {@likLogistic}; hyp.lik = [];
  ell = -1; sf = 6; seed = 1;
 
case 'housing';
  % approximation method for E(log p(y|f))
  hyp.approx_method = 'gauss_hermite'; hyp.step_size = 1;
  % numer of iterations
  hyp.max_iters = 50; hyp.verbose = 1;
  % likelihood function
  lik_func = {@likLaplace}; hyp.lik = -2;
  ell = 2.5; sf = 5; sn = -2; seed = 1;
  
otherwise
  error('no such data name');
end

% get data
[y, X, y_te, X_te] = get_data_gp(data_name, seed);

% set the GP prior with covSEiso Kernel
cov_func = {@covSEiso}; 
hyp.cov = [ell; sf];
mean_func = {@meanZero}; 
hyp.mean = [];

% run algos 
algos = {'infKL_PG','infEP'}; % compare against EP
setSeed(1);
for i = 1:length(algos)
  tic;
  [~,~,m_hat,v_hat,log_p_hat,~,nlZ(i)] = gp(hyp, algos{i}, mean_func, cov_func, lik_func, X, y, X_te, y_te);
  tt(i) = toc;
  
  % compute log_loss
  log_loss(i) = -mean(log_p_hat);
  fprintf('%s, log_loss = %0.4f, nlZ_EP = %0.4f, took %1.1fs\n', algos{i}, log_loss(i), nlZ(i), tt(i));
end

%{
% for Wu's code
n = length(y);
hyp.is_cached = 0;
hyp.save_iter = 0;
hyp.is_save = 0;
hyp.snu2 = 1e-4;
hyp.mini_batch_size = n;
hyp.init_m = 0*ones(n,1);
hyp.init_V = eye(n);
hyp.max_pass = 1000;
hyp.learning_rate = 0.1;
hyp.stochastic_approx = 1;
hyp.sample_size = 500;
%}


