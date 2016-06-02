function method_comparison_step_size_sprox(varargin)
if nargin<1 
	disp('Usage method_comparison_step_size_sprox(dataset_name, output_path, check_idx)');
	return
else
	dataset_name=char(varargin(1));
end
path='../dataset';
if nargin>1
	output_path=char(varargin(2));
end
check=-1;
if nargin>2
	check=varargin(3);
	check=check{:};
end
if nargin>3
	path=char(varargin(4));
end
switch dataset_name
case 'ionosphere'
	name=sprintf('%s/ionosphere.data-',path);
	hyp.cov=[1.0, 2.5];
	hyp.snu2=1e-2;
otherwise
	error('unsupported dataset')
end

switch dataset_name
case {'ionosphere'}
	train_name=strcat(name,'tra.csv');
	data=csvread(train_name);
	X_train=data(:,1:end-1);
	Y_train=data(:,end);
	test_name=strcat(name,'tst.csv');
	data=csvread(test_name);
	X_test=data(:,1:end-1);
	Y_test=data(:,end);
otherwise
	error('unsupported dataset')
end

covfuncF = @covSEiso;
meanfunc = @meanConst;
n_test=size(X_test,1);
n_train=size(X_train,1);
hyp.mean = 0;
seed=5;

switch dataset_name
case {'ionosphere'}
	likfunc = @likLogistic;
	hyp.lik=[];
otherwise
	error('unsupported dataset')
end

%hyp.stochastic_approx=0 disable stochastic approxmation for the expectation
%hyp.stochastic_approx=1 enable stochastic approxmation for the expectation
%hyp.snu2 is used to correct the kernel matrix (eg, K_corrected=K+hyp.snu2*eye(n))

algos = {'infKL_sprox','infKL_sprox','infKL_sprox','infKL_sprox','infKL_sprox','infKL_sprox','infKL_sprox','infKL_sprox','infKL_sprox','infKL_sprox','infKL_sprox','infKL_sprox','infKL_sprox','infKL_sprox','infKL_sprox','infKL_sprox','infKL_sprox','infKL_sprox','infKL_sprox','infKL_sprox','infKL_sprox', 'infKL_sprox','infKL_sprox','infKL_sprox', 'infKL_sprox', 'infKL_sprox', 'infKL_sprox', 'infKL_sprox'};

hyp.init_m = feval(meanfunc, hyp.mean, X_train); 
n=size(X_train,1);
hyp.init_V = feval(covfuncF, hyp.cov, X_train);
hyp.init_V=hyp.init_V+hyp.snu2*eye(n); 


hyp.stochastic_approx=0;
[a b c d log_p_hat, e, nlZ0] = gp(hyp, @infKL_init, meanfunc, covfuncF, likfunc, X_train, Y_train, X_test, Y_test);

log_loss0=-mean(log_p_hat);
res_name=sprintf('./%s/%s.data.init.mat',output_path,dataset_name);
save(res_name,'nlZ0', 'log_loss0');

hyp.stochastic_approx=1;
hyp.sample_size=500;
hyp.mini_batch_size=5;
for i=1:length(algos) 
	if check>0 && i~=check
		continue
	end
	switch i
	case 1
		disp('check1')
		hyp.learning_rate=10^-5;
		switch dataset_name
		case 'ionosphere'
			max_pass=3500;
		otherwise
			error('unsupported dataset')
		end
	case 2
		disp('check2')
		hyp.learning_rate=10^-4.85;
		switch dataset_name
		case 'ionosphere'
			max_pass=3500;
		otherwise
			error('unsupported dataset')
		end
	case 3
		disp('check3')
		hyp.learning_rate=10^-4.7;
		switch dataset_name
		case 'ionosphere'
			max_pass=3500;
		otherwise
			error('unsupported dataset')
		end
	case 4
		disp('check4')
		hyp.learning_rate=10^-4.55;
		switch dataset_name
		case 'ionosphere'
			max_pass=3000;
		otherwise
			error('unsupported dataset')
		end
	case 5
		disp('check5')
		hyp.learning_rate=10^-4.4;
		switch dataset_name
		case 'ionosphere'
			max_pass=3000;
		otherwise
			error('unsupported dataset')
		end
	case 6
		disp('check6')
		hyp.learning_rate=10^-4.25;
		switch dataset_name
		case 'ionosphere'
			max_pass=2500;
		otherwise
			error('unsupported dataset')
		end
	case 7
		disp('check7')
		hyp.learning_rate=10^-4.1;
		switch dataset_name
		case 'ionosphere'
			max_pass=2500;
		otherwise
			error('unsupported dataset')
		end
	case 8
		disp('check8')
		hyp.learning_rate=10^-3.95;
		switch dataset_name
		case 'ionosphere'
			max_pass=2000;
		otherwise
			error('unsupported dataset')
		end
	case 9
		disp('check9')
		hyp.learning_rate=10^-3.8;
		switch dataset_name
		case 'ionosphere'
			max_pass=2000;
		otherwise
			error('unsupported dataset')
		end
	case 10
		disp('check10')
		hyp.learning_rate=10^-3.65;
		switch dataset_name
		case 'ionosphere'
			max_pass=2000;
		otherwise
			error('unsupported dataset')
		end
	case 11
		disp('check11')
		hyp.learning_rate=10^-3.5;
		switch dataset_name
		case 'ionosphere'
			max_pass=2000;
		otherwise
			error('unsupported dataset')
		end
	case 12
		disp('check12')
		hyp.learning_rate=10^-3.35;
		switch dataset_name
		case 'ionosphere'
			max_pass=2000;
		otherwise
			error('unsupported dataset')
		end
	case 13
		disp('check13')
		hyp.learning_rate=10^-3.2;
		switch dataset_name
		case 'ionosphere'
			max_pass=2000;
		otherwise
			error('unsupported dataset')
		end
	case 14
		disp('check14')
		hyp.learning_rate=10^-3.05;
		switch dataset_name
		case 'ionosphere'
			max_pass=2000;
		otherwise
			error('unsupported dataset')
		end
	case 15
		disp('check15')
		hyp.learning_rate=10^-2.9;
		switch dataset_name
		case 'ionosphere'
			max_pass=2000;
		otherwise
			error('unsupported dataset')
		end
	case 16
		disp('check16')
		hyp.learning_rate=10^-2.75;
		switch dataset_name
		case 'ionosphere'
			max_pass=2000;
		otherwise
			error('unsupported dataset')
		end
	case 17
		disp('check17')
		hyp.learning_rate=10^-2.6;
		switch dataset_name
		case 'ionosphere'
			max_pass=2000;
		otherwise
			error('unsupported dataset')
		end
	case 18
		disp('check18')
		hyp.learning_rate=10^-2.45;
		switch dataset_name
		case 'ionosphere'
			max_pass=2000;
		otherwise
			error('unsupported dataset')
		end
	case 19
		disp('check19')
		hyp.learning_rate=10^-2.3;
		switch dataset_name
		case 'ionosphere'
			max_pass=2000;
		otherwise
			error('unsupported dataset')
		end
	case 20
		disp('check20')
		hyp.learning_rate=10^-2.15;
		switch dataset_name
		case 'ionosphere'
			max_pass=2000;
		otherwise
			error('unsupported dataset')
		end
	case 21
		disp('check21')
		hyp.learning_rate=10^-2;
		switch dataset_name
		case 'ionosphere'
			max_pass=2000;
		otherwise
			error('unsupported dataset')
		end
	case 22
		disp('check22')
		hyp.learning_rate=10^-1.85;
		switch dataset_name
		case 'ionosphere'
			max_pass=2000;
		otherwise
			error('unsupported dataset')
		end
	case 23
		disp('check23')
		hyp.learning_rate=10^-3;
		switch dataset_name
		case 'ionosphere'
			max_pass=2000;
		otherwise
			error('unsupported dataset')
		end
	case 24
		disp('check24')
		hyp.learning_rate=10^-1;
		switch dataset_name
		case 'ionosphere'
			max_pass=2000;
		otherwise
			error('unsupported dataset')
		end
	case 25
		disp('check25')
		hyp.learning_rate=10^-4;
		switch dataset_name
		case 'ionosphere'
			max_pass=2500;
		otherwise
			error('unsupported dataset')
		end
	case 26
		disp('check26')
		hyp.learning_rate=10^-1.95;
		switch dataset_name
		case 'ionosphere'
			max_pass=2000;
		otherwise
			error('unsupported dataset')
		end
	case 27
		disp('check27')
		hyp.learning_rate=10^-1.9;
		switch dataset_name
		case 'ionosphere'
			max_pass=2000;
		otherwise
			error('unsupported dataset')
		end
	otherwise
		error('unsupported method\n')
	end
	inffunc=algos{i};

	hyp.init_m = feval(meanfunc, hyp.mean, X_train); 
	n=size(X_train,1);
	hyp.init_V = feval(covfuncF, hyp.cov, X_train);
	hyp.init_V=hyp.init_V+hyp.snu2*eye(n); 

	hyp.is_cached=0; %if 1, then load result from cache_post and cache_nlz
	hyp.is_save=0;%if 1, then save result into cache_post and cache_nlz

	hyp.max_pass=max_pass; 
	global cache_nlz_iter;
	global cache_iter;
	global num_iters_at_pass;
	num_iters_at_pass=1;
	cache_nlz_iter=[nlZ0]; cache_iter=[0];
	hyp.save_iter=1;%if 1, then save result into cache_iter and cache_nlz_iter

	rng(seed);
	feval(inffunc, hyp, {meanfunc}, {covfuncF}, {likfunc}, X_train, Y_train);

	assert( length(cache_nlz_iter) == max_pass*num_iters_at_pass+1 );
	assert( length(cache_nlz_iter) == length(cache_iter) );
	method_nlz_iter=cache_nlz_iter;
	method_iter=cache_iter;

	res_name=sprintf('./%s/%s.data.plot-sprox-%d.mat',output_path,dataset_name,i);
	if isfield(hyp,'save_iter') && hyp.save_iter==1
		save(res_name,'method_nlz_iter','method_iter','num_iters_at_pass');
	else
		error('do not support')
	end
end
