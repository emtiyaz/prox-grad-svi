function method_comparison_single(varargin)
%Wu Lin
if nargin<1 
	disp('Usage method_comparison_online(dataset_name, output_path, check_idx)');
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
case 'sonar'
	name=sprintf('%s/sonar.all-data-',path);
	hyp.cov=[-1.0,6.0];
	hyp.snu2=1e-4;
	max_pass=5000;
case 'ionosphere'
	name=sprintf('%s/ionosphere.data-',path);
	hyp.cov=[1.0, 2.5];
	hyp.snu2=1e-2;
	max_pass=500;
case 'usps3v5'
	name=sprintf('%s/usps_resampled.mat',path);
	hyp.cov=[2.5, 5.0];
	hyp.snu2=1e-4;
	%max_pass=2000;%for sgd
	max_pass=1000;%for sprox and prox
otherwise
	error('unsupported dataset')
end

switch dataset_name
case {'sonar','ionosphere'}
	train_name=strcat(name,'tra.csv');
	data=csvread(train_name);
	X_train=data(:,1:end-1);
	Y_train=data(:,end);
	test_name=strcat(name,'tst.csv');
	data=csvread(test_name);
	X_test=data(:,1:end-1);
	Y_test=data(:,end);
case 'usps3v5'
	load(name)
	train_positive_idx=train_labels(3,:)==1;
	train_negative_idx=train_labels(5,:)==1;
	test_positive_idx=test_labels(3,:)==1;
	test_negative_idx=test_labels(5,:)==1;
	X_train=[ train_patterns(:,train_positive_idx),  train_patterns(:,train_negative_idx) ]';
	Y_train=[ ones(1, sum(train_positive_idx==1)), -1 .* ones(1, sum(train_negative_idx==1)) ]';
	X_test=[ test_patterns(:,test_positive_idx),  test_patterns(:,test_negative_idx) ]';
	Y_test=[ ones(1, sum(test_positive_idx==1)), -1 .* ones(1, sum(test_negative_idx==1)) ]';
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
case {'sonar','ionosphere','usps3v5'}
	likfunc = @likLogistic;
	hyp.lik=[];
otherwise
	error('unsupported dataset')
end

%hyp.stochastic_approx=0 disable stochastic approxmation for the expectation
%hyp.stochastic_approx=1 enable stochastic approxmation for the expectation
%hyp.snu2 is used to correct the kernel matrix (eg, K_corrected=K+hyp.snu2*eye(n))

algos = {'infKL_sprox','infKL_sprox_pcg'};
hyp.init_m = feval(meanfunc, hyp.mean, X_train); 
n=size(X_train,1);
hyp.init_V = feval(covfuncF, hyp.cov, X_train);
hyp.init_V=hyp.init_V+hyp.snu2*eye(n); 

hyp.stochastic_approx=0;
[a b c d log_p_hat, e, nlZ0] = gp(hyp, @infKL_init, meanfunc, covfuncF, likfunc, X_train, Y_train, X_test, Y_test);

log_loss0=-mean(log_p_hat);
res_name=sprintf('./%s/%s.data.init.mat',output_path,dataset_name);
save(res_name,'nlZ0', 'log_loss0');

hyp.mini_batch_size=1;
hyp.sample_size=50; 
for i=1:length(algos) 
	if check>0 && i~=check
		continue
	end
	switch i
	case 1
		%%%doubly sprox
		hyp.stochastic_approx=1;
		switch dataset_name
		case 'sonar'
			hyp.learning_rate=0.01/n_train;
		case 'ionosphere'
			hyp.learning_rate=0.2/n_train;
		case 'usps3v5'
			hyp.learning_rate=0.1/n_train;
		otherwise
			error('unsupported dataset')
		end
	case 2
		%%%doubly sprox
		hyp.stochastic_approx=1;
		switch dataset_name
		case 'sonar'
			hyp.learning_rate=0.01/n_train;
		case 'ionosphere'
			hyp.learning_rate=0.2/n_train;
		case 'usps3v5'
			hyp.learning_rate=0.1/n_train;
		otherwise
			error('unsupported dataset')
		end
	otherwise
		error('unsupported method\n')
	end
	method_loss=[log_loss0];
	method_nlz=[nlZ0];
	method_pass=[0];

	inffunc=algos{i};
	
	hyp.init_m = feval(meanfunc, hyp.mean, X_train); 
	n=size(X_train,1);
	hyp.init_V = feval(covfuncF, hyp.cov, X_train);
	hyp.init_V=hyp.init_V+hyp.snu2*eye(n); 

	hyp.is_cached=0; %if 1, then load result from cache_post and cache_nlz
	hyp.is_save=1;%if 1, then save result into cache_post and cache_nlz

	hyp.max_pass=max_pass; 
	global cache_idx;
	global cache_post;
	global cache_nlz;
	cache_post=[]; cache_nlz=[];
	rng(seed);
	feval(inffunc, hyp, {meanfunc}, {covfuncF}, {likfunc}, X_train, Y_train);
	assert( length(cache_nlz) == max_pass );
	assert( length(cache_post) == max_pass );

	global g_pass;
	g_pass=0;
	skipped=0;
	for pass=1:max_pass
		rng(seed);
		hyp.max_pass=pass;
		cache_idx=pass;
		hyp.is_cached=1; %if 1, then load result from cache_post and cache_nlz
		hyp.is_save=0;%if 1, then save result into cache_post and cache_nlz

		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		%just in case
		hyp.init_m = feval(meanfunc, hyp.mean, X_train); 
		n=size(X_train,1);
		hyp.init_V = feval(covfuncF, hyp.cov, X_train);
		hyp.init_V=hyp.init_V+hyp.snu2*eye(n); 
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		if pass>g_pass && skipped==0
			g_pass=pass;
			[a b c d log_p_hat, e, nlZ] = gp(hyp, inffunc, meanfunc, covfuncF, likfunc, X_train, Y_train, X_test, Y_test);
			log_loss=-mean(log_p_hat);
			g_pass=max(pass,g_pass);
			if g_pass<pass
				fprintf('found the optmial using g_pass=%d at %d-th pass\n',g_pass,pass);
				skipped=1;
			end
			g_pass=max(pass,g_pass);
		else
			if skipped==1
				g_pass=pass;
			else
				fprintf('skipped %d-th pass due to g_pass=%d\n',pass,g_pass);
			end
		end
		method_pass=[method_pass; g_pass];
		method_loss=[method_loss; log_loss];
		method_nlz=[method_nlz; nlZ];
	end
	res_name=sprintf('./%s/%s.data.resv2-%d.mat',output_path,dataset_name,i);
	save(res_name,'method_pass', 'method_loss', 'method_nlz');
end

load_results_and_plot_nips(dataset_name,output_path,length(algos));
