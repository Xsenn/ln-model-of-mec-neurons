%% Description
% The model: r = exp(W*theta), where r is the predicted # of spikes, W is a
% matrix of one-hot vectors describing variable (P, H, S, or T) values, and
% theta is the learned vector of parameters.

%% compute the position, head direction, speed, and theta phase matrices

% initialize the number of bins that position, head direction, speed, and
% theta phase will be divided into
n_pos_bins = 20;
n_dir_bins = 18;
n_speed_bins = 10;
% n_theta_bins = 18;

% compute position matrix
[posgrid, posVec] = pos_map([posx_c posy_c], n_pos_bins, boxSize);

% compute head direction matrix
[hdgrid,hdVec,direction] = hd_map(posx,posx2,posy,posy2,n_dir_bins);

% compute speed matrix
[speedgrid,speedVec,speed] = speed_map(posx_c,posy_c,n_speed_bins);

% compute theta matrix
% [thetagrid,thetaVec,phase] = theta_map(filt_eeg,post,eeg_sample_rate,n_theta_bins);

% remove times when the animal ran > 50 cm/s (these data points may contain artifacts)
too_fast = find(speed >= 50);
posgrid(too_fast,:) = []; hdgrid(too_fast,:) = []; 
speedgrid(too_fast,:) = []; % thetagrid(too_fast,:) = [];
spiketrain(too_fast) = [];

%% Fit all 15 LN models

numModels = 7;
testFit = cell(numModels,1);
trainFit = cell(numModels,1);
param = cell(numModels,1);
A = cell(numModels,1);
modelType = cell(numModels,1);

% ALL VARIABLES
A{1} = [ posgrid hdgrid speedgrid]; modelType{1} = [1 1 1];
% TWO VARIABLES
A{2} = [ posgrid hdgrid]; modelType{2} = [1 1 0 ];
A{3} = [ posgrid  speedgrid ]; modelType{3} = [1 0 1];
A{4} = [  hdgrid speedgrid ]; modelType{4} = [0 1 1];
% ONE VARIABLE
A{5} = posgrid; modelType{5} = [1 0 0];
A{6} = hdgrid; modelType{6} = [0 1 0];
A{7} = speedgrid; modelType{7} = [0 0 1];

% compute a filter, which will be used to smooth the firing rate
filter = gaussmf(-4:4,[2 0]); filter = filter/sum(filter); 
dt = post(3)-post(2); fr = spiketrain/dt;
smooth_fr = conv(fr,filter,'same');

% compute the number of folds we would like to do
numFolds = 10;

for n = 1:numModels
    fprintf('\t- Fitting model %d of %d\n', n, numModels);
    [testFit{n},trainFit{n},param{n}] = fit_model(A{n},dt,spiketrain,filter,modelType{n},numFolds);
end

