%% Description of run_me
dbstop if error;
warning('off','all')
clearvars; close all; clc
% This script is segmented into several parts. First, the data (an
% example cell) is loaded. Then, 15 LN models are fit to the
% cell's spike train. Each model uses information about 
% position, head direction, running speed, theta phase,
% or some combination thereof, to predict a section of the
% spike train. Model fitting and model performance is computed through
% 10-fold cross-validation, and the minimization procedure is carried out
% through fminunc. Next, a forward-search procedure is
% implemented to find the simplest 'best' model describing this spike
% train. Following this, the firing rate tuning curves are computed, and
% these - along with the model-derived response profiles and the model
% performance and results of the selection procedure are plotted.

% Code as implemented in Hardcastle, Maheswaranthan, Ganguli, Giocomo,
% Neuron 2017
% V1: Kiah Hardcastle, March 16, 2017


%% Clear the workspace and load the data

files = subdir('C:\Users\elhan\Github\Cells\6_Y2\*.mat');
previousFilesToProcess = [276 279 285 286];

for iiFile = 1:length(files)
fprintf('Processing file %i / %i\n', iiFile, length(files));

% load the data
load(files(iiFile).name);

if iiFile < 31 && ~ismember(c.cell_number, previousFilesToProcess)
    continue;
end

boxSize = 100;
post = cVt.timestamps;

    % make spiketrain
    dt = mean(diff(post));
    timebins = [post; (post(end) + dt)];
    spiketrain = histcounts(c.timestamps, timebins)';

posx = cVt.posx;
posx2 = cVt.posx2;
posx_c = cVt.posx_c;
posy = cVt.posy;
posy2 = cVt.posy2;
posy_c = cVt.posy_c;
sampleRate = 30;

fprintf('(1/5) Loading data from example cell \n')
% load data_for_cell77
% description of variables included:
% boxSize = length (in cm) of one side of the square box
% post = vector of time (seconds) at every 20 ms time bin
% spiketrain = vector of the # of spikes in each 20 ms time bin
% posx = x-position of left LED every 20 ms
% posx2 = x-position of right LED every 20 ms
% posx_c = x-position in middle of LEDs
% posy = y-position of left LED every 20 ms
% posy2 = y-posiiton of right LED every 20 ms
% posy_c = y-position in middle of LEDs
% filt_eeg = local field potential, filtered for theta frequency (4-12 Hz)
% eeg_sample_rate = sample rate of filt_eeg (250 Hz)
% sampleRate = sampling rate of neural data and behavioral variable (50Hz)


%% fit the model
fprintf('(2/5) Fitting all linear-nonlinear (LN) models\n')
fit_all_ln_models

%% find the simplest model that best describes the spike train
fprintf('(3/5) Performing forward model selection\n')
select_best_model

%% Compute the firing-rate tuning curves
fprintf('(4/5) Computing tuning curves\n')
compute_all_tuning_curves

%% plot the results
if ~isnan(selected_model)
    fprintf('(5/5) Plotting performance and parameters\n')
    plot_performance_and_parameters
end
end
