%% OTFS Modulation

%% Generate data sets for training DL-based OTFS Signal Detector
% Create a labeled data set for training and testing a DL-based signal
% detector for an OTFS modulated signal. The training data is the received
% signal at the input of the receiver (i.e., after transmision of the OTFS
% modulated signal through a noisy channel). The labels are the actual
% symbols (QPSK, in this case) transmitted over the channel. Two frames of
% data are passed over the channel. The first frame is a single pilot
% symbol and the second frame is a full OTFS data frame.

%% Simulation Setup
% Configure the simulation parameters. For demonstrating basic OTFS concepts, 
% let $M$ = 64 and $N$ = 30. Set the SNR to a high value to show the effects of 
% ISI and ICI in different modulations and channel conditions.

clear all; close all; clc;

rng(42);

M = 64;          % number of subcarriers
N = 30;          % number of subsymbols/frame
df = 15e3;       % make this the frequency bin spacing of LTE
fc = 5e9;        % carrier frequency in Hz
padLen = 10;     % make this larger than the channel delay spread channel in samples
padType = 'ZP';  % this example requires ZP for ISI mitigation
SNRdB = 40;      % Use a hign SNR to generate training data

fsamp = M*df;                % sampling frequency at the Nyquist rate
if strcmp(padType,'ZP') || strcmp(padType,'CP')
    Meff = M + padLen;       % number of samples per OTFS subsymbol
    numSamps = Meff * N;     % number of samples per OTFS symbol
    T = ((M+padLen)/(M*df)); % symbol time (seconds)
else
    Meff = M;                % number of samples per OTFS subsymbol
    numSamps = M*N + padLen; % number of samples per OTFS symbol
    T = 1/df;                % symbol time (seconds)
end

Ndata = 10000;       % number of labeled data samples to generate

%% Grid Population for Channel Sounding and Data
% For each sample, create an empty array of size $M$-by-$N$ where the $M$ rows
% correspond to the delay bins and the $N$ columns map to the Doppler bins. 
% To demonstrate how data in the delay-Doppler domain propagates through a  
% high-mobility channel, place a pilot signal at grid position (1,16) to sound
% the channel. Leave the other grid elements empty so that the scatterer echoes
% will appear in the received delay-Doppler grid. Then append another M-by-N
% grid of random QPSK symbols. Pass both frames through a randomly generated
% instance of the high-Doppler channel.

% Pilot generation and grid population. This is the same for every sample,
% so it can stay outside the data generation loop.
pilotSymbol = exp(1i*pi/4);
pilotBin = floor(N/2)+1;
Pdd = zeros(M,N);
Pdd(1,pilotBin) = pilotSymbol; % populate just one bin to see the effect through the channel

%% Data Generation Loop

% Create arrays to hold the data
XData = zeros(M,N,Ndata);       % 3-way array to hold data frames
Rx = zeros(Ndata,2*numSamps);   % Received data for pilot frame and data frame
Tdata = zeros(M,2*N,Ndata);     % Target data: pilot frame concatenated with data frame
xdata = cell(Ndata,1);
tdata = cell(Ndata,1);
chanParamsData = cell(Ndata,1);

tic;

for n = 1:Ndata
    % Data generation
    Xgrid = zeros(M,N);
    Xdata = randi([0,1],2*M,N);
    Xgrid(1:M,:) = pskmod(Xdata,4,pi/4,InputType="bit");
    XData(:,:,n) = Xgrid;
    
    % % % Randomly select the SNR for this run
    % % SNRdB = randi(40);

    %% OTFS Modulation
    % Modulate the DD data grid with the single pilot symbol using OTFS 
    % modulation, followed by OTFS modulation of the grid containing the data 
    % symbols. Use the |helperOTFSmod| function to apply the inverse Zak 
    % transform on the pilot grid |_Pdd_| and data grid.
    
    % OTFS modulation
    txPilotOut = helperOTFSmod(Pdd,padLen,padType);
    txDataOut = helperOTFSmod(Xgrid,padLen,padType);
    
    %% High-Mobility Channel
    % Create an AWGN high-mobility channel with stationary transmitter and mobile 
    % receiver, and moving scatterers of different delays and Doppler shifts:
    %% 
    % 1) Randomly select a small number of paths, P.
    % 2) Create a line-of-sight path representing the main propagation path
    %   from a base station to the receiver with zero delay and zero normalized Doppler. 
    %   The line-of-sight path has zero delay and zero Doppler since the receiver is 
    %   synchronized in time and frequency to the base station.
    % 3) Create a random channel with P - 1 scatterers; one path will be the
    % direct path. Use padLen and Dmax as maximum delay and Doppler shifts.
    Dmax = floor(N/2);
    
    Pmax = 5;
    P = randi(Pmax);                     % Randomly choose number of paths
    De = sort(randi(padLen,1,P-1));      % In ascending order of delay
    Do = randi([-Dmax,Dmax],1,P-1);
    Gains = sort(rand(1,P-1),'descend'); % In descending order of gain
    
    % Configure paths. Paths with longer delay have lower gain.
    chanParams.numPaths        = P;
    chanParams.pathDelays      = [0 De];    % number of samples that path is delayed
    chanParams.pathGains       = [1 Gains]; % complex path gain
    chanParams.pathDopplers    = [0 Do];    % Doppler index as a multiple of fsamp/MN
    
    % Calculate the actual Doppler frequencies from the Doppler indices
    chanParams.pathDopplerFreqs = chanParams.pathDopplers * 1/(N*T); % Hz
    
    % Send the OTFS modulated signals (pilot and data) through the channel
    dopplerPilotOut = dopplerChannel(txPilotOut,fsamp,chanParams);
    dopplerDataOut = dopplerChannel(txDataOut,fsamp,chanParams);
    
    % Add white Gaussian noise
    Es = mean(abs(pskmod(0:3,4,pi/4).^ 2));
    n0 = Es/(10^(SNRdB/10));
    chPilotOut = awgn(dopplerPilotOut,SNRdB,'measured');
    chDataOut = awgn(dopplerDataOut,SNRdB,'measured');

    rxWindow = [chPilotOut(1:numSamps);chDataOut(1:numSamps)]; % Get data for the pilot frame and the data frame
    Rx(n,:) = rxWindow;
    xdata{n} = rxWindow;   % The n-th cell of xdata contains the complex time series at the receiver

    temp = squeeze(XData(:,:,n));
    Tdata(:,:,n) = [Pdd,temp];
    tdata{n} = reshape(squeeze(Tdata(:,:,n)),1,[]);  % Cell array with Ndata cells of 1x2*N*M complex vectors

    chanParamsData{n} = chanParams;
end

% Save the data (Rx) and labels (XData) and as cell arrays
save("OTFSData.mat","Rx","Tdata","xdata","tdata","chanParamsData")

toc;


%% Support Functions

function y = dopplerChannel(x,fs,chanParams)
    % Form an output vector y comprising paths of x with different
    % delays, Dopplers, and complex gains
    numPaths = length(chanParams.pathDelays);
    maxPathDelay = max(chanParams.pathDelays);
    txOutSize = length(x);
    
    y = zeros(txOutSize+maxPathDelay,1);
    
    for k = 1:numPaths
        pathOut = zeros(txOutSize+maxPathDelay,1);

        % Doppler
        pathShift = frequencyOffset(x,fs,chanParams.pathDopplerFreqs(k));
    
        % Delay and gain
        pathOut(1+chanParams.pathDelays(k):chanParams.pathDelays(k)+txOutSize) = ...
            pathShift * chanParams.pathGains(k);
            
        y = y + pathOut;
    end
end

function G = getG(M,N,chanParams,padLen,padType)
    % Form time domain channel matrix from detected DD paths
    if strcmp(padType,'ZP') || strcmp(padType,'CP')
        Meff = M + padLen;  % account for subsymbol pad length in forming channel
        lmax = padLen;      % max delay
    else
        Meff = M;
        lmax = max(chanParams.pathDelays);  % max delay
    end
    MN = Meff*N;
    P = length(chanParams.pathDelays);  % number of paths
    
    % Form an array of channel responses for each path
    g = zeros(lmax+1,MN);
    for p = 1:P
        gp = chanParams.pathGains(p);
        lp = chanParams.pathDelays(p);
        vp = chanParams.pathDopplers(p); 

        % For each DD path, compute the channel response.
        % Each path is a complex sinusoid at the Doppler frequency (kp)
        % shifted by a delay (lp) and scaled by the path gain (gp)
        g(lp+1,:) = g(lp+1,:) + gp*exp(1i*2*pi/MN * vp*((0:MN-1)-lp));
    end    

    % Form the MN-by-MN channel matrix G
    G = zeros(MN,MN);
    % Each DD path is a diagonal in G offset by its path delay l
    for l = unique(chanParams.pathDelays).'
        G = G + diag(g(l+1,l+1:end),-l);
    end
end
%% 
% _Copyright 2023-2024 The MathWorks, Inc._
% 
%
