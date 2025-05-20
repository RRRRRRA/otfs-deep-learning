function [y,isfftout] = helperOTFSmod(x,padlen,varargin)
%HELPEROTFSMOD Modulate the delay-Doppler domain input signal
%
%   [Y,ISFFT] = HELPEROTFSMOD(X,PADLEN,PADTYPE) performs OTFS modulation
%   on X using a rectangular pulse shaping window and outputs the results
%   in Y. Specify X as an M-by-N array of real or complex values. M is
%   the number of subcarriers and N is the number of OTFS subsymbols.
%   X - OTFS grid (M x N)
%   PADLEN - CP or ZP length (in samples)
%   PADTYPE (optional)
%     'CP'   = cyclic prefix (default)
%     'RCP'  = reduced cyclic prefix
%     'ZP'   = zero padding
%     'RZP'  = reduced zero padding
%     'NONE' = no CP/ZP
%   Y - vector of time domain output with selected cyclic prefix of size
%   M*N+PADLEN if PADTYPE is RCP or RZP, or of size (M+PADLEN)*N if
%   PADTYPE is CP or ZP, else size M*N if PADTYPE is NONE
%   ISFFTOUT - TF domain output of size M-by-N produced during the
%   two-step OTFS modulation.
%
%   See also helperOTFSdemod and OTFSModulationExample.mlx.

%   Copyright 2023-2024 The MathWorks, Inc.

M = size(x,1);
if isempty(varargin)
    padtype = 'CP';
else
    padtype = varargin{1};
end

% Inverse Zak transform
y = ifft(x.').' / M;

% ISFFT to produce the TF grid output
isfftout = fft(y);

% Add cyclic prefix/zero padding according to padtype
switch padtype
    case 'CP'
        % % CP before each OTFS column (like OFDM) then serialize
        y = [y(end-padlen+1:end,:); y];  % cyclic prefix
        y = y(:);                        % serialize
    case 'ZP'
        % Zeros after each OTFS column then serialize
        N = size(x,2);
        y = [y; zeros(padlen,N)];    % zero padding
        y = y(:);                    % serialize
    case 'RZP'
        % Serialize then append OTFS symbol with zeros
        y = y(:);                    % serialize
        y = [y; zeros(padlen,1)];    % zero padding
    case 'RCP'
        % Reduced CP
        % Serialize then prepend cyclic prefix
        y = y(:);                        % serialize
        y = [y(end-padlen+1:end); y];    % cyclic prefix
    case 'NONE'
        y = y(:);                   % no CP/ZP
    otherwise
        error('Invalid pad type');
end
end