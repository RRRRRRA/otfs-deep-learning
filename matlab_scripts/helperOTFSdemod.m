function [y,tfout] = helperOTFSdemod(x,M,padlen,offset,varargin)
%HELPEROTFSDEMOD OTFS demodulate the time-domain input received signal
%
%   [Y,TFOUT] = HELPEROTFSDEMOD(X,M,PADLEN,SYMOFFSET,PADTYPE) performs
%   OTFS demodulation on X using a rectangular pulse shaping window and
%   outputs the results in Y. Specify X as a vector of real or complex
%   values. M is the number of subcarriers. The number of OTFS
%   subsymbols is inferred from the input size, PADTYPE, and PADLEN.
%   X - time-domain input including CP of size M*N+PADLEN if PADTYPE is
%   RCP or RZP, or of size (M+PADLEN)*N if PADTYPE is CP or ZP
%   M - number of subcarriers
%   PADLEN - CP or ZP length (in samples)
%   OFFSET - sample offset from beginning/end of OTFS symbol/subsymbol
%   PADTYPE (optional)
%     'CP'   = cyclic prefix (default)
%     'RCP'  = reduced cyclic prefix
%     'ZP'   = zero padding
%     'RZP'  = reduced zero padding
%     'NONE' = no CP/ZP
%   Y - DD domain output of size M-by-N
%   TFOUT - TF domain output produced during the two-step OTFS
%   demodulation of size M-by-N
%
%   See also helperOTFSmod and OTFSModulationExample.mlx.

%   Copyright 2023-2024 The MathWorks, Inc.

    if isempty(varargin)
        padtype = 'CP';
    else
        padtype = varargin{1};
    end

    % Remove CP and form delay-Doppler grid
    if strcmp(padtype,'CP') || strcmp(padtype,'ZP')
        % Full CP (offset = padlen) or Zero Padding (offset = 0)
        N = size(x,1)/(M+padlen);
        assert((N - round(N)) < sqrt(eps)); % check that M*N is an integer
        
        rx = reshape(x,M+padlen,N);
        Y = rx(1+offset:M+offset,:); % remove CPs
    elseif strcmp(padtype,'NONE')
        N = size(x,1)/M;
        assert((N - round(N)) < sqrt(eps)); % check that M*N is an integer
        
        Y = reshape(x,M,N);
    elseif strcmp(padtype,'RCP') || strcmp(padtype,'RZP')
        % Reduced CP (offset = padlen) or Reduced ZP (offset = 0)
        N = (size(x,1)-padlen)/M;
        assert((N - round(N)) < sqrt(eps)); % check that M*N is an integer

        rx = x(1+offset:M*N+offset); % remove CP
        Y = reshape(rx,M,N);
    else
        error('Invalid pad type');
    end

    % This code segment shows the SFFT/OFDM demod representation
    % Wigner transform with rectangular window (OFDM demodulator)
    tfout = fft(Y);

    % This code segment shows the simpler Zak transform representation
    y = fft(Y.').' * M;
end