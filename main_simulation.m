clc;
clear;
close all;

figure();

% The current and complete implementation of this simulation requires 
% 10 minutes and 30 minutes to run on a standard laptop from start to 
% finish respectively. This is a direct result of using a high number of 
% repetitions of the stimulus in the auditory nerve fiber model combined 
% with a relatively long audio signal. To obtain a sample of the results 
% in a shorter run, the number of repetitions may be reduced in 'line 181'.

% Programmer's log: This file generates 8 figures in its current form and
% 27 figures in its complete form from start to finish. It spans 490 lines.
% A guide to run the code blocks efficiently can be found on 'line 400'.

function [spike_train, mean_firing_rate] = model_func(pin, BF, ...
    Fs, T, rt, nrep, psthbinwidth, cohc, cihc, fiberType, implnt, count)

    % Defined a single function to run the catmodel as and when required

    mean_firing_rate = []; %Initialization
    mxpts = length(pin);
    irpts = round(rt*Fs);
    rf_up = (0:irpts-1)/(irpts-1); %onset ramp
    rf_down = (irpts:-1:0)/(irpts-1); %offset ramp

    pin(1:irpts) = pin(1:irpts).*rf_up;
    pin((mxpts - irpts):mxpts) = pin((mxpts-irpts):mxpts).*rf_down;

    %CAT model implementation

    vihc = catmodel_IHC(pin, BF, nrep, 1e-5, T*2, cohc, cihc);
    [~, psth] = catmodel_Synapse(vihc, BF, nrep, 1e-5, fiberType, implnt);

    spike_train = psth;
    if count == 1
        return; %In order to efficiently return the required value
    end

    psthbins = round(psthbinwidth*Fs);
    nbins = round(length(psth)/psthbins);
    psth = psth(1:nbins*psthbins);
    pr = reshape(psth, psthbins, nbins);
    Psth = pr/psthbinwidth; %True Peri-Stimulus Time Histogram
    mean_firing_rate = mean(Psth(:));

end

% Q.1: Tuning Curves

cohc = 1;
cihc = 1;
fiberType = 3;
implnt = 0; %Model Parameters

BF = [4e2, 5e3];
F0 = 100*2.^(0:1/8:7);
Fs = 100e3;
T = 100e-3;
rt = 10e-3;
stimdb = -10:10:80; 

t = 0:1/Fs:T-1/Fs;
nrep = 10;
psthbinwidth = 0.5e-3; %Stimulus parameters

f_tun = zeros(1,length(F0));
avg_spike_rate = zeros(1, length(F0));
ri = zeros(2, length(stimdb)); %Initialization

for i = 1:length(BF)

    [~, bf_idx] = min(abs(F0 - BF(i))); %Index of frequency closest to BF

    figure(i);
    sgtitle(sprintf('Tuning Curves at BF = %d Hz', BF(i)));

    for j = 1:length(stimdb)
        for k = 1:length(F0)
            f_tun(k) = log10(F0(k));
            pin = sqrt(2)*20e-6*10^(stimdb(j)/20)*sin((2*pi*F0(k))*(t));

            %I chose sin as the arbitrary input stimulus due to it having a
            %single frequency in every time signal.

            [~,mean_firing_rate(k)] = model_func(pin, BF(i), Fs, T, rt, ...
                nrep, psthbinwidth, cohc, cihc, fiberType, implnt, 0);

        end

        ri(i,j) = mean_firing_rate(bf_idx);

        subplot(2,length(stimdb)/2,j);
        plot(f_tun, mean_firing_rate);
        hold on;
        plot(f_tun, mean_firing_rate, '.');
        xlabel('log(f)');
        ylabel('Mean Firing Rate');
        title(sprintf('Intensity = %d dB SPL', stimdb(j))); %Tuning curves

    end

    hold off;
    figure(3);
    sgtitle('Rate v/s Intensity Function');
    hold on;
    plot(stimdb, ri(i,:), 'DisplayName', sprintf('BF = %d Hz', BF(i)));
    plot(stimdb, ri(i,:), '.');
    xlabel('Intensity(in dB SPL)');
    ylabel('Mean Firing Rate');
    legend show; %Rate v/s intensity graph
end

hold off;

clearvars pin BF stimdb F0 Fs T rt nrep psthbinwidth f_tun avg_spike_rate
clearvars ri cohc

%Q.2: Speech Analysis
%Programmer's log: 3rd Feb 2026: The fact that the ANF model is so
%precise can be slightly annoying at times. The problem I faced was that 
% the sampling rate of the audio signal was 97.656 kHz, but the ANF model 
% expects aperfect 100kHz. The number of times MATLAB crashed!

function Psth_fft = compare(cohc, BF_bank, counter)

cihc = 1;
fiberType = 3;
implnt = 0;
cohc_n = mean(cohc); %Model Parameters (Mean of cohc is used only for 
                     %simplicity in the rate v/s intensity plot)

BF = 550;
rt = 10e-3;
stimdb = -20:5:100;
nrep = 10;
psthbinwidth = 0.5e-3; %Stimulus Parameters

[audio, Fs] = audioread('fivewo.wav');
t_start = 1.08;
t_end = 1.18;
n1 = round(t_start*Fs);
n2 = round(t_end*Fs); %Stimulus function

audio_2 = audio';
T_audio_2 = length(audio_2)*1/Fs;
sssl_audio_2 = 20*log10(rms(audio_2)/20e-6);
func_2 = (audio(n1:n2))';  %New Stimulus function

T_func_2 = length(func_2)*1/Fs;
sssl_func_2 = 20*log10(rms(func_2/20e-6));
ri = zeros(1, length(stimdb)); %Initialization

figure(counter+1);

for j = 1:length(stimdb)
    pin = func_2*10^(((stimdb(j)-sssl_func_2)/20));
    [~,ri(j)] = model_func(pin, BF, Fs, T_func_2, rt, nrep, ...
        psthbinwidth, cohc_n, cihc, fiberType, implnt, 0);
end

plot(stimdb, ri, 'DisplayName', sprintf('BF = %d Hz', BF));
hold on;
plot(stimdb, ri, '.');
hold off;
xlabel('Intensity(in dB SPL)');
ylabel('Mean Firing Rate');
title('Rate v/s Intensity Function');
legend show;

SL_array = [-10, 30, 80]; %Chosen heuristically
spike_train = cell(length(SL_array), length(BF_bank));
cohc_l = zeros(1, length(BF_bank)); %Further initialization

for i = 1:length(SL_array)
    pin = audio_2*10^(((SL_array(i)-sssl_audio_2)/20));
    for j = 1:length(BF_bank)
        cohc_l(j) = cohc(j);
        [st, ~] = model_func(pin, BF_bank(j), Fs, T_audio_2, rt, 80, ...
            psthbinwidth, cohc(j), cihc, fiberType, implnt, 1);
        spike_train{i,j} = st; 
        % Reduce nrep above to see faster outputs.

        % Tried really hard to avoid using cell arrays, 
        % but 4 dimensional matrices are too hard to handle!
    end
end

t_w = 25.6e-3;
nfft = round(Fs*t_w);
w_ham = 0.54 - 0.46*cos(2*pi*(0:nfft-1)/(nfft)); %Designing the Hamming win

noverlap = round(Fs*t_w*0.5);
[s,f,t] = spectrogram(audio, w_ham, noverlap, nfft, Fs);

[~, l_idx] = min(abs(f - 3e2));
[~, h_idx] = min(abs(f - 3e3));
f_speech = f(l_idx:h_idx); %Speech signal parameters

figure(counter+2);
imagesc(t, log10(f_speech), 20*log10(abs(s))); %Spectrogram plot
axis xy;
xlabel('time');
ylabel('frequency');
title('Spectrogram of the vowel segment');
colorbar;

bin_width = 3.2e-3*2.^(0:5);
spike_length = size(spike_train{1,1}, 2);

figure(counter+3);
title('Average Rate v/s Time')

for i = 1:length(bin_width)
    bin_pts = round(bin_width(i)*Fs);
    step_pts = round(bin_pts*0.5);
    nbins = floor((spike_length-bin_pts)/step_pts) + 1;
    avg_rate = zeros(1, nbins);
    time = zeros(1, nbins);

    for j = 1:nbins
        total_spikes = 0;
        idx_start = (j-1)*step_pts + 1;
        idx_end = idx_start + bin_pts - 1;
        for k = 1:length(SL_array)
            for l = 1:length(BF_bank)
                total_spikes = (total_spikes + ...
                    sum(spike_train{k,l}(idx_start:idx_end)));
            end
        end
        avg_rate(i,j) = total_spikes/bin_width(i);
        time(i,j) = (idx_start + idx_end)/(2*bin_pts);
    end

    subplot(2,3,i);
    sgtitle('Average Firing Rate v/s Time');
    hold on;
    plot(time(i,:), avg_rate(i,:), '.');
    plot(time(i,:), avg_rate(i,:)); %Plotting avarage firing rate over time
    hold off;
    xlabel('time(in s)');
    ylabel('Average Firing Rate');
    title(sprintf('Window = %.1f ms', bin_width(i)*10e2));

end

% Uncomment to compare the new spectrogram with the old one
% [~, l_index] = min(abs(BF_bank - 3e2));
% [~, h_index] = min(abs(BF_bank - 3e3));
% f_speech = f(l_index:h_index);

% imagesc(time(:), f_speech, 20*log10(avg_rate(:)));
% colorbar;

clearvars w_ham t_w bin_pts step_pts nbins idx_start idx_end

%Q.3: Phase Locking

idx1 = 1:4:21;
idx2 = 3:2:13;
BF_bn = [BF_bank(idx1); BF_bank(idx2)]; %New BF bank

psthbins = round(psthbinwidth*Fs);
nbins = floor(length(st)/psthbins);
st = st(1:nbins*psthbins);
pr = reshape(st, psthbins, nbins);
Psth = pr/psthbinwidth; %Finding the histogram from the spike train

N = length(Psth);
t_w = 25.6e-3;
w_ham = 0.54 - 0.46*cos(2*pi*(0:N-1)/(N));

win_pts = min(round(t_w/psthbinwidth), size(Psth, 1));
step_pts = round(win_pts*0.5);
nw = floor((N-win_pts)/step_pts) + 1; %Building the overlaping window

f_Psth = (0:win_pts-1)*(1/(win_pts*psthbinwidth));
t_Psth = ((0:nw-1)*step_pts + win_pts/2)*psthbinwidth; %Axes definitions

Psth_fft = zeros(win_pts, nw, length(BF_bn(1,:)));
S_fa = zeros(win_pts, nw, length(BF_bn(1,:)));
f_d = zeros(nw, length(BF_bn(1,:)));
fiber_d = zeros(2, length(BF_bn(1,:))); %Initialization

for i = 1:2
    figure(i+counter+3);
    for j = 1:length(BF_bn(i,:))
        for w = 1:nw
            idx_start = (w-1)*step_pts + 1;
            idx_end = idx_start + win_pts - 1;
            range_idx = idx_start:idx_end;
            sub = Psth(range_idx);
            if all(sub == 0)
                f_d(w,j) = -1; % Took almost an hour before I realised the
                               % the first window doesn't have any spikes!
                continue;
            end
            w_hn = w_ham(range_idx); %New Hamming window
        
            Psth_fft(:, w, j) = abs(fftshift(fft(sub.*w_hn)));
            S_fa(:,w, j) = abs((Psth_fft(:,w,j)).^2);
    
            locs = find(S_fa(:, w, j)>0);
            peaks = S_fa(locs,w,j);
            [peaks_s, f_idx] = sort(peaks, 'descend'); % Sorted Peaks
    
            e = 1e-10; %Small epsilon to avoid log(0/0) errors
            delta = 1; 

            %'delta' is a bandwidth parameter, smaller
            %'delta' gives a sharper frequency resolution. 
            % In this question, delta = 1 so that the
            % neighbourhood around the peaks spans consecutive
            % windows, as asked in the question.
    
            first_peak = peaks_s(1);
            nb_1 = locs(f_idx(1)):locs(f_idx(1))+delta;
            % Neighbourhood around first_peak
            if any(nb_1 > win_pts)
                continue;
            end

            second_peak = peaks_s(2);
            nb_2 = locs(f_idx(2)):locs(f_idx(2))+delta; 
            % Neighbourhood around second_peak
            if any(nb_2 > win_pts)
                continue;
            end

            P = S_fa(nb_1, w, j) + e;
            P = P/sum(P);  
            Q = S_fa(nb_2, w, j) + e;
            Q = Q/sum(Q); % Normalizing to shift the perspective from a 
                          % continuous spectrum to a probability vector
    
            M = 0.5*(P+Q);
    
            KL_PM = sum(P.*log(P/M));
            KL_QM = sum(Q.*log(Q/M));

    
            JS_PQ = abs(0.5*KL_PM + 0.5*KL_QM);
            JS_PQ = JS_PQ(1);
    
            % Definitely inspired by the discussions in the lab and the
            % reading I had done over the winter! The reason I went a step
            % further and chose Jensen-Shannon distance over the standard 
            % KL distance is because I observed that:
            % D_KL(P||Q) != D_KL(Q||P)
            % whereas in the JS formulation, it remains commutable. 
    
            threshold = 0.25; % Empirical value obtained through testing.
            if (JS_PQ) > threshold
                f_d(w,j) = f_Psth(locs(f_idx(1))); % Dominant Frequency_win
            else
                f_d(w,j) = -1;
            end
        end

        fd = f_d(f_d(:,j)>0,j); % For edge case where -1 is the mode
        fiber_d(i,j) = mode(fd); % Dominant Frequency across the ANF

        hold on;
        %Plotting the spectrogram
        imagesc(t_Psth, f_Psth, 20*log10(Psth_fft(:,:,j)));
        axis xy;
        xlabel('time');
        ylabel('frequency');
        title(sprintf('Phase Locking for BF bank %d ', i));
        colorbar;
       
    end

    scatter(t_Psth(floor(12*nw/25)), fiber_d(i,:), 200, 'r', '*'); %Fun!
    hold off;
end

% Extra plot to compare phase locking [Uncomment]
f_phase = (-win_pts/2:win_pts/2-1)*(1/(win_pts*psthbinwidth));
for i = 1:2
    for j = 1:length(BF_bn(i,:))
        for w = 1:nw
            figure(counter+6);
            hold on;
            plot(f_phase, Psth_fft(:,w,j)); 
            % Uncomment to see the visualization of phase locking. 
            % Too beautiful to skip!
            hold off;
            xlabel('frequency(in Hz)');
            ylabel('fft(Psth)');
            title('Phase Locking in 2D');
        end
    end
end

end

% Guide to running the code blocks:

% In the lines below, you will find four distinct blocks of code (1, 2, 3a,
% 3b). 1,2 can be run independently. In order to run either 3a, 3b or both 
% simultaneously, please uncomment 1 AND 2 before executing.

%Q.4: Hearing Loss and Restoration

%cohc - OHC Scaling Factor
fig_start_1 = -3;

% 1.Normal ear: Uncomment this block to simulate the normal ear
fig_start_1 = -fig_start_1;
BF_bank_normal = 100*2.^(0:1/4:6);
cohc_normal = ones(1, length(BF_bank_normal));
Psth_fft_normal = compare(cohc_normal, BF_bank_normal, fig_start_1);

% 2.Deafened ear: Uncomment this block to simulate the deafened ear %
fig_start_2 = 6;
BF_bank_deaf = 100*2.^(0:1/4:6);
cohc_deaf = zeros(1,length(BF_bank_deaf));
for j = 1:length(BF_bank_deaf)
    BF_deaf = BF_bank_deaf(j);
    if BF_deaf > 400
        threshold_increase = 20*log2(BF_deaf/400);
        cohc_deaf(j) = 10^(-threshold_increase/20);
    else
        cohc_deaf(j) = 1;
    end
end
Psth_fft_deaf = compare(cohc_deaf,BF_bank_deaf,fig_start_1+fig_start_2); 

% Slightly convoluted logic for the figures but the general idea is have 
% a contiguous figure output throughout.


% Extra Credit(Run only after uncommenting both blocks above)

% 3a. Application of simple selective amplification
% One idea that popped up immediately is to just calculate the
% transformation matrix from cohc_normal to cohc_deaf and invert it.
% It became obvious pretty quickly that we don't have enough 
% equations to find the matrix in the first place. However, we can make a 
% reasonable assumption that the transformation matrix is a diagonal
% matrix. Given that, every A(i,i) = cohc_deaf(i) (since cihc(i) = 1)

fig_start_3 = 6;
A = zeros(length(BF_bank_deaf));

for i = 1:length(BF_bank_deaf)
    A(i,i) = cohc_deaf(i);
end

% Now we just find the inverse of the matrix. Pre-multiplying the same to
% cohc_deaf should give the reconstructed version

A_inv = inv(A);
d = diag(A_inv);
cohc_rec = d'.*cohc_deaf; %Reconstructed cohc

compare(cohc_rec, BF_bank_normal, fig_start_1+fig_start_2+fig_start_3);

%However, as mentioned in the question, I realized that the linear
%selective amplification approach doesn't work. It introduces some edge
% spikes making it harder to distinguish the true phase locked frequency.
% This could be because the catmodel is inherently non-linear in its code.
% Further, finding matrix inverses in hardware is time-consuming.

% So, my initial instinct was to treat this as a signal processing question 
% where we need to design a filter was wrong. I slowly realised that any 
% signal processing will fail if it is not driving the system towards
% stability. 

% Reading some more convinced me that a combination of a
% digital filter and a feedback circuit would be the best method to 
% approach the problem. This part of the project is the one that resonates
% the most with me, because it helped me combine concepts learnt through 
% coursework at different levels to find an optimal solution.

% The insight I had was to combine stochastic processes, DSP and control to 
% create a Causal Wiener Feedback Filter(CWFF) model that works towards
% restoring the phase locking ability of the deaf ear.

% 3b. Causal Wiener Feedback Filter (WFF) model

fig_start_4 = 6;

cohc_recn = cwff_model(cohc_deaf, cohc_normal);

compare(cohc_recn, BF_bank_normal, ...
    fig_start_1+fig_start_2+fig_start_3 + fig_start_4);

disp('Script Completed Successfully!');