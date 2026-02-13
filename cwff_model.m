% This CWFF (Causal Wiener Feedback Filter) section has been modularized
% into a seperate function (cwff_model.m) in order to clearly isolate the
% core hearing restoration algorithm designed independently(extra credit) 
% beyond what the project required.

function cohc_recn = cwff_model(cohc_deaf, cohc_normal)

% i) We start with defining the Causal Wiener filter:-

e = 1e-10;
X = cohc_deaf;
Y = cohc_normal;

S_xx = X.*conj(X);
S_yy = Y.*conj(Y); %Auto Power Spectral Density

S_xy = X.*conj(Y); 
S_yx = Y.*conj(X); %Cross Power Spectral Density

log_1 = 0.5*log(abs(S_yy + e));
phi_1 = imag(hilbert(abs(log_1 + e)));
S_yy_c = exp(log_1 + 1i*phi_1);
S_yy_nc = exp(log_1 - 1i*phi_1);

log_2 = 0.5*log(abs(S_xy./(S_yy_nc + e)));
phi_2 = imag(hilbert(abs(log_2 + e)));
q_c = exp(log_2 + 1i*phi_2); 

% The above steps are a result of the Bode Criterion (quantifies the 
% relationship between magnitude and phase of a causal system) combined 
% with the Wiener Hopf Equation. A detailed derivation can be found in the 
% write up accompanying the code. 

% Programmer's log: 10th Feb 2026
% There is a very interesting story behind this visualization: I was
% thinking about different ways to go about this for a few days and on a
% random night, I dreamt about the Bode Criterion proof we had done in 
% digital signal processing. It was a really cool proof from start to 
% finish and had had a huge effect on me when it was first derived, so 
% naturally, when I woke up, I asked myself whether I could translate it 
% for this problem. 

% However, when I saw the Hilbert transforms, my initial enthusiasm was 
% quashed. I assumed that it's probably never realizable in hardware and 
% moved on, without even checking with an LLM. To my surprise, we learnt 
% about Weaver's circuit in the communication-I class on the same day! 
% It definitely felt like the solution had come full circle.

% Next, we compute the transfer function.

H_wiener = q_c./S_yy_c;

%This completes the open loop part. 


%ii) Feedback function(closed loop system)

% This part focusses on choosing the right G such that the closed loop gain
% H_cl = H/(1+GH) is stable. By defining the cost function as:-
% J(G) = E(|Y-GX|^2) and minimizing it, we get G = S_yx/S_xx (Detailed
% proof can be found in the write up)

G = S_yx./(S_xx + e);

H_cl = H_wiener./(1+abs(G.*H_wiener));

h_t = real(fftshift(ifft(abs(H_cl)))); 

cohc_recn = normalize(abs((h_t)'.*cohc_deaf), 'range'); 
%Normalizing to give a value between 0 and 1 for cohc (model defined)