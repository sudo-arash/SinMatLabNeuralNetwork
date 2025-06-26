function [y1] = sin_pi_neural_network(x1)
% ===== NEURAL NETWORK CONSTANTS =====

% Input 1
x1_step1.xoffset = 0;
x1_step1.gain = 0.0636619772367581;
x1_step1.ymin = -1;

% Layer 1
b1 = [-8.0467546158515119714;-4.8212245974541021454;2.1546142428463657126;-4.4490415864909076404;-0.27094771814405615995;-0.097956027296563372153;2.0189507094411793808;-1.8826734308625059366;-1.1873057664296693403;4.2945639524887253557];
IW1_1 = [8.2462153474372179573;6.0988816007534580876;-5.3851011436353166673;7.3890708024977040935;-2.4986648969022327016;-3.3187086714053188885;-6.9014243883604473595;-3.2413588238013519849;-2.3384258396344073105;4.3928309014909912023];

% Layer 2
b2 = 9.6502592036524443841;
LW2_1 = [2.1619529208258212449 -6.8053012457533386126 17.170975327822613821 1.5344173524915534212 -453.71621710304998487 234.35317840677993217 2.3937062766066579123 -269.10505970684255317 509.37640526403106378 36.828499096308881633];

% Output 1
y1_step1.ymin = -1;
y1_step1.gain = 1.00012588817196;
y1_step1.xoffset = -0.999874127673875;

% ===== SIMULATION ========

% Dimensions
Q = size(x1,2); % samples

% Input 1
xp1 = mapminmax_apply(x1,x1_step1);

% Layer 1
a1 = tansig_apply(repmat(b1,1,Q) + IW1_1*xp1);

% Layer 2
a2 = repmat(b2,1,Q) + LW2_1*a1;

% Output 1
y1 = mapminmax_reverse(a2,y1_step1);
end

% ===== MODULE FUNCTIONS ========

% Map Minimum and Maximum Input Processing Function
function y = mapminmax_apply(x,settings)
y = bsxfun(@minus,x,settings.xoffset);
y = bsxfun(@times,y,settings.gain);
y = bsxfun(@plus,y,settings.ymin);
end

% Sigmoid Symmetric Transfer Function
function a = tansig_apply(n,~)
a = 2 ./ (1 + exp(-2*n)) - 1;
end

% Map Minimum and Maximum Output Reverse-Processing Function
function x = mapminmax_reverse(y,settings)
x = bsxfun(@minus,y,settings.ymin);
x = bsxfun(@rdivide,x,settings.gain);
x = bsxfun(@plus,x,settings.xoffset);
end
