% Illustration of a RNN translated from Python to Matlab
% Source: https://pythonalgos.com/build-a-recurrent-neural-network-from-scratch-in-python-3/

% This script is only for self learning purposes.

% Clear workspace and command window
clear; clc;

% Create data
x = (0:199)';
sin_wave = sin(x);

% Access constant properties from simple_rnn.m class
seq_len = simple_rnn.seq_len;
hidden_dim = simple_rnn.hidden_dim;
output_dim = simple_rnn.output_dim;

% training data - preallocate
num_records = size(sin_wave,1) - seq_len; % 150
X = nan(seq_len, num_records-seq_len);
Y = nan(seq_len,1);

for idx = 1:num_records-seq_len
    X(:,idx) = sin_wave(idx:idx+seq_len-1);
    Y(idx) = sin_wave(idx+seq_len);
end

% validation data
X_validation = nan(seq_len, seq_len);
Y_validation = nan(seq_len,1);
range = (1:seq_len)';
for idx=num_records-seq_len+1:num_records
    
    X_validation(:,idx-100) = sin_wave(idx:idx+seq_len-1);
    Y_validation(idx-100) = sin_wave(idx+seq_len-1);
    
end

% Transpose X and X_validation
X = X';
X_validation = X_validation';

rng(2);
U = rand(hidden_dim, seq_len); % weights from input to hidden layer
V = rand(output_dim, hidden_dim); % weights from hidden to output layer
W = rand(hidden_dim, hidden_dim); % recurrent weights for layer (RNN weigts)

% Train model

[U, V, W] = simple_rnn.train(U, V, W, X, Y, X_validation, Y_validation);
% The training loss and the validation loss are not to a 100%
% exact, if I compare the solution to the implementation in Python. Maybe
% there is still a bug in my implementation.

% prediction on the training set
predictions = nan(size(Y));

for ii=1:size(Y,1)
   
    x = X(ii,:)'; y = Y(ii);
    prev_activation = zeros(hidden_dim,1);
    
    % forward pass
    for timestep=1:seq_len
       
        mulu = U*x;
        mulw = W*prev_activation;
        sumTemp = mulu + mulw;
        activation = simple_rnn.sigmoid(sumTemp);
        mulv = V*activation;
        prev_activation = activation;
        
    end
    
    predictions(ii) =  mulv;
    
end

figure;
plot(predictions, 'g')
hold on
plot(Y(:),'r')
legend('Predictions', 'Actual')


