% Illustration of a RNN translated from Python to Matlab
% Source: https://pythonalgos.com/build-a-recurrent-neural-network-from-scratch-in-python-3/

classdef simple_rnn
    
    properties(Constant)
        % create RNN architecture
        learning_rate = 0.0001;
        seq_len = 50;
        max_epochs = 25;
        hidden_dim = 100;
        output_dim = 1;
        bptt_truncate = 5; % backprop through time --> lasts 5 iterations
        min_clip_val = -10;
        max_clip_val = 10;
    end
    
    methods (Static)
        
        function y = sigmoid(x)
            
            y = 1./(1+exp(-x));
            
        end
        
        function [loss, activation] = calculate_loss(X, Y, U, V, W)
            
            % X = Data matrix
            % Y = Result matrix
            % U = Weight matrix, representing the weights from the input layer to the
            % hidden layer
            % V = Weight matrix from the hidden layer to the output layer
            % W = Weight matrix from the hidden layer to itself
            
            loss = 0;
            
            for ii=1:size(Y,1)
                
                x = X(ii,:)'; y = Y(ii);
                prev_activation = zeros(simple_rnn.hidden_dim,1);
                for timestep=1:simple_rnn.seq_len
                    
                    new_input = zeros(size(x)); % forward pass, done for reach step in the sequence
                    new_input(timestep) = x(timestep); % define a single input for that timestep
                    mulu = U*new_input;
                    mulw = W*prev_activation;
                    sumTemp = mulu + mulw;
                    activation = simple_rnn.sigmoid(sumTemp);
                    mulv = V*activation;
                    prev_activation = activation;
                    
                end
                % Calculate and add loss per record
                loss_per_record = ((y-mulv).^2).*0.5;
                loss  = loss+loss_per_record;
                % Calculate loss after first Y pass
            end
        end
        
        function [layers, mulu, mulw, mulv] = calc_layers(x, U, V,W, prev_activation)
            
            % takes x values and the weights matrices
            % returns layer struct, final layers (mulu, mulw, mulv)
            
            % Preallocate layers with nan
            layers = struct;
            layers.activation = nan(size(U));
            layers.prev_activation = nan(size(U));
            
            for timestep=1:simple_rnn.seq_len
                
                new_input = zeros(size(x));
                new_input(timestep) = x(timestep);
                mulu = U*new_input;
                mulw = W*prev_activation;
                sumTemp = mulw + mulu;
                activation = simple_rnn.sigmoid(sumTemp);
                mulv = V*activation;
                
                % Add to struct
                layers.activation(:,timestep) = activation;
                layers.prev_activation(:,timestep) = prev_activation;
                
                prev_activation = activation;
                
            end
            
        end
        
        function [dU, dV, dW] = backprop(x, U, V, W, dmulv, mulu, mulw, layers)
            
            dU = zeros(size(U));
            dV = zeros(size(V));
            dW = zeros(size(W));
            
            dU_t = zeros(size(U));
            dW_t = zeros(size(W));
            
            sumTemp = mulu + mulw;
            dsv = V'.*dmulv;
            
            function out = get_previous_activation_differential(sumTemp, ds, W)
                
                d_sum = sumTemp .* (1-sumTemp) .* ds;
                dmulw = d_sum .* ones(size(ds));
                out = W'*dmulw;
                
            end
            
            for timestep=1:simple_rnn.seq_len
                
                dV_t = dmulv.*transpose(layers.activation(:,timestep));
                ds = dsv;
                dprev_activation = get_previous_activation_differential(sumTemp, ds, W);
                
                if timestep ~= 1
                    
                    % Convert Python to Matlab indexing
                    convert2MatlabCounter = 0:simple_rnn.seq_len-1 ;
                    tempTimeStep = convert2MatlabCounter(timestep);
                    
                    nRepeatStart = tempTimeStep-1;
                    nRepeatEnd = max(-1, tempTimeStep-simple_rnn.bptt_truncate-1);
                    nRange = size((nRepeatStart:nRepeatEnd:-1),2);
                    
                    % Repeat loop nRange times
                    for temp = 1:nRange-1
                        
                        ds = dsv + dprev_activation;
                        dprev_activation = get_previous_activation_differential(sumTemp, ds, W);
                        dW_i = W*layers.prev_activation(:,timestep);
                        
                        new_input = zeros(size(x));
                        new_input(timestep) = x(timestep);
                        dU_i = U*new_input;
                        
                        dU_t = dU_t + repmat(dU_i,1,size(dU_t,2));
                        dW_t = dW_t + repmat(dW_i,1,size(dW_t,2));
                        
                    end
                end
                
                dU = dU + dU_t;
                dV = dV + dV_t;
                dW = dW + dW_t;
                
                % take care of possible exploiding gradients
                if max(dU(:)) > simple_rnn.max_clip_val
                    dU(dU>simple_rnn.max_clip_val) = simple_rnn.max_clip_val;
                end
                if max(dV(:)) > simple_rnn.max_clip_val
                    dV(dV > simple_rnn.max_clip_val) = simple_rnn.max_clip_val;
                end
                if max(dW(:)) > simple_rnn.max_clip_val
                    dW(dW > simple_rnn.max_clip_val) = simple_rnn.max_clip_val;
                end
                
                if min(dU(:)) < simple_rnn.min_clip_val
                    dU(dU < simple_rnn.min_clip_val) = simple_rnn.min_clip_val;
                end
                if min(dV(:)) < simple_rnn.min_clip_val
                    dV(dV < simple_rnn.min_clip_val) = simple_rnn.min_clip_val;
                end
                if min(dW(:)) < simple_rnn.min_clip_val
                    dW(dW < simple_rnn.min_clip_val) = simple_rnn.min_clip_val;
                end
                
            end
        end
        
        % training
        function [U, V, W] = train(U,V,W,X,Y,X_validation, Y_validation)
            
            for epoch=1:simple_rnn.max_epochs
                
                % calculate initial loss, ie what the output is given a random set of weigths
                [loss, ~] = simple_rnn.calculate_loss(X, Y, U, V, W);
                
                % eheck validation loss
                [val_loss, ~] = simple_rnn.calculate_loss(X_validation, Y_validation, U, V, W);
                
                fprintf('Epoch: %d, Loss: %.8f, Validation Loss: %.8f \n ', epoch, loss, val_loss);
                
                % train model/forward pass
                for ii=1:size(Y,1)
                    x = X(ii,:)';
                    y = Y(ii);
                    prev_activation = zeros(simple_rnn.hidden_dim, 1);
                    [layers, mulu, mulw, mulv] = simple_rnn.calc_layers(x, U, V, W, prev_activation);
                    
                    % difference to the prediction
                    dmulv = mulv - y;
                    [dU, dV, dW] = simple_rnn.backprop(x, U, V, W, dmulv, mulu, mulw, layers);
                    
                    % update weights
                    U = U - simple_rnn.learning_rate .* dU;
                    V = V - simple_rnn.learning_rate .* dV;
                    W = W - simple_rnn.learning_rate .* dW;
                    
                end
            end
        end
    end
end

