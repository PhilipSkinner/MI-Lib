clear;
close all;
clc;

[X, X_test, X_valid, Y, Y_test, Y_valid] = loadData('../data/out.csv', '../data/out.y.csv');

settings.lambda 			= 0.003;
settings.epsilon 			= 1.2;
settings.iterations 		= 250;
settings.hidden_layer_size 	= 9;
settings.num_layers 		= 6;
settings.modifier			= 0;
settings.random				= 2;
		
% standard settings
settings.input_layer_size	= size(X, 2);
settings.num_labels			= max(Y);

% retry settings
settings.retries			= 300;

% train the net and get the error
Thetas = train(X, Y, settings);

disp(sprintf('Prediction: %.4f%%', mean(double(Y == predict(Thetas, X, settings))) * 100));
disp(sprintf('Test prediction: %.4f%%', mean(double(Y_test == predict(Thetas, X_test, settings))) * 100));
disp(sprintf('Validation prediction: %.4f%%', mean(double(Y_valid == predict(Thetas, X_valid, settings))) * 100));