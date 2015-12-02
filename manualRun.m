clear;
close all;
clc;

%[X, X_test, X_valid, Y, Y_test, Y_valid] = loadData('../data/train_x.csv.original', '../data/train_y.csv');

[X, X_test, X_valid, Y, Y_test, Y_valid] = loadData('../data/philip.csv', '../data/philipY.csv');

settings 					= generateSettings(X, Y, X_test, Y_test, X_valid, Y_valid)
	
Thetas = train(X, Y, settings);

pred = predict(Thetas, X, settings);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == Y)) * 100);

pred = predict(Thetas, X_test, settings);

fprintf('\nTest Set Accuracy: %f\n', mean(double(pred == Y_test)) * 100);

pred = predict(Thetas, X_valid, settings);

fprintf('\nCV Set Accuracy: %f\n', mean(double(pred == Y_valid)) * 100);