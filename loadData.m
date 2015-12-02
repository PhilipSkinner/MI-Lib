function [X, X_test, X_valid, Y, Y_test, Y_valid] = loadData(X_file, Y_file)
	X_size = 0.7;
	X_test_size = 0.85;	
	
	X_all = load(X_file);		
	Y_all = load(Y_file);
	
	nRows = size(X_all, 1);
	randomIndex = randi(nRows, nRows, 1);
	
	X_all = X_all(randomIndex, :);
	Y_all = Y_all(randomIndex, :);
	
	nSample = ceil(nRows * X_size);
	nCVSample = ceil(nRows * X_test_size);
	
	X = X_all(1:nSample, :);
	Y = Y_all(1:nSample, :);
	
	X_test = X_all(nSample:nCVSample, :);
	Y_test = Y_all(nSample:nCVSample, :);
	
	X_valid = X_all(nCVSample:end, :);
	Y_valid = Y_all(nCVSample:end, :);			
end