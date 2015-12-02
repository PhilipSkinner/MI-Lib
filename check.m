clear;
close all;
clc;

filename = argv(){1};
fromFile = load(filename);
[X, X_test, X_valid, Y, Y_test, Y_valid] = loadData('../data/philip.csv', '../data/philipY.csv');

% find the first entry where the last item is -1
count = 0
resultCol = size(fromFile, 2);
for i = 1:size(fromFile, 1)	
	if fromFile(i, resultCol) == -1	
		% read the settings
		settings.lambda 			= fromFile(i, 1);
		settings.epsilon 			= fromFile(i, 2);
		settings.iterations 		= fromFile(i, 3);
		settings.hidden_layer_size 	= fromFile(i, 4);
		settings.num_layers 		= fromFile(i, 5);
		
		% standard settings
		settings.input_layer_size	= size(X, 2);
		settings.num_labels			= max(Y);
		
		% train the net and get the error
		Thetas = train(X, Y, settings);
		nn_params = [];
		for j = 1:size(Thetas, 1)
			nn_params = [nn_params(:) ; cell2mat(Thetas(j, 1))(:) ];
		end	
		
		error = (cost(nn_params, settings, X_valid, Y_valid) + \
				cost(nn_params, settings, X_test, Y_test)) * (1/2);
				
		fromFile(i, resultCol) = error;		
		
		count += 1
		
		if count == 50
			csvwrite(filename, fromFile);
			count = 0
		end
	end
end

csvwrite(filename, fromFile);