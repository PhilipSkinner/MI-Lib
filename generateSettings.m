function settings = generateSettings(X, Y, X_test, Y_test, X_valid, Y_valid)
	settings.input_layer_size 	= size(X, 2);	
	settings.num_labels 		= max(Y);
	settings.retries			= 1;
	settings.modifier			= 0;
	
	lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';
	lambda_types = size(lambda_vec, 1);
	
	epsilon_vec = [0 0.0012 0.0036 0.012 0.036 0.12 0.36 1.2 3.6]';
	epsilon_types = size(epsilon_vec, 1);
	
	iteration_vec = [10 20 30 40 50 60 70 80 90 100]';
	iteration_types = size(iteration_vec, 1);
	
	layer_size_vec = [1 2 3 6 9]';
	layer_size_types = size(layer_size_vec, 1);
	
	num_layers_vec = [2 3 4 5 6 7]';
	num_layer_types = size(num_layers_vec, 1);
	
	settings.lambda 			= 10;
	settings.iterations 		= 10;
	settings.epsilon 			= 0.12;
	settings.hidden_layer_size 	= settings.input_layer_size;
	settings.num_layers 		= 2;
	
	Thetas = train(X, Y, settings);
	nn_params = [];
	for i = 1:size(Thetas, 1)
		nn_params = [nn_params(:) ; cell2mat(Thetas(i, 1))(:) ];
	end	
	
	error = (cost(nn_params, settings, X_valid, Y_valid) + \
			cost(nn_params, settings, X_test, Y_test)) * (1/2);	
	
	disp(sprintf('Using vars:\nlambda=%.4f\nepsilon=%.4f\niterations=%.0f\nError rating of %.4f\n', settings.lambda, settings.epsilon, settings.iterations, error));
	
	chosenLambda 		= settings.lambda;
	chosenIterations 	= settings.iterations;
	chosenEpsilon 		= settings.epsilon;
	chosenLayerSize		= settings.hidden_layer_size;
	chosenLayers		= settings.num_layers;
	
	for i = 1:lambda_types
		settings.lambda = lambda_vec(i);		
		for j = 1:epsilon_types
			settings.epsilon = epsilon_vec(j);					
			for k = 1:iteration_types										
				settings.iterations = iteration_vec(k);
				for l = 1:layer_size_types
					settings.hidden_layer_size = settings.input_layer_size * layer_size_vec(l);
					for g = 1:num_layer_types
						settings.num_layers = num_layers_vec(g);
								
						Thetas = train(X, Y, settings);	
								
						nn_params = [];
						for i = 1:size(Thetas, 1)
							nn_params = [nn_params(:) ; cell2mat(Thetas(i, 1))(:) ];
						end
						
						this_error = (cost(nn_params, settings, X_valid, Y_valid) + \
										cost(nn_params, settings, X_test, Y_test)) * (1/2);
						
						disp(sprintf('lambda: %.4f', 			settings.lambda));
						disp(sprintf('epsilon: %.4f', 			settings.epsilon));
						disp(sprintf('iterations: %.0f', 		settings.iterations));
						disp(sprintf('layer size: %.0f',		settings.hidden_layer_size));
						disp(sprintf('layer count: %.0f',		settings.num_layers));		
						disp(sprintf('Error rating of %.4f', 	this_error));
						disp(sprintf('Current Best: %.4f\n', 	error));
						
						if this_error < error
							chosenLambda 		= settings.lambda;
							chosenIterations 	= settings.iterations;
							chosenEpsilon 		= settings.epsilon;
							chosenLayerSize		= settings.hidden_layer_size;
							chosenLayers		= settings.num_layers;
							
							error = this_error;
							
							disp(sprintf('New lambda: %.4f', 		settings.lambda));
							disp(sprintf('New epsilon: %.4f', 		settings.epsilon));
							disp(sprintf('New iterations: %.0f', 	settings.iterations));
							disp(sprintf('New layer size: %.0f',	settings.hidden_layer_size));
							disp(sprintf('New layer count: %.0f',	settings.num_layers));					
						end
					end
				end								
			end
		end				
	end
	
	settings.lambda 			= chosenLambda;
	settings.iterations 		= chosenIterations;
	settings.epsilon 			= chosenEpsilon;
	settings.hidden_layer_size	= chosenLayerSize;
	settings.num_layers			= chosenLayers;
end