function Thetas = train(X, y, settings)	
	bestError = 99999999999999999999999;
	Thetas = [];
 		
	for attempt = 1:settings.retries
		fprintf('\nAttempt %.0f', attempt)
	
		[nn_params, cost] = _train(X, y, settings);
				
		cost = min(cost);
				
		fprintf('Cost of %.2f compared to %.2f\n', cost, bestError);
				
		if cost < bestError
			bestError = cost;
			Thetas = uncompress(nn_params, settings);
			fprintf('New thetas acquired!\n');
		end		
	end
end

function [nn_params, cost] = _train(X, y, settings)
	initial_nn_params = [];
		
	for i = 1:settings.num_layers
		layer_size_y = settings.hidden_layer_size;
		layer_size_x = settings.hidden_layer_size;
		if i == settings.num_layers
			layer_size_y = settings.num_labels;			
		end
		if i == 1
			layer_size_x = settings.input_layer_size;
		end		
				
		initial = initTheta(layer_size_x, layer_size_y, settings);		
		initial_nn_params = [initial_nn_params(:) ; initial(:) ];
	end	
		
	fprintf('\nTraining Neural Network... \n');
	options = optimset('MaxIter', settings.iterations);
	costFunction = @(p) cost(p, settings, X, y);

	[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);			
end