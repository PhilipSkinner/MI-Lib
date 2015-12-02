function Thetas = uncompress(nn_params, settings)
	startPosition = 1;			
	endPosition = settings.hidden_layer_size * (settings.input_layer_size + 1);
	totalX = settings.hidden_layer_size;
	totalY = settings.input_layer_size + 1;

	Thetas = cell(settings.num_layers, 1);
		
	for i = 1:settings.num_layers
		if i == settings.num_layers
			totalX = settings.num_labels;
			endPosition = startPosition + ((settings.num_labels * (settings.hidden_layer_size + 1)) - 1);
			
			if i == 1
				endPosition = startPosition + ((settings.num_labels * (settings.input_layer_size + 1)) - 1);
			end
		end						
		
		Thetas(i, 1) = reshape(nn_params(startPosition:endPosition), totalX, totalY);
		
		% now increment our values
		startPosition = endPosition + 1;
		endPosition = startPosition + (settings.hidden_layer_size * (settings.hidden_layer_size + 1)) - 1;
		totalY = settings.hidden_layer_size + 1;
	end				 
end