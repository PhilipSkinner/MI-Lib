function grad = generateGradients(Sigma, Thetas, A, m, settings)
	grad = [];	
	for i = 1:size(Thetas, 1)
		Theta = cell2mat(Thetas(i, 1));						
				
		Delta = cell2mat(Sigma(i, 1))' * cell2mat(A(i, 1));
																	
		ThetaGrad =	Delta ./ m + (settings.lambda/m) * [zeros(size(Theta, 1), 1) Theta(:, 2:end)];
				
		grad = [grad(:) ; ThetaGrad(:) ];
	end			
end