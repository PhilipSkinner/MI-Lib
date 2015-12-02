function [J grad] = cost(nn_params, settings, X, y) 		
	Thetas = uncompress(nn_params, settings);		
	m = size(X, 1);			
	
	J = 0;
	
	A = cell(size(Thetas, 1), 1);		
	Z = cell(size(Thetas, 1), 1);	
	grads = cell(size(Thetas, 1), 1);		
	Deltas = cell(size(Thetas, 1), 1);	
		
	I = eye(settings.num_labels);	
	Y = zeros(m, settings.num_labels);		
	
	for i = 1:m
		Y(i, :) = I(y(i), :);
	end	
		
	penaltyTot = 0;			
	
	for i = 1:size(Thetas, 1)
		
		if i == 1
			% set the As to be the input stuff			
			A(i, 1) = [ones(m, 1) X];										 
		else			
			A(i, 1) = [ones(size(cell2mat(Z(i - 1, 1)), 1), 1) sigmoid(cell2mat(Z(i - 1, 1)), settings.modifier)];			
		end
				
		Z(i, 1) = cell2mat(A(i, 1)) * cell2mat(Thetas(i, 1))';
				
		penaltyTot += sum(sum(cell2mat(Thetas(i, 1)))(:, 2:end).^2, 2);		
	end				
	
	H = sigmoid(cell2mat(Z(size(Thetas, 1), 1)), settings.modifier);		
	
	penalty = (settings.lambda / (2 * m)) * penaltyTot;
			
	J = (1/m)*sum(sum((-Y).*log(H) - (1-Y).*log(1-H), 2));	
	J += penalty;		
	
	Sigma = generateSigma(A, Y, Z, Thetas, H, settings);		
	grad = generateGradients(Sigma, Thetas, A, m, settings);					
end

