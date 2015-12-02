function Sigma = generateSigma(A, Y, Z, Thetas, H, settings)
	Sigma = cell(size(Thetas, 1), 1);		

	for fakeI = 1:size(Thetas, 1)
		i = size(Thetas, 1) - (fakeI - 1);				
		
		if i == size(Thetas, 1)
			Sigma(i, 1) = H - Y;								
		else									
			eelcosResult = sigmoidGradient([ones(size(cell2mat(Z(i, 1)), 1), 1) cell2mat(Z(i, 1))], settings.modifier);
						
			Sigma2 = cell2mat(Sigma(i + 1, 1));
			Theta2 = cell2mat(Thetas(i + 1, 1));
																																
			Sigma(i, 1) = (Sigma2 * Theta2 .* eelcosResult )(:, 2:end);						
		end				
	end		
end