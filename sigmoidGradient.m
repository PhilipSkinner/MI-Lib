function g = sigmoidGradient(z, modifier)	
	Gz = sigmoid(z, modifier);
	g = Gz .* (1 - Gz);
end
