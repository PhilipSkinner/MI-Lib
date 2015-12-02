function g = sigmoid(z, modifier)	
	g = (1.0 ./ (1.0 + exp(-z))) + modifier;
end
