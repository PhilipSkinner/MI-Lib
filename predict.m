function p = predict(Thetas, X, settings)
	m = size(X, 1);	 
	p = zeros(size(X, 1), 1);

	h = cell(size(Thetas, 1), 1);
	
	for i = 1:size(Thetas, 1)
		if i == 1
			h(i, 1) = sigmoid([ones(m, 1) X] * cell2mat(Thetas(i, 1))', settings.modifier);
		else
			h(i, 1) = sigmoid([ones(m, 1) cell2mat(h(i - 1, 1))] * cell2mat(Thetas(i, 1))', settings.modifier);
		end	
	end	
	
	hPredict = cell2mat(h(size(Thetas, 1), 1));
	
	[dummy, p] = max(hPredict, [], 2);
end
