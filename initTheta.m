function W = initTheta(L_in, L_out, settings) 
	W = [];
		
	if settings.random == 1
		W = betarnd(L_out, 1+L_in, L_out, 1+L_in) * 2 * settings.epsilon - settings.epsilon;			
	elseif settings.random == 2
		W = cauchy_rnd(0, 0.2, L_out, 1+L_in) * 2 * settings.epsilon - settings.epsilon;	
	else
		W = rand(L_out, 1+L_in) * 2 * settings.epsilon - settings.epsilon;
	end	
end
