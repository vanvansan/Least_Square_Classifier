% ------------------------------------------------------------------------ 
%
%  ECE174: Mini Project # 1
%  Writen by Zhengyu Huang
%  Fall, 2023
%
% ------------------------------------------------------------------------ 


load("mnist.mat");
% fun = @(beta_i) ;
y = is_zero(trainY);

% disp(trainY(1, 2) == 0);
disp("complete");

function [result] = is_zero(y)
    result = zeros([1 length(y)]);
    for i = 1:length(y)
        result(1 , i) = y(1,i)== 0;
    end
end

