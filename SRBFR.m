function [ Accuracy ] = SRBFR( numTrainee, path )
%SRBFR Summary of this function goes here
% numTrainee is the number of totoal training samples
% path is the path to training samples
%Detailed explanation goes here
%t0 = cputime;
X_total = [];															
for m = [1:13,15:39]
	x_path = path;
	if (m < 10)
		x_path = strcat(path,'\','yaleB0',num2str(m),'\');
	else
		x_path = strcat(path,'\','yaleB',num2str(m),'\');
    end
	img_path_list = dir(strcat(x_path,'*.pgm'));	
    X = [];  										% X is the matrix containing all images.  
	for j = 1:numTrainee                              
        image_name = img_path_list(j).name;								
        A = imread(strcat(x_path,image_name));  					
        B = reshape(A, [], 1);
        X(:,j) = B;                                % Augument every matrix to X
    end
    X_total = [X_total X];  
end

[n_t, m_t] = size(X_total);
t0 = cputime;
[coeff, score, ~] = PCA(X_total);
t1 = cputime - t0;
t1
X_total_mean = X_total - ones(n_t,1) * mean(X_total);
solve_A = (X_total_mean' * coeff)';
% read test figures:
num = '07'; % change num to test different files
test_path = strcat(path, '\', 'yaleB',num,'\');
%-----------------------------------^^--------
%-------------change this two parameters to change test cases-------
img_path_list = dir(strcat(test_path,'*.pgm'));
y_pre = [];
rank_r = zeros(38,1);
for i = 30:60
    image_name = img_path_list(i).name;
    A = imread(strcat(test_path,image_name));
    B = reshape(A, [], 1);
    y_pre = [y_pre B]; 
    y_pre = double(y_pre);
    y = coeff' * y_pre;   
% imshow(mat2gray(reshape(coeff, 168, 192)));
% img_test = imread('F:\ShanghaiTech\Sophomore\Semester 1\Linear Algebra\Coding Project\CroppedYale\CroppedYale\yaleB38\yaleB38_P00A+070E+00.pgm');
% img_test_a = reshape(img_test, [], 1);
% size(img_test)
% size(img_test_a)
% img_test_a = double(img_test_a);
    solve_A = bsxfun(@times, solve_A, 1./sqrt(sum(solve_A.^2,1))) ;
    y = bsxfun(@times, y, 1./sqrt(sum(y.^2,1))) ;
    [x] = feature_sign(solve_A, y, 0.0007);
% size(after)
% Serr = coeff * x - img_test_a;
%imshow(mat2gray(reshape(after,192,[])));
%imshow(mat2gray(img_test));
    sum_r = zeros(38,1);

    for k = 1:20
        for j = (numTrainee * (k-1) + 1):(numTrainee * k)
            sum_r(k) = sum_r(k) + x(j,1);
        end
    end
    [~,I] = max(sum_r);
    rank_r(I) = rank_r(I) + 1;
    y_pre = [];
end
rank_r
num = str2num(num)
if num > 14
    num = num + 1;
end
Accuracy = rank_r(num) / 30;
%t1 = cputime - t0;
%t1
end