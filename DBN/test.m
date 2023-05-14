load mnist_uint8;


train_x = double(train_x) / 255;
train_y = double(train_y);
test_x = gpuArray(double(test_x)  / 255);
test_y = gpuArray(double(test_y));

% 保证算力，取舍数据集
% train_x=train_x(1:20000,:);
% train_y=train_y(1:20000,:);
% test_x=test_x(1:1000,:);
% test_y=test_y(1:1000,:);

% part = zeros(10, 2);
% for i = 1:10
%    [~, col, ~] = find(train_y);
%    col = find(col == i);
%    part(i, :) = [col(1) col(end)];
% end

rng('default');
% model parameters
% sizes = [500 500 2000]; 
sizes = [60 200 200];  % 要求
opts.numepochs =   10;
opts.batchsize =   10;
opts.momentum  =   0;
opts.alpha     =   0.1;
opts.decay     =   0.00001;
opts.k         =   1;
% opts.part      =   part;

dbn = DBN(train_x, train_y, sizes, opts);
tic
train(dbn, train_x, train_y);
toc

%% compute most probable class label given test data
probs = dbn.predict(test_x, test_y);
[~, I] = max(probs, [], 2);
pred = bsxfun(@eq, I, 1:10);
mis = find(~all(pred == test_y,2));

err = length(mis) / size(test_y, 1);

fprintf('Classification error is %3.2f%%\n',err*100);

%% plot MNIST examples
figure('Color','black');
[row,col,~] = find(train_y);

for i = 1:4
    for j = 1:10
        ix = (i-1)*10 + j;
        idx = row(col == j);
        subplot(4,10,ix), imshow(reshape(train_x(idx(ix),:), 28, 28)');
    end
end

%% plot some misclassified test cases
figure('Color','black');
idx = mis(1:10:100);
for i = 1:10
    subplot(2,5,i), imshow(reshape(test_x(idx(i),:), 28, 28)');
    [~, predicted] = max(probs(idx(i),:));
    [~, actual] = max(test_y(idx(i),:));
    t = title(sprintf('Predicted %d\nActual %d', predicted - 1, actual - 1), 'Color', 'white');
    set(t, 'horizontalAlignment', 'left');
    set(t, 'units', 'normalized');
    h1 = get(t, 'position');
    set(t, 'position', [0 1 0]);
end

%% plot samples as iterations of gibbs sampling increases
figure('Color','black');
gibbSteps = [1, 10, 100, 1000];
for i = 1:10
    for j = 1:length(gibbSteps)
        subplot(length(gibbSteps),10,(j-1)*10+i), imshow(reshape(dbn.generate2(i, 10, gibbSteps(j)), 28, 28)');
    end
end

%% visualize the weights of the first layer
% figure('Color','black');
% for i = 1:100
%     subplot(10,10,i), imshow(reshape(dbn.rbm(1).W(i,:), 28, 28)', [-1, 1]);
% end

%% save a sequence of samples generated by each step of gibbs sampling
for i = 1:10
    imageseq(dbn, i, 10, 200);
end