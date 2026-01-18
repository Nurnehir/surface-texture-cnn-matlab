dataPath = fullfile(pwd, 'dataset');

imds = imageDatastore(dataPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
numClasses = numel(categories(imds.Labels));

[imdsTrain, imdsVal] = splitEachLabel(imds, 0.8, 'randomized');

net = mobilenetv2;
inputSize = net.Layers(1).InputSize;

augmenter = imageDataAugmenter( ...
    'RandRotation', [-20 20], ...
    'RandXTranslation', [-10 10], ...
    'RandYTranslation', [-10 10], ...
    'RandXReflection', true, ...
    'RandScale', [0.9 1.1]);

augTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, 'DataAugmentation', augmenter);
augVal = augmentedImageDatastore(inputSize(1:2), imdsVal);

lgraph = layerGraph(net);
fcName = find_layer_name(lgraph, 'nnet.cnn.layer.FullyConnectedLayer');
softmaxName = find_layer_name(lgraph, 'nnet.cnn.layer.SoftmaxLayer');
className = find_layer_name(lgraph, 'nnet.cnn.layer.ClassificationLayer');

newFc = fullyConnectedLayer(numClasses, 'Name', 'new_fc', ...
    'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10);
newSoftmax = softmaxLayer('Name', 'new_softmax');
newClass = classificationLayer('Name', 'new_classoutput');

lgraph = replaceLayer(lgraph, fcName, newFc);
lgraph = replaceLayer(lgraph, softmaxName, newSoftmax);
lgraph = replaceLayer(lgraph, className, newClass);

options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-4, ...
    'MaxEpochs', 6, ...
    'MiniBatchSize', 32, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augVal, ...
    'ValidationFrequency', 20, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

trainedNet = trainNetwork(augTrain, lgraph, options);

if ~exist('results', 'dir')
    mkdir('results');
end

save(fullfile('results', 'trainedNet.mat'), 'trainedNet');

YPred = classify(trainedNet, augVal);
YTrue = imdsVal.Labels;
acc = mean(YPred == YTrue);
disp(['Val Accuracy: ', num2str(acc)]);

figure;
confusionchart(YTrue, YPred);
exportgraphics(gcf, fullfile('results', 'confusion_matrix.png'));
close(gcf);

numSamples = 12;
valCount = numel(imdsVal.Files);
sampleIdx = randperm(valCount, min(numSamples, valCount));
figure;
for i = 1:numel(sampleIdx)
    idx = sampleIdx(i);
    img = readimage(imdsVal, idx);
    img = imresize(img, inputSize(1:2));
    pred = classify(trainedNet, img);
    subplot(3, 4, i);
    imshow(img);
    title(sprintf('T:%s P:%s', string(imdsVal.Labels(idx)), string(pred)));
end
exportgraphics(gcf, fullfile('results', 'sample_predictions.png'));
close(gcf);

function name = find_layer_name(lgraph, className)
layers = lgraph.Layers;
match = arrayfun(@(l) isa(l, className), layers);
idx = find(match, 1, 'last');
if isempty(idx)
    error('Layer not found: %s', className);
end
name = layers(idx).Name;
end
