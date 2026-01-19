% matlab/train_texture_cnn.m
% Texture classification with MobileNetV2 (Transfer Learning)
% Dataset: dataset/{wood,metal,fabric,plastic}

clear; clc;

% --- Load dataset ---
dataPath = fullfile(pwd, 'dataset');
imds = imageDatastore(dataPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

if isempty(imds.Files)
    error("Dataset not found. Expected images under: %s", dataPath);
end

disp("Classes:");
disp(categories(imds.Labels));
disp("Counts:");
disp(countEachLabel(imds));

numClasses = numel(categories(imds.Labels));
[imdsTrain, imdsVal] = splitEachLabel(imds, 0.8, 'randomized');

% --- MobileNetV2 + augmentation ---
net = mobilenetv2;
inputSize = net.Layers(1).InputSize;

augmenter = imageDataAugmenter( ...
    'RandRotation', [-20 20], ...
    'RandXTranslation', [-10 10], ...
    'RandYTranslation', [-10 10], ...
    'RandXReflection', true, ...
    'RandScale', [0.9 1.1]);

augTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
    'DataAugmentation', augmenter);

augVal = augmentedImageDatastore(inputSize(1:2), imdsVal);

% --- Transfer learning: replace last layers robustly (class-type based) ---
lgraph = layerGraph(net);

fcName       = find_layer_name_any(lgraph, ["nnet.cnn.layer.FullyConnectedLayer"]);
softmaxName  = find_layer_name_any(lgraph, ["nnet.cnn.layer.SoftmaxLayer"]);
classLayName = find_layer_name_any(lgraph, ["nnet.cnn.layer.ClassificationLayer", ...
                                           "nnet.cnn.layer.ClassificationOutputLayer"]);

newFc = fullyConnectedLayer(numClasses, 'Name', 'new_fc', ...
    'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10);
newSoftmax = softmaxLayer('Name', 'new_softmax');
newClass = classificationLayer('Name', 'new_classoutput');

lgraph = replaceLayer(lgraph, fcName, newFc);
lgraph = replaceLayer(lgraph, softmaxName, newSoftmax);
lgraph = replaceLayer(lgraph, classLayName, newClass);

% --- Training options ---
options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-4, ...
    'MaxEpochs', 6, ...
    'MiniBatchSize', 32, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augVal, ...
    'ValidationFrequency', 20, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% --- Train ---
trainedNet = trainNetwork(augTrain, lgraph, options);

% --- Save results ---
if ~exist('results', 'dir')
    mkdir('results');
end

save(fullfile('results', 'trainedNet.mat'), 'trainedNet');

% --- Validate ---
YPred = classify(trainedNet, augVal);
YTrue = imdsVal.Labels;
acc = mean(YPred == YTrue);
disp(['Val Accuracy: ', num2str(acc)]);

% Confusion matrix (saved)
fig1 = figure('Visible','off');
confusionchart(YTrue, YPred);
exportgraphics(fig1, fullfile('results', 'confusion_matrix.png'));
close(fig1);

% Sample predictions figure (saved)
numSamples = 12;
valCount = numel(imdsVal.Files);
sampleIdx = randperm(valCount, min(numSamples, valCount));

fig2 = figure('Visible','off');
for i = 1:numel(sampleIdx)
    idx = sampleIdx(i);
    img = readimage(imdsVal, idx);
    imgResized = imresize(img, inputSize(1:2));
    pred = classify(trainedNet, imgResized);

    subplot(3, 4, i);
    imshow(imgResized);
    title(sprintf('T:%s  P:%s', string(imdsVal.Labels(idx)), string(pred)));
end
exportgraphics(fig2, fullfile('results', 'sample_predictions.png'));
close(fig2);

disp("Done. Saved to results/: trainedNet.mat, confusion_matrix.png, sample_predictions.png");

% ---------- helper ----------
function name = find_layer_name_any(lgraph, classNames)
layers = lgraph.Layers;
layerClasses = arrayfun(@(l) string(class(l)), layers);

idx = [];
for k = 1:numel(classNames)
    cand = find(layerClasses == string(classNames(k)), 1, 'last');
    if ~isempty(cand)
        idx = cand;
        break;
    end
end

if isempty(idx)
    error("Layer not found. Tried: %s", strjoin(string(classNames), ", "));
end

name = layers(idx).Name;
end
