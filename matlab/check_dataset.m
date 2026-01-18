dataPath = fullfile(pwd, 'dataset');
imds = imageDatastore(dataPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

if isempty(imds.Files)
    disp('Dataset klasorlerine gorseller ekleyin');
    return;
end

disp(countEachLabel(imds));

labels = categories(imds.Labels);
numClasses = numel(labels);
numPerClass = 2;

figure;
plotIndex = 1;
for i = 1:numClasses
    classLabel = labels{i};
    classIdx = find(imds.Labels == classLabel);
    selectCount = min(numPerClass, numel(classIdx));
    classIdx = classIdx(randperm(numel(classIdx), selectCount));

    for j = 1:selectCount
        subplot(2, 4, plotIndex);
        img = readimage(imds, classIdx(j));
        imshow(img);
        title(char(classLabel));
        plotIndex = plotIndex + 1;
    end
end
