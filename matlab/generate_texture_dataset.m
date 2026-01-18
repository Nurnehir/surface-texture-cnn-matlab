dataPath = fullfile(pwd, 'dataset');
classes = {'wood', 'metal', 'fabric', 'plastic'};
numPerClass = 300;
imgSize = 128;

for i = 1:numel(classes)
    classDir = fullfile(dataPath, classes{i});
    if ~exist(classDir, 'dir')
        mkdir(classDir);
    end
end

rng(0);

for c = 1:numel(classes)
    className = classes{c};
    classDir = fullfile(dataPath, className);

    for i = 1:numPerClass
        switch className
            case 'wood'
                img = generate_wood(imgSize);
            case 'metal'
                img = generate_metal(imgSize);
            case 'fabric'
                img = generate_fabric(imgSize);
            case 'plastic'
                img = generate_plastic(imgSize);
        end

        img = apply_random_transforms(img, imgSize);
        outName = sprintf('%s_%04d.png', className, i);
        imwrite(img, fullfile(classDir, outName));
    end
end

for i = 1:numel(classes)
    classDir = fullfile(dataPath, classes{i});
    files = dir(fullfile(classDir, '*.png'));
    fprintf('%s: %d\n', classes{i}, numel(files));
end

function img = generate_wood(sz)
    [x, y] = meshgrid(1:sz, 1:sz);
    freq1 = 0.08 + 0.04 * rand;
    freq2 = 0.02 + 0.02 * rand;
    wave = sin(2 * pi * freq1 * y + 2 * pi * rand) + ...
           0.5 * sin(2 * pi * freq2 * y + 2 * pi * rand);
    noise = low_freq_noise(sz, 8);
    base = wave + 0.6 * noise;
    base = mat2gray(base);
    base = imgaussfilt(base, 0.8);
    r = 0.55 + 0.1 * rand;
    g = 0.40 + 0.1 * rand;
    b = 0.25 + 0.1 * rand;
    img = cat(3, base * r, base * g, base * b);
    img = im2uint8(img);
end

function img = generate_metal(sz)
    [x, y] = meshgrid(1:sz, 1:sz);
    freq = 0.20 + 0.15 * rand;
    lines = sin(2 * pi * freq * x + 2 * pi * rand);
    grad = repmat(linspace(0.3, 0.9, sz)', 1, sz);
    base = 0.35 * lines + grad;
    base = base + 0.08 * randn(sz, sz);
    base = mat2gray(base);
    r = 0.70 + 0.05 * rand;
    g = 0.72 + 0.05 * rand;
    b = 0.78 + 0.05 * rand;
    img = cat(3, base * r, base * g, base * b);
    img = im2uint8(img);
end

function img = generate_fabric(sz)
    [x, y] = meshgrid(1:sz, 1:sz);
    f1 = 0.08 + 0.05 * rand;
    f2 = 0.08 + 0.05 * rand;
    pattern = sin(2 * pi * f1 * (x + y)) + sin(2 * pi * f2 * (x - y));
    base = pattern + 0.2 * randn(sz, sz);
    base = mat2gray(base);
    base = imsharpen(base, 'Radius', 1, 'Amount', 0.8);
    r = 0.45 + 0.1 * rand;
    g = 0.45 + 0.1 * rand;
    b = 0.50 + 0.1 * rand;
    img = cat(3, base * r, base * g, base * b);
    img = im2uint8(img);
end

function img = generate_plastic(sz)
    [x, y] = meshgrid(1:sz, 1:sz);
    angle = rand * 2 * pi;
    grad = cos(angle) * x + sin(angle) * y;
    grad = mat2gray(grad);
    base = 0.5 + 0.3 * (grad - 0.5) + 0.05 * randn(sz, sz);
    base = mat2gray(base);
    base = 0.4 + 0.2 * base;
    r = 0.60 + 0.05 * rand;
    g = 0.62 + 0.05 * rand;
    b = 0.65 + 0.05 * rand;
    img = cat(3, base * r, base * g, base * b);
    img = im2uint8(img);
end

function img = apply_random_transforms(img, targetSize)
    img = im2double(img);

    scale = 0.9 + 0.2 * rand;
    img = imresize(img, scale, 'bilinear');
    img = center_crop_or_pad(img, targetSize);

    angle = -20 + 40 * rand;
    img = imrotate(img, angle, 'bilinear', 'crop');

    contrast = 0.9 + 0.2 * rand;
    brightness = -0.1 + 0.2 * rand;
    img = img * contrast + brightness;
    img = min(max(img, 0), 1);

    if rand < 0.3
        img = imgaussfilt(img, 0.6);
    end

    img = im2uint8(img);
end

function out = center_crop_or_pad(img, targetSize)
    [h, w, c] = size(img);
    if h > targetSize
        startRow = floor((h - targetSize) / 2) + 1;
        img = img(startRow:startRow + targetSize - 1, :, :);
    elseif h < targetSize
        padTop = floor((targetSize - h) / 2);
        padBottom = targetSize - h - padTop;
        img = padarray(img, [padTop, 0], 'replicate', 'pre');
        img = padarray(img, [padBottom, 0], 'replicate', 'post');
    end

    if w > targetSize
        startCol = floor((w - targetSize) / 2) + 1;
        img = img(:, startCol:startCol + targetSize - 1, :);
    elseif w < targetSize
        padLeft = floor((targetSize - w) / 2);
        padRight = targetSize - w - padLeft;
        img = padarray(img, [0, padLeft], 'replicate', 'pre');
        img = padarray(img, [0, padRight], 'replicate', 'post');
    end

    out = reshape(img, targetSize, targetSize, c);
end

function noise = low_freq_noise(sz, scale)
    small = rand(ceil(sz / scale), ceil(sz / scale));
    noise = imresize(small, [sz, sz], 'bilinear');
end
