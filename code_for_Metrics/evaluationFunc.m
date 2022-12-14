function [ meanMetric, allMetrics, frames] = evaluationFunc( options, metricName )
%EVALUATIONFUNC Evaluate result with the metric
%   result: array of cells containing the predicted saliency map
%   data: the ground truth data
%   metricName: the name of metric
%       -"similarity": Similarity
%       -"CC": CC
%       -"AUC_Borji": AUC_Borji
%       -"AUC_Judd": AUC_Judd
%       -"AUC_shuffled": sAUC
%   if the ground truth cannot be found, e.g. testing data, the central
%   gaussian will be taken as ground truth automatically.

% GlobalParameters;
%SALICONGlobalParameters;
%addpath(METRIC_DIR);
% assert(length(result)==length(data));
%availableMetric = {'similarity','CC', 'AUC_Judd', 'AUC_Borji', 'AUC_shuffled'};
% assert(any(strcmp(metricName, availableMetric)));
% if strcmp(metricName, 'AUC_shuffled')
%     try
%         load(TRAIN_DATA_PATH);
%     catch
%         fprintf('Training data missing!\n');
%     end
% end

postfix = '.png';

fh = str2func(metricName);
frames = dir(fullfile([options.IMG_DIR '*' postfix]));


nframe = length(frames);
    
%% we evaluate at most 50000 randomly selected frames in one dataset for efficiency
if nframe>50000
    k = randperm(nframe);
    frames = frames(k(1:50000));
end

allMetrics = zeros(length(frames),1);
for i = 1:min(nframe, 50000)

    gt_fold = frames(i).folder;
    gt_fold = strrep(gt_fold, '\','/');
    gt_name = frames(i).name;
    
    map_gt_path = strrep(gt_fold,'/saliency', '/maps/');
    fix_gt_path = strrep(gt_fold,'/saliency', '/fixation/maps/');
%     map_eval_path = strrep(gt_fold, options.DS_GT_DIR, options.SALIENCY_DIR);
    map_eval_path = gt_fold;
    
    saliency_path = [map_gt_path, gt_name];
    
%     if ~exist(saliency_path,'file')
%         continue;
%     end
    
    fixation_path = [fix_gt_path, strrep(gt_name, postfix, '.mat')];
    
    load(fixation_path);
    
    result = double(imread([map_eval_path,'/',gt_name]));
    
    result = result(:,:,1);
    result = imresize(result, [size(I,1) size(I,2)]);

%     disp([map_eval_path, '/',gt_name]);
%     disp(saliency_path);
%     disp(fixation_path);
    if any(strcmp(metricName, {'similarity','CC', 'EMD'}))
        if exist(saliency_path, 'file')
            I = double(imread(saliency_path))/255;
            
            allMetrics(i) = fh( result, I);
        else       
            allMetrics(i) = nan;
        end
    elseif any(strcmp(metricName, {'AUC_Judd', 'AUC_Borji','NSS'}))
        if exist(fixation_path, 'file')
            load(fixation_path);
            I = double(I);
            allMetrics(i) = fh( result, I);
        else       
            allMetrics(i) = nan;
        end       
    elseif strcmp(metricName, 'AUC_shuffled')
        if exist(fixation_path, 'file')
            load(fixation_path);
            I = I>0;
            I = double(I);
            ids = randsample(length(frames), min(10,length(frames)));
            fixation_point = zeros(0,2);
            for k = 1:min(10,length(frames))
                fx_name = frames(ids(k)).name;
                fx_fold = frames(ids(k)).folder;
                fx_fold = strrep(fx_fold, '\','/');

                fix_path = strrep(fx_fold,'/saliency', '/fixation/maps/');
                fixation_pathx = [fix_path, strrep(fx_name, postfix, '.mat')];
               
                Ix = load(fixation_pathx);
                Ix = double(Ix.I);
                training_resolution = size(Ix);
                rescale = size(result)./training_resolution;
                [fx, fy]= find(Ix);
                pts = vertcat([fy fx]);
                fixation_point = [fixation_point; pts.*repmat(rescale, size(pts,1), 1)];
            end
            otherMap = makeFixationMap(size(result), fixation_point);
            allMetrics(i) = fh( result, I, otherMap);
        else       
            allMetrics(i) = nan;
        end 
    else
        allMetrics(i) = nan;
    end
end
allMetrics(isnan(allMetrics)) = [];
meanMetric = mean(allMetrics);
end

