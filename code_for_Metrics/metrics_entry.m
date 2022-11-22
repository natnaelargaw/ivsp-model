function [allmetrics, meanmetrics, frames] = metrics_entry()
%     folders = dir('../DHF1K/annotation/');
%     allmetrics = 0;
%     meanmetrics = 0;
%     frames = 0;
%     disp(length(folders))
%     for i=1: length(folders)
%         disp(folders(i));
%     end

    
    options.IMG_DIR = '/home/natnael/Documents/datasets/DHF1K/val_images/*/saliency/';
    options.DS_GT_DIR = '/home/natnael/Documents/datasets/DHF1K/val_images/';
    options.SALIENCY_DIR = '/home/natnael/Documents/datasets/DHF1K/val_images/results';

%    [allmetrics, meanmetrics, frames] = evaluationFunc(options, 'NSS');
%     [allmetrics, meanmetrics, frames] = evaluationFunc(options, 'CC');
    [allmetrics, meanmetrics, frames] = evaluationFunc(options, 'AUC_shuffled');
%      [allmetrics, meanmetrics, frames] = evaluationFunc(options, 'AUC_Judd');

%      [allmetrics, meanmetrics, frames] = evaluationFunc(options, 'similarity');



% [score,tp,fp,allthreshes] = evaluationFunc(options, 'AUC_Judd');
%     [allmetrics, meanmetrics, frames]









%     for j = 5:5% 1:length(Metrics)
%             if ~exist([CACHE Datasets{i} '_' Results{k} '_' Metrics{j} '.mat'], 'file')                 
%                 [result, allMetric, ~] = evaluationFunc(options, Metrics{j});
%                 % save([CACHE Datasets{i} '_' Results{k} '_' Metrics{j} '.mat'], 'result');
%                 % save([CACHE Datasets{i} '_' Results{k} '_' Metrics{j} '_all.mat'], 'result');
%                 % std_value = std(allMetric); % calculate std value if you want to
%             else
%                 load([CACHE Datasets{i} '_' Results{k} '_' Metrics{j} '.mat']);
%             end
%             meanMetric{i}(k,j) = result;
%             fprintf('%s :%.4f \n', Metrics{j}, result);
%         end
%     end
% end


end












% 
% imagefiles = dir('*.png');   
% 
% nfiles = length(imagefiles);    % Number of files found
% % for ii=1:nfiles
%    currentfilename = imagefiles(ii).name;
%    currentimage = imread(currentfilename);
%    imshow(currentimage)
% end

% function avg = untitled(x)
%     avg = sum(x(:))/numel(x);
% end

% function [allmetrics, meanmetrics, frames] = mertics_entry()
%     postfix = ".png";
%     pred_dir = "../../DHF1K/val_images/";
%     true_dir = "../../DHF1K/annotation/";
% 
% 
%     fixation_map = "fixation/";
%     fix_mat = "fixation/maps/"; % other map
%     saliency_map="maps/";
% 
%     true_dir_list = dir(true_dir);
%     pred_dir_list = dir(pred_dir);
% 
%     true_dirFlags = [true_dir_list.isdir];
%     pred_dirflags = [pred_dir_list.isdir];
% 
% 
%     true_subfolders = true_dir_list(true_dirFlags); % A structure with extra info.
%     pred_subfolders = pred_dir_list(pred_dirflags);
% 
% 
%     true_subFolderNames = {true_subfolders(3:end).name}; % Start at 3 to skip . and ..
%     pred_subFolderNames = {pred_subfolders(3:end).name}; % Start at 3 to skip . and ..
% 
% 
%     files_true = true_subFolderNames;
%     files_pred_map = pred_subFolderNames;
%     files_pred_fix = pred_subFolderNames;
% 
%     meanmetrics =0;
%     counter = 1;
%     
%     for i=3: length(files_pred_map)
%         pred_map_base = files_pred_map(i);
% 
% %         disp(pred_map_base)
%         % read files to array
% 
% 
%         pred_folder_path = pred_dir + pred_map_base + "/";
% 
%         frames = dir( fullfile(pred_folder_path, '*.png') );
% 
%         truth_base = true_dir +'0'+pred_map_base+'/';
% 
%         disp(truth_base);
% 
% 
%         for j=1: length(frames)
% %             disp(frames(j).name);
% %             disp(frames(j).folder);
% %             disp(frames(j).name(1:end-4)); % for fixations and maps
%             
%             frame_map_pred = [frames(j).folder,'/',frames(j).name];
%             saliency_map_pred = double(imread(frame_map_pred))/255;
% 
% 
% 
% 
%             frame_saliency_map = truth_base+ saliency_map +frames(j).name;
% %             disp(frame_saliency_map);
%             sal_map_truth = double(imread(frame_saliency_map))/255;
% 
% 
%             frame_fixaion_map_truth = truth_base+ fixation_map+ frames(j).name;
%             fix_map_truth = double(imread(frame_fixaion_map_truth))/255; 
% 
% 
% % 
%             frame_fixmat = truth_base + fix_mat+ strrep(frames(j).name, postfix, ".mat");
% %             frame_fix_mat = imread(frame_fixmat);
% %             disp(frame_fixmat);
% 
% 
%             score = CC(saliency_map_pred,sal_map_truth);
%             disp(score);
% 
%             allmetrics(counter) = score;
%             counter= counter+1;
% 
%         end
% 
% 
%         meanmetrics = mean(allmetrics);
% 
%         disp(meanmetrics);
% 
% 
% 
% 
% 
% 
% 
% %         disp(length(img_folder));
% 
% 
% 
% 
%         
% 
% 
% 
% 
% 
% 
% 
%     end
% 





%     for i=1:length(true_subFolderNames)
%         disp(true_subFolderNames);








 % to change metrics to function
%   fh = str2func(metrics_type);
%   disp(fh)
%   
%   dirData = dir(dirName);      %# Get the data for the current directory
%   dirIndex = [dirData.isdir];  %# Find the index for directories
%   fileList = {dirData(~dirIndex).name}';  %'# Get a list of the files
%   if ~isempty(fileList)
%     fileList = cellfun(@(x) fullfile(dirName,x),...  %# Prepend path to files
%                        fileList,'UniformOutput',false);
%   end
%   subDirs = {dirData(dirIndex).name};  %# Get a list of the subdirectories
%   validIndex = ~ismember(subDirs,{'.','..'});  %# Find index of subdirectories
%                                                %#   that are not '.' or '..'
%   for iDir = find(validIndex)                  %# Loop over valid subdirectories
%     nextDir = fullfile(dirName,subDirs{iDir});    %# Get the subdirectory path
%     fileList = [fileList; mertics_entry(nextDir)];  %# Recursively call getAllFiles
%   end

% end




% % true_dir = "../../DHF1K/val_images/";
% pred_dir = "../../DHF1K/annotation/";


% 
% 
% true_dire_read = dir(fullfile(true_dir,'sub*'));
% F = {true_dire_read([true_dire_read.isdir]).name};
% N = numel(F);
% % disp('tolist')
% for k = 1:N
%     T = fullfile(P,F{k});
%     disp(N)
% 
% end
% 
% x = imread("../../DHF1K/val_images/601/saliency/0010.png", "png");
% y = imread("../../DHF1K/annotation/0601/maps/0010.png", "png");
% z = imread("../../DHF1K/annotation/0601/fixation/0010.png", "png");
% 
% 
% [score, tf, tp, althreshs] = AUC_Judd(x,y,1,0);
% 
% score = similarity(x,y,0);
% 
% score = AUC_shuffled(x,z,y,3,4, 0);
% 
% score = CC(x,y);
% 
% score = NSS(x,y);
% 
% % disp(score);
% end