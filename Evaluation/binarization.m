clear all; close all; clc;

%set your dataset path and saliency map result path.
% dataset = 'SIP';
% salPath = 'D:\Research Group\Research circle\Dr. Saeed Anwar\VS via Saliency\Experimental results\Output results\SIP Test Data Results\Temp/';
% gtPath = 'D:\My Research\Datasets\Saliency Detection\SIP\Test\Labels\';

% dataset = 'NLPR';
% salPath = 'D:\Research Group\Research circle\Dr. Saeed Anwar\VS via Saliency\Experimental results\Output results\NLPR Data Results\Temp/';
% gtPath = 'D:\My Research\Datasets\Saliency Detection\NLPR\Test\Labels\';

% dataset = 'NJU2K';
% salPath = 'D:\Research Group\Research circle\Dr. Saeed Anwar\VS via Saliency\Experimental results\Output results\NJU2K Data Results\Temp/';
% gtPath = 'D:\My Research\Datasets\Saliency Detection\NJU2K\Test\Labels\';

% Testing DUTRGBD
% dataset = 'DUTRGB-D';
% salPath = 'D:\Research Group\Research circle\Dr. Saeed Anwar\VS via Saliency\Experimental results\Output results\DUTRGB-D Test Data Results\Temp/';
% gtPath = 'D:\My Research\Datasets\Saliency Detection\DUT-RGBD\Test\Labels/';

dataset = 'Choke';
salPath = 'D:\PycharmProjects\SOD_SOTA\2_TNNLS2020_D3NetBenchmark-master\crossdata_output\Output1\';

%obtain the total number of image (ground-truth)
imgFiles = dir(salPath);
imgNUM = length(imgFiles)-2;


tic;
for i = 1:imgNUM
    
    fprintf('Evaluating: %d/%d\n',i,imgNUM);
    
    name =  imgFiles(i+2).name;
    %name = name(:,3:10);
  
    
    %load salency
    sal  = imread([salPath name]);
    
    if numel(size(sal))>2
        sal = rgb2gray(sal);
    end
    if ~islogical(sal)
        sal = sal(:,:,1) > 128;
    end

    
    sal = im2double(sal(:,:,1));
    
    %normalize sal to [0, 1]
    sal = reshape(mapminmax(sal(:)',0,1),size(sal));
    
  
    
    %You can change the method of binarization method. As an example, here just use adaptive threshold.
    threshold =  2* mean(sal(:)) ;
    if ( threshold > 1 )
        threshold = 1;
    end
    Bi_sal = zeros(size(sal));
    Bi_sal(sal>threshold)=1;
    imwrite(Bi_sal,[salPath name])
    
end

toc;
