clear; clc;

project_root = 'D:\PhD work\HumanEVA dataset';

source_root = fullfile(project_root, 'data', 'humaneva', 'source_code', 'Release_Code_v1_1_beta');
addpath(genpath(source_root));

subject = 'S1';
action = 'Walking_1';
camera = 'C1';

raw_root = fullfile(project_root, 'data', 'humaneva', 'raw');

mocap_dir = fullfile(raw_root, subject, 'Mocap_Data');
c3d_path = fullfile(mocap_dir, [action '.c3d']);
static_path = fullfile(mocap_dir, 'Static.c3d');
mp_path = fullfile(mocap_dir, [subject '.mp']);
cal_path = fullfile(raw_root, subject, 'Calibration_Data', [camera '.cal']);

% IMPORTANT TEST:
% scaling = 2 means requested frame_id approximately indexes video frames.
ms = mocap_stream(c3d_path, static_path, mp_path, 1, 2);

frame_ids = [100, 200, 300, 400, 500];

point_names = {
    'torsoProximal'
    'torsoDistal'
    'upperLArmProximal'
    'upperLArmDistal'
    'lowerLArmProximal'
    'lowerLArmDistal'
    'upperRArmProximal'
    'upperRArmDistal'
    'lowerRArmProximal'
    'lowerRArmDistal'
    'upperLLegProximal'
    'upperLLegDistal'
    'lowerLLegProximal'
    'lowerLLegDistal'
    'upperRLegProximal'
    'upperRLegDistal'
    'lowerRLegProximal'
    'lowerRLegDistal'
    'headProximal'
    'headDistal'
};

pose_3d = zeros(numel(frame_ids), 20, 3);
pose_2d = zeros(numel(frame_ids), 20, 2);
valid = zeros(numel(frame_ids), 1);

for f = 1:numel(frame_ids)
    frame_id = frame_ids(f);

    [ms, p3d, is_valid] = cur_frame(ms, frame_id, 'body_pose');
    p2d = project2d(p3d, cal_path);

    valid(f) = is_valid;

    for j = 1:numel(point_names)
        name = point_names{j};
        pose_3d(f, j, :) = p3d.(name)(:).';
        pose_2d(f, j, :) = p2d.(name)(:).';
    end
end

out_dir = fullfile(project_root, 'outputs', 'debug');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

out_path = fullfile(out_dir, 'video_rate_pose_debug.mat');

save(out_path, ...
    'pose_3d', ...
    'pose_2d', ...
    'valid', ...
    'frame_ids', ...
    'subject', ...
    'action', ...
    'camera');

disp(['Saved: ' out_path]);
disp('frame_ids:');
disp(frame_ids);
disp('valid:');
disp(valid.');