clear; clc;

project_root = 'D:\PhD work\HumanEVA dataset';

source_root = fullfile(project_root, 'data', 'humaneva', 'source_code', 'Release_Code_v1_1_beta');
addpath(genpath(source_root));

subject = 'S1';
action = 'Walking_1';
camera = 'C1';
frame_id = 704;

raw_root = fullfile(project_root, 'data', 'humaneva', 'raw');

c3d_path = fullfile(raw_root, subject, 'Mocap_Data', [action '.c3d']);
static_path = fullfile(raw_root, subject, 'Mocap_Data', 'Static.c3d');
mp_path = fullfile(raw_root, subject, 'Mocap_Data', [subject '.mp']);
cal_path = fullfile(raw_root, subject, 'Calibration_Data', [camera '.cal']);

ms = mocap_stream(c3d_path, static_path, mp_path, 1, 1);

frame_id = 1;
is_valid = 0;

while ~is_valid && frame_id < 3000
    [ms, pose3d, is_valid] = cur_frame(ms, frame_id, 'body_pose');
    if ~is_valid
        frame_id = frame_id + 1;
    end
end

if ~is_valid
    error('Could not find a valid frame.');
end

disp(['Selected valid frame: ' num2str(frame_id)]);

pose2d = project2d(pose3d, cal_path);

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

pose_3d = zeros(20, 3);
pose_2d_matlab = zeros(20, 2);

for i = 1:numel(point_names)
    name = point_names{i};
    pose_3d(i, :) = pose3d.(name)(:).';
    pose_2d_matlab(i, :) = pose2d.(name)(:).';
end

out_dir = fullfile(project_root, 'outputs', 'debug');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

out_path = fullfile(out_dir, 'official_project2d_one_frame.mat');

save(out_path, ...
    'pose_3d', ...
    'pose_2d_matlab', ...
    'point_names', ...
    'is_valid', ...
    'subject', ...
    'action', ...
    'camera', ...
    'frame_id', ...
    'cal_path');

disp(['Saved: ' out_path]);
disp(['Valid: ' num2str(is_valid)]);
disp('MATLAB official 2D min/max:');
disp(min(pose_2d_matlab));
disp(max(pose_2d_matlab));