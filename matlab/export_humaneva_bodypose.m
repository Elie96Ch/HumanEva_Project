function export_humaneva_bodypose(dataset_root, support_root, output_root)
% Export HumanEva body_pose sequences to MAT files for Python use.

    if nargin < 3
        error('Usage: export_humaneva_bodypose(dataset_root, support_root, output_root)');
    end

    addpath(support_root);
    addpath(fullfile(support_root, 'TOOLBOX_dxAvi'));
    addpath(fullfile(support_root, 'TOOLBOX_calib'));
    addpath(fullfile(support_root, 'TOOLBOX_readc3d'));
    addpath(fullfile(support_root, 'TOOLBOX_common'));

    subjects = {'S1', 'S2', 'S3', 'S4'};
    static_name = 'Static.c3d';

    point_names = { ...
        'torsoProximal', 'torsoDistal', ...
        'upperLArmProximal', 'upperLArmDistal', ...
        'lowerLArmProximal', 'lowerLArmDistal', ...
        'upperRArmProximal', 'upperRArmDistal', ...
        'lowerRArmProximal', 'lowerRArmDistal', ...
        'upperLLegProximal', 'upperLLegDistal', ...
        'lowerLLegProximal', 'lowerLLegDistal', ...
        'upperRLegProximal', 'upperRLegDistal', ...
        'lowerRLegProximal', 'lowerRLegDistal', ...
        'headProximal', 'headDistal'};

    for s = 1:numel(subjects)
        subject = subjects{s};
        mocap_dir = fullfile(dataset_root, subject, 'Mocap_Data');

        if ~exist(mocap_dir, 'dir')
            fprintf('Skipping missing folder: %s\n', mocap_dir);
            continue;
        end

        files = dir(fullfile(mocap_dir, '*.c3d'));
        out_dir = fullfile(output_root, subject);
        if ~exist(out_dir, 'dir')
            mkdir(out_dir);
        end

        static_path = fullfile(mocap_dir, static_name);
        mp_path = fullfile(mocap_dir, [subject '.mp']);

        for i = 1:numel(files)
            trial_name = files(i).name;

            if strcmpi(trial_name, static_name)
                continue;
            end

            c3d_path = fullfile(mocap_dir, trial_name);
            trial_stem = trial_name(1:end-4);

            fprintf('\nProcessing %s / %s\n', subject, trial_stem);

            try
                ms = mocap_stream(c3d_path, static_path, mp_path, 1, 1);

                T = n_frames(ms);
                pose_3d = zeros(T, 20, 3);
                valid = zeros(T, 1);

                for t = 1:T
                    [ms, pose, is_valid] = cur_frame(ms, t, 'body_pose');
                    valid(t) = is_valid;

                    pose_3d(t,1,:)  = pose.torsoProximal;
                    pose_3d(t,2,:)  = pose.torsoDistal;
                    pose_3d(t,3,:)  = pose.upperLArmProximal;
                    pose_3d(t,4,:)  = pose.upperLArmDistal;
                    pose_3d(t,5,:)  = pose.lowerLArmProximal;
                    pose_3d(t,6,:)  = pose.lowerLArmDistal;
                    pose_3d(t,7,:)  = pose.upperRArmProximal;
                    pose_3d(t,8,:)  = pose.upperRArmDistal;
                    pose_3d(t,9,:)  = pose.lowerRArmProximal;
                    pose_3d(t,10,:) = pose.lowerRArmDistal;
                    pose_3d(t,11,:) = pose.upperLLegProximal;
                    pose_3d(t,12,:) = pose.upperLLegDistal;
                    pose_3d(t,13,:) = pose.lowerLLegProximal;
                    pose_3d(t,14,:) = pose.lowerLLegDistal;
                    pose_3d(t,15,:) = pose.upperRLegProximal;
                    pose_3d(t,16,:) = pose.upperRLegDistal;
                    pose_3d(t,17,:) = pose.lowerRLegProximal;
                    pose_3d(t,18,:) = pose.lowerRLegDistal;
                    pose_3d(t,19,:) = pose.headProximal;
                    pose_3d(t,20,:) = pose.headDistal;
                end

                subject_name = subject;
                action_name = trial_stem;
                out_path = fullfile(out_dir, [trial_stem '.mat']);

                save(out_path, 'pose_3d', 'valid', 'point_names', ...
                    'subject_name', 'action_name', '-v7');

                fprintf('Saved: %s | shape=(%d, %d, %d)\n', out_path, size(pose_3d,1), size(pose_3d,2), size(pose_3d,3));

            catch ME
                fprintf('FAILED: %s / %s\n', subject, trial_stem);
                fprintf('%s\n', ME.message);
                for k = 1:numel(ME.stack)
                    fprintf('  at %s (line %d)\n', ME.stack(k).name, ME.stack(k).line);
                end
            end

        end
    end

    fprintf('\nDone.\n');
end
