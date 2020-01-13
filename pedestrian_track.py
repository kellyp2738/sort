import sys
sys.path.insert(0, '..')
from sort import Sort
import pandas as pd
import multiprocessing as mp
import re
import os
import time


TIMESTAMP_PATTERN = r'\d{12}'
LOGFILE_NAME = 'out.log'


def find_files(parent_dir, pattern=TIMESTAMP_PATTERN, file_name=LOGFILE_NAME):

    data_paths = []
    pattern = re.compile(pattern)
    for root, dirs, files in os.walk(parent_dir, topdown=False):
        for name in files:
            if file_name in name:
                log_path = os.path.join(root, name)
                ts = re.findall(pattern, log_path)
                if ts:
                    data_paths.append(log_path)

    return data_paths


def generate_track_filename(datapath, pattern=TIMESTAMP_PATTERN):

    ts = re.findall(pattern, datapath)
    if not ts:
        raise ValueError('No timestamp detected in log file path {}.'.format(datapath))

    outfile_name = '{}_pedestrian_tracks.csv'.format(ts)

    return outfile_name


def log_reader(filepath):

    yolo_log_columns = [
        'frame_idx',
        'object_trajectory_idx',
        'unclear_1',
        'unclear_2',
        'class_label',
        'class_confidence',
        'x_min',
        'y_min',
        'x_max',
        'y_max',
        'dx',
        'dy',
    ]
    seq_dets = pd.read_csv(filepath)
    seq_dets.columns = yolo_log_columns
    ped_dets = seq_dets[(seq_dets['class_label'] == 0) & (seq_dets['class_confidence'] > 50)]

    return ped_dets


def ped_tracker(ped_data, age, hits, threshold):

    total_time = 0.0
    total_frames = 0

    # create instance of the SORT tracker
    mot_tracker = Sort(
        max_age=age,
        min_hits=hits,
        iou_threshold=threshold
    )

    # build the tracks by iterating over frames
    tracks = []
    for f in range(1, int(ped_data['frame_idx'].max())):

        frame_dets = ped_data[ped_data['frame_idx'] == f]
        dets = frame_dets[['x_min', 'y_min', 'x_max', 'y_max']].values
        total_frames += 1

        start_time = time.time()
        trackers = mot_tracker.update(dets)
        cycle_time = time.time() - start_time
        total_time += cycle_time

        for t in trackers.tolist():
            tracks.append(
                {'frame_idx': f, 'x_min': t[0], 'y_min': t[1], 'x_max': t[2], 'y_max': t[3], 'track_idx': t[4]}
            )

    # organize track outputs
    tracks_df = pd.DataFrame.from_dict(tracks)
    tracks_df['x_center'] = [(row['x_max'] + row['x_min']) / 2 for i, row in tracks_df.iterrows()]
    tracks_df['y_center'] = [(row['y_max'] + row['y_min']) / 2 for i, row in tracks_df.iterrows()]

    return tracks_df


def tracker_workflow(data_path, age, hits, threshold, outdir):

    data = log_reader(data_path)
    if data is not None:
        tracks = ped_tracker(data, age, hits, threshold)
        save_name = generate_track_filename(data_path)
        tracks.to_csv(os.path.join(outdir, save_name))


def launch_tracker(data_dir, age, hits, threshold, outdir):

    # find yolo logfiles
    file_paths = find_files(parent_dir=data_dir)

    # configure parallel run based on compute environment
    avail_cpus = mp.cpu_count()
    if avail_cpus == 272:
        req_cpus = 64 # no hyperthreading on stampede 2
    else:
        req_cpus = 1  # avail_cpus-2  # if running locally, save some processing power for other stuff

    # assemble parallel tasks
    pool = mp.Pool(req_cpus)
    parallel_tasks = [(f, age, hits, threshold, outdir) for f in file_paths]
    results = [pool.apply_async(tracker_workflow, p) for p in parallel_tasks]
    pool.close()

    # execute parallel tasks
    for r in results:
        r.get()

    return

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '-p',
        '--parent_dir',
        help='directory containing log files and background images'
    )
    parser.add_argument(
        '-o',
        '--out_dir',
        help='output directory'
    )
    parser.add_argument(
        '-a',
        '--max_age',
        help='max_age parameter for SORT'
    )
    parser.add_argument(
        '-m',
        '--min_hits',
        help='min_hits parameter for SORT'
    )
    parser.add_argument(
        '-t',
        '--iou_threshold',
        help='iou_threshold parameter for SORT'
    )

    opts = parser.parse_args()

    # setup output directory
    if not os.path.exists(opts.out_dir):
        os.makedirs(opts.out_dir)

    launch_tracker(opts.parent_dir, int(opts.max_age), int(opts.min_hits), float(opts.iou_threshold), opts.out_dir)
