import concurrent.futures
from ..detectors import run_detection

from micoes.microclusters.clustream import update_clu_microcluster
from micoes.microclusters.denstream import update_den_microcluster

#from timeit import default_timer as timer

import time

import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)


def run_detector_clustream(X, clustream, detector_type='lof', profiling=False):
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        if profiling:
            start = time.perf_counter()
        # submit the process
        clustream_process = executor.submit(update_clu_microcluster, clustream, X, profiling=profiling)
        detection_process = executor.submit(run_detection, X, detector_type, profiling=profiling)

        # collect the results
        results = []
        for f in concurrent.futures.as_completed((clustream_process, detection_process)):
            try:
                results.append(f.result())
            except Exception as exc:
                logging.info(f'thread exception {exc}')
            # else:
            #     logging.info(f'Complete thread : {len(results)}')

    if profiling:
        duration = time.perf_counter() - start
        return (results, duration)
    else:
        return (results,)


def run_detector_microcluster_serial(X, clustream, denstream, detector_type='lof', profiling=False):
    results = list()

    # logging.info('run detection process')
    detection_process = run_detection(X, detector_type, profiling=profiling)
    results.append(detection_process)

    # logging.info('run clustream microcluster process')
    clustream_process = update_clu_microcluster(clustream, X, profiling=profiling)
    results.append(clustream_process)

    # logging.info('run denstream microcluster process')
    denstream_process = update_den_microcluster(denstream, X, profiling=profiling)
    results.append(denstream_process)

    return results


def run_detector_denstream(X, denstream, detector_type='lof', profiling=False):
    if profiling:
        start = time.perf_counter()
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # submit the process
        denstream_process = executor.submit(update_den_microcluster, denstream, X, profiling)
        detection_process = executor.submit(run_detection, X, detector_type, profiling)

        # collect the results
        for f in concurrent.futures.as_completed((denstream_process, detection_process)):
            results.append(f.result())

    if profiling:
        duration = time.perf_counter() - start
        return (results, duration)
    else:
        return (results,)
