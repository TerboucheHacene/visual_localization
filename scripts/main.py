import logging
from pathlib import Path
from pprint import pprint

from svl.keypoint_pipeline.detection_and_description import SuperPointAlgorithm
from svl.keypoint_pipeline.matcher import SuperGlueMatcher
from svl.keypoint_pipeline.typing import SuperGlueConfig, SuperPointConfig
from svl.localization.drone_streamer import DroneImageStreamer
from svl.localization.map_reader import SatelliteMapReader
from svl.localization.pipeline import Pipeline, PipelineConfig
from svl.localization.preprocessing import QueryProcessor
from svl.tms.data_structures import CameraModel

if __name__ == "__main__":

    # logging.basicConfig(level=logging.INFO)
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")
    # set to debug for more information

    # Initialize the keypoint detector
    superpoint_config = SuperPointConfig(
        device="cuda",
        nms_radius=4,
        keypoint_threshold=0.01,
        max_keypoints=-1,
    )
    superpoint_algorithm = SuperPointAlgorithm(superpoint_config)

    # Initialize the keypoint matcher
    superglue_config = SuperGlueConfig(
        device="cuda",
        weights="outdoor",
        sinkhorn_iterations=20,
        match_threshold=0.5,
    )
    superglue_matcher = SuperGlueMatcher(superglue_config)

    # Initialize the map reader
    map_reader = SatelliteMapReader(
        db_path="data/map/",
        resize_size=(800,),
        logger=logging.getLogger("%s.SatelliteMapReader" % __name__),  # noqa
    )
    map_reader.initialize_db()
    map_reader.setup_db()
    map_reader.resize_db_images()
    map_reader.describe_db_images(superpoint_algorithm)

    # Initialize the drone image streamer
    streamer = DroneImageStreamer(
        image_folder="data/query/",
        has_gt=True,
        logger=logging.getLogger("%s.DroneImageStreamer" % __name__),  # noqa
    )
    print(len(streamer))

    # Initialize the query processor
    camera_model = CameraModel(
        focal_length=4.5 / 1000,  # 4.5mm
        resolution_height=4056,
        resolution_width=3040,
        hfov_deg=82.9,
    )
    query_processor = QueryProcessor(
        processings=["resize"],
        camera_model=camera_model,
        satellite_resolution=None,
        size=(800,),
    )

    # Initialize the pipeline
    logger = logging.getLogger("%s.Pipeline" % __name__)  # noqa
    logger.setLevel(logging.DEBUG)
    pipeline = Pipeline(
        map_reader=map_reader,
        drone_streamer=streamer,
        detector=superpoint_algorithm,
        matcher=superglue_matcher,
        query_processor=query_processor,
        config=PipelineConfig(),
        # logger=logging.getLogger("%s.Pipeline" % __name__),  # noqa
        logger=logger,
    )
    output_path = "data/output"
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    preds = pipeline.run(
        output_path=output_path,
    )
    metrics = pipeline.compute_metrics(preds)
    pprint(metrics)
