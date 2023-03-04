from stream.stream_processor import StreamProcessor

if __name__ == '__main__':

    config = {
        'models_dir': '/Users/mustafas/.insightface/models/buffalo_l/',
        'det_model_file': 'det_10g.onnx',
        'det_size': (640, 640),
        'landmark_model_file': '1k3d68.onnx'
    }

    streamer = StreamProcessor(config)
    streamer.run()
