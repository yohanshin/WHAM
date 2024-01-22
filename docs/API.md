## Python API

To use python API of WHAM, please finish the basic installation first ([Installation](INSTALL.md) or [Docker](DOCKER.md)).

If you use Docker environment, please run:

```bash
cd /path/to/WHAM
docker run -it -v .:/code/ --rm yusun9/wham-vitpose-dpvo-cuda11.3-python3.9 python
```

Then you can run wham via python code like 
```bash
from wham_api import WHAM_API
wham_model = WHAM_API()
input_video_path = 'examples/IMG_9732.mov'
results, tracking_results, slam_results = wham_model(input_video_path)
```