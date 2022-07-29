# clothes-detector

* Make sure you have `git lfs` installed in order to pull in the model weights.
* Install the required modules from `requirements.txt` file (same as that used in official YoloV5 repo).
* Run `detect.py` that takes `test.jpg` image as input inside the `test_images` folder.
* The predictions are saved inside the `output` folder.

Pipeline is as follows:
1. Run a faster RCNN model for detecting person.
2. Crop each person, and run YoloV5 model on each person for detecting clothes.
