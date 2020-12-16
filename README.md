# HOLOGRAPHIC AR MEETINGS

![Fox](https://github.com/lliu12/holomeetings/blob/main/gifs/cropped_single_fox.gif)

## Running SLAM
To work with the playroom dataset, follow these steps:
1. Download the SLAM_playroom.pynb file and upload to Jupyter notebook (or you can run locally as a Python file if you download as a .py file).
2. Download the playroom_lowres.zip and the util.py files and include in the same directory as the Python notebook.
3. Run the notebook. At the end you should see a GIF produced of a pyramid being tracked on one of the Aprilboards in the images. 

To work with one of our video datasets follow these steps:
1. Download the SLAM_video.pynb file and upload to Jupyter notebook (or you can run locally as a Python file if you download as a .py file).
2. Download the SLAM_videos.zip and the util.py files and include in the same directory as the Python notebook.
3. Run the notebook. At the end you should see a GIF produced of a pyramid being tracked on sampled frames from a video (which includes a painting of a fish). 

To work with your own dataset follow these steps:
1. Download the SLAM_video.pynb file and upload to Jupyter notebook (or you can run locally as a Python file if you download as a .py file).
2. Download the SLAM_videos.zip and the AprilBoards.pickle file.
3. Include this file with one calibration video and one video of what you would like to place the pyramid object onto into a zip file called SLAM_videos.zip and put this zip file in the same working directory as the notebook. This requires taking a calibration video of an AprilBoard, which is most easily done using any smartphone (make sure to turn off auto exposure/focus) and then using the same exposure/focus take another video of some flat surface in an environment - ideally with some patterns/distinguishing features on it.
4. Change the fourth code block to include the path to your calibration video and sample video.
5. Run the notebook.

The output GIF of the fish dataset should look like:
![Fish](https://github.com/lliu12/holomeetings/blob/main/fish_pyramid_slow.gif)
