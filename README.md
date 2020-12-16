# HOLOGRAPHIC AUGMENTED REALITY

## Motivation
The onset of the coronavirus pandemic has forced people into their homes and away from the day-to-day human interaction that defined regular life. What has taken its place is an ever-expanding digital world, dominated by virtual classes over Zoom and video calls over Google Meet. Yet even as each of these platforms grows in popularity, it's clear that they are not a replacement for face-to-face social interaction.

For our project, we were inspired by this need for more personal digital interaction among the general public to create a new kind of social interaction via Augmented Reality. Another source of inspiration are the depictions of futuristic digital meetings in popular culture, namely holographic meetings as seen in popular films such as Star Wars. We aimed to recreate a holographic meeting by using our mobile phones as a “looking glass” into an augmented world where we could see and interact with a person on the other side. (Simultaneously, our own phone would scan and project a version of ourselves into the other person’s space.) 

To accomplish our goals, we relied on techniques from ES-143, including camera calibration, camera matrices, plane detection, homographies, projective geometry, and SLAM. We used our coding exercise Box AR as a starting point, and then made modular and incremental changes until we could augment a video taken on our own smartphones with 3D animated holographic models of ourselves. Along the way, we explored additional technologies, including a machine learning model that generates 3D models from a 2D image of a person, Blender, Adobe Mixamo, and various SLAM and SFM packages that will all be discussed.

![Fox](https://github.com/lliu12/holomeetings/blob/main/gifs/compressed_single_fox.gif?raw=true)

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
![Fish](https://github.com/lliu12/holomeetings/blob/main/gifs/compressed_fish_pyramid_every30.gif?raw=true)
