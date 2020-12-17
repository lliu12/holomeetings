# HOLOGRAPHIC AUGMENTED REALITY

## Motivation
The onset of the coronavirus pandemic has forced people into their homes and away from the day-to-day human interaction that defined regular life. What has taken its place is an ever-expanding digital world, dominated by virtual classes over Zoom and video calls over Google Meet. Yet even as each of these platforms grows in popularity, it's clear that they are not a replacement for face-to-face social interaction.

For our project, we were inspired by this need for more personal digital interaction among the general public to create a new kind of social interaction via Augmented Reality. Another source of inspiration are the depictions of futuristic digital meetings in popular culture, namely holographic meetings as seen in popular films such as Star Wars. We aimed to recreate a holographic meeting by using our mobile phones as a “looking glass” into an augmented world where we could see and interact with a person on the other side. (Simultaneously, our own phone would scan and project a version of ourselves into the other person’s space.) 

To accomplish our goals, we relied on techniques from ES-143, including camera calibration, camera matrices, plane detection, homographies, projective geometry, and SLAM. We used our coding exercise Box AR as a starting point, and then made modular and incremental changes until we could augment a video taken on our own smartphones with 3D animated holographic models of ourselves. Along the way, we explored additional technologies, including a machine learning model that generates 3D models from a 2D image of a person, Blender, Adobe Mixamo, and various SLAM and SFM packages that will all be discussed.

![Fox](https://github.com/lliu12/holomeetings/blob/main/gifs/compressed_single_fox.gif?raw=true)

See more of our AR results here: https://lliu12.github.io/holomeetings/

## How to Generate a GIF with a Hologram of your own
The main steps are as follows:
1. Run Generate-OBJ.ipynb or download your own .obj file.
2. Process the output (a .obj file) using Adobe Mixamo and/or Blender.
3. Run Generate-GIF.ipynb.

Further details below:

If you would like to project yourself on to your AprilBoard (or someone who you can take a picture of), dowload the notebook Generate-OBJ.ipynb and open it in Google Colab (in order to have access to GPUs). Follow the directions in there: upload a full-body image of the person and it will create a 3D model in .obj format. It is best if the person is in a "T-pose" with their arms in the shape of a T and their legs shoulder-width apart.

If you are downloading your own .obj file, or using the ones provided in this repo, skip step 1. To use the models provided, which include an animated fox (animated-fox.zip), a dancing person (dance.zip), and stationary person (person.obj), first unzip the files if needed, then skip to the last paragraph in this section.

If you would like to animate a stationary 3D model, you can process the model with Adobe Mixamo, which helps you add a skeleton to your model and choose from a wide range of animations. You can download this animation and open it in Blender. 

Animation or no animation, you may want to use Blender to adjust the model. Some frequent adjustments we used were the Decimate tool and the Triangulate tool so that the model is more easily processed by our notebook. Another adjustment is using rotations and translations to make sure that the bottom of the model is level to the x=0, y=0 plane. Finally, make sure to export your animation or stationary model as a Wavefront Object (.obj). If it is an animation, Blender will create a series of .obj files that correspond to each frame of the animation

Once your model is ready, run Generate-GIF.ipynb. Inside, you will need to specify the source video, the Aprilboard type (coarse or fine), and a directory for the .obj files. Currently, the function in the notebook is generateGIF3(), which is meant for animated objects. If you would like to project a stationary object, use generateGIF2(). There is also generateGIF(), which produces a black and white video. (See utils.py for the exact implementation.)

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
